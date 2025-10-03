# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

# Standard
from concurrent.futures import Future
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio
import time
import threading
import urllib.parse

# Third Party
import aiohttp
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.abstract_backend import ConfigurableStorageBackendInterface

from .kv_service_sm_config import KVServiceSMConfig

if TYPE_CHECKING:
    from lmcache.v1.storage_backend import LocalCPUBackend

logger = init_logger(__name__)


@dataclass
class LeaseInfo:
    """Lease returned by the KVServiceSM daemon."""

    lease_id: str
    offsets: List[Tuple[int, int]]
    total_size: int


class KVServiceSMBackend(ConfigurableStorageBackendInterface):
    """Simplified configurable backend that talks to KVServiceSM over HTTP."""

    def __init__(
        self,
        dst_device: str = "cuda",
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(dst_device, config, metadata, local_cpu_backend, loop)

        if config is None:
            raise ValueError("Config is required for KVServiceSMBackend")
        if local_cpu_backend is None:
            raise ValueError("Local CPU backend is required for KVServiceSMBackend")
        if loop is None:
            raise ValueError("Event loop is required for KVServiceSMBackend")

        self.config = config
        self.metadata = metadata
        self.local_cpu_backend = local_cpu_backend
        self.loop = loop

        extra_config = getattr(config, "extra_config", None) or {}
        self.kv_config = KVServiceSMConfig.from_extra_config(extra_config)
        self._get_timeout_ms = extra_config.get(
            "kv_service_sm_get_timeout_ms",
            max(2000, self.kv_config.http_read_timeout_ms),
        )

        # Memory management helpers
        self.memory_allocator = local_cpu_backend.get_memory_allocator()

        # Shared resources
        self._control_session: Optional[aiohttp.ClientSession] = None
        self._control_session_lock = asyncio.Lock()
        self._data_session: Optional[aiohttp.ClientSession] = None
        self._data_session_lock = asyncio.Lock()
        self._shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self._shared_memory_map: Optional[memoryview] = None
        self._shared_memory_lock = threading.Lock()

        # PUT concurrency & tracking
        self._put_sema = asyncio.Semaphore(max(1, self.kv_config.max_concurrent_puts))
        self._put_lock = threading.Lock()
        self._put_futures: Dict[str, Future] = {}

        self._closed = False

        # Short-lived lease cache so contains() can be followed by get() without
        # re-acquiring the lease. TTL 5 seconds by default.
        self._lease_cache: Dict[str, tuple[LeaseInfo, float]] = {}
        self._lease_cache_lock = asyncio.Lock()
        self._lease_cache_ttl_ms = 5000

    # ------------------------------------------------------------------
    # Public API expected by the base interface
    # ------------------------------------------------------------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:  # noqa: ARG002
        """Return True when KVServiceSM reports the key exists."""
        if self._closed:
            return False
        try:
            future = asyncio.run_coroutine_threadsafe(
                self._contains_async(key), self.loop
            )
            timeout_s = (self.kv_config.lease_timeout_ms + 200) / 1000.0
            return future.result(timeout=timeout_s)
        except Exception as exc:  # pragma: no cover - best effort logging
            logger.error(f"contains() failed for key {key}: {exc}")
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        key_str = key.to_string()
        with self._put_lock:
            return key_str in self._put_futures

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,  # noqa: D417, ARG002 - reserved for future use
    ) -> Optional[List[Future]]:
        if self._closed:
            return None
        futures: List[Future] = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(key, memory_obj)
            if future is not None:
                futures.append(future)
        return futures or None

    @_lmcache_nvtx_annotate
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        if self._closed:
            return None

        key_str = key.to_string()
        with self._put_lock:
            existing = self._put_futures.get(key_str)
            if existing is not None:
                memory_obj.ref_count_down()
                return existing

            memory_obj.ref_count_up()
            future = asyncio.run_coroutine_threadsafe(
                self._put_once(key, memory_obj), self.loop
            )
            self._put_futures[key_str] = future
            return future

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        if self._closed:
            return None
        return asyncio.run_coroutine_threadsafe(
            self._get_memory_obj(key), self.loop
        )

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        if self._closed:
            return None
        future = asyncio.run_coroutine_threadsafe(
            self._get_memory_obj(key), self.loop
        )
        try:
            return future.result(timeout=self._get_timeout_ms / 1000.0)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error(f"get_blocking() failed for key {key}: {exc}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        return self.submit_prefetch_task(key)

    def pin(self, key: CacheEngineKey) -> bool:  # noqa: ARG002 - no-op
        return True

    def unpin(self, key: CacheEngineKey) -> bool:  # noqa: ARG002 - no-op
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:  # noqa: ARG002
        # Backend does not currently expose delete; treat as best-effort success.
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        try:
            with self._put_lock:
                self._put_futures.clear()

            # Close HTTP sessions
            if self._control_session is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._control_session.close(), self.loop
                    ).result(timeout=5)
                except Exception:
                    pass
                self._control_session = None

            if self._data_session is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self._data_session.close(), self.loop
                    ).result(timeout=5)
                except Exception:
                    pass
                self._data_session = None

            # Release shared memory handles
            with self._shared_memory_lock:
                if self._shared_memory_map is not None:
                    try:
                        self._shared_memory_map.release()
                    except Exception:
                        pass
                    self._shared_memory_map = None
                if self._shared_memory_obj is not None:
                    try:
                        self._shared_memory_obj.close()
                    except Exception:
                        pass
                    self._shared_memory_obj = None
        except Exception as exc:  # pragma: no cover - shutdown should not raise
            logger.error(f"Error while closing KVServiceSMBackend: {exc}")

    def get_allocator_backend(self):
        return self.local_cpu_backend

    async def async_batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> None:
        futures = self.batched_submit_put_task(keys, memory_objs, transfer_spec)
        if futures:
            await asyncio.gather(*[asyncio.wrap_future(f) for f in futures], return_exceptions=True)

    async def batched_async_contains(
        self,
        lookup_id: str,  # noqa: ARG002 - reserved for compatibility
        keys: List[CacheEngineKey],
        pin: bool = False,  # noqa: ARG002
    ) -> int:
        if not keys:
            return 0

        # Process in concurrent windows to reduce total latency when many
        # leading keys exist. Early stop at first miss.
        window = max(1, min(32, self.kv_config.control_max_connections_per_host))
        total = 0
        i = 0
        while i < len(keys):
            batch = keys[i : i + window]
            leases: list[Optional[LeaseInfo]] = [None] * len(batch)
            missing_indices: list[int] = []
            missing_tasks = []

            for idx, key in enumerate(batch):
                cached = await self._cache_peek_lease(key)
                if cached is not None:
                    leases[idx] = cached
                    continue
                missing_indices.append(idx)
                missing_tasks.append(asyncio.create_task(self._acquire_lease(key)))

            if missing_tasks:
                results = await asyncio.gather(*missing_tasks, return_exceptions=True)
                for idx, result in zip(missing_indices, results, strict=False):
                    leases[idx] = None if isinstance(result, Exception) else result

            for key, lease in zip(batch, leases, strict=False):
                if lease is None:
                    return total
                # Cache lease for a short time; get() will reuse and release it
                await self._cache_put_lease(key, lease)
                total += 1
            i += window
        return total

    async def batched_get_non_blocking(
        self,
        lookup_id: str,  # noqa: ARG002 - reserved for compatibility
        keys: List[CacheEngineKey],
        transfer_spec=None,
    ) -> List[MemoryObj]:
        results: List[MemoryObj] = []
        for key in keys:
            memory_obj = await self._get_memory_obj(key)
            if memory_obj is not None:
                results.append(memory_obj)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _contains_async(self, key: CacheEngineKey) -> bool:
        # Fast path: if a valid cached lease exists, treat as contained.
        lease = await self._cache_peek_lease(key)
        if lease is not None:
            return True

        lease = await self._acquire_lease(key)
        if lease is None:
            return False
        # Cache lease; do not release now. A following get() will consume it.
        await self._cache_put_lease(key, lease)
        return True

    async def _put_once(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        key_str = key.to_string()
        acquired = False
        try:
            # Fast path: if we already have a valid cached lease for this key,
            # we know it exists. Skip the PUT without any HTTP call.
            cached = await self._cache_peek_lease(key)
            if cached is not None:
                return

            await self._put_sema.acquire()
            acquired = True

            payload = await asyncio.to_thread(self._memory_obj_to_bytes, memory_obj)
            url = self._build_kv_url(key)
            response = await self._data_http_request(
                "PUT",
                url,
                data=payload,
                timeout=self.kv_config.put_timeout_ms / 1000.0,
            )
            if not response or response.get("status") != 200:
                status = None if response is None else response.get("status")
                logger.error(f"PUT failed for key {key}: HTTP {status}")
        except Exception as exc:  # pragma: no cover - log and continue
            logger.error(f"PUT exception for key {key}: {exc}")
        finally:
            if acquired:
                self._put_sema.release()
            memory_obj.ref_count_down()
            with self._put_lock:
                self._put_futures.pop(key_str, None)

    async def _get_memory_obj(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        lease = await self._cache_take_lease(key)
        if lease is None:
            lease = await self._acquire_lease(key)
        if lease is None:
            return None
        try:
            return await self._read_from_shared_memory(key, lease)
        finally:
            await self._release_lease(lease.lease_id)

    async def _cache_put_lease(self, key: CacheEngineKey, lease: LeaseInfo) -> None:
        key_str = key.to_string()
        expiry = time.time() + (self._lease_cache_ttl_ms / 1000.0)
        async with self._lease_cache_lock:
            self._lease_cache[key_str] = (lease, expiry)
        # Schedule expiry; if not consumed by then, release it.
        asyncio.create_task(self._expire_lease_later(key_str, lease.lease_id, expiry))

    async def _cache_take_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        key_str = key.to_string()
        async with self._lease_cache_lock:
            entry = self._lease_cache.pop(key_str, None)
        if entry is None:
            return None
        lease, expiry = entry
        if time.time() > expiry:
            # Already expired; make sure it is released asynchronously
            asyncio.create_task(self._release_lease(lease.lease_id))
            return None
        return lease

    async def _cache_peek_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Return a cached lease without consuming it, if still valid.

        If the cached lease is expired, remove it and schedule a release.
        """
        key_str = key.to_string()
        stale_lease_id: Optional[str] = None
        async with self._lease_cache_lock:
            entry = self._lease_cache.get(key_str)
            if entry is None:
                return None
            lease, expiry = entry
            if time.time() > expiry:
                self._lease_cache.pop(key_str, None)
                stale_lease_id = lease.lease_id
            else:
                return lease

        if stale_lease_id is not None:
            asyncio.create_task(self._release_lease(stale_lease_id))
        return None

    async def _expire_lease_later(self, key_str: str, lease_id: str, expiry: float) -> None:
        try:
            delay = max(0.0, expiry - time.time())
            if delay > 0:
                await asyncio.sleep(delay)
            async with self._lease_cache_lock:
                # If still cached and expired, pop and release
                entry = self._lease_cache.get(key_str)
                if entry is None:
                    return
                cached_lease, cached_expiry = entry
                if time.time() >= cached_expiry:
                    self._lease_cache.pop(key_str, None)
                else:
                    return
        finally:
            # Release regardless; server treats duplicate/late releases as harmless
            await self._release_lease(lease_id)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        url = f"{self.kv_config.base_url}/v1/kv/{self.kv_config.bucket_name}/{self._key_to_string(key)}/leases"
        params = {"timeout_ms": self.kv_config.lease_timeout_ms}
        response = await self._control_http_request(
            "POST",
            url,
            params=params,
            timeout=self.kv_config.lease_timeout_ms / 1000.0,
        )
        if not response or response.get("status") != 200 or not response.get("json"):
            return None
        data = response["json"]
        offsets = [(chunk["offset"], chunk["len"]) for chunk in data.get("offsets", [])]
        if not offsets:
            return None
        return LeaseInfo(
            lease_id=data["id"],
            offsets=offsets,
            total_size=sum(length for _, length in offsets),
        )

    async def _release_lease(self, lease_id: str) -> bool:
        url = f"{self.kv_config.base_url}/v1/leases/{lease_id}/release"
        response = await self._control_http_request(
            "POST",
            url,
            timeout=self.kv_config.release_timeout_ms / 1000.0,
        )
        return bool(response) and response.get("status") in (200, 404)
    async def _ensure_control_session(self) -> aiohttp.ClientSession:
        if self._control_session and not self._control_session.closed:
            return self._control_session
        async with self._control_session_lock:
            if self._control_session and not self._control_session.closed:
                return self._control_session
            connector = aiohttp.TCPConnector(
                limit=self.kv_config.control_max_connections,
                limit_per_host=self.kv_config.control_max_connections_per_host,
                ttl_dns_cache=self.kv_config.dns_ttl,
                keepalive_timeout=self.kv_config.connection_keepalive,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=self.kv_config.http_connect_timeout_ms / 1000.0,
                sock_read=self.kv_config.http_read_timeout_ms / 1000.0,
            )
            self._control_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "LMCache-KVServiceSM-Control"},
            )
            return self._control_session

    async def _ensure_data_session(self) -> aiohttp.ClientSession:
        if self._data_session and not self._data_session.closed:
            return self._data_session
        async with self._data_session_lock:
            if self._data_session and not self._data_session.closed:
                return self._data_session
            connector = aiohttp.TCPConnector(
                limit=self.kv_config.put_max_connections,
                limit_per_host=self.kv_config.put_max_connections_per_host,
                ttl_dns_cache=self.kv_config.dns_ttl,
                keepalive_timeout=self.kv_config.connection_keepalive,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=self.kv_config.http_connect_timeout_ms / 1000.0,
                sock_read=self.kv_config.http_read_timeout_ms / 1000.0,
            )
            self._data_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "LMCache-KVServiceSM-Put"},
            )
            return self._data_session

    async def _control_http_request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[bytes] = None,
        params: Optional[dict] = None,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        return await self._http_request_common(
            self._ensure_control_session, method, url, data=data, params=params, timeout=timeout
        )

    async def _data_http_request(
        self,
        method: str,
        url: str,
        *,
        data: Optional[bytes] = None,
        params: Optional[dict] = None,
        timeout: float = 5.0,
    ) -> Optional[dict]:
        return await self._http_request_common(
            self._ensure_data_session, method, url, data=data, params=params, timeout=timeout
        )

    async def _http_request_common(
        self,
        session_getter,
        method: str,
        url: str,
        *,
        data: Optional[bytes] = None,
        params: Optional[dict] = None,
        timeout: float,
    ) -> Optional[dict]:
        try:
            session = await session_getter()
            request_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.request(
                method,
                url,
                data=data,
                params=params,
                timeout=request_timeout,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                body_json = None
                if content_type.startswith("application/json"):
                    try:
                        body_json = await response.json()
                    except Exception:
                        body_json = None
                body_data = None
                if method in {"PUT", "POST"}:
                    try:
                        body_data = await response.read()
                    except Exception:
                        body_data = None
                return {"status": response.status, "json": body_json, "data": body_data}
        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} timeout talking to {url}")
            return None
        except aiohttp.ClientError as exc:
            logger.error(f"HTTP {method} client error for {url}: {exc}")
            return None
        except Exception as exc:  # pragma: no cover - unexpected failures
            logger.error(f"HTTP {method} request failed for {url}: {exc}")
            return None

    async def _ensure_shared_memory(self) -> bool:
        with self._shared_memory_lock:
            if self._shared_memory_map is not None:
                return True
            name = self.kv_config.shared_memory_name
            if not name:
                logger.error("Shared memory name is not configured")
                return False
            try:
                self._shared_memory_obj = shared_memory.SharedMemory(name=name, create=False)
                self._shared_memory_map = memoryview(self._shared_memory_obj.buf)
                return True
            except FileNotFoundError:
                logger.error(
                    "Shared memory segment '%s' not found. Is KVServiceSM daemon running?",
                    name,
                )
                return False
            except Exception as exc:
                logger.error(f"Failed to open shared memory '{name}': {exc}")
                return False

    async def _read_from_shared_memory(
        self,
        key: CacheEngineKey,
        lease_info: LeaseInfo,
    ) -> Optional[MemoryObj]:
        if not await self._ensure_shared_memory() or self._shared_memory_map is None:
            return None

        # RemoteMetadata header is 7 int32 = 28 bytes
        metadata_size = 4 * 7

        # 1) Read only the header across offsets
        header = bytearray(metadata_size)
        filled = 0
        for off, ln in lease_info.offsets:
            if filled >= metadata_size:
                break
            take = min(ln, metadata_size - filled)
            header[filled : filled + take] = self._shared_memory_map[off : off + take]
            filled += take

        if filled < metadata_size:
            logger.error("Lease payload too small for metadata for key %s", key)
            return None

        # 2) Parse metadata
        metadata = RemoteMetadata.deserialize(header)
        payload_len = metadata.length
        if payload_len <= 0:
            logger.error("Invalid payload length %d for key %s", payload_len, key)
            return None

        # 3) Allocate destination
        memory_obj = self.memory_allocator.allocate(
            metadata.shape,
            metadata.dtype,
            metadata.fmt,
        )
        if memory_obj is None:
            logger.error("Failed to allocate memory for key %s", key)
            return None

        view = (
            memory_obj.byte_array
            if isinstance(memory_obj.byte_array, memoryview)
            else memoryview(memory_obj.byte_array)
        )
        if getattr(view, "format", None) == "<B":
            view = view.cast("B")

        # 4) Direct copy from SHM to destination, skipping header bytes
        copied = 0
        bytes_to_skip = metadata_size
        for off, ln in lease_info.offsets:
            if copied >= payload_len:
                break
            # Skip header bytes first
            if bytes_to_skip > 0:
                if ln <= bytes_to_skip:
                    bytes_to_skip -= ln
                    continue
                off += bytes_to_skip
                ln -= bytes_to_skip
                bytes_to_skip = 0
            if ln <= 0:
                continue
            take = min(payload_len - copied, ln)
            view[copied : copied + take] = self._shared_memory_map[off : off + take]
            copied += take

        if copied != payload_len:
            logger.error("Data size mismatch: expected %d, got %d", payload_len, copied)
            return None

        return memory_obj

    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        kv_bytes = memory_obj.byte_array
        shape = list(memory_obj.get_shape())
        padded_shape = (shape + [0] * 4)[:4]
        metadata = RemoteMetadata(
            len(kv_bytes),
            torch.Size(padded_shape),
            memory_obj.get_dtype(),
            memory_obj.get_memory_format(),
        )
        return metadata.serialize() + kv_bytes

    def _key_to_string(self, key: CacheEngineKey) -> str:
        return urllib.parse.quote(key.to_string(), safe="")

    def _build_kv_url(self, key: CacheEngineKey) -> str:
        return f"{self.kv_config.base_url}/v1/kv/{self.kv_config.bucket_name}/{self._key_to_string(key)}"
