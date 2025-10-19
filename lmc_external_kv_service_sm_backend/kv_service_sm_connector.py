# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.

from __future__ import annotations

# Standard
from dataclasses import dataclass
from enum import IntEnum, auto
from multiprocessing import shared_memory
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple
import asyncio
import threading
import time
import urllib.parse

# Third Party
import aiohttp
import torch

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.connector.base_connector import RemoteConnector
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend
from lmcache.utils import CacheEngineKey

from .kv_service_sm_config import KVServiceSMConfig


logger = init_logger(__name__)


@dataclass
class LeaseInfo:
    """Lease returned by the KVServiceSM daemon."""

    lease_id: str
    offsets: List[Tuple[int, int]]


class _TTLCache:
    """Simple TTL-bound cache used for leases and recent PUT tracking."""

    def __init__(self, ttl_s: float, max_size: int) -> None:
        self.ttl_s = float(ttl_s)
        self.max_size = max_size
        self._store: Dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Optional[Any]:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, cached_at = entry
        if time.monotonic() - cached_at > self.ttl_s:
            self._store.pop(key, None)
            return None
        return value

    def contains(self, key: str) -> bool:
        return self.get(key) is not None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = (value, time.monotonic())
        self._evict_if_needed()

    def take(self, key: str) -> Optional[Any]:
        value = self.get(key)
        if value is None:
            return None
        self._store.pop(key, None)
        return value

    def clear(self) -> None:
        self._store.clear()

    def _evict_if_needed(self) -> None:
        if self.max_size <= 0 or len(self._store) <= self.max_size:
            return
        num_to_evict = self.max_size // 10
        if num_to_evict <= 0:
            num_to_evict = len(self._store) - self.max_size
        oldest = sorted(self._store.items(), key=lambda item: item[1][1])
        for key, _ in oldest[:num_to_evict]:
            self._store.pop(key, None)


class _KVSMTaskPriority(IntEnum):
    """Priority buckets for control-plane scheduling."""

    LEASE = 0
    PREFETCH = auto()
    PUT = auto()


class KVServiceSMConnector(RemoteConnector):
    """
    Remote connector that talks to a KVServiceSM daemon.

    Control plane: HTTP (leases and PUT uploads)
    Data plane (GET): local OS shared memory using offsets described in the lease.

    Note: This connector allocates destination buffers via LocalCPUBackend.
    For compatibility, use a "naive" remote serde (no compression) so that the
    serialized bytes length matches the tensor shape/dtype/format.
    """

    def __init__(
        self,
        *,
        loop: asyncio.AbstractEventLoop,
        local_cpu_backend: LocalCPUBackend,
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
    ) -> None:
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        self.config = config
        self.metadata = metadata

        extra_config = getattr(config, "extra_config", None) or {}
        self.kv_config = KVServiceSMConfig.from_extra_config(extra_config)

        # HTTP session state
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._http_session_lock = asyncio.Lock()

        # Concurrency gates
        self._control_inflight = asyncio.Semaphore(
            max(1, self.kv_config.control_max_connections_per_host)
        )
        self._put_inflight = asyncio.Semaphore(
            max(1, self.kv_config.put_max_connections_per_host)
        )
        prefetch_limit = max(1, int(self.kv_config.control_max_connections_per_host))
        self._prefetch_gate = asyncio.Semaphore(prefetch_limit)

        # Prioritized executor
        total_workers = max(
            1,
            self.kv_config.control_max_connections_per_host
            + self.kv_config.put_max_connections_per_host,
        )
        self._executor = AsyncPQExecutor(loop, max_workers=total_workers)

        # Shared memory mapping (lazy)
        self._shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self._shared_memory_map: Optional[memoryview] = None
        self._shared_memory_lock = threading.Lock()

        # Caches
        lease_cache_ttl_s = float(self.kv_config.lease_ttl_s)
        bridge_ms = extra_config.get("kv_service_sm_client_bridge_ttl_ms")
        if bridge_ms is not None:
            try:
                lease_cache_ttl_s = max(0.0, float(bridge_ms) / 1000.0)
            except Exception:
                pass
        self._lease_cache = _TTLCache(
            ttl_s=lease_cache_ttl_s,
            max_size=self.kv_config.lease_cache_max_size,
        )
        self._recent_puts = _TTLCache(
            ttl_s=float(self.kv_config.put_cache_ttl_s),
            max_size=self.kv_config.put_cache_max_size,
        )

        # Diagnostics
        if config is not None and getattr(config, "remote_serde", "naive") != "naive":
            logger.warning(
                "KVServiceSMConnector is best used with remote_serde='naive' for size"
                " consistency between metadata shape and byte payload."
            )

    # --------------
    # RemoteConnector API
    # --------------
    async def exists(self, key: CacheEngineKey) -> bool:
        lease = self._cache_peek_lease(key)
        if lease is not None:
            return True

        lease = await self._executor.submit_job(
            self._acquire_lease, key=key, priority=_KVSMTaskPriority.LEASE
        )
        if lease is None:
            return False
        self._cache_put_lease(key, lease)
        return True

    def exists_sync(self, key: CacheEngineKey) -> bool:
        fut = asyncio.run_coroutine_threadsafe(self.exists(key), self.loop)
        return bool(fut.result())

    async def get(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        lease = self._cache_take_lease(key)
        if lease is None:
            lease = await self._executor.submit_job(
                self._acquire_lease, key=key, priority=_KVSMTaskPriority.LEASE
            )
        if lease is None:
            return None
        return self._read_from_shared_memory(key, lease)

    def support_batched_get(self) -> bool:
        return True

    async def batched_get(self, keys: List[CacheEngineKey]) -> List[Optional[MemoryObj]]:
        results: List[Optional[MemoryObj]] = []
        # Simple sequential inside a prefetch gate to avoid saturating
        async with self._prefetch_gate:
            for key in keys:
                results.append(await self.get(key))
        return results

    def support_batched_put(self) -> bool:
        return True

    async def batched_put(self, keys: List[CacheEngineKey], memory_objs: List[MemoryObj]):
        # Submit individual PUTs concurrently via executor; gather
        await asyncio.gather(
            *(
                self.put(key, mem)
                for key, mem in zip(keys, memory_objs, strict=False)
            )
        )

    async def put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        # Fast-paths to avoid redundant PUTs
        if self._cache_peek_lease(key) is not None:
            return
        if self._check_recent_put(key):
            return

        # Serialize header + KV bytes, then stream via HTTP
        payload_len, payload_iter = self._build_put_stream(memory_obj)
        url = self._build_kv_url(key)
        await self._http_request(
            "PUT",
            url,
            data=payload_iter,
            timeout=self.kv_config.put_timeout_ms / 1000.0,
            gate=self._put_inflight,
            headers={"Content-Length": str(payload_len)},
        )
        self._mark_recent_put(key)

    def support_batched_async_contains(self) -> bool:
        return True

    async def batched_async_contains(
        self, lookup_id: str, keys: List[CacheEngineKey], pin: bool = False
    ) -> int:
        if not keys:
            return 0
        total = 0
        for key in keys:
            lease = self._cache_peek_lease(key)
            if lease is None:
                try:
                    lease = await self._executor.submit_job(
                        self._acquire_lease,
                        key=key,
                        priority=_KVSMTaskPriority.LEASE,
                    )
                except Exception as exc:
                    logger.debug(f"Lease acquisition failed for {key}: {exc}")
                    lease = None
            if lease is None:
                return total
            self._cache_put_lease(key, lease)
            total += 1
        return total

    def support_batched_get_non_blocking(self) -> bool:
        return True

    async def batched_get_non_blocking(
        self, lookup_id: str, keys: List[CacheEngineKey]
    ) -> List[MemoryObj]:
        async with self._prefetch_gate:
            results: List[MemoryObj] = []
            for key in keys:
                mem = await self.get(key)
                if mem is not None:
                    results.append(mem)
            return results

    async def list(self) -> List[str]:
        # Not supported by the KV service; return empty list
        return []

    def remove_sync(self, key: CacheEngineKey) -> bool:
        # Current service does not expose deletion; treat as success
        return True

    async def close(self):
        try:
            await self._executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            if self._http_session is not None:
                try:
                    await self._http_session.close()
                except Exception:
                    pass
                self._http_session = None
        finally:
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

    # --------------
    # Internals
    # --------------
    def _cache_put_lease(self, key: CacheEngineKey, lease: LeaseInfo) -> None:
        self._lease_cache.put(key.to_string(), lease)

    def _cache_take_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        return self._lease_cache.take(key.to_string())

    def _cache_peek_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        return self._lease_cache.get(key.to_string())

    def _check_recent_put(self, key: CacheEngineKey) -> bool:
        return self._recent_puts.contains(key.to_string())

    def _mark_recent_put(self, key: CacheEngineKey) -> None:
        self._recent_puts.put(key.to_string(), None)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        url = self._build_lease_url(key)
        params = {
            "timeout_ms": self.kv_config.lease_timeout_ms,
            "ttl_s": self.kv_config.lease_ttl_s,
        }
        response = await self._http_request(
            "POST",
            url,
            params=params,
            timeout=self.kv_config.lease_timeout_ms / 1000.0,
            gate=self._control_inflight,
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
        )

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session and not self._http_session.closed:
            return self._http_session
        async with self._http_session_lock:
            if self._http_session and not self._http_session.closed:
                return self._http_session
            connector = aiohttp.TCPConnector(
                limit=max(
                    1,
                    self.kv_config.control_max_connections
                    + self.kv_config.put_max_connections,
                ),
                limit_per_host=max(
                    1,
                    self.kv_config.control_max_connections_per_host
                    + self.kv_config.put_max_connections_per_host,
                ),
                ttl_dns_cache=self.kv_config.dns_ttl,
                keepalive_timeout=self.kv_config.connection_keepalive,
                enable_cleanup_closed=True,
            )
            timeout = aiohttp.ClientTimeout(
                total=30,
                connect=self.kv_config.http_connect_timeout_ms / 1000.0,
                sock_read=self.kv_config.http_read_timeout_ms / 1000.0,
            )
            self._http_session = aiohttp.ClientSession(
                connector=connector,
                timeout=timeout,
                headers={"User-Agent": "LMCache-KVServiceSM-Connector"},
            )
            return self._http_session

    async def _http_request(
        self,
        method: str,
        url: str,
        *,
        data=None,
        params: Optional[dict] = None,
        timeout: float = 5.0,
        gate: Optional[asyncio.Semaphore] = None,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        if gate is None:
            return await self._http_request_common(
                method,
                url,
                data=data,
                params=params,
                timeout=timeout,
                headers=headers,
            )
        async with gate:
            return await self._http_request_common(
                method,
                url,
                data=data,
                params=params,
                timeout=timeout,
                headers=headers,
            )

    async def _http_request_common(
        self,
        method: str,
        url: str,
        *,
        data=None,
        params: Optional[dict] = None,
        timeout: float,
        headers: Optional[dict[str, str]] = None,
    ) -> Optional[dict]:
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.request(
                method,
                url,
                data=data,
                params=params,
                timeout=request_timeout,
                headers=headers,
            ) as response:
                content_type = response.headers.get("Content-Type", "")
                body_json = None
                if content_type.startswith("application/json"):
                    try:
                        body_json = await response.json()
                    except Exception:
                        body_json = None
                return {"status": response.status, "json": body_json}
        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} timeout talking to {url}")
            return None
        except aiohttp.ClientError as exc:
            logger.error(f"HTTP {method} client error for {url}: {exc}")
            return None
        except Exception as exc:
            logger.error(f"HTTP {method} request failed for {url}: {exc}")
            return None

    def _ensure_shared_memory(self) -> bool:
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

    def _read_from_shared_memory(
        self,
        key: CacheEngineKey,
        lease_info: LeaseInfo,
    ) -> Optional[MemoryObj]:
        if not self._ensure_shared_memory() or self._shared_memory_map is None:
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

        # 3) Allocate destination using LocalCPUBackend
        memory_obj = self.local_cpu_backend.allocate(
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

    def _build_put_stream(
        self,
        memory_obj: MemoryObj,
    ) -> tuple[int, AsyncIterator[object]]:
        """Build streaming payload (header + KV bytes) using a detached buffer."""
        kv_bytes = bytes(memory_obj.byte_array)
        kv_view = memoryview(kv_bytes)
        if getattr(kv_view, "format", None) == "<B":
            kv_view = kv_view.cast("B")

        kv_len = len(kv_view)
        shape = list(memory_obj.get_shape())
        padded_shape = (shape + [0] * 4)[:4]
        metadata = RemoteMetadata(
            kv_len,
            torch.Size(padded_shape),
            memory_obj.get_dtype(),
            memory_obj.get_memory_format(),
        )

        header = bytearray(4 * 7)
        metadata.serialize_into(header)
        header_bytes = bytes(header)
        total_len = len(header_bytes) + kv_len
        chunk_size = max(1, int(self.kv_config.put_stream_chunk_bytes))

        async def generator() -> AsyncIterator[object]:
            yield header_bytes
            offset = 0
            while offset < kv_len:
                next_offset = min(kv_len, offset + chunk_size)
                yield kv_view[offset:next_offset]
                offset = next_offset

        return total_len, generator()

    def _key_to_string(self, key: CacheEngineKey) -> str:
        return urllib.parse.quote(key.to_string(), safe="")

    def _build_kv_url(self, key: CacheEngineKey) -> str:
        return f"{self.kv_config.base_url}/v1/kv/{self.kv_config.bucket_name}/{self._key_to_string(key)}"

    def _build_lease_url(self, key: CacheEngineKey) -> str:
        return f"{self._build_kv_url(key)}/leases"

