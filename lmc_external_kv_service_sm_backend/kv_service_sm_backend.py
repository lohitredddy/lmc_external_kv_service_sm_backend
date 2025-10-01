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

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
import asyncio
import logging
import threading
import time
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
from lmcache.v1.storage_backend.job_executor.pq_executor import AsyncPQExecutor

# Local imports
from .cache import KeyCache, LeaseInfo
from .kv_service_sm_config import KVServiceSMConfig
from .utils import LeaseManager, OperationMetrics, OperationPriority

if TYPE_CHECKING:
    from lmcache.v1.storage_backend import LocalCPUBackend

logger = init_logger(__name__)


class KVServiceSMBackend(ConfigurableStorageBackendInterface):
    """
    High-performance storage backend for KVServiceSM daemon.

    Critical fixes included:
    - Lazy SHM (re)mapping on GET path
    - Robust JSON detection
    - True single-copy zero-copy reads even when header spans blocks
    - Soft-fail shm init in initialize()
    - Public-op auto-initialize guard
    - Timeout for get_blocking()
    - Graceful maintenance task cancellation
    """

    def __init__(
        self,
        dst_device: str = "cuda",
        config: Optional[LMCacheEngineConfig] = None,
        metadata: Optional[LMCacheEngineMetadata] = None,
        local_cpu_backend: Optional["LocalCPUBackend"] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(dst_device, config, metadata, local_cpu_backend, loop)

        if config is None:
            raise ValueError("Config is required for KVServiceSMBackend")
        if loop is None:
            raise ValueError("Event loop is required for KVServiceSMBackend")
        if local_cpu_backend is None:
            raise ValueError("Local CPU backend is required for KVServiceSMBackend")

        extra_config = getattr(config, "extra_config", None) or {}
        self.kv_config = KVServiceSMConfig.from_extra_config(extra_config)

        # reasonable default if config class doesn't provide it
        if not hasattr(self.kv_config, "get_timeout_ms"):
            self.kv_config.get_timeout_ms = max(2000, self.kv_config.http_read_timeout_ms)

        self.memory_allocator = local_cpu_backend.get_memory_allocator()

        # Core components (lazy init)
        self._http_session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()
        self._shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self._shared_memory_map: Optional[memoryview] = None
        self._shared_memory_lock = threading.Lock()

        # Thread pool for CPU-bound work
        self._executor: Optional[ThreadPoolExecutor] = None

        # Priority queue executor
        self._pq_executor: Optional[AsyncPQExecutor] = None

        # Caching
        self._key_cache: Optional[KeyCache] = None
        if self.kv_config.enable_key_cache:
            self._key_cache = KeyCache(
                max_entries=self.kv_config.max_cache_entries,
                ttl=self.kv_config.cache_ttl_seconds,
            )

        # Resource tracking
        self._inflight_semaphore = asyncio.Semaphore(self.kv_config.max_inflight_requests)
        self._lease_manager = LeaseManager(cache_ttl=30.0)

        # PUT task tracking
        self._put_lock = threading.Lock()
        self._put_tasks: set[CacheEngineKey] = set()

        # Metrics & state
        self._metrics = OperationMetrics()
        self._initialized = False
        self._closed = False
        self._maintenance_task: Optional[asyncio.Task] = None

        logger.info(
            f"KVServiceSMBackend new 448 created with URL: {self.kv_config.base_url}, "
            f"bucket: {self.kv_config.bucket_name}, "
            f"max_connections: {self.kv_config.max_connections}"
        )

    def __str__(self):
        return self.__class__.__name__

    async def initialize(self):
        """Post-construction initialization of async components."""
        if self._initialized:
            return

        try:
            await self._ensure_http_session()

            # SOFT-Fail shm init at startup; lazy retry will happen on GET
            ok = await self._ensure_shared_memory()
            if not ok:
                logger.warning(
                    "Shared memory not available at init; will retry lazily on first GET."
                )

            self._executor = ThreadPoolExecutor(
                max_workers=self.kv_config.serialization_threads,
                thread_name_prefix="kvservice-sm",
            )
            self._pq_executor = AsyncPQExecutor(self.loop, max_workers=self.kv_config.pq_executor_workers)

            self._maintenance_task = asyncio.create_task(self._periodic_maintenance())

            self._initialized = True
            logger.info("KVServiceSMBackend initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize KVServiceSMBackend: {e}")
            await self.close()
            raise

    # ---------- helper: ensure initialized if caller forgot ----------
    def _ensure_initialized_blocking(self):
        if not self._initialized and not self._closed:
            fut = asyncio.run_coroutine_threadsafe(self.initialize(), self.loop)
            # Small timeout to avoid deadlocks
            fut.result(timeout=5)

    # ============== Required Interface Methods ==============

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in KVServiceSM cache - highest priority."""
        self._ensure_initialized_blocking()
        try:
            # Key cache fast path
            if self._key_cache:
                future = asyncio.run_coroutine_threadsafe(
                    self._key_cache.get(key.to_string()), self.loop
                )
                try:
                    cache_entry = future.result(timeout=0.1)
                except Exception:
                    cache_entry = None  # treat as cache miss, not an error
                if cache_entry is not None:
                    self._metrics.cache_hits += 1
                    return cache_entry.exists
                self._metrics.cache_misses += 1

            # PQ path
            if self._pq_executor:
                future = asyncio.run_coroutine_threadsafe(
                    self._pq_executor.submit_job(
                        self._contains_async_with_lease_cache,
                        key,
                        priority=OperationPriority.EXISTENCE_CHECK,
                    ),
                    self.loop,
                )
            else:
                future = asyncio.run_coroutine_threadsafe(
                    self._contains_async_with_lease_cache(key), self.loop
                )

            # Give a little dispatch slack beyond lease timeout
            timeout_s = (self.kv_config.lease_timeout_ms + 200) / 1000.0
            return future.result(timeout=timeout_s)

        except Exception as e:
            logger.error(f"Exception during contains check for key {key}: {type(e).__name__}: {str(e)}")
            return False

    async def _contains_async_with_lease_cache(self, key: CacheEngineKey) -> bool:
        """Async contains that caches leases for subsequent get()."""
        lease_info = await self._lease_manager.get_cached_lease(key)
        if not lease_info:
            lease_info = await self._acquire_lease(key)

        if lease_info:
            await self._lease_manager.cache_lease(key, lease_info)
            if self._key_cache:
                await self._key_cache.put(key.to_string(), exists=True)
            return True
        else:
            if self._key_cache:
                await self._key_cache.put(key.to_string(), exists=False)
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self._put_lock:
            return key in self._put_tasks

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        self._ensure_initialized_blocking()
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)
        return None

    @_lmcache_nvtx_annotate
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        self._ensure_initialized_blocking()
        memory_obj.ref_count_up()
        with self._put_lock:
            self._put_tasks.add(key)

        if self._pq_executor:
            asyncio.run_coroutine_threadsafe(
                self._pq_executor.submit_job(
                    self._put_optimized, key, memory_obj, priority=OperationPriority.PUT
                ),
                self.loop,
            )
        else:
            self.loop.call_soon_threadsafe(asyncio.create_task, self._put_optimized(key, memory_obj))
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        self._ensure_initialized_blocking()
        if self._pq_executor:
            return asyncio.run_coroutine_threadsafe(
                self._pq_executor.submit_job(
                    self._get_memory_obj_zero_copy, key, priority=OperationPriority.PREFETCH
                ),
                self.loop,
            )
        else:
            return asyncio.run_coroutine_threadsafe(self._get_memory_obj_zero_copy(key), self.loop)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking GET operation."""
        self._ensure_initialized_blocking()
        try:
            if self._pq_executor:
                future = asyncio.run_coroutine_threadsafe(
                    self._pq_executor.submit_job(
                        self._get_memory_obj_zero_copy, key, priority=OperationPriority.ACTIVE_GET
                    ),
                    self.loop,
                )
            else:
                future = asyncio.run_coroutine_threadsafe(self._get_memory_obj_zero_copy(key), self.loop)
            # IMPORTANT: avoid indefinite hang
            return future.result(timeout=self.kv_config.get_timeout_ms / 1000.0)
        except Exception as e:
            logger.error(f"GET operation exception for key {key}: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        return self.submit_prefetch_task(key)

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        if self._key_cache:
            asyncio.run_coroutine_threadsafe(self._key_cache.remove(key.to_string()), self.loop).result()
        return True

    def close(self) -> None:
        """Close the backend and release resources."""
        if self._closed:
            return
        self._closed = True
        try:
            # Cancel background maintenance and await it to avoid warnings
            if self._maintenance_task:
                self._maintenance_task.cancel()
                try:
                    asyncio.run_coroutine_threadsafe(self._await_task(self._maintenance_task), self.loop).result(
                        timeout=2.0
                    )
                except Exception:
                    pass

            # Release all leases
            lease_ids = asyncio.run_coroutine_threadsafe(
                self._lease_manager.get_all_lease_ids(), self.loop
            ).result()
            for lease_id in lease_ids:
                try:
                    asyncio.run_coroutine_threadsafe(self._release_lease_async(lease_id), self.loop).result(
                        timeout=2.0
                    )
                except Exception as e:
                    logger.error(f"Error releasing lease {lease_id}: {e}")

            # Shutdown PQ
            if self._pq_executor:
                self._pq_executor.shutdown(wait=True)

            # Shutdown thread pool
            if self._executor:
                self._executor.shutdown(wait=True)

            # Close HTTP
            if self._http_session:
                try:
                    asyncio.run_coroutine_threadsafe(self._http_session.close(), self.loop).result(timeout=5.0)
                except Exception:
                    pass

            # Close SHM
            with self._shared_memory_lock:
                if self._shared_memory_map:
                    try:
                        self._shared_memory_map.release()
                    except Exception:
                        pass
                    self._shared_memory_map = None
                if self._shared_memory_obj:
                    try:
                        self._shared_memory_obj.close()
                    except Exception:
                        pass
                    self._shared_memory_obj = None

            logger.info("KVServiceSMBackend closed successfully")

        except Exception as e:
            logger.error(f"Error during backend closure: {e}")

    # ============== HTTP Operations ==============

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self._http_session is None or self._http_session.closed:
            async with self._session_lock:
                if self._http_session is None or self._http_session.closed:
                    connector = aiohttp.TCPConnector(
                        limit=self.kv_config.max_connections,
                        limit_per_host=self.kv_config.max_connections_per_host,
                        ttl_dns_cache=self.kv_config.dns_ttl,
                        keepalive_timeout=self.kv_config.connection_keepalive,
                        enable_cleanup_closed=True,
                        force_close=False,
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=30,
                        connect=self.kv_config.http_connect_timeout_ms / 1000,
                        sock_read=self.kv_config.http_read_timeout_ms / 1000,
                    )

                    self._http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": "LMCache-KVServiceSM/2.0", "Connection": "keep-alive"},
                    )

                    logger.info(f"Created HTTP session with {self.kv_config.max_connections} connections")
        return self._http_session

    async def _http_request(self, method: str, url: str, data=None, params=None, timeout=5.0):
        """HTTP request with robust JSON detection."""
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.request(method, url, data=data, params=params, timeout=request_timeout) as response:
                ctype = response.headers.get("Content-Type", "") or ""
                is_json = ctype.startswith("application/json")
                result = {
                    "status": response.status,
                    "data": await response.read() if method in ["PUT", "POST"] else None,
                    "json": await response.json() if is_json else None,
                }
                return result

        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} timeout for {url} after {timeout}s")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP {method} client error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"HTTP {method} request failed for {url}: {e}")
            return None

    async def _http_request_with_retry(self, method: str, url: str, **kwargs):
        """HTTP request with exponential backoff retry."""
        last_error = None
        for attempt in range(self.kv_config.max_retries):
            try:
                result = await self._http_request(method, url, **kwargs)
                if result is not None:
                    return result
                last_error = "Request failed or timed out"
            except Exception as e:
                last_error = e

            if attempt < self.kv_config.max_retries - 1:
                backoff = self.kv_config.retry_backoff_ms * (2 ** attempt) / 1000
                await asyncio.sleep(backoff)
                self._metrics.http_retries += 1
                logger.debug(
                    f"Retry {attempt + 1}/{self.kv_config.max_retries} after {backoff}s for {url}"
                )

        self._metrics.http_failures += 1
        logger.error(f"All {self.kv_config.max_retries} retries failed: {last_error}")
        raise RuntimeError(f"HTTP request failed after {self.kv_config.max_retries} retries")

    # ============== Core Ops ==============

    async def _get_memory_obj_zero_copy(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Zero-copy GET that reuses cached leases and lazily (re)maps SHM."""
        start_time = time.time()
        try:
            if self._key_cache:
                cache_entry = await self._key_cache.get(key.to_string())
                if cache_entry and not cache_entry.exists:
                    self._metrics.cache_hits += 1
                    self._metrics.record_get(success=False)
                    return None

            lease_info = await self._lease_manager.get_cached_lease(key)
            if not lease_info:
                lease_info = await self._acquire_lease(key)
                if not lease_info:
                    if self._key_cache:
                        await self._key_cache.put(key.to_string(), exists=False)
                    self._metrics.record_get(success=False)
                    return None

            try:
                memory_obj = await self._read_zero_copy(key, lease_info)
                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.record_get(
                    success=memory_obj is not None,
                    size=memory_obj.get_size() if memory_obj else 0,
                    time_ms=elapsed_ms,
                )
                return memory_obj
            finally:
                if self._pq_executor:
                    await self._pq_executor.submit_job(
                        self._release_lease_async, lease_info.lease_id, priority=OperationPriority.LEASE_RELEASE
                    )
                else:
                    await self._release_lease_async(lease_info.lease_id)

        except Exception as e:
            logger.error(f"Error in zero-copy GET for key {key}: {e}")
            self._metrics.record_get(success=False)
            return None

    async def _put_optimized(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Optimized PUT with minimal copies and early memory release."""
        start_time = time.time()
        memory_released = False
        try:
            lease_info = await self._lease_manager.get_cached_lease(key)
            if lease_info:
                logger.info(f"Key {key} has cached lease, skipping PUT")
                memory_obj.ref_count_down()
                return
            key_str = self._key_to_string(key)
            url = f"{self.kv_config.base_url}/v1/kv/{self.kv_config.bucket_name}/{key_str}"

            await self._inflight_semaphore.acquire()
            try:
                if self._executor:
                    loop = asyncio.get_running_loop()
                    data = await loop.run_in_executor(self._executor, self._memory_obj_to_bytes, memory_obj)
                else:
                    data = self._memory_obj_to_bytes(memory_obj)

                memory_obj.ref_count_down()
                memory_released = True

                result = await self._http_request("PUT", url, data=data, timeout=self.kv_config.put_timeout_ms / 1000)
                success = result and result["status"] == 200

                elapsed_ms = (time.time() - start_time) * 1000
                self._metrics.record_put(success=success, size=len(data) if success else 0, time_ms=elapsed_ms)

                if not success:
                    status = result["status"] if result else "UNKNOWN"
                    logger.error(f"Failed to store key {key}: HTTP {status}")
            finally:
                self._inflight_semaphore.release()

        except Exception as e:
            logger.exception(f"Exception during PUT for key {key}: {e}")
            self._metrics.record_put(success=False)
            if not memory_released:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass
        finally:
            with self._put_lock:
                self._put_tasks.discard(key)

    # ============== Lease Management ==============

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Acquire lease WITHOUT retry."""
        key_str = self._key_to_string(key)

        existing_lease = await self._lease_manager.get_lease(key)
        if existing_lease:
            return existing_lease

        url = f"{self.kv_config.base_url}/v1/kv/{self.kv_config.bucket_name}/{key_str}/leases"
        params = {"timeout_ms": self.kv_config.lease_timeout_ms}

        try:
            result = await self._http_request("POST", url, params=params, timeout=self.kv_config.lease_timeout_ms / 1000)
            if result and result["status"] == 200 and result["json"]:
                lease_data = result["json"]
                lease_info = LeaseInfo(
                    lease_id=lease_data["id"],
                    offsets=[(o["offset"], o["len"]) for o in lease_data["offsets"]],
                    total_size=sum(o["len"] for o in lease_data["offsets"]),
                )
                await self._lease_manager.register_lease(key, lease_info)
                self._metrics.lease_acquisitions += 1
                return lease_info
        except Exception as e:
            logger.error(f"Failed to acquire lease for {key_str}: {e}")
            self._metrics.lease_timeouts += 1
        return None

    async def _release_lease_async(self, lease_id: str) -> bool:
        url = f"{self.kv_config.base_url}/v1/leases/{lease_id}/release"
        try:
            result = await self._http_request("POST", url, timeout=self.kv_config.release_timeout_ms / 1000)
            success = result and result["status"] in (200, 404)
            if success:
                await self._lease_manager.unregister_lease(lease_id)
                self._metrics.lease_releases += 1
            return success
        except Exception as e:
            logger.error(f"Failed to release lease {lease_id}: {e}")
            return False

    # ============== Zero-Copy (with lazy SHM remap) ==============

    async def _ensure_shared_memory(self) -> bool:
        """Ensure shared memory is initialized and accessible."""
        with self._shared_memory_lock:
            if self._shared_memory_map is not None:
                return True

            if not self.kv_config.shared_memory_name:
                logger.error("No shared memory name configured")
                return False

            # Try exact name; if creator used /name, consider allowing a config that includes the slash.
            name = self.kv_config.shared_memory_name
            try:
                self._shared_memory_obj = shared_memory.SharedMemory(name=name, create=False)
                self._shared_memory_map = memoryview(self._shared_memory_obj.buf)
                logger.info(
                    f"Successfully opened shared memory: {name} (size: {len(self._shared_memory_map)} bytes)"
                )
                return True
            except FileNotFoundError:
                logger.error(
                    f"Shared memory segment '{name}' not found. Is KVServiceSM daemon running?"
                )
                return False
            except Exception as e:
                logger.error(f"Failed to initialize shared memory: {e}")
                return False

    async def _read_zero_copy(self, key: CacheEngineKey, lease_info: LeaseInfo) -> Optional[MemoryObj]:
        """True single-copy read from shared memory with header-split handling and lazy remap."""
        if not lease_info.offsets:
            logger.error(f"No offsets in lease for key {key}")
            return None

        # Lazy (re)map if needed
        if self._shared_memory_map is None:
            ok = await self._ensure_shared_memory()
            if not ok or self._shared_memory_map is None:
                logger.error("Shared memory not initialized (lazy init failed)")
                return None

        try:
            shm = self._shared_memory_map
            metadata_size = 4 * 7  # bytes

            # 1) Accumulate metadata header across blocks (without mutating lease_info)
            hdr = bytearray()
            block_idx = 0
            while len(hdr) < metadata_size and block_idx < len(lease_info.offsets):
                off, ln = lease_info.offsets[block_idx]
                take = min(metadata_size - len(hdr), ln)
                # shm slices are memoryview; use .tobytes() to avoid keeping refs
                hdr += shm[off : off + take].tobytes()
                block_idx += 1 if take == ln else 0  # move to next block only if fully consumed
                if take < ln:
                    # not fully consumed this block; next phase will start from off+take
                    break

            if len(hdr) < metadata_size:
                logger.error("Insufficient data for metadata")
                return None

            metadata = RemoteMetadata.deserialize(bytes(hdr))
            payload_len = metadata.length
            if payload_len < 0:
                logger.error("Invalid payload length")
                return None

            # Infer actual shape (strip zeros after first real dim)
            original_shape = metadata.shape
            actual_dims: List[int] = []
            for i, d in enumerate(original_shape):
                if d == 0 and i > 0:
                    break
                actual_dims.append(d)
            actual_shape = torch.Size(actual_dims) if actual_dims else torch.Size([1])

            # 2) Allocate destination buffer
            memory_obj = self.memory_allocator.allocate(actual_shape, metadata.dtype, metadata.fmt)
            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key}")
                return None

            view = memory_obj.byte_array if isinstance(memory_obj.byte_array, memoryview) else memoryview(
                memory_obj.byte_array
            )
            if getattr(view, "format", None) == "<B":
                view = view.cast("B")

            # 3) Copy payload directly into destination, skipping the metadata bytes
            copied = 0
            bytes_to_skip = metadata_size
            for (off, ln) in lease_info.offsets:
                # skip metadata bytes first
                if bytes_to_skip > 0:
                    if ln <= bytes_to_skip:
                        bytes_to_skip -= ln
                        continue
                    else:
                        off += bytes_to_skip
                        ln -= bytes_to_skip
                        bytes_to_skip = 0

                if ln <= 0 or copied >= payload_len:
                    continue

                take = min(payload_len - copied, ln)
                src_slice = shm[off : off + take]
                view[copied : copied + take] = src_slice
                copied += take
                if copied >= payload_len:
                    break

            if copied != payload_len:
                logger.error(f"Data size mismatch: expected {payload_len}, got {copied}")
                return None

            return memory_obj

        except BufferError as e:
            # Daemon may have rotated shm; drop mapping and let next call remap
            logger.warning(f"BufferError during shm read; dropping mapping: {e}")
            with self._shared_memory_lock:
                try:
                    if self._shared_memory_map is not None:
                        self._shared_memory_map.release()
                except Exception:
                    pass
                self._shared_memory_map = None
            return None
        except Exception as e:
            logger.error(f"Error reading from shared memory: {e}")
            return None

    # ============== Helper Methods ==============

    def _key_to_string(self, key: CacheEngineKey) -> str:
        key_str = key.to_string()
        return urllib.parse.quote(key_str, safe="")

    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded_shape) > 4:
            logger.warning(f"Shape has {len(kv_shape)} dimensions, truncating to 4")
            padded_shape = list(kv_shape[:4])

        padded_torch_shape = torch.Size(padded_shape)
        metadata = RemoteMetadata(len(kv_bytes), padded_torch_shape, kv_dtype, memory_format)
        metadata_bytes = metadata.serialize()
        return metadata_bytes + kv_bytes

    async def _periodic_maintenance(self):
        """Periodic maintenance tasks."""
        while not self._closed:
            try:
                await asyncio.sleep(60)
                if self._closed:
                    break
                if self._key_cache:
                    expired_count = await self._key_cache.cleanup_expired()
                    if expired_count > 0:
                        logger.debug(f"Cleaned {expired_count} expired cache entries")

                expired_leases = await self._lease_manager.cleanup_expired_cache()
                if expired_leases > 0:
                    logger.debug(f"Cleaned {expired_leases} expired lease cache entries")

                summary = self._metrics.get_summary()
                logger.info(f"Metrics summary: {summary}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic maintenance: {e}")

    async def _await_task(self, t: asyncio.Task):
        try:
            await t
        except asyncio.CancelledError:
            pass

    # ============== Required Abstract Methods ==============

    def get_allocator_backend(self):
        return self.local_cpu_backend

    # ============== Batched Operations ==============

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: List[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        """
        Check whether keys are in the storage backend.
        Returns the number of CONSECUTIVE existing keys from the beginning.
        Stops at the first non-existing key.
        """
        if not keys:
            return 0

        cache_results = {}
        if self._key_cache:
            cache_tasks = [self._key_cache.get(k.to_string()) for k in keys]
            cache_entries = await asyncio.gather(*cache_tasks)
            for i, (key, cache_entry) in enumerate(zip(keys, cache_entries)):
                if cache_entry is not None:
                    cache_results[i] = cache_entry.exists
                    self._metrics.cache_hits += 1
                else:
                    self._metrics.cache_misses += 1

        consecutive_count = 0
        for i, key in enumerate(keys):
            if i in cache_results:
                exists = cache_results[i]
            else:
                if self._pq_executor:
                    exists = await self._pq_executor.submit_job(
                        self._contains_async_with_lease_cache, key, priority=OperationPriority.EXISTENCE_CHECK
                    )
                else:
                    exists = await self._contains_async_with_lease_cache(key)

            if not exists:
                return consecutive_count
            consecutive_count += 1

        return consecutive_count
