# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.

# Standard
import asyncio
import threading
import time
import concurrent.futures
from collections import OrderedDict, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, Set, Deque

# Third Party
import aiohttp
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    MemoryAllocatorInterface,
    MemoryObj,
)
from lmcache.v1.protocol import RemoteMetadata
from lmcache.v1.storage_backend.abstract_backend import StorageBackendInterface

logger = init_logger(__name__)


@dataclass
class LeaseInfo:
    """Information about a lease obtained from KVServiceSM daemon."""
    lease_id: str
    offsets: List[Tuple[int, int]]  # (offset, length) pairs
    total_size: int


class _LRUExpiring:
    """Simple LRU with per-entry expiry (monotonic millis)."""
    __slots__ = ("_cap", "_map", "_lock")

    def __init__(self, capacity: int):
        self._cap = max(1, int(capacity))
        self._map: OrderedDict[CacheEngineKey, float] = OrderedDict()
        self._lock = threading.Lock()

    def _now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def put(self, key: CacheEngineKey, ttl_ms: int) -> None:
        exp = self._now_ms() + max(0, int(ttl_ms))
        with self._lock:
            if key in self._map:
                self._map.move_to_end(key)
            self._map[key] = exp
            if len(self._map) > self._cap:
                self._map.popitem(last=False)

    def has_valid(self, key: CacheEngineKey) -> bool:
        now = self._now_ms()
        with self._lock:
            exp = self._map.get(key)
            if exp is None:
                return False
            if exp > now:
                # refresh recency
                self._map.move_to_end(key)
                return True
            # expired → evict
            try:
                del self._map[key]
            except KeyError:
                pass
            return False

    def clear(self):
        with self._lock:
            self._map.clear()


class KVServiceSMBackend(StorageBackendInterface):
    """
    External KV cache backend (layer-wise) via KVServiceSM daemon:
      - Shared memory leases for zero-copy GET
      - HTTP control plane for leases/PUT/RELEASE
      - Fast-path contains() with TTL caches and single-flight
    """

    def __init__(
        self,
        config,
        metadata,
        loop,
        memory_allocator: MemoryAllocatorInterface,
        local_cpu_backend,
        dst_device,
        lookup_server=None,
    ):
        super().__init__(dst_device)

        self.config: LMCacheEngineConfig = config
        self.metadata = metadata
        self.loop: asyncio.AbstractEventLoop = loop
        self.memory_allocator = memory_allocator
        self.local_cpu_backend = local_cpu_backend
        self.lookup_server = lookup_server

        # KVServiceSM configuration
        extra_config = getattr(config, "extra_config", None) or {}
        self.base_url = extra_config.get("kv_service_sm_url", "http://localhost:9200")
        self.shared_memory_name = extra_config.get(
            "kv_service_sm_shared_memory_name", "shared_memory"
        )
        self.bucket_name = extra_config.get("kv_service_sm_bucket", "lmcache")

        # Timeouts (ms)
        self.lease_timeout_ms = int(extra_config.get("kv_service_sm_lease_timeout_ms", 500))
        self.put_timeout_ms = int(extra_config.get("kv_service_sm_put_timeout_ms", 15000))
        self.release_timeout_ms = int(extra_config.get("kv_service_sm_release_timeout_ms", 2000))

        # contains() fast-path tuning
        self.contains_timeout_ms = int(
            extra_config.get(
                "kv_service_sm_contains_timeout_ms",
                min(self.lease_timeout_ms, 15),
            )
        )
        self.contains_pos_ttl_ms = int(extra_config.get("kv_service_sm_contains_ttl_ms", 250))
        self.contains_neg_ttl_ms = int(extra_config.get("kv_service_sm_negative_ttl_ms", 50))
        self.contains_cache_capacity = int(extra_config.get("kv_service_sm_contains_cache_cap", 65536))

        # Performance config
        self.max_connections = int(extra_config.get("kv_service_sm_max_connections", 256))
        self.max_connections_per_host = int(
            extra_config.get("kv_service_sm_max_connections_per_host", 128)
        )
        self.serialization_threads = int(
            extra_config.get("kv_service_sm_serialization_threads", 16)
        )

        # PUT admission/backpressure
        self.max_inflight_puts = int(extra_config.get("kv_service_sm_max_inflight_puts", 8))
        self.max_pending_put_queue = int(extra_config.get("kv_service_sm_max_pending_put_queue", 512))
        self.put_pending_ttl_ms = int(extra_config.get("kv_service_sm_put_pending_ttl_ms", 5000))
        self.put_circuit_open_ms = int(extra_config.get("kv_service_sm_put_circuit_open_ms", 3000))
        self.put_circuit_error_threshold = int(extra_config.get("kv_service_sm_put_circuit_errors", 8))
        self.put_circuit_latency_ms = int(extra_config.get("kv_service_sm_put_circuit_latency_ms", 12000))

        # Streamed PUT
        self.stream_put = bool(extra_config.get("kv_service_sm_stream_put", True))
        self.stream_put_chunk_bytes = int(extra_config.get("kv_service_sm_stream_put_chunk_bytes", 4 * 1024 * 1024))

        # HTTP connection pool
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()

        # CPU-bound serialization pool (kept for non-stream mode or other tasks)
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.serialization_threads,
            thread_name_prefix="kv-service-sm-serialize",
        )

        # Leases
        self.lease_lock = threading.Lock()
        self.leases: Dict[CacheEngineKey, LeaseInfo] = {}
        self.lease_id_to_key: Dict[str, CacheEngineKey] = {}

        # PUT tracking & backpressure
        self.put_lock = threading.Lock()
        self.put_tasks: Set[CacheEngineKey] = set()
        self._put_sem = asyncio.Semaphore(self.max_inflight_puts)
        self._put_pending_queue: Deque[Tuple[CacheEngineKey, MemoryObj]] = deque()
        self._put_pending_until: Dict[CacheEngineKey, float] = {}
        self._put_metrics_lock = threading.Lock()
        self._put_recent_errors = 0
        self._put_circuit_open_until_ms: float = 0.0

        # Shared memory mapping
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        self.shared_memory_lock = threading.Lock()

        # contains() caches & inflight dedupe
        self._contains_pos_cache = _LRUExpiring(self.contains_cache_capacity)
        self._contains_neg_cache = _LRUExpiring(self.contains_cache_capacity)
        self._contains_inflight: Set[CacheEngineKey] = set()
        self._contains_lock = threading.Lock()

        # Warm HTTP pool early
        self.loop.call_soon_threadsafe(asyncio.create_task, self._ensure_http_session())

        logger.info(
            f"KVServiceSMBackend initialized url={self.base_url} bucket={self.bucket_name} shm={self.shared_memory_name} "
            f"conn(total={self.max_connections},/host={self.max_connections_per_host}) serialize_threads={self.serialization_threads} "
            f"timeouts(ms): lease={self.lease_timeout_ms} put={self.put_timeout_ms} release={self.release_timeout_ms} contains={self.contains_timeout_ms} "
            f"stream_put={self.stream_put} chunk={self.stream_put_chunk_bytes}B"
        )

    # --------------------------
    # Helpers
    # --------------------------
    def __str__(self):
        return self.__class__.__name__

    def _now_ms(self) -> float:
        return time.monotonic() * 1000.0

    def _key_to_string(self, key: CacheEngineKey) -> str:
        # Use URL encoding for complete safety
        import urllib.parse
        key_str = key.to_string()
        return urllib.parse.quote(key_str, safe="")

    def _put_circuit_open(self) -> bool:
        return self._now_ms() < self._put_circuit_open_until_ms

    def _put_trip_circuit(self):
        self._put_circuit_open_until_ms = self._now_ms() + self.put_circuit_open_ms

    def _put_record_ok(self, latency_ms: float):
        with self._put_metrics_lock:
            if latency_ms > self.put_circuit_latency_ms:
                self._put_recent_errors += 1
            else:
                self._put_recent_errors = max(0, self._put_recent_errors - 1)
            if self._put_recent_errors >= self.put_circuit_error_threshold:
                self._put_trip_circuit()

    def _put_record_error(self):
        with self._put_metrics_lock:
            self._put_recent_errors += 1
            if self._put_recent_errors >= self.put_circuit_error_threshold:
                self._put_trip_circuit()

    # --------------------------
    # contains(): ultra-fast path
    # --------------------------
    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Bounded-latency existence check.
        Order: local lease → positive TTL cache → negative TTL cache → single-flight short lease try.
        """
        try:
            lookup_key = CacheEngineKey(
                key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
            )

            # 0) already have a lease → exists
            with self.lease_lock:
                if lookup_key in self.leases:
                    return True

            # 1) positive cache
            if self._contains_pos_cache.has_valid(lookup_key):
                return True

            # 2) negative cache
            if self._contains_neg_cache.has_valid(lookup_key):
                return False

            # 3) single-flight: if someone else is checking, don't pile on
            with self._contains_lock:
                if lookup_key in self._contains_inflight:
                    # micro negative TTL to collapse stampede
                    self._contains_neg_cache.put(lookup_key, ttl_ms=min(20, self.contains_neg_ttl_ms))
                    return False
                self._contains_inflight.add(lookup_key)

            # 4) strictly bounded lease attempt
            try:
                fut = asyncio.run_coroutine_threadsafe(
                    self._acquire_lease_with_timeout(lookup_key, self.contains_timeout_ms),
                    self.loop,
                )
                # IMPORTANT: run_coroutine_threadsafe returns concurrent.futures.Future
                lease_info = fut.result(timeout=max(1, self.contains_timeout_ms) / 1000.0)
                if lease_info:
                    self._contains_pos_cache.put(lookup_key, self.contains_pos_ttl_ms)
                    return True
                else:
                    self._contains_neg_cache.put(lookup_key, self.contains_neg_ttl_ms)
                    return False
            except concurrent.futures.TimeoutError:
                # Hard bound respected → treat as miss (fast return)
                self._contains_neg_cache.put(lookup_key, self.contains_neg_ttl_ms)
                return False
            except Exception as e:
                # Non-fatal: treat as miss
                logger.debug(f"contains() non-fatal error: {e}")
                self._contains_neg_cache.put(lookup_key, self.contains_neg_ttl_ms)
                return False
            finally:
                with self._contains_lock:
                    self._contains_inflight.discard(lookup_key)

        except Exception as e:
            logger.debug(f"contains() exception: {e}")
            return False

    async def _acquire_lease_with_timeout(self, key: CacheEngineKey, timeout_ms: int) -> Optional[LeaseInfo]:
        """Acquire lease but clamp total time to a very small budget (for contains)."""
        try:
            return await asyncio.wait_for(self._acquire_lease(key), timeout=max(1, timeout_ms) / 1000.0)
        except asyncio.TimeoutError:
            return None

    # --------------------------
    # PUT path
    # --------------------------
    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)
        return None

    @_lmcache_nvtx_annotate
    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Optional[Future]:
        """Non-blocking admission of a PUT with de-dup, backpressure, and pending window."""
        # Avoid duplicate in-flight
        with self.put_lock:
            if key in self.put_tasks:
                return None

        # Avoid hammering same key while a recent PUT is pending
        now = self._now_ms()
        exp = self._put_pending_until.get(key)
        if exp is not None and exp > now:
            return None

        # Circuit breaker open → skip
        if self._put_circuit_open():
            return None

        # Bounded queue
        if len(self._put_pending_queue) >= self.max_pending_put_queue:
            # mark short pending so caller won't requeue instantly
            self._put_pending_until[key] = now + self.put_pending_ttl_ms
            return None

        # retain while queued
        memory_obj.ref_count_up()
        self._put_pending_queue.append((key, memory_obj))
        self._put_pending_until[key] = now + self.put_pending_ttl_ms

        # start draining
        self.loop.call_soon_threadsafe(asyncio.create_task, self._drain_put_queue())
        return None

    async def _drain_put_queue(self):
        while self._put_pending_queue:
            if self._put_circuit_open():
                return
            try:
                await self._put_sem.acquire()
            except Exception:
                return

            try:
                key, mem = self._put_pending_queue.popleft()
            except IndexError:
                try:
                    self._put_sem.release()
                except Exception:
                    pass
                return

            with self.put_lock:
                if key in self.put_tasks:
                    # someone started it already
                    try:
                        mem.ref_count_down()
                    except Exception:
                        pass
                    try:
                        self._put_sem.release()
                    except Exception:
                        pass
                    continue
                self.put_tasks.add(key)

            # Run guarded PUT
            self.loop.call_soon_threadsafe(
                asyncio.create_task, self._guarded_async_put(key, mem)
            )

    async def _guarded_async_put(self, key: CacheEngineKey, memory_obj: MemoryObj):
        start = self._now_ms()
        try:
            await self._async_put(key, memory_obj)
            self._put_record_ok(self._now_ms() - start)
        except Exception:
            self._put_record_error()
            logger.debug("PUT failed", exc_info=True)
        finally:
            # queue-level ref (submit_put_task) always balanced here
            try:
                memory_obj.ref_count_down()
            except Exception:
                pass
            with self.put_lock:
                self.put_tasks.discard(key)
            try:
                self._put_sem.release()
            except Exception:
                pass

    # --------------------------
    # HTTP
    # --------------------------
    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        if self.http_session is None:
            async with self.session_lock:
                if self.http_session is None:
                    connector = aiohttp.TCPConnector(
                        limit=self.max_connections,
                        limit_per_host=self.max_connections_per_host,
                        ttl_dns_cache=300,
                        use_dns_cache=True,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True,
                    )
                    timeout = aiohttp.ClientTimeout(
                        total=30,
                        connect=5,
                        sock_read=10,
                    )
                    self.http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": "LMCache-KVServiceSMBackend/1.0"},
                    )
                    logger.info(f"HTTP session ready (pool={self.max_connections}/{self.max_connections_per_host})")
        return self.http_session

    async def _http_request(self, method: str, url: str, data=None, params=None, timeout=5.0):
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)
            async with session.request(method, url, data=data, params=params, timeout=request_timeout) as response:
                result = {
                    "status": response.status,
                    "data": await response.read() if method in ["PUT", "POST"] else None,
                    "json": await response.json() if response.content_type == "application/json" else None,
                }
                return result
        except asyncio.TimeoutError:
            logger.debug(f"HTTP {method} timeout {timeout}s {url}")
            return None
        except aiohttp.ClientError as e:
            logger.debug(f"HTTP {method} client error {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"HTTP {method} failed {url}: {e}")
            return None

    # --------------------------
    # PUT implementation (streaming by default)
    # --------------------------
    @_lmcache_nvtx_annotate
    async def _async_put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        store_key = CacheEngineKey(
            key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
        )
        key_str = self._key_to_string(store_key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}"

        # Build metadata (small, cheap)
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded_shape) > 4:
            padded_shape = list(kv_shape[:4])
        metadata = RemoteMetadata(len(kv_bytes), torch.Size(padded_shape), kv_dtype, memory_format)
        metadata_bytes = metadata.serialize()

        if self.stream_put:
            # Stream metadata + payload in chunks without concatenation
            async def gen():
                # metadata first
                yield metadata_bytes
                view = memoryview(kv_bytes)
                step = self.stream_put_chunk_bytes
                for i in range(0, len(view), step):
                    yield view[i : i + step]

            result = await self._http_request(
                "PUT", url, data=gen(), timeout=self.put_timeout_ms / 1000.0
            )
        else:
            # Fallback: serialize in thread, early release (legacy path)
            loop = asyncio.get_running_loop()
            data = await loop.run_in_executor(self.thread_pool, self._memory_obj_to_bytes, memory_obj)
            # Early release after copy (legacy behavior)
            try:
                memory_obj.ref_count_down()
            except Exception:
                pass
            result = await self._http_request("PUT", url, data=data, timeout=self.put_timeout_ms / 1000.0)

        if not (result and result["status"] == 200):
            status = result["status"] if result else "TIMEOUT"
            raise RuntimeError(f"PUT failed for {key} (HTTP {status})")

    # --------------------------
    # GET path
    # --------------------------
    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        try:
            return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop).result()
        except Exception as e:
            logger.error(f"GET exception for {key}: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    async def _get_memory_obj(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        lookup_key = CacheEngineKey(
            key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
        )

        with self.lease_lock:
            lease_info = self.leases.get(lookup_key)

        if lease_info is None:
            lease_info = await self._acquire_lease(lookup_key)
            if lease_info is None:
                return None

        try:
            result = await self._read_tensor_from_lease_incremental(key, lease_info)
            return result
        finally:
            await self._release_lease(lease_info.lease_id)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
        params = {"timeout_ms": self.lease_timeout_ms}
        result = await self._http_request("POST", url, params=params, timeout=self.lease_timeout_ms / 1000.0)

        if result and result["status"] == 200 and result["json"]:
            lease_data = result["json"]
            lease_info = LeaseInfo(
                lease_id=lease_data["id"],
                offsets=[(o["offset"], o["len"]) for o in lease_data["offsets"]],
                total_size=sum(o["len"] for o in lease_data["offsets"]),
            )
            with self.lease_lock:
                self.leases[key] = lease_info
                self.lease_id_to_key[lease_info.lease_id] = key
            return lease_info

        return None

    async def _release_lease(self, lease_id: str) -> bool:
        url = f"{self.base_url}/v1/leases/{lease_id}/release"
        result = await self._http_request("POST", url, timeout=self.release_timeout_ms / 1000.0)

        success = False
        if result:
            if result["status"] == 200:
                success = True
            elif result["status"] == 404:
                success = True
            else:
                logger.debug(f"Lease release failed {lease_id} status={result['status']} resp={result.get('json', {})}")

        if success:
            with self.lease_lock:
                if lease_id in self.lease_id_to_key:
                    key = self.lease_id_to_key.pop(lease_id)
                    self.leases.pop(key, None)

        return success

    # Incremental reader: parse metadata first, then copy payload directly
    async def _read_tensor_from_lease_incremental(
        self, key: CacheEngineKey, lease_info: LeaseInfo
    ) -> Optional[MemoryObj]:
        if not await self._ensure_shared_memory():
            return None
        if not lease_info.offsets:
            logger.error(f"No offsets in lease for key {key}")
            return None

        smap = self.shared_memory_map
        if smap is None:
            logger.error("Shared memory map not initialized")
            return None

        # Helper to pull up to n bytes across (offset,len) segments
        def read_prefix(n: int) -> bytes:
            out = bytearray()
            remain = n
            for off, ln in lease_info.offsets:
                if remain <= 0:
                    break
                take = min(ln, remain)
                out += smap[off : off + take].tobytes()
                remain -= take
            return bytes(out)

        metadata_size = 4 * 7  # RemoteMetadata is 7 32-bit ints
        header = read_prefix(metadata_size)
        if len(header) < metadata_size:
            logger.error("Insufficient data for metadata header")
            return None

        try:
            metadata = RemoteMetadata.deserialize(header)
        except Exception as e:
            logger.error(f"Failed to parse RemoteMetadata: {e}")
            return None

        expected_len = metadata.length
        if expected_len <= 0:
            logger.error("Invalid metadata.length")
            return None

        # Allocate target memory
        actual_shape = self._restore_shape(metadata.shape)
        memory_obj = self.memory_allocator.allocate(actual_shape, metadata.dtype, metadata.fmt)
        if memory_obj is None:
            logger.error(f"Allocation failed for key {key}")
            return None

        # Copy payload directly into destination buffer (no big temp)
        dest = memoryview(memory_obj.byte_array)
        if isinstance(dest, memoryview) and dest.format == "<B":
            dest = dest.cast("B")

        bytes_to_skip = metadata_size
        written = 0
        for off, ln in lease_info.offsets:
            # segment might contain header+payload or only payload
            seg_off = off
            seg_len = ln
            if bytes_to_skip > 0:
                # drop header bytes first
                drop = min(seg_len, bytes_to_skip)
                seg_off += drop
                seg_len -= drop
                bytes_to_skip -= drop
                if seg_len <= 0:
                    continue
            if written >= expected_len:
                break
            copy_len = min(seg_len, expected_len - written)
            segment = smap[seg_off : seg_off + copy_len]
            dest[written : written + copy_len] = segment
            written += copy_len

        if written != expected_len:
            logger.error(f"Size mismatch: expected {expected_len}, wrote {written}")
            return None

        return memory_obj

    def _restore_shape(self, shape_4d: torch.Size) -> torch.Size:
        # RemoteMetadata stores 4D shape with trailing zeros
        actual: List[int] = []
        for dim in shape_4d:
            if dim == 0 and len(actual) > 0:
                break
            actual.append(int(dim))
        return torch.Size(actual if actual else [1])

    async def _ensure_shared_memory(self) -> bool:
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                return True
            if self.shared_memory_name is None:
                logger.error("No shared memory name configured")
                return False
            try:
                self.shared_memory_obj = shared_memory.SharedMemory(
                    name=self.shared_memory_name, create=False
                )
                self.shared_memory_map = memoryview(self.shared_memory_obj.buf)
                logger.info(
                    f"Opened shared memory '{self.shared_memory_name}' size={len(self.shared_memory_map)}"
                )
                return True
            except FileNotFoundError:
                logger.error(
                    f"Shared memory '{self.shared_memory_name}' not found. Is KVServiceSM running?"
                )
                return False
            except Exception as e:
                logger.error(f"Failed to initialize shared memory: {e}")
                return False

    # --------------------------
    # No-op pins/removes (not required for external daemon)
    # --------------------------
    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return True

    # --------------------------
    # Shutdown
    # --------------------------
    def close(self) -> None:
        # Release leases
        with self.lease_lock:
            lease_ids = list(self.lease_id_to_key.keys())
        try:
            for lid in lease_ids:
                try:
                    asyncio.run_coroutine_threadsafe(self._release_lease(lid), self.loop).result(timeout=2.0)
                except Exception:
                    pass
            logger.info(f"Released {len(lease_ids)} leases")
        except Exception as e:
            logger.error(f"Error releasing leases: {e}")

        # Close HTTP
        if self.http_session is not None:
            try:
                asyncio.run_coroutine_threadsafe(self.http_session.close(), self.loop).result(timeout=5.0)
                logger.info("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
            self.http_session = None

        # Shutdown thread pool
        try:
            self.thread_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Thread pool shutdown error: {e}")

        # Close SHM
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                try:
                    self.shared_memory_map.release()
                except Exception:
                    pass
                self.shared_memory_map = None
            if self.shared_memory_obj is not None:
                try:
                    self.shared_memory_obj.close()
                except Exception:
                    pass
                self.shared_memory_obj = None

        # Clear caches
        self._contains_pos_cache.clear()
        self._contains_neg_cache.clear()
        logger.info("KVServiceSMBackend closed cleanly.")

    # --------------------------
    # Legacy serializer (kept for non-stream mode)
    # --------------------------
    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()
        padded = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded) > 4:
            padded = list(kv_shape[:4])
        metadata = RemoteMetadata(len(kv_bytes), torch.Size(padded), kv_dtype, memory_format)
        mbytes = metadata.serialize()
        return mbytes + kv_bytes
