# SPDX-License-Identifier: Apache-2.0
# Copyright 2024-2025 LMCache Authors.
#
# See header in user's original for license details.

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple, Set
from collections import OrderedDict
import asyncio
import threading
import time
import urllib.parse

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


class KVServiceSMBackend(StorageBackendInterface):
    """
    KVServiceSM storage backend (layerwise) with:
      - Fast probe-only contains() (single-flight, TTL smoothing, tiny deadlines)
      - Dual HTTP lane (control-plane vs data-plane)
      - Bounded PUT concurrency (after serialization)
      - Zero-copy-ish GET: read header directly, copy chunks straight into target
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata,
        loop: asyncio.AbstractEventLoop,
        memory_allocator: MemoryAllocatorInterface,
        local_cpu_backend,
        dst_device,
        lookup_server=None,
    ):
        super().__init__(dst_device)

        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.memory_allocator = memory_allocator
        self.local_cpu_backend = local_cpu_backend
        self.lookup_server = lookup_server

        # === Config ===
        extra = getattr(config, "extra_config", None) or {}
        self.base_url = extra.get("kv_service_sm_url", "http://localhost:9200")
        self.shared_memory_name = extra.get("kv_service_sm_shared_memory_name", "shared_memory")
        self.bucket_name = extra.get("kv_service_sm_bucket", "lmcache")

        # Original timeouts still honored for non-contains calls (GET/PUT/release)
        self.lease_timeout_ms = int(extra.get("kv_service_sm_lease_timeout_ms", 500))
        self.put_timeout_ms = int(extra.get("kv_service_sm_put_timeout_ms", 5000))
        self.release_timeout_ms = int(extra.get("kv_service_sm_release_timeout_ms", 2000))

        # NEW: contains() dedicated timeouts (tiny)
        self.contains_probe_timeout_ms = int(extra.get("kv_service_sm_contains_probe_timeout_ms", 15))
        self.contains_result_timeout_ms = int(extra.get("kv_service_sm_contains_result_timeout_ms",
                                                        max(20, self.contains_probe_timeout_ms + 5)))

        # NEW: cache smoothing TTLs (ms) + capacities
        self.recently_put_ttl_ms = int(extra.get("kv_service_sm_recently_put_ttl_ms", 3000))
        self.negative_ttl_ms = int(extra.get("kv_service_sm_negative_ttl_ms", 100))
        self.positive_capacity = int(extra.get("kv_service_sm_positive_capacity", 65536))
        self.negative_capacity = int(extra.get("kv_service_sm_negative_capacity", 65536))

        # NEW: probe policy
        self.prelease_mode = bool(extra.get("kv_service_sm_prelease_mode", False))  # False = probe-only (recommended)
        self.probe_hold_local_ttl_ms = int(extra.get("kv_service_sm_probe_hold_local_ttl_ms", 150))

        # NEW: dual sessions + limits (control plane is huge to avoid HOL during PUTs)
        self.ctrl_limit = int(extra.get("kv_service_sm_ctrl_max_connections", 2048))
        self.ctrl_limit_per_host = int(extra.get("kv_service_sm_ctrl_max_connections_per_host", 1024))
        self.data_limit = int(extra.get("kv_service_sm_data_max_connections", 512))
        self.data_limit_per_host = int(extra.get("kv_service_sm_data_max_connections_per_host", 256))
        self.keepalive_timeout_s = int(extra.get("kv_service_sm_keepalive_timeout_s", 30))

        # NEW: bound in-flight PUTs (after serialization)
        self.put_concurrency = int(extra.get("kv_service_sm_put_concurrency", 128))
        self.put_http_sema = asyncio.Semaphore(self.put_concurrency)

        # Serialization threads
        self.serialization_threads = int(extra.get("kv_service_sm_serialization_threads", 16))

        # Sessions (created lazily but warmed on first use)
        self.ctrl_session: Optional[aiohttp.ClientSession] = None  # control plane: leases / release / contains probes
        self.data_session: Optional[aiohttp.ClientSession] = None  # data plane: PUTs
        self.session_lock = asyncio.Lock()

        # Thread pool for CPU-bound serialization
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.serialization_threads,
            thread_name_prefix="kv-service-sm-serialize",
        )

        # Lease tracking (for prelease mode / get path)
        self.lease_lock = threading.Lock()
        self.leases: Dict[CacheEngineKey, LeaseInfo] = {}
        self.lease_id_to_key: Dict[str, CacheEngineKey] = {}

        # PUT tracking
        self.put_lock = threading.Lock()
        self.put_tasks: Set[CacheEngineKey] = set()

        # NEW: in-flight probe single-flight map (event-loop domain)
        self._probe_futures: Dict[CacheEngineKey, asyncio.Future] = {}
        self._probe_lock = asyncio.Lock()

        # NEW: TTL caches (OrderedDict for cheap LRU-ish cleanup)
        self._recently_put = OrderedDict()   # key -> expiry (monotonic)
        self._negative_ttl = OrderedDict()   # key -> expiry (monotonic)
        self._ttl_lock = threading.Lock()

        # Shared memory mapping
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        self.shared_memory_lock = threading.Lock()

        logger.info(
            "KVServiceSMBackend init:"
            f" url={self.base_url}, bucket={self.bucket_name}, shm={self.shared_memory_name}; "
            f"timeouts(ms): lease={self.lease_timeout_ms}, put={self.put_timeout_ms}, "
            f"release={self.release_timeout_ms}, contains_probe={self.contains_probe_timeout_ms}, "
            f"contains_result={self.contains_result_timeout_ms}; "
            f"sessions(ctrl={self.ctrl_limit}/{self.ctrl_limit_per_host}, "
            f"data={self.data_limit}/{self.data_limit_per_host}); "
            f"put_concurrency={self.put_concurrency}, threads={self.serialization_threads}"
        )

    def __str__(self):
        return self.__class__.__name__

    # --------------- Public API ---------------

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Fast, bounded probe with TTL smoothing and single-flight."""
        try:
            lookup_key = CacheEngineKey(
                key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
            )

            # 1) Local fast checks
            if self._ttl_positive(lookup_key):
                return True
            if self._ttl_negative(lookup_key):
                return False
            if self.exists_in_put_tasks(lookup_key):
                return True
            with self.lease_lock:
                if lookup_key in self.leases:
                    return True

            # 2) One-shot bounded probe via event loop (single-flight)
            fut = asyncio.run_coroutine_threadsafe(
                self._probe_contains(lookup_key), self.loop
            )

            try:
                ok = fut.result(timeout=self.contains_result_timeout_ms / 1000.0)
                return ok
            except Exception as e:
                logger.debug(f"contains(): probe result timeout/error -> {e}")
                # Conservatively return False on probe delay; keeps critical path unblocked
                return False

        except Exception as e:
            logger.error(f"contains() exception: {e}")
            return False

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
        memory_obj.ref_count_up()
        with self.put_lock:
            self.put_tasks.add(key)
        self.loop.call_soon_threadsafe(asyncio.create_task, self._async_put(key, memory_obj))
        return None

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        try:
            return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop).result()
        except Exception as e:
            logger.error(f"GET blocking exception for key {key}: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    # --------------- Internals ---------------

    async def _ensure_sessions(self) -> None:
        """Create both control and data sessions (idempotent)."""
        if self.ctrl_session is not None and self.data_session is not None:
            return
        async with self.session_lock:
            if self.ctrl_session is None:
                ctrl_connector = aiohttp.TCPConnector(
                    limit=self.ctrl_limit,
                    limit_per_host=self.ctrl_limit_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=self.keepalive_timeout_s,
                    enable_cleanup_closed=True,
                )
                self.ctrl_session = aiohttp.ClientSession(
                    connector=ctrl_connector,
                    timeout=aiohttp.ClientTimeout(total=10, connect=2, sock_read=5),
                    headers={"User-Agent": "LMCache-KVServiceSMBackend/ctrl"},
                )
                logger.info(f"Created CTRL session ({self.ctrl_limit}/{self.ctrl_limit_per_host})")
            if self.data_session is None:
                data_connector = aiohttp.TCPConnector(
                    limit=self.data_limit,
                    limit_per_host=self.data_limit_per_host,
                    ttl_dns_cache=300,
                    use_dns_cache=True,
                    keepalive_timeout=self.keepalive_timeout_s,
                    enable_cleanup_closed=True,
                )
                self.data_session = aiohttp.ClientSession(
                    connector=data_connector,
                    timeout=aiohttp.ClientTimeout(total=60, connect=5, sock_read=30),
                    headers={"User-Agent": "LMCache-KVServiceSMBackend/data"},
                )
                logger.info(f"Created DATA session ({self.data_limit}/{self.data_limit_per_host})")

    async def _http_request_ctrl(self, method: str, url: str, *, data=None, params=None, timeout_s: float = 1.0):
        try:
            await self._ensure_sessions()
            session = self.ctrl_session  # type: ignore
            req_timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.request(method, url, data=data, params=params, timeout=req_timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                is_json = ctype.lower().startswith("application/json")
                return {
                    "status": resp.status,
                    "json": (await resp.json()) if is_json else None,
                    "data": None if method in ("GET", "HEAD") else (await resp.read()),
                }
        except asyncio.TimeoutError:
            logger.debug(f"CTRL {method} timeout: {url} ({timeout_s:.3f}s)")
            return None
        except aiohttp.ClientError as e:
            logger.debug(f"CTRL {method} client error: {url}: {e}")
            return None
        except Exception as e:
            logger.debug(f"CTRL {method} error: {url}: {e}")
            return None

    async def _http_request_data(self, method: str, url: str, *, data=None, params=None, timeout_s: float = 5.0):
        try:
            await self._ensure_sessions()
            session = self.data_session  # type: ignore
            req_timeout = aiohttp.ClientTimeout(total=timeout_s)
            async with session.request(method, url, data=data, params=params, timeout=req_timeout) as resp:
                ctype = resp.headers.get("Content-Type", "")
                is_json = ctype.lower().startswith("application/json")
                return {
                    "status": resp.status,
                    "json": (await resp.json()) if is_json else None,
                    "data": await resp.read(),
                }
        except asyncio.TimeoutError:
            logger.warning(f"DATA {method} timeout: {url} ({timeout_s:.3f}s)")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"DATA {method} client error: {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"DATA {method} error: {url}: {e}")
            return None

    # ---------- PUT ----------

    @_lmcache_nvtx_annotate
    async def _async_put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        serialization_released = False
        try:
            store_key = CacheEngineKey(
                key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
            )
            url = f"{self.base_url}/v1/kv/{self.bucket_name}/{self._key_to_string(store_key)}"

            loop = asyncio.get_running_loop()
            t0 = loop.time()
            data = await loop.run_in_executor(self.thread_pool, self._memory_obj_to_bytes, memory_obj)
            t_ser = loop.time() - t0

            # Early release GPU/host tensor memory
            memory_obj.ref_count_down()
            serialization_released = True

            # Bound data-plane concurrency AFTER serialization
            async with self.put_http_sema:
                t1 = loop.time()
                result = await self._http_request_data(
                    "PUT", url, data=data, timeout_s=self.put_timeout_ms / 1000.0
                )
                t_http = loop.time() - t1

            if result and result["status"] == 200:
                self._ttl_set_positive(store_key, self.recently_put_ttl_ms)
                logger.debug(f"PUT ok key={store_key} bytes={len(data)} ser={t_ser*1000:.1f}ms http={t_http*1000:.1f}ms")
            else:
                status = result["status"] if result else "TIMEOUT"
                logger.error(f"PUT fail key={store_key}: HTTP {status}")

        except Exception as e:
            logger.exception(f"PUT exception key={key}: {e}")
        finally:
            if not serialization_released:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass
            with self.put_lock:
                self.put_tasks.discard(key)

    # ---------- GET (lease → read → reconstruct → release) ----------

    async def _get_memory_obj(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        lookup_key = CacheEngineKey(
            key.fmt, key.model_name, key.world_size, key.worker_id, key.chunk_hash, key.request_configs
        )

        lease_info: Optional[LeaseInfo] = None
        with self.lease_lock:
            if lookup_key in self.leases:
                lease_info = self.leases.get(lookup_key)

        if lease_info is None:
            lease_info = await self._acquire_lease(lookup_key, timeout_ms=self.lease_timeout_ms)
            if lease_info is None:
                return None

        try:
            result = await self._read_tensor_from_lease(lookup_key, lease_info)
            return result
        finally:
            await self._release_lease(lease_info.lease_id)

    # ---------- Lease mgmt ----------

    async def _probe_contains(self, key: CacheEngineKey) -> bool:
        """Single-flight, tiny-deadline contains probe."""
        # If another probe is in-flight, await it (with bounded wait)
        async with self._probe_lock:
            fut = self._probe_futures.get(key)
            if fut is None:
                fut = asyncio.create_task(self._acquire_lease(key, timeout_ms=self.contains_probe_timeout_ms))
                self._probe_futures[key] = fut

        try:
            lease_info = await asyncio.wait_for(fut, timeout=self.contains_probe_timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            lease_info = None
        except Exception as e:
            logger.debug(f"_probe_contains error for {key}: {e}")
            lease_info = None
        finally:
            # Cleanup from map once resolved
            async with self._probe_lock:
                cur = self._probe_futures.get(key)
                if cur is fut:
                    self._probe_futures.pop(key, None)

        if lease_info:
            if self.prelease_mode:
                # Keep it for a short while for a following GET; auto-evict if unused soon.
                with self.lease_lock:
                    self.leases[key] = lease_info
                    self.lease_id_to_key[lease_info.lease_id] = key
                self._ttl_set_positive(key, self.probe_hold_local_ttl_ms)
            else:
                # Probe-only mode: do not hold server lease
                self._ttl_set_positive(key, self.probe_hold_local_ttl_ms)
                await self._release_lease(lease_info.lease_id)
            return True

        # Miss or timeout
        self._ttl_set_negative(key, self.negative_ttl_ms)
        return False

    async def _acquire_lease(self, key: CacheEngineKey, *, timeout_ms: int) -> Optional[LeaseInfo]:
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
        params = {"timeout_ms": timeout_ms}
        result = await self._http_request_ctrl("POST", url, params=params, timeout_s=timeout_ms / 1000.0)
        if result and result["status"] == 200 and result["json"]:
            lease_data = result["json"]
            lease_info = LeaseInfo(
                lease_id=lease_data["id"],
                offsets=[(o["offset"], o["len"]) for o in lease_data["offsets"]],
                total_size=sum(o["len"] for o in lease_data["offsets"]),
            )
            return lease_info
        return None

    async def _release_lease(self, lease_id: str) -> bool:
        url = f"{self.base_url}/v1/leases/{lease_id}/release"
        result = await self._http_request_ctrl("POST", url, timeout_s=self.release_timeout_ms / 1000.0)
        success = False
        if result:
            if result["status"] == 200:
                success = True
            elif result["status"] == 404:
                logger.debug(f"Lease {lease_id} already gone (404)")
                success = True
            else:
                logger.warning(f"Release lease {lease_id} failed: {result['status']}, body={result.get('json', {})}")

        if success:
            with self.lease_lock:
                if lease_id in self.lease_id_to_key:
                    key = self.lease_id_to_key.pop(lease_id, None)
                    if key and key in self.leases:
                        self.leases.pop(key, None)
        return success

    # ---------- Shared-memory read path (zero-copy-ish) ----------

    async def _read_tensor_from_lease(self, key: CacheEngineKey, lease_info: LeaseInfo) -> Optional[MemoryObj]:
        if not await self._ensure_shared_memory():
            return None
        if not lease_info.offsets:
            logger.error(f"No offsets in lease for key {key}")
            return None
        if self.shared_memory_map is None:
            logger.error("Shared memory map not initialized")
            return None

        try:
            # 1) Read fixed-size RemoteMetadata header directly from offsets
            metadata_size = 4 * 7  # RemoteMetadata encodes 7 int32 (keep in sync with protocol)
            header = self._read_exact_from_offsets(lease_info.offsets, 0, metadata_size)
            if header is None or len(header) < metadata_size:
                logger.error("Failed to read metadata header")
                return None

            metadata = RemoteMetadata.deserialize(header)
            data_len = metadata.length

            # 2) Allocate target MemoryObj using allocator
            memory_obj = self.memory_allocator.allocate(
                self._actual_shape_from_padded(metadata.shape),
                metadata.dtype,
                metadata.fmt,
            )
            if memory_obj is None:
                logger.error(f"Allocation failed for key {key}")
                return None

            # 3) Directly copy payload into memory_obj.byte_array (no big temp aggregation)
            view = memory_obj.byte_array if isinstance(memory_obj.byte_array, memoryview) else memoryview(memory_obj.byte_array)
            try:
                if getattr(view, "format", None) == "<B":
                    view = view.cast("B")
            except Exception:
                pass

            copied = self._copy_from_offsets_to_view(
                lease_info.offsets,
                start_offset=metadata_size,
                total_len=data_len,
                target=view,
            )
            if not copied:
                logger.error(f"Copy from shared memory failed for key {key}")
                return None

            logger.debug(f"GET reconstructed key={key} bytes={data_len}")
            return memory_obj

        except Exception as e:
            logger.error(f"Read tensor from lease error for key {key}: {e}")
            return None

    def _read_exact_from_offsets(self, offsets: List[Tuple[int, int]], start: int, n: int) -> Optional[bytes]:
        """Read exactly n bytes beginning at 'start' relative to the lease's first byte."""
        if self.shared_memory_map is None:
            return None
        if n <= 0:
            return b""

        # Map logical [0 .. total_size) to the physical offsets list
        # We'll walk offsets accumulating until we cover [start : start+n)
        buf = bytearray(n)
        want_start = start
        want_end = start + n
        wrote = 0
        logical_cursor = 0

        for off, length in offsets:
            seg_start = logical_cursor
            seg_end = logical_cursor + length
            if seg_end <= want_start:
                logical_cursor += length
                continue
            if seg_start >= want_end:
                break

            # overlap of [seg_start, seg_end) with [want_start, want_end)
            read_start_logical = max(seg_start, want_start)
            read_end_logical = min(seg_end, want_end)
            take = read_end_logical - read_start_logical
            if take <= 0:
                logical_cursor += length
                continue

            # Map logical to physical shared-memory range
            # physical offset = off + (read_start_logical - seg_start)
            shm_off = off + (read_start_logical - seg_start)
            shm_end = shm_off + take

            buf[wrote:wrote + take] = self.shared_memory_map[shm_off:shm_end]  # type: ignore
            wrote += take
            logical_cursor += length
            if wrote >= n:
                break

        return bytes(buf) if wrote == n else None

    def _copy_from_offsets_to_view(
        self,
        offsets: List[Tuple[int, int]],
        start_offset: int,
        total_len: int,
        target: memoryview,
    ) -> bool:
        """Copy total_len bytes from lease offsets (starting at start_offset) into target."""
        if self.shared_memory_map is None:
            return False
        if total_len <= 0:
            return True

        want_start = start_offset
        want_end = start_offset + total_len
        written = 0
        logical_cursor = 0

        for off, length in offsets:
            seg_start = logical_cursor
            seg_end = logical_cursor + length
            if seg_end <= want_start:
                logical_cursor += length
                continue
            if seg_start >= want_end:
                break

            read_start_logical = max(seg_start, want_start)
            read_end_logical = min(seg_end, want_end)
            take = read_end_logical - read_start_logical
            if take <= 0:
                logical_cursor += length
                continue

            shm_off = off + (read_start_logical - seg_start)
            shm_end = shm_off + take

            target[written:written + take] = self.shared_memory_map[shm_off:shm_end]  # type: ignore
            written += take
            logical_cursor += length
            if written >= total_len:
                break

        return written == total_len

    # ---------- Helpers ----------

    async def _ensure_shared_memory(self) -> bool:
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                return True
            if self.shared_memory_name is None:
                logger.error("No shared memory name configured")
                return False
            try:
                self.shared_memory_obj = shared_memory.SharedMemory(name=self.shared_memory_name, create=False)
                self.shared_memory_map = memoryview(self.shared_memory_obj.buf)
                logger.info(f"Opened shared memory '{self.shared_memory_name}' size={len(self.shared_memory_map)}")
                return True
            except FileNotFoundError:
                logger.error(f"Shared memory '{self.shared_memory_name}' not found")
                return False
            except Exception as e:
                logger.error(f"Failed to initialize shared memory: {e}")
                return False

    def pin(self, key: CacheEngineKey) -> bool:
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        return True

    def close(self) -> None:
        # Release leases
        with self.lease_lock:
            lease_ids = list(self.lease_id_to_key.keys())
        try:
            for lease_id in lease_ids:
                asyncio.run_coroutine_threadsafe(self._release_lease(lease_id), self.loop).result(timeout=2.0)
            logger.info(f"Released {len(lease_ids)} leases")
        except Exception as e:
            logger.error(f"Error releasing leases: {e}")

        # Close sessions
        for sess_name in ("ctrl_session", "data_session"):
            sess = getattr(self, sess_name)
            if sess is not None:
                try:
                    asyncio.run_coroutine_threadsafe(sess.close(), self.loop).result(timeout=5.0)
                    logger.info(f"{sess_name} closed")
                except Exception as e:
                    logger.error(f"Error closing {sess_name}: {e}")
                setattr(self, sess_name, None)

        # Shutdown thread pool
        try:
            self.thread_pool.shutdown(wait=True)
            logger.info("Thread pool shutdown complete")
        except Exception as e:
            logger.error(f"Thread pool shutdown error: {e}")

        # Close shared memory
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                try:
                    self.shared_memory_map.release()
                except Exception as e:
                    logger.error(f"Shared memory map release error: {e}")
                self.shared_memory_map = None
            if self.shared_memory_obj is not None:
                try:
                    self.shared_memory_obj.close()
                except Exception as e:
                    logger.error(f"Shared memory close error: {e}")
                self.shared_memory_obj = None

        logger.info("KVServiceSMBackend closed cleanly.")

    # --- TTL caches (thread-safe) ---

    def _ttl_now(self) -> float:
        return time.monotonic()

    def _ttl_cleanup(self, od: OrderedDict, capacity: int):
        # Drop expired from the front; cap size
        now = self._ttl_now()
        # Remove a small batch to keep cost bounded
        for _ in range(64):
            if not od:
                break
            k, exp = next(iter(od.items()))
            if exp < now or len(od) > capacity:
                od.popitem(last=False)
            else:
                break

    def _ttl_set_positive(self, key: CacheEngineKey, ttl_ms: int):
        with self._ttl_lock:
            expiry = self._ttl_now() + ttl_ms / 1000.0
            self._recently_put[key] = expiry
            self._recently_put.move_to_end(key)
            self._ttl_cleanup(self._recently_put, self.positive_capacity)

    def _ttl_positive(self, key: CacheEngineKey) -> bool:
        with self._ttl_lock:
            exp = self._recently_put.get(key)
            if exp is None:
                return False
            if exp < self._ttl_now():
                self._recently_put.pop(key, None)
                return False
            # keep LRU fresh
            self._recently_put.move_to_end(key)
            return True

    def _ttl_set_negative(self, key: CacheEngineKey, ttl_ms: int):
        with self._ttl_lock:
            expiry = self._ttl_now() + ttl_ms / 1000.0
            self._negative_ttl[key] = expiry
            self._negative_ttl.move_to_end(key)
            self._ttl_cleanup(self._negative_ttl, self.negative_capacity)

    def _ttl_negative(self, key: CacheEngineKey) -> bool:
        with self._ttl_lock:
            exp = self._negative_ttl.get(key)
            if exp is None:
                return False
            if exp < self._ttl_now():
                self._negative_ttl.pop(key, None)
                return False
            self._negative_ttl.move_to_end(key)
            return True

    # --- Misc helpers ---

    def _key_to_string(self, key: CacheEngineKey) -> str:
        return urllib.parse.quote(key.to_string(), safe="")

    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        padded = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded) > 4:
            logger.warning(f"Shape has {len(kv_shape)} dims, truncating to first 4: {kv_shape}")
            padded = list(kv_shape[:4])

        metadata = RemoteMetadata(len(kv_bytes), torch.Size(padded), kv_dtype, memory_format)
        meta_bytes = metadata.serialize()
        return meta_bytes + kv_bytes

    def _actual_shape_from_padded(self, padded_shape: torch.Size) -> torch.Size:
        # Remove trailing zeros after first real dim
        actual: List[int] = []
        for d in padded_shape:
            if d == 0 and len(actual) > 0:
                break
            actual.append(int(d))
        return torch.Size(actual) if actual else torch.Size([1])
