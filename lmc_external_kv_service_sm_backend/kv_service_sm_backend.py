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
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Dict, List, Optional, Tuple
import asyncio
import threading

# Third Party
from aiohttp import payload
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


class _KVStream(payload.Payload):
    """Two-part streaming body: [metadata][payload memoryview].

    Avoids concatenating multi-GB buffers into a single bytes object.
    """

    def __init__(self, meta: bytes, buf_mv: memoryview, chunk_bytes: int):
        super().__init__(None, content_type="application/octet-stream")
        self._meta = meta
        self._buf = buf_mv
        self._step = max(1, int(chunk_bytes))
        # Expose size so aiohttp can set Content-Length (no chunked encoding)
        self.size = len(self._meta) + buf_mv.nbytes

    async def write(self, writer):
        # metadata is tiny; write at once
        await writer.write(self._meta)
        # stream payload in slices to avoid large Python copies
        mv = self._buf
        step = self._step
        total = mv.nbytes
        for start in range(0, total, step):
            end = min(start + step, total)
            await writer.write(mv[start:end])


class KVServiceSMBackend(StorageBackendInterface):
    """
    A storage backend that uses KVServiceSM KV cache daemon for layerwise caching.

    This backend is designed for layerwise mode operations and provides:
    - Direct shared memory access via leases (no local_cpu_backend buffer)
    - HTTP API integration with KVServiceSM daemon
    - Efficient batch operations for layer-by-layer processing
    - Memory-mapped file access for zero-copy operations

    Configuration requires:
    - kv_service_sm_url: URL of the KVServiceSM daemon (e.g., "http://localhost:9200")
    - shared_memory_name: Optional name for shared memory segment
    - bucket_name: Bucket name for priority/organization (default: "lmcache")
    """

    def __init__(
        self,
        config,
        metadata,
        loop,
        memory_allocator,
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

        # KVServiceSM configuration
        extra_config = getattr(config, "extra_config", None) or {}
        self.base_url = extra_config.get("kv_service_sm_url", "http://localhost:9200")
        self.shared_memory_name = extra_config.get(
            "kv_service_sm_shared_memory_name", "shared_memory"
        )
        self.bucket_name = extra_config.get("kv_service_sm_bucket", "lmcache")
        
        # Separate timeout configurations for different operations
        self.lease_timeout_ms = extra_config.get("kv_service_sm_lease_timeout_ms", 500)
        self.put_timeout_ms = extra_config.get("kv_service_sm_put_timeout_ms", 5000)
        self.release_timeout_ms = extra_config.get("kv_service_sm_release_timeout_ms", 2000)

        # Performance optimizations for scale
        # Size of each write slice when streaming PUTs. 4–16MB are good defaults.
        self.put_chunk_bytes = int(
            extra_config.get("kv_service_sm_put_chunk_bytes", 8 * 1024 * 1024)
        )

        # Bounded deserialization pool for GET copies (keeps event loop free)
        self.deserialization_threads = int(
            extra_config.get("kv_service_sm_deserialization_threads", 16)
        )

        self.max_connections = extra_config.get("kv_service_sm_max_connections", 256)
        self.max_connections_per_host = extra_config.get(
            "kv_service_sm_max_connections_per_host", 128
        )
        self.serialization_threads = extra_config.get(
            "kv_service_sm_serialization_threads", 16
        )

        # HTTP connection pool for high-scale performance
        self.http_session: Optional[aiohttp.ClientSession] = None
        self.session_lock = asyncio.Lock()

        # Thread pool for CPU-bound serialization operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.serialization_threads,
            thread_name_prefix="kv-service-sm-serialize",
        )

        # Separate pool for GET reconstruction/copies
        self.deser_pool = ThreadPoolExecutor(
            max_workers=self.deserialization_threads,
            thread_name_prefix="kv-service-sm-deser",
        )

        # Cached size of serialized RemoteMetadata headers. Fallback to the
        # legacy fixed-width (7 * int32) layout if the runtime calculation
        # fails for any reason so older daemons remain compatible.
        try:
            sample_metadata = RemoteMetadata(
                0,
                torch.Size([0, 0, 0, 0]),
                torch.float32,
                getattr(torch, "contiguous_format", 0),
            )
            self.metadata_header_bytes = len(sample_metadata.serialize())
        except Exception:
            self.metadata_header_bytes = 4 * 7

        self.lease_lock = threading.Lock()
        self.leases: Dict[CacheEngineKey, LeaseInfo] = {}
        self.lease_id_to_key: Dict[str, CacheEngineKey] = {}

        # Put task tracking - required by interface
        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        # Shared memory mapping (lazy initialization)
        self.shared_memory_obj: Optional[shared_memory.SharedMemory] = None
        self.shared_memory_map: Optional[memoryview] = None
        self.shared_memory_lock = threading.Lock()

        # Note: No CUDA streams needed for simple byte array approach

        logger.info(
            f"KVServiceSMBackend initialized with URL: {self.base_url}, "
            f"bucket: {self.bucket_name}, shared_memory: {self.shared_memory_name}, "
            f"max_connections: {self.max_connections}, "
            f"max_connections_per_host: {self.max_connections_per_host}, "
            f"serialization_threads: {self.serialization_threads}, "
            f"timeouts (ms): lease={self.lease_timeout_ms}, "
            f"put={self.put_timeout_ms}, release={self.release_timeout_ms}"
        )

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        """Check if key exists in KVServiceSM cache."""
        try:
            # Use original worker_id for cache lookup
            lookup_key = CacheEngineKey(
                key.fmt,
                key.model_name,
                key.world_size,
                key.worker_id,
                key.chunk_hash,
                key.request_configs,
            )

            # DEBUG: Log the cache key lookup details
            key_str = self._key_to_string(lookup_key)
            if lookup_key in self.leases:
                logger.debug(f"Key found in local leases - CACHE HIT")
                return True

            lease_info = asyncio.run_coroutine_threadsafe(
                self._acquire_lease(lookup_key), self.loop
            ).result()

            result = lease_info is not None
            logger.debug(f"Lease acquisition result: {'SUCCESS' if result else 'FAILED'}")
            return result
        except Exception as e:
            logger.error(f"Exception during key existence check: {e}")
            return False

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        """Check if key is currently being stored."""
        with self.put_lock:
            return key in self.put_tasks

    @_lmcache_nvtx_annotate
    def batched_submit_put_task(
        self,
        keys: List[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
    ) -> Optional[List[Future]]:
        """Submit a batch of PUT tasks to KVServiceSM."""
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)
        return None

    @_lmcache_nvtx_annotate
    def submit_put_task(
        self, key: CacheEngineKey, memory_obj: MemoryObj
    ) -> Optional[Future]:
        """Submit a single PUT task to KVServiceSM."""
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        self.loop.call_soon_threadsafe(
            asyncio.create_task, self._async_put(key, memory_obj)
        )
        return None

    async def _ensure_http_session(self) -> aiohttp.ClientSession:
        """Ensure HTTP session with connection pooling is initialized."""
        if self.http_session is None:
            async with self.session_lock:
                if self.http_session is None:  # Double-check locking
                    connector = aiohttp.TCPConnector(
                        limit=self.max_connections,  # Total connection pool size
                        limit_per_host=self.max_connections_per_host,  # Per-host connection limit  # noqa: E501
                        ttl_dns_cache=300,  # DNS cache TTL (5 min)
                        use_dns_cache=True,  # Enable DNS caching
                        keepalive_timeout=30,  # Keep connections alive
                        enable_cleanup_closed=True,  # Clean up closed connections
                    )

                    timeout = aiohttp.ClientTimeout(
                        total=30,  # Total timeout for request
                        connect=5,  # Connection timeout
                        sock_read=10,  # Socket read timeout
                    )

                    self.http_session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": "LMCache-KVServiceSMBackend/1.0"},
                    )
                    logger.info(
                        f"Created HTTP session with {self.max_connections} max connections"  # noqa: E501
                    )

        return self.http_session

    async def _http_request(
        self, method: str, url: str, data=None, params=None, timeout=5.0
    ):
        """Optimized HTTP request with connection pooling."""
        try:
            session = await self._ensure_http_session()
            request_timeout = aiohttp.ClientTimeout(total=timeout)

            async with session.request(
                method, url, data=data, params=params, timeout=request_timeout
            ) as response:
                result = {
                    "status": response.status,
                    "data": await response.read()
                    if method in ["PUT", "POST"]
                    else None,
                    "json": await response.json()
                    if response.content_type == "application/json"
                    else None,
                }
                return result
        except asyncio.TimeoutError:
            logger.warning(f"HTTP {method} request timeout for {url} (timeout: {timeout}s)")
            return None
        except aiohttp.ClientError as e:
            logger.error(f"HTTP {method} client error for {url}: {e}")
            return None
        except Exception as e:
            logger.error(f"HTTP {method} request failed for {url}: {e}")
            return None

    @_lmcache_nvtx_annotate
    async def _async_put(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        """Async PUT with zero-copy(ish) streaming: send [meta][payload] without concatenation."""
        serialization_start = None
        http_start = None
        buf_mv: Optional[memoryview] = None

        try:
            # Use original worker_id for cache storage
            store_key = CacheEngineKey(
                key.fmt,
                key.model_name,
                key.world_size,
                key.worker_id,
                key.chunk_hash,
                key.request_configs,
            )

            key_str = self._key_to_string(store_key)
            url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}"

            # Build *only* metadata bytes on the pool (tiny); DO NOT materialize [meta]+[payload]
            loop = asyncio.get_running_loop()
            serialization_start = loop.time()
            meta_bytes = await loop.run_in_executor(
                self.thread_pool, self._metadata_only_bytes, memory_obj
            )
            serialization_time = loop.time() - serialization_start

            # Prepare a *zero-copy view* over the existing payload buffer.
            buf_mv = self._buffer_to_byte_view(memory_obj.byte_array)

            # Stream the HTTP body: [meta][payload] in slices; no giant concat.
            data = _KVStream(meta_bytes, buf_mv, self.put_chunk_bytes)

            # HTTP request on event loop (I/O-bound operation)
            http_start = loop.time()
            result = await self._http_request(
                "PUT", url, data=data, timeout=self.put_timeout_ms / 1000.0
            )
            http_time = loop.time() - http_start

            if result and result["status"] == 200:
                logger.debug(
                    f"Successfully stored key {key}: {len(meta_bytes) + buf_mv.nbytes} bytes, "
                    f"serialize: {serialization_time * 1000:.1f}ms, "
                    f"http: {http_time * 1000:.1f}ms"
                )
            else:
                status = result["status"] if result else "TIMEOUT"
                logger.error(f"Failed to store key {key}: HTTP {status}")
        except Exception as e:
            logger.exception(f"Exception during PUT for key {key}: {e}")
        finally:
            if buf_mv is not None:
                try:
                    buf_mv.release()
                except (AttributeError, BufferError):
                    pass
            try:
                memory_obj.ref_count_down()
            except Exception:
                pass
            # Always cleanup task tracking
            with self.put_lock:
                self.put_tasks.discard(key)

    def submit_prefetch_task(self, key: CacheEngineKey) -> Optional[Future]:
        """Submit prefetch task - unified with other GET operations."""
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    def get_blocking(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Blocking GET operation from KVServiceSM."""
        try:
            return asyncio.run_coroutine_threadsafe(
                self._get_memory_obj(key), self.loop
            ).result()
        except Exception as e:
            logger.error(f"GET operation exception for key {key}: {e}")
            return None

    def get_non_blocking(self, key: CacheEngineKey) -> Optional[Future]:
        """Non-blocking GET operation."""
        return asyncio.run_coroutine_threadsafe(self._get_memory_obj(key), self.loop)

    async def _get_memory_obj(self, key: CacheEngineKey) -> Optional[MemoryObj]:
        """Unified GET method: lease → read → reconstruct → release."""
        # Use original worker_id for cache operations
        lookup_key = CacheEngineKey(
            key.fmt,
            key.model_name,
            key.world_size,
            key.worker_id,
            key.chunk_hash,
            key.request_configs,
        )
        
        lease_info = None

        # Step 1: Acquire lease
        with self.lease_lock:
            if lookup_key in self.leases:
                lease_info = self.leases.get(lookup_key)

        if lease_info is None:
            lease_info = await self._acquire_lease(lookup_key)

        if lease_info is None:
            return None

        try:
            # Step 2: Ensure SHM then reconstruct off the event loop (deser pool)
            if not await self._ensure_shared_memory():
                return None
            loop = asyncio.get_running_loop()
            metadata = await loop.run_in_executor(
                self.deser_pool, self._read_metadata_from_lease_sync, lease_info
            )
            if metadata is None:
                return None

            actual_shape = self._unpadded_shape(metadata.shape)
            try:
                memory_obj = self.memory_allocator.allocate(
                    actual_shape, metadata.dtype, metadata.fmt
                )
            except Exception as exc:
                logger.error(f"Failed to allocate memory for key {key}: {exc}")
                return None

            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key}")
                return None

            success = await loop.run_in_executor(
                self.deser_pool,
                self._copy_payload_from_lease_sync,
                lease_info,
                metadata.length,
                memory_obj,
            )
            if not success:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass
                return None

            logger.debug(
                f"Reconstructed key={key} shape={actual_shape} "
                f"dtype={metadata.dtype} fmt={metadata.fmt}"
            )
            return memory_obj
        finally:
            # Step 3: Always release lease
            await self._release_lease(lease_info.lease_id)

    async def _acquire_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Acquire a lease for the given key from KVServiceSM daemon."""
        key_str = self._key_to_string(key)
        url = f"{self.base_url}/v1/kv/{self.bucket_name}/{key_str}/leases"
        params = {"timeout_ms": self.lease_timeout_ms}
        
        logger.debug(f"Acquiring lease for key {key_str} with timeout {self.lease_timeout_ms}ms")

        result = await self._http_request(
            "POST", url, params=params, timeout=self.lease_timeout_ms / 1000.0
        )

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

            logger.debug(f"Lease acquired successfully: {lease_info.lease_id}")
            return lease_info
        
        logger.debug(f"Failed to acquire lease for {key_str} - timeout after {self.lease_timeout_ms}ms")
        return None

    async def _release_lease(self, lease_id: str) -> bool:
        url = f"{self.base_url}/v1/leases/{lease_id}/release"
        result = await self._http_request("POST", url, timeout=self.release_timeout_ms / 1000.0)

        # Consider both 200 OK and 404 Not Found as "success" since in both cases
        # the lease is no longer active on the server
        success = False
        if result:
            if result["status"] == 200:
                success = True
            elif result["status"] == 404:
                logger.debug(
                    f"Lease {lease_id} not found on server, already released or expired"
                )
                success = True
            else:
                logger.warning(
                    f"Failed to release lease {lease_id}, "
                    f"status: {result['status']}, "
                    f"response: {result.get('json', {})}"
                )

        if success:
            with self.lease_lock:
                if lease_id in self.lease_id_to_key:
                    key = self.lease_id_to_key[lease_id]
                    if key in self.leases:
                        del self.leases[key]
                    del self.lease_id_to_key[lease_id]

        return success

    def _read_metadata_from_lease_sync(self, lease_info: LeaseInfo) -> Optional[RemoteMetadata]:
        """Read and deserialize the RemoteMetadata header for a lease."""
        mv = self.shared_memory_map
        if mv is None:
            return None
        if not lease_info.offsets:
            logger.error("No offsets present in lease to read metadata header")
            return None

        header = bytearray(self.metadata_header_bytes)
        filled = 0
        for off, ln in lease_info.offsets:
            if filled >= self.metadata_header_bytes:
                break
            n = min(ln, self.metadata_header_bytes - filled)
            header[filled : filled + n] = mv[off : off + n]
            filled += n

        if filled < self.metadata_header_bytes:
            logger.error(
                "Insufficient data for metadata header: "
                f"expected {self.metadata_header_bytes}, got {filled}"
            )
            return None

        try:
            return RemoteMetadata.deserialize(header)
        except Exception as exc:
            logger.error(f"Failed to parse metadata header: {exc}")
            return None

    def _unpadded_shape(self, padded_shape: torch.Size) -> torch.Size:
        """Remove trailing zero padding from a 4D RemoteMetadata shape."""
        actual: List[int] = []
        for dim in padded_shape:
            if dim == 0 and actual:
                break
            actual.append(dim)
        return torch.Size(actual) if actual else torch.Size([1])

    def _copy_payload_from_lease_sync(
        self, lease_info: LeaseInfo, payload_len: int, memory_obj: MemoryObj
    ) -> bool:
        """Copy payload bytes from shared memory offsets into ``memory_obj``."""
        mv = self.shared_memory_map
        if mv is None:
            return False

        dst_view: Optional[memoryview] = None
        try:
            dst_view = self._buffer_to_byte_view(memory_obj.byte_array)
            if payload_len > dst_view.nbytes:
                logger.error(
                    "Data size mismatch: expected buffer >= %d, got %d",
                    payload_len,
                    dst_view.nbytes,
                )
                return False

            written = 0
            payload_skip = self.metadata_header_bytes
            for off, ln in lease_info.offsets:
                if payload_skip:
                    if ln <= payload_skip:
                        payload_skip -= ln
                        continue
                    off += payload_skip
                    ln -= payload_skip
                    payload_skip = 0
                if ln <= 0:
                    continue
                remaining = payload_len - written
                if remaining <= 0:
                    break
                n = min(ln, remaining)
                dst_view[written : written + n] = mv[off : off + n]
                written += n

            if written != payload_len:
                logger.error(
                    "Size mismatch while copying payload: expected %d, wrote %d",
                    payload_len,
                    written,
                )
                return False
            return True
        finally:
            if dst_view is not None:
                try:
                    dst_view.release()
                except (AttributeError, BufferError):
                    pass

    async def _ensure_shared_memory(self) -> bool:
        """Ensure shared memory is initialized and accessible."""
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                return True

            if self.shared_memory_name is None:
                logger.error("No shared memory name configured")
                return False

            try:
                # Try to open existing shared memory segments created by
                # KVServiceSM daemon
                self.shared_memory_obj = shared_memory.SharedMemory(
                    name=self.shared_memory_name, create=False
                )
                self.shared_memory_map = memoryview(self.shared_memory_obj.buf)

                logger.info(
                    f"KVServiceSMBackend: Successfully opened "
                    f"shared memory: {self.shared_memory_name} "
                    f"(size: {len(self.shared_memory_map)} bytes)"
                )
                return True

            except FileNotFoundError:
                logger.error(
                    f"KVServiceSMBackend: CRITICAL - "
                    f"Shared memory segment '{self.shared_memory_name}' not found. "
                    f"Is KVServiceSM daemon running and creating shared memory?"
                )
                return False
            except Exception as e:
                logger.error(
                    f"KVServiceSMBackend: CRITICAL - "
                    f"Failed to initialize shared memory: {e}"
                )
                return False
        return None

    def pin(self, key: CacheEngineKey) -> bool:
        """Pin operation - not implemented for KVServiceSM."""
        return True

    def unpin(self, key: CacheEngineKey) -> bool:
        """Unpin operation - not implemented for KVServiceSM."""
        return True

    def remove(self, key: CacheEngineKey, force: bool = True) -> bool:
        """Remove operation - not implemented for KVServiceSM."""
        return True

    def close(self) -> None:
        """Close the backend and release resources."""
        # Release all leases
        with self.lease_lock:
            lease_ids_to_release = list(self.lease_id_to_key.keys())

        try:
            for lease_id in lease_ids_to_release:
                asyncio.run_coroutine_threadsafe(
                    self._release_lease(lease_id), self.loop
                ).result(timeout=2.0)

            logger.info(f"Released {len(lease_ids_to_release)} key leases")
        except Exception as e:
            logger.error(f"Error releasing key leases: {e}")

        # Close HTTP session and connection pool
        if self.http_session is not None:
            try:
                # Schedule closure on the event loop
                asyncio.run_coroutine_threadsafe(
                    self.http_session.close(), self.loop
                ).result(timeout=5.0)
                logger.info("HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing HTTP session: {e}")
            self.http_session = None

        # Shutdown thread pool
        if self.thread_pool is not None:
            try:
                self.thread_pool.shutdown(wait=True)
                logger.info("Thread pool shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")

        # Shutdown deserialization pool
        if self.deser_pool is not None:
            try:
                self.deser_pool.shutdown(wait=True)
                logger.info("Deserialization thread pool shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down deserialization pool: {e}")

        # Note: No CUDA streams to cleanup in simple approach

        # Close shared memory resources
        with self.shared_memory_lock:
            if self.shared_memory_map is not None:
                try:
                    self.shared_memory_map.release()
                except Exception as e:
                    logger.error(f"Error releasing shared memory map: {e}")
                self.shared_memory_map = None

            if self.shared_memory_obj is not None:
                try:
                    self.shared_memory_obj.close()
                except Exception as e:
                    logger.error(f"Error closing shared memory: {e}")
                self.shared_memory_obj = None

        logger.info("KVServiceSMBackend closed with all resources cleaned up.")

    # Helper methods

    def _key_to_string(self, key: CacheEngineKey) -> str:
        """Convert CacheEngineKey to string format for HTTP API.

        Use URL encoding for complete safety instead of character replacement.
        This avoids conflicts with existing underscores in keys.
        """
        # Standard
        import urllib.parse

        key_str = key.to_string()
        encoded_key = urllib.parse.quote(key_str, safe="")

        return encoded_key

    def _buffer_nbytes(self, buffer) -> int:
        """Return the number of bytes exposed by a buffer-compatible object.

        Falls back to copying via ``bytes()`` if the object does not implement the
        buffer protocol.
        """
        try:
            mv = memoryview(buffer)
        except TypeError:
            return len(bytes(buffer))

        try:
            return mv.nbytes
        finally:
            try:
                mv.release()
            except (AttributeError, BufferError):
                pass

    def _buffer_to_byte_view(self, buffer) -> memoryview:
        """Return a 1-D unsigned-byte memoryview over ``buffer``.

        Attempts to avoid copies by casting contiguous buffers to ``'B'``.
        When casting fails (e.g., non-contiguous exports), falls back to copying
        the data into a new ``bytes`` object.
        """
        try:
            mv = memoryview(buffer)
        except TypeError:
            return memoryview(bytes(buffer))

        if mv.format in ("B", "b") and mv.ndim == 1 and getattr(mv, "c_contiguous", True):
            return mv

        try:
            cast_mv = mv.cast("B")
        except (TypeError, ValueError):
            cast_mv = None

        if cast_mv is not None and getattr(cast_mv, "c_contiguous", True):
            try:
                mv.release()
            except (AttributeError, BufferError):
                pass
            return cast_mv

        data = mv.tobytes()
        try:
            mv.release()
        except (AttributeError, BufferError):
            pass
        return memoryview(data)

    def _metadata_only_bytes(self, memory_obj: MemoryObj) -> bytes:
        """Return just the RemoteMetadata header bytes (no payload).

        Used by streaming PUT to avoid concatenating multi-GB payloads.
        """

        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        # Compute payload length from current buffer view
        payload_len = self._buffer_nbytes(memory_obj.byte_array)

        # RemoteMetadata expects 4D shape; pad/truncate accordingly
        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded_shape) > 4:
            padded_shape = list(kv_shape[:4])
        padded_torch_shape = torch.Size(padded_shape)

        metadata = RemoteMetadata(payload_len, padded_torch_shape, kv_dtype, memory_format)
        return metadata.serialize()

    def _memory_obj_to_bytes(self, memory_obj: MemoryObj) -> bytes:
        """Ultra-simple serialization using Redis-style approach.

        Format: [RemoteMetadata struct][byte_array]
        - Uses existing LMCache protocol
        - No tensor conversion needed
        - Works with all dtypes including BFloat16
        """
        # Simple approach: use the existing byte_array from MemoryObj
        kv_bytes = memory_obj.byte_array
        kv_shape = memory_obj.get_shape()
        kv_dtype = memory_obj.get_dtype()
        memory_format = memory_obj.get_memory_format()

        # RemoteMetadata expects exactly 4 dimensions - pad with zeros if needed
        # Following the protocol.py comment:
        # "Pass in shape [x, 0, 0, 0] if it is a bytes memory object"
        padded_shape = list(kv_shape) + [0] * (4 - len(kv_shape))
        if len(padded_shape) > 4:
            # If shape has more than 4 dimensions, we need to flatten or
            # handle differently
            logger.warning(
                f"Shape has {len(kv_shape)} dimensions, "
                f"truncating to first 4: {kv_shape}"
            )
            padded_shape = list(kv_shape[:4])

        # Convert to torch.Size with exactly 4 dimensions
        padded_torch_shape = torch.Size(padded_shape)

        # Use existing RemoteMetadata from protocol.py
        metadata = RemoteMetadata(
            len(kv_bytes), padded_torch_shape, kv_dtype, memory_format
        )
        metadata_bytes = metadata.serialize()

        # Format: [metadata][byte_array]
        result = metadata_bytes + kv_bytes

        logger.debug(
            f"Simple serialization: original_shape={kv_shape}, "
            f"padded_shape={padded_torch_shape}, dtype={kv_dtype}, "
            f"size={len(result)} bytes (metadata: {len(metadata_bytes)}, "
            f"data: {len(kv_bytes)})"
        )
        return result
