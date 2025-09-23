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

        # Performance optimizations for scale - deprecated but kept for backward compatibility
        self.max_connections = extra_config.get("kv_service_sm_max_connections", 256)
        self.max_connections_per_host = extra_config.get(
            "kv_service_sm_max_connections_per_host", 128
        )
        
        # Separate connection pool configurations for lease and PUT operations
        self.lease_max_connections = extra_config.get(
            "kv_service_sm_lease_max_connections", 128
        )
        self.lease_max_connections_per_host = extra_config.get(
            "kv_service_sm_lease_max_connections_per_host", 64
        )
        self.put_max_connections = extra_config.get(
            "kv_service_sm_put_max_connections", 64
        )
        self.put_max_connections_per_host = extra_config.get(
            "kv_service_sm_put_max_connections_per_host", 32
        )
        
        self.serialization_threads = extra_config.get(
            "kv_service_sm_serialization_threads", 16
        )

        # Separate HTTP sessions for lease and PUT operations
        self.lease_http_session: Optional[aiohttp.ClientSession] = None
        self.put_http_session: Optional[aiohttp.ClientSession] = None
        self.lease_session_lock = asyncio.Lock()
        self.put_session_lock = asyncio.Lock()

        # Thread pool for CPU-bound serialization operations
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.serialization_threads,
            thread_name_prefix="kv-service-sm-serialize",
        )

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
            f"lease connections: {self.lease_max_connections} (per-host: {self.lease_max_connections_per_host}), "
            f"put connections: {self.put_max_connections} (per-host: {self.put_max_connections_per_host}), "
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

    async def _ensure_http_session(self, session_type: str = "lease") -> aiohttp.ClientSession:
        """Ensure HTTP session with connection pooling is initialized.
        
        Args:
            session_type: Either "lease" or "put" to get the appropriate session
        """
        if session_type == "put":
            if self.put_http_session is None:
                async with self.put_session_lock:
                    if self.put_http_session is None:  # Double-check locking
                        connector = aiohttp.TCPConnector(
                            limit=self.put_max_connections,
                            limit_per_host=self.put_max_connections_per_host,
                            ttl_dns_cache=300,  # DNS cache TTL (5 min)
                            use_dns_cache=True,  # Enable DNS caching
                            keepalive_timeout=15,  # Shorter keepalive for PUT connections
                            enable_cleanup_closed=True,  # Clean up closed connections
                        )

                        timeout = aiohttp.ClientTimeout(
                            total=30,  # Total timeout for request
                            connect=5,  # Connection timeout
                            sock_read=10,  # Socket read timeout
                        )

                        self.put_http_session = aiohttp.ClientSession(
                            connector=connector,
                            timeout=timeout,
                            headers={"User-Agent": "LMCache-KVServiceSMBackend/1.0"},
                        )
                        logger.info(
                            f"Created PUT HTTP session with {self.put_max_connections} max connections, "
                            f"{self.put_max_connections_per_host} per host"
                        )

            return self.put_http_session
        else:  # Default to lease session
            if self.lease_http_session is None:
                async with self.lease_session_lock:
                    if self.lease_http_session is None:  # Double-check locking
                        connector = aiohttp.TCPConnector(
                            limit=self.lease_max_connections,
                            limit_per_host=self.lease_max_connections_per_host,
                            ttl_dns_cache=300,  # DNS cache TTL (5 min)
                            use_dns_cache=True,  # Enable DNS caching
                            keepalive_timeout=60,  # Longer keepalive for lease connections
                            enable_cleanup_closed=True,  # Clean up closed connections
                        )

                        timeout = aiohttp.ClientTimeout(
                            total=30,  # Total timeout for request
                            connect=5,  # Connection timeout
                            sock_read=10,  # Socket read timeout
                        )

                        self.lease_http_session = aiohttp.ClientSession(
                            connector=connector,
                            timeout=timeout,
                            headers={"User-Agent": "LMCache-KVServiceSMBackend/1.0"},
                        )
                        logger.info(
                            f"Created Lease HTTP session with {self.lease_max_connections} max connections, "
                            f"{self.lease_max_connections_per_host} per host"
                        )

            return self.lease_http_session

    async def _http_request(
        self, method: str, url: str, data=None, params=None, timeout=5.0
    ):
        """Optimized HTTP request with connection pooling.
        
        Automatically routes to the appropriate session based on operation type:
        - PUT operations -> put_http_session
        - Lease operations (acquire/release) -> lease_http_session
        """
        try:
            # Determine session type based on operation
            if method == "PUT":
                session_type = "put"
            else:
                # All other operations (lease acquire, release, etc.) use lease session
                session_type = "lease"
            
            session = await self._ensure_http_session(session_type)
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
        """Optimized async PUT operation with thread pool for serialization and early memory release."""  # noqa: E501
        serialization_start = None
        http_start = None
        memory_released = False

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

            # OPTIMIZATION 1: Serialize tensor on thread pool (CPU-bound operation)
            loop = asyncio.get_running_loop()
            serialization_start = loop.time()
            data = await loop.run_in_executor(
                self.thread_pool, self._memory_obj_to_bytes, memory_obj
            )
            serialization_time = loop.time() - serialization_start

            # OPTIMIZATION 2: Early memory release - tensor copied to bytes, release GPU memory  # noqa: E501
            memory_obj.ref_count_down()
            memory_released = True

            # HTTP request on event loop (I/O-bound operation)
            http_start = loop.time()
            result = await self._http_request(
                "PUT", url, data=data, timeout=self.put_timeout_ms / 1000.0
            )
            http_time = loop.time() - http_start

            if result and result["status"] == 200:
                logger.debug(
                    f"Successfully stored key {key}: {len(data)} bytes, "
                    f"serialize: {serialization_time * 1000:.1f}ms, "
                    f"http: {http_time * 1000:.1f}ms"
                )
            else:
                status = result["status"] if result else "TIMEOUT"
                logger.error(f"Failed to store key {key}: HTTP {status}")
        except Exception as e:
            logger.exception(f"Exception during PUT for key {key}: {e}")
            # Ensure memory is released even on error
            if not memory_released:
                try:
                    memory_obj.ref_count_down()
                except Exception:
                    pass  # May have already been released or failed for other reasons
        finally:
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
            # Step 2: Read and reconstruct tensor from shared memory
            result = await self._read_tensor_from_lease(key, lease_info)
            return result
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

    async def _read_tensor_from_lease(
        self, key: CacheEngineKey, lease_info: LeaseInfo
    ) -> Optional[MemoryObj]:
        """Unified tensor reading from lease - handles both single and multi-block cases."""  # noqa: E501
        if not await self._ensure_shared_memory():
            return None

        if not lease_info.offsets:
            logger.error(f"No offsets in lease for key {key}")
            return None

        try:
            # Read all data (single block is just multi-block with length=1)
            total_data = bytearray()
            # Ensure shared_memory_map is not None (guaranteed by _ensure_shared_memory check above)  # noqa: E501
            if self.shared_memory_map is None:
                logger.error(
                    "Shared memory map is None despite successful initialization check"  # noqa: E501
                )
                return None

            for offset, length in lease_info.offsets:
                chunk = bytes(self.shared_memory_map[offset : offset + length])
                total_data.extend(chunk)

            # Validate total size
            if len(total_data) != lease_info.total_size:
                logger.error(
                    f"Size mismatch: expected {lease_info.total_size}, got {len(total_data)}"  # noqa: E501
                )
                return None

            # Parse simple format: [RemoteMetadata struct][byte_array]
            metadata_size = 4 * 7  # RemoteMetadata is 7 integers
            if len(total_data) < metadata_size:
                logger.error("Insufficient data for metadata header")
                return None

            try:
                # Parse using existing RemoteMetadata
                metadata = RemoteMetadata.deserialize(total_data[:metadata_size])
                kv_bytes = total_data[metadata_size : metadata_size + metadata.length]

                if len(kv_bytes) != metadata.length:
                    logger.error(
                        f"Data size mismatch: expected {metadata.length}, "
                        f"got {len(kv_bytes)}"
                    )
                    return None

                # Simple reconstruction using existing allocator
                return await self._create_simple_tensor_from_metadata(
                    key, metadata, kv_bytes
                )

            except Exception as e:
                logger.error(f"Failed to parse simple metadata: {e}")
                return None

        except Exception as e:
            logger.error(f"Error reading tensor from lease for key {key}: {e}")
            return None

    async def _create_simple_tensor_from_metadata(
        self, key: CacheEngineKey, metadata: RemoteMetadata, kv_bytes: bytes
    ) -> Optional[MemoryObj]:
        """Simple tensor reconstruction using Redis-style approach."""
        try:
            # RemoteMetadata uses 4D padded shape - restore original shape
            # by removing trailing zeros
            original_shape = metadata.shape
            # Remove trailing zeros to get the actual shape
            actual_shape_list: List[int] = []
            for dim in original_shape:
                if dim == 0 and len(actual_shape_list) > 0:
                    # Stop at first zero after we have at least one dimension
                    break
                actual_shape_list.append(dim)

            # Convert back to torch.Size
            actual_shape = (
                torch.Size(actual_shape_list) if actual_shape_list else torch.Size([1])
            )

            # Allocate memory object using existing allocator with actual shape
            memory_obj = self.memory_allocator.allocate(
                actual_shape, metadata.dtype, metadata.fmt
            )
            if memory_obj is None:
                logger.error(f"Failed to allocate memory for key {key}")
                return None

            # Direct byte copy - no tensor conversion needed!
            if isinstance(memory_obj.byte_array, memoryview):
                view = memory_obj.byte_array
                if view.format == "<B":
                    view = view.cast("B")
            else:
                view = memoryview(memory_obj.byte_array)

            # Copy data directly to byte array
            view[: metadata.length] = kv_bytes

            logger.debug(
                f"Simple reconstruction: actual_shape={actual_shape}, "
                f"dtype={metadata.dtype}, format={metadata.fmt}"
            )
            return memory_obj

        except Exception as e:
            logger.error(f"Error in simple tensor reconstruction for key {key}: {e}")
            return None

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

        # Close both HTTP sessions and their connection pools
        if self.lease_http_session is not None:
            try:
                # Schedule closure on the event loop
                asyncio.run_coroutine_threadsafe(
                    self.lease_http_session.close(), self.loop
                ).result(timeout=5.0)
                logger.info("Lease HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing lease HTTP session: {e}")
            self.lease_http_session = None

        if self.put_http_session is not None:
            try:
                # Schedule closure on the event loop
                asyncio.run_coroutine_threadsafe(
                    self.put_http_session.close(), self.loop
                ).result(timeout=5.0)
                logger.info("PUT HTTP session closed")
            except Exception as e:
                logger.error(f"Error closing PUT HTTP session: {e}")
            self.put_http_session = None

        # Shutdown thread pool
        if self.thread_pool is not None:
            try:
                self.thread_pool.shutdown(wait=True)
                logger.info("Thread pool shutdown complete")
            except Exception as e:
                logger.error(f"Error shutting down thread pool: {e}")

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
