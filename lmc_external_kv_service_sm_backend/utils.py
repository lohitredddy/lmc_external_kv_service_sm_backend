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
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, Optional
import asyncio
import time

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey

from .cache import LeaseInfo

logger = init_logger(__name__)


class OperationPriority(IntEnum):
    """Priority levels for operations (lower number = higher priority)."""
    EXISTENCE_CHECK = 0  # Highest - contains() operations
    LEASE_RELEASE = 1    # Prevent resource leaks
    ACTIVE_GET = 2       # User-facing operations
    PREFETCH = 3         # Background prefetch
    PUT = 4              # Storage operations
    MAINTENANCE = 5      # Cleanup, stats


@dataclass
class OperationMetrics:
    """Metrics tracking for backend operations."""
    get_requests: int = 0
    get_successes: int = 0
    get_failures: int = 0
    get_total_bytes: int = 0
    get_total_time_ms: float = 0
    
    put_requests: int = 0
    put_successes: int = 0
    put_failures: int = 0
    put_total_bytes: int = 0
    put_total_time_ms: float = 0
    
    lease_acquisitions: int = 0
    lease_releases: int = 0
    lease_timeouts: int = 0
    
    cache_hits: int = 0
    cache_misses: int = 0
    
    http_retries: int = 0
    http_failures: int = 0
    
    start_time: float = field(default_factory=time.time)
    
    def record_get(self, success: bool, size: int = 0, time_ms: float = 0):
        """Record a GET operation."""
        self.get_requests += 1
        if success:
            self.get_successes += 1
            self.get_total_bytes += size
        else:
            self.get_failures += 1
        self.get_total_time_ms += time_ms
    
    def record_put(self, success: bool, size: int = 0, time_ms: float = 0):
        """Record a PUT operation."""
        self.put_requests += 1
        if success:
            self.put_successes += 1
            self.put_total_bytes += size
        else:
            self.put_failures += 1
        self.put_total_time_ms += time_ms
    
    def get_summary(self) -> Dict[str, any]:
        """Get a summary of metrics."""
        uptime = time.time() - self.start_time
        
        get_success_rate = (
            self.get_successes / self.get_requests 
            if self.get_requests > 0 else 0
        )
        put_success_rate = (
            self.put_successes / self.put_requests 
            if self.put_requests > 0 else 0
        )
        cache_hit_rate = (
            self.cache_hits / (self.cache_hits + self.cache_misses)
            if (self.cache_hits + self.cache_misses) > 0 else 0
        )
        
        return {
            "uptime_seconds": uptime,
            "get": {
                "requests": self.get_requests,
                "successes": self.get_successes,
                "failures": self.get_failures,
                "success_rate": get_success_rate,
                "total_bytes": self.get_total_bytes,
                "avg_time_ms": (
                    self.get_total_time_ms / self.get_requests
                    if self.get_requests > 0 else 0
                ),
            },
            "put": {
                "requests": self.put_requests,
                "successes": self.put_successes,
                "failures": self.put_failures,
                "success_rate": put_success_rate,
                "total_bytes": self.put_total_bytes,
                "avg_time_ms": (
                    self.put_total_time_ms / self.put_requests
                    if self.put_requests > 0 else 0
                ),
            },
            "leases": {
                "acquisitions": self.lease_acquisitions,
                "releases": self.lease_releases,
                "timeouts": self.lease_timeouts,
            },
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": cache_hit_rate,
            },
            "http": {
                "retries": self.http_retries,
                "failures": self.http_failures,
            },
        }


class LeaseManager:
    """Manages active leases with tracking, cleanup, and caching."""
    
    def __init__(self, cache_ttl: float = 0.1):
        """
        Initialize the lease manager.
        
        Args:
            cache_ttl: Time to cache leases between contains and get operations (seconds)
        """
        self.active_leases: Dict[str, tuple[CacheEngineKey, LeaseInfo]] = {}
        self.lease_to_key: Dict[str, CacheEngineKey] = {}
        self.lease_cache: Dict[str, tuple[LeaseInfo, float]] = {}  # key -> (lease, expiry_time)
        self.cache_ttl = cache_ttl
        self.lock = asyncio.Lock()
        
    async def register_lease(
        self, 
        key: CacheEngineKey, 
        lease_info: LeaseInfo
    ) -> None:
        """Register a new active lease."""
        async with self.lock:
            key_str = key.to_string()
            self.active_leases[key_str] = (key, lease_info)
            self.lease_to_key[lease_info.lease_id] = key
    
    async def get_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """Get lease info for a key if it exists."""
        async with self.lock:
            key_str = key.to_string()
            if key_str in self.active_leases:
                _, lease_info = self.active_leases[key_str]
                return lease_info
            return None
    
    async def get_cached_lease(self, key: CacheEngineKey) -> Optional[LeaseInfo]:
        """
        Get cached lease if not expired.
        
        This is used to reuse leases between contains() and get() operations
        to avoid double lease acquisition.
        """
        async with self.lock:
            key_str = key.to_string()
            if key_str in self.lease_cache:
                lease_info, expiry_time = self.lease_cache[key_str]
                if time.time() < expiry_time:
                    # DON'T remove from cache - let it be reused multiple times
                    # It will be removed either:
                    # 1. When it expires naturally
                    # 2. When the lease is released via unregister_lease
                    # 3. During periodic cleanup
                    return lease_info
                else:
                    # Expired, remove from cache
                    del self.lease_cache[key_str]
            return None
    
    async def cache_lease(self, key: CacheEngineKey, lease_info: LeaseInfo) -> None:
        """
        Cache lease for short duration.
        
        Used to cache leases from contains() for potential subsequent get() calls.
        """
        async with self.lock:
            key_str = key.to_string()
            expiry_time = time.time() + self.cache_ttl
            self.lease_cache[key_str] = (lease_info, expiry_time)
    
    async def unregister_lease(self, lease_id: str) -> Optional[CacheEngineKey]:
        """Unregister a lease and return the associated key."""
        async with self.lock:
            if lease_id not in self.lease_to_key:
                return None
            
            key = self.lease_to_key[lease_id]
            key_str = key.to_string()
            
            if key_str in self.active_leases:
                del self.active_leases[key_str]
            del self.lease_to_key[lease_id]
            
            # Also remove from cache if present
            if key_str in self.lease_cache:
                del self.lease_cache[key_str]
            
            return key
    
    async def get_all_lease_ids(self) -> list[str]:
        """Get all active lease IDs."""
        async with self.lock:
            return list(self.lease_to_key.keys())
    
    async def cleanup_expired_cache(self) -> int:
        """Remove expired entries from lease cache."""
        async with self.lock:
            now = time.time()
            expired_keys = [
                key for key, (_, expiry) in self.lease_cache.items()
                if now >= expiry
            ]
            for key in expired_keys:
                del self.lease_cache[key]
            return len(expired_keys)
    
    async def clear(self) -> None:
        """Clear all lease records and cache."""
        async with self.lock:
            self.active_leases.clear()
            self.lease_to_key.clear()
            self.lease_cache.clear()
