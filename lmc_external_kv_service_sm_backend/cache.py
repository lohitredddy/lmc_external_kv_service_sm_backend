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
from collections import OrderedDict
from dataclasses import dataclass
from typing import Dict, Optional
import asyncio
import time

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)


@dataclass
class LeaseInfo:
    """Information about a lease obtained from KVServiceSM daemon."""
    lease_id: str
    offsets: list[tuple[int, int]]  # (offset, length) pairs
    total_size: int


@dataclass
class CacheEntry:
    """Cache entry for key existence and metadata."""
    key: str
    exists: bool
    size: Optional[int]
    lease_info: Optional[LeaseInfo]
    created_at: float
    last_accessed: float
    access_count: int = 0
    
    def is_expired(self, ttl: float) -> bool:
        """Check if cache entry has expired."""
        return time.time() - self.created_at > ttl
    
    def touch(self):
        """Update last accessed time and increment access count."""
        self.last_accessed = time.time()
        self.access_count += 1


class KeyCache:
    """LRU cache for key existence and metadata with TTL support."""
    
    def __init__(self, max_entries: int, ttl: float):
        """
        Initialize the key cache.
        
        Args:
            max_entries: Maximum number of entries to cache
            ttl: Time-to-live for cache entries in seconds
        """
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_entries = max_entries
        self.ttl = ttl
        self.lock = asyncio.Lock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    async def get(self, key: str) -> Optional[CacheEntry]:
        """
        Get a cache entry if it exists and is not expired.
        
        Args:
            key: The cache key to lookup
            
        Returns:
            The cache entry if found and valid, None otherwise
        """
        async with self.lock:
            entry = self.cache.get(key)
            if entry is None:
                self.misses += 1
                return None
                
            if entry.is_expired(self.ttl):
                # Remove expired entry
                del self.cache[key]
                self.misses += 1
                return None
            
            # Update access info and move to end (LRU)
            entry.touch()
            self.cache.move_to_end(key)
            self.hits += 1
            return entry
    
    async def put(
        self, 
        key: str, 
        exists: bool, 
        size: Optional[int] = None,
        lease_info: Optional[LeaseInfo] = None
    ) -> None:
        """
        Put an entry in the cache.
        
        Args:
            key: The cache key
            exists: Whether the key exists in storage
            size: Optional size of the cached data
            lease_info: Optional lease information
        """
        async with self.lock:
            now = time.time()
            
            # Check if we need to evict
            if len(self.cache) >= self.max_entries and key not in self.cache:
                # Evict least recently used
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                self.evictions += 1
            
            # Create or update entry
            entry = CacheEntry(
                key=key,
                exists=exists,
                size=size,
                lease_info=lease_info,
                created_at=now,
                last_accessed=now,
                access_count=1 if key not in self.cache else self.cache[key].access_count + 1
            )
            
            self.cache[key] = entry
            self.cache.move_to_end(key)
    
    async def remove(self, key: str) -> bool:
        """
        Remove an entry from the cache.
        
        Args:
            key: The cache key to remove
            
        Returns:
            True if the key was removed, False if it didn't exist
        """
        async with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        """Clear all entries from the cache."""
        async with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries from the cache.
        
        Returns:
            Number of entries removed
        """
        async with self.lock:
            now = time.time()
            expired_keys = [
                key for key, entry in self.cache.items()
                if now - entry.created_at > self.ttl
            ]
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "size": len(self.cache),
            "max_size": self.max_entries,
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "hit_rate": hit_rate,
        }
