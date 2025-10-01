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
from dataclasses import dataclass
from typing import Optional


@dataclass
class KVServiceSMConfig:
    """Configuration for KVServiceSM storage backend."""
    
    # Connection settings
    base_url: str = "http://localhost:9200"
    shared_memory_name: str = "shared_memory"
    bucket_name: str = "lmcache"
    
    # Performance settings
    max_connections: int = 256
    max_connections_per_host: int = 256
    connection_keepalive: int = 30
    dns_ttl: int = 300
    
    # Thread pool settings
    serialization_threads: int = 16
    
    # Timeout settings (milliseconds)
    lease_timeout_ms: int = 500
    put_timeout_ms: int = 5000
    release_timeout_ms: int = 2000
    http_connect_timeout_ms: int = 5000
    http_read_timeout_ms: int = 10000
    
    # Cache settings
    enable_key_cache: bool = True
    cache_ttl_seconds: int = 300
    max_cache_entries: int = 10000
    
    # Operation limits
    max_inflight_requests: int = 100
    max_batch_size: int = 50
    pq_executor_workers: int = 16  # Number of workers for priority queue executor
    
    # Retry settings
    max_retries: int = 3
    retry_backoff_ms: int = 100
    
    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> "KVServiceSMConfig":
        """Create config from LMCache extra_config dictionary."""
        if extra_config is None:
            return cls()
        
        # Extract values with proper defaults
        return cls(
            base_url=extra_config.get("kv_service_sm_url", cls.base_url),
            shared_memory_name=extra_config.get("kv_service_sm_shared_memory_name", cls.shared_memory_name),
            bucket_name=extra_config.get("kv_service_sm_bucket", cls.bucket_name),
            
            # Performance settings
            max_connections=extra_config.get("kv_service_sm_max_connections", cls.max_connections),
            max_connections_per_host=extra_config.get("kv_service_sm_max_connections_per_host", cls.max_connections_per_host),
            connection_keepalive=extra_config.get("kv_service_sm_connection_keepalive", cls.connection_keepalive),
            dns_ttl=extra_config.get("kv_service_sm_dns_ttl", cls.dns_ttl),
            
            # Thread pool
            serialization_threads=extra_config.get("kv_service_sm_serialization_threads", cls.serialization_threads),
            
            # Timeouts
            lease_timeout_ms=extra_config.get("kv_service_sm_lease_timeout_ms", cls.lease_timeout_ms),
            put_timeout_ms=extra_config.get("kv_service_sm_put_timeout_ms", cls.put_timeout_ms),
            release_timeout_ms=extra_config.get("kv_service_sm_release_timeout_ms", cls.release_timeout_ms),
            http_connect_timeout_ms=extra_config.get("kv_service_sm_http_connect_timeout_ms", cls.http_connect_timeout_ms),
            http_read_timeout_ms=extra_config.get("kv_service_sm_http_read_timeout_ms", cls.http_read_timeout_ms),
            
            # Cache settings
            enable_key_cache=extra_config.get("kv_service_sm_enable_key_cache", cls.enable_key_cache),
            cache_ttl_seconds=extra_config.get("kv_service_sm_cache_ttl_seconds", cls.cache_ttl_seconds),
            max_cache_entries=extra_config.get("kv_service_sm_max_cache_entries", cls.max_cache_entries),
            
            # Operation limits
            max_inflight_requests=extra_config.get("kv_service_sm_max_inflight_requests", cls.max_inflight_requests),
            max_batch_size=extra_config.get("kv_service_sm_max_batch_size", cls.max_batch_size),
            pq_executor_workers=extra_config.get("kv_service_sm_pq_executor_workers", cls.pq_executor_workers),
            
            # Retry settings
            max_retries=extra_config.get("kv_service_sm_max_retries", cls.max_retries),
            retry_backoff_ms=extra_config.get("kv_service_sm_retry_backoff_ms", cls.retry_backoff_ms),
        )
