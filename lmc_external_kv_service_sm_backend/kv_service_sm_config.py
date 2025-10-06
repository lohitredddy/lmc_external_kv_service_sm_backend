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
    
    # Connection management
    connection_keepalive: int = 30
    dns_ttl: int = 300

    control_max_connections: int = 256
    control_max_connections_per_host: int = 256
    put_max_connections: int = 256
    put_max_connections_per_host: int = 256

    # Timeout settings (milliseconds)
    lease_timeout_ms: int = 500
    put_timeout_ms: int = 20000
    http_connect_timeout_ms: int = 5000
    http_read_timeout_ms: int = 10000

    # Lease expiration (seconds)
    lease_ttl_s: int = 30

    # Cache configuration
    lease_cache_max_size: int = 20000      # Max lease cache entries
    put_cache_ttl_s: int = 10              # Recent PUT cache TTL (seconds)
    put_cache_max_size: int = 20000        # Max PUT cache entries

    # Client-side streaming / diagnostics
    put_stream_concurrency: int = 8
    put_stream_chunk_bytes: int = 4 * 1024 * 1024
    trace_enabled: bool = True

    @classmethod
    def from_extra_config(cls, extra_config: Optional[dict]) -> "KVServiceSMConfig":
        """Create config from LMCache extra_config dictionary."""
        if extra_config is None:
            return cls()
        
        # Helper converters with sane defaults
        def _to_int(value, default):
            try:
                return int(value)
            except (TypeError, ValueError):
                return default

        def _to_bool(value, default):
            if isinstance(value, bool):
                return value
            if value is None:
                return default
            return str(value).strip().lower() in {"1", "true", "yes", "on"}

        # Extract values with proper defaults
        return cls(
            base_url=extra_config.get("kv_service_sm_url", cls.base_url),
            shared_memory_name=extra_config.get("kv_service_sm_shared_memory_name", cls.shared_memory_name),
            bucket_name=extra_config.get("kv_service_sm_bucket", cls.bucket_name),
            connection_keepalive=extra_config.get("kv_service_sm_connection_keepalive", cls.connection_keepalive),
            dns_ttl=extra_config.get("kv_service_sm_dns_ttl", cls.dns_ttl),

            control_max_connections=extra_config.get("kv_service_sm_control_max_connections", cls.control_max_connections),
            control_max_connections_per_host=extra_config.get("kv_service_sm_control_max_connections_per_host", cls.control_max_connections_per_host),
            put_max_connections=extra_config.get("kv_service_sm_put_max_connections", cls.put_max_connections),
            put_max_connections_per_host=extra_config.get("kv_service_sm_put_max_connections_per_host", cls.put_max_connections_per_host),

            # Timeouts
            lease_timeout_ms=extra_config.get("kv_service_sm_lease_timeout_ms", cls.lease_timeout_ms),
            put_timeout_ms=extra_config.get("kv_service_sm_put_timeout_ms", cls.put_timeout_ms),
            http_connect_timeout_ms=extra_config.get("kv_service_sm_http_connect_timeout_ms", cls.http_connect_timeout_ms),
            http_read_timeout_ms=extra_config.get("kv_service_sm_http_read_timeout_ms", cls.http_read_timeout_ms),
            lease_ttl_s=extra_config.get("kv_service_sm_lease_ttl_s", cls.lease_ttl_s),

            # Cache config
            lease_cache_max_size=extra_config.get("lease_cache_max_size", cls.lease_cache_max_size),
            put_cache_ttl_s=extra_config.get("put_cache_ttl_s", cls.put_cache_ttl_s),
            put_cache_max_size=extra_config.get("put_cache_max_size", cls.put_cache_max_size),

            # Client-side streaming / diagnostics
            put_stream_concurrency=_to_int(
                extra_config.get("kv_service_sm_put_stream_concurrency", cls.put_stream_concurrency),
                cls.put_stream_concurrency,
            ),
            put_stream_chunk_bytes=_to_int(
                extra_config.get("kv_service_sm_put_stream_chunk_bytes", cls.put_stream_chunk_bytes),
                cls.put_stream_chunk_bytes,
            ),
            trace_enabled=_to_bool(
                extra_config.get("kv_service_sm_trace", cls.trace_enabled),
                cls.trace_enabled,
            ),
        )
