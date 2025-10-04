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
    max_concurrent_puts: int = 32
    
    # Timeout settings (milliseconds)
    lease_timeout_ms: int = 500
    put_timeout_ms: int = 5000
    http_connect_timeout_ms: int = 5000
    http_read_timeout_ms: int = 10000

    # Lease expiration (seconds)
    lease_ttl_s: int = 30
    
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
            connection_keepalive=extra_config.get("kv_service_sm_connection_keepalive", cls.connection_keepalive),
            dns_ttl=extra_config.get("kv_service_sm_dns_ttl", cls.dns_ttl),

            control_max_connections=extra_config.get("kv_service_sm_control_max_connections", cls.control_max_connections),
            control_max_connections_per_host=extra_config.get("kv_service_sm_control_max_connections_per_host", cls.control_max_connections_per_host),
            put_max_connections=extra_config.get("kv_service_sm_put_max_connections", cls.put_max_connections),
            put_max_connections_per_host=extra_config.get("kv_service_sm_put_max_connections_per_host", cls.put_max_connections_per_host),
            max_concurrent_puts=extra_config.get("kv_service_sm_max_concurrent_puts", cls.max_concurrent_puts),

            # Timeouts
            lease_timeout_ms=extra_config.get("kv_service_sm_lease_timeout_ms", cls.lease_timeout_ms),
            put_timeout_ms=extra_config.get("kv_service_sm_put_timeout_ms", cls.put_timeout_ms),
            http_connect_timeout_ms=extra_config.get("kv_service_sm_http_connect_timeout_ms", cls.http_connect_timeout_ms),
            http_read_timeout_ms=extra_config.get("kv_service_sm_http_read_timeout_ms", cls.http_read_timeout_ms),
            lease_ttl_s=extra_config.get("kv_service_sm_lease_ttl_s", cls.lease_ttl_s),
        )
