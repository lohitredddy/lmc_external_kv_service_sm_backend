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

"""
KVServiceSM Backend for LMCache.

A high-performance storage backend that integrates with KVServiceSM daemon
for distributed KV cache storage with zero-copy shared memory access.
"""

from typing import TYPE_CHECKING

__version__ = "0.1.0"
__all__ = (
    "KVServiceSMBackend",
    "KVServiceSMConnector",
    "KVServiceSMConfig",
    "LeaseInfo",
)

if TYPE_CHECKING:  # pragma: no cover - for type checkers only
    # Import actual symbols only for static type checking
    from .kv_service_sm_backend import KVServiceSMBackend, LeaseInfo  # noqa: F401
    from .kv_service_sm_connector import KVServiceSMConnector  # noqa: F401
    from .kv_service_sm_config import KVServiceSMConfig  # noqa: F401


def __getattr__(name: str):
    """Lazily expose public symbols to avoid heavy imports at module import.

    This prevents triggering torch/LMCache imports during installation or
    when merely importing the top-level package.
    """
    if name in ("KVServiceSMBackend", "LeaseInfo"):
        from . import kv_service_sm_backend as _mod

        return getattr(_mod, name)
    if name == "KVServiceSMConnector":
        from . import kv_service_sm_connector as _mod

        return getattr(_mod, name)
    if name == "KVServiceSMConfig":
        from . import kv_service_sm_config as _mod

        return getattr(_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
