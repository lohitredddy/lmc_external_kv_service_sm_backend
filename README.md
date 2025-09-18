# LMCache External KV Service SM Backend

This project provides an external KVServiceSM backend implementation for LMCache.

## Features
- Implements the `StorageBackendInterface` from LMCache
- Zero-copy shared memory access for high performance
- HTTP API integration with KVServiceSM daemon
- Easy to integrate with existing LMCache

## Installation

```bash
pip install lmc_external_kv_service_sm_backend
```

## Usage

To use this backend in your LMCache,

Add the following to your LMCache Configuration:
```yaml
chunk_size: 64
local_cpu: False
max_local_cpu_size: 5
external_backends: "kv_service_sm_backend"
extra_config:
  external_backend.kv_service_sm_backend.module_path: lmc_external_kv_service_sm_backend.kv_service_sm_backend
  external_backend.kv_service_sm_backend.class_name: KVServiceSMBackend
  kv_service_sm_url: "http://localhost:9200"
```

## Development

To build the package:
```bash
python setup.py sdist bdist_wheel
```

To install locally:
```bash
pip install -e .
```

## License

Apache-2.0 License
