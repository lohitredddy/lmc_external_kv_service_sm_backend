# KVServiceSM Timeout Configuration for Parallel Requests

## Issue
When running multiple prompts in parallel, lease operations timeout at 500ms while actual operations take 1500-1700ms.

## Root Cause
The KVServiceSM daemon requires more processing time when handling multiple concurrent lease requests. Each additional parallel request increases the response time due to:
- Lock contention in the daemon
- Sequential processing of lease requests
- Shared memory access serialization

## Solution
Increase the lease timeout to accommodate parallel request processing:

```yaml
lmcache:
  ...
  extra_config:
    # Increased from 500ms to 2500ms to handle parallel requests
    kv_service_sm_lease_timeout_ms: 2500
    
    # Keep other timeouts as is
    kv_service_sm_put_timeout_ms: 5000
    kv_service_sm_release_timeout_ms: 2000
```

## Performance Impact
- Single requests: May wait longer before timing out (but will still complete quickly)
- Parallel requests: Will successfully complete instead of timing out
- Overall throughput: Improved due to fewer retries and failures

## Monitoring
Look for these log patterns:
- Success: `Lease acquired successfully` within 2500ms
- Warning: Operations taking >2000ms indicate daemon overload
- Error: Timeouts at 2500ms indicate severe overload

## Further Optimization Options
If timeouts persist with many parallel requests:
1. Increase `kv_service_sm_lease_timeout_ms` to 3000-5000ms
2. Consider batching lease requests
3. Implement exponential backoff for retries
4. Scale KVServiceSM daemon horizontally
