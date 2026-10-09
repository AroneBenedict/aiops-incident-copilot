# Cache eviction storm
<!-- fault: cache_eviction_storm -->

## Symptoms
The cache has reached maxmemory and is evicting keys at a high rate. The hit ratio drops, so
callers rebuild hot keys at the same time and cause a miss stampede. Command latency on the
cache rises, the slowlog grows and CPU increases. Services that read through the cache slow
down while their own error rate stays low.

## Diagnosis
Check memory usage against maxmemory, the eviction policy and the key count trend. Look for
a new key pattern with large values or missing TTLs. Compare the hit ratio before and after
the latency increase.

## Remediation
1. Identify the key pattern that grew the most.
2. Increase the cache memory limit. (requires approval)
3. Enable request coalescing for hot keys in the callers. (requires approval)
4. Add a TTL to the offending key pattern and redeploy the writer. (requires approval)

## Escalation
Escalate to the platform team if the hit ratio has not recovered 20 minutes after resizing.
