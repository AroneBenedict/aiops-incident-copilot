# CPU saturation on a service
<!-- fault: cpu_saturation -->

## Symptoms
CPU utilisation sits close to its limit and latency climbs with it. Worker or thread pools are
saturated, requests queue up, and health probes start timing out because the process is too
busy to answer. The autoscaler reports utilisation above its target. Error rate rises only
slightly, and throughput dips because the service cannot keep up.

## Diagnosis
Check whether load actually increased. If throughput is flat or lower while CPU is high, the
cause is the service itself: a hot loop, an expensive request type, or cgroup throttling.
Compare the load average with the number of cores and look at the top CPU consumers.

## Remediation
1. Confirm the autoscaler is not stuck at its maximum replica count.
2. Scale the service out by adding replicas. (requires approval)
3. Enable load shedding for low-priority requests. (requires approval)
4. Capture a CPU profile from one busy replica for the owning team.

## Escalation
Page the owning application team if utilisation stays above 90% for 15 minutes after scaling.
