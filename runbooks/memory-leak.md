# Memory leak and out-of-memory restarts
<!-- fault: memory_leak -->

## Symptoms
Memory grows steadily instead of following traffic. Heap usage approaches the limit, garbage
collection pauses get longer and the old generation does not shrink after a full GC. Latency
degrades gradually over tens of minutes rather than in one step. Eventually the container is
OOMKilled and restarts with exit code 137, which shows up as bursts of errors.

## Diagnosis
Plot resident memory against request rate. A leak keeps growing when traffic is flat. Check
restart counts and the last termination reason of the container. Compare with the previous
release: leaks usually arrive with a deploy but take time to show.

## Remediation
1. Capture a heap dump from one affected replica before it restarts.
2. Perform a rolling restart of the service to reclaim memory. (requires approval)
3. Raise the memory limit temporarily if restarts are too frequent. (requires approval)
4. Open a defect for the owning team with the heap dump attached.

## Escalation
Escalate to the application team if more than two OOM restarts happen within one hour.
