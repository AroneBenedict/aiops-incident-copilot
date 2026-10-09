# Log volume spike filling a disk
<!-- fault: none -->

## Symptoms
A service writes far more log lines than usual and the log partition fills up. Disk usage
alerts fire for the host. The service itself may be healthy, with normal latency.

## Diagnosis
Find the logger that produces the most lines. Check whether debug logging was left enabled
after a change, and whether log rotation is running.

## Remediation
1. Identify the noisiest logger on the host.
2. Set the log level back to its default. (requires approval)
3. Force a log rotation and compress old files. (requires approval)

## Escalation
Escalate to the platform team if rotation is failing.
