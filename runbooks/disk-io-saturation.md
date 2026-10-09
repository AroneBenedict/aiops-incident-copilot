# Disk I/O saturation on a data store
<!-- fault: disk_io_saturation -->

## Symptoms
Storage latency is high. Writes are slow: fsync and write-ahead log flushes take seconds,
checkpoints run long with many buffers pending, and iowait is elevated with a deep device
queue. Queries are slow because reads block on disk. Latency rises a lot while the error rate
barely moves.

## Diagnosis
Check device utilisation, queue depth and iowait on the host. Look for a competing workload:
a backup, compaction, a reindex or a large sequential scan. Confirm the volume has not hit
its provisioned IOPS limit.

## Remediation
1. Identify the top I/O consumer on the host.
2. Pause or reschedule the backup or compaction job that overlaps with peak traffic. (requires approval)
3. Cancel the long-running sequential scan if one is found. (requires approval)
4. Request a higher IOPS tier for the volume. (requires approval)

## Escalation
Escalate to the storage team when the device queue stays saturated with no competing workload.
