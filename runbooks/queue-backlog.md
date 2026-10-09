# Message queue backlog and consumer lag
<!-- fault: queue_backlog -->

## Symptoms
Consumer lag keeps growing on one or more partitions and the age of the oldest
unacknowledged message exceeds its SLO. Consumers rebalance repeatedly. When the queue
reaches its depth limit the broker rejects new publishes. End-to-end latency grows steadily
while throughput through the queue falls.

## Diagnosis
Compare the publish rate with the consume rate. Check how many consumers are active in the
group and whether one of them is stuck on a poison message. Look for a rebalance loop caused
by slow processing or session timeouts.

## Remediation
1. Check consumer group membership and the lag per partition.
2. Scale the consumer group out by adding consumers. (requires approval)
3. Move the poison message to the dead-letter queue. (requires approval)
4. Raise the consumer session timeout if processing is legitimately slow. (requires approval)

## Escalation
Page the messaging on-call when the broker starts rejecting publishes.
