# Packet loss on a network device
<!-- fault: packet_loss -->

## Symptoms
An interface on the router shows rising input errors, CRC errors and drops. Packet loss on the
uplink causes TCP retransmissions, so every service behind the device sees higher latency and
some errors at the same moment. Routing sessions may flap: a BGP neighbor resets and
re-establishes. CPU on the device and on the affected services stays normal.

## Diagnosis
Compare error counters across interfaces to isolate the faulty link. Check optical power
levels and the interface error history. Many unrelated services degrading together, with a
shared network path, points at the network rather than any one application.

## Remediation
1. Identify the interface with rising error counters.
2. Shift traffic to the redundant uplink. (requires approval)
3. Administratively disable the faulty interface once traffic has moved. (requires approval)
4. Raise a ticket to replace the optic or cable on the faulty link.

## Escalation
Page the network on-call immediately if there is no redundant path.
