# Planned maintenance window
<!-- fault: none -->

## Symptoms
Alerts fire for a service during an announced maintenance window. Instances are drained and
restarted one at a time, so capacity is briefly reduced and latency rises a little.

## Diagnosis
Check the change calendar for an approved maintenance window that covers the service and the
time of the alert.

## Remediation
1. Confirm the alert falls inside an approved maintenance window.
2. Silence the alert for the remainder of the window. (requires approval)

## Escalation
Escalate to the change manager if the work overruns its window.
