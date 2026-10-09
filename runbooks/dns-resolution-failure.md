# DNS resolution failure
<!-- fault: none -->

## Symptoms
Services log that a hostname could not be resolved or that the name server timed out.
Failures are intermittent and affect outbound calls to many different dependencies at once.
Calls that use a cached address keep working.

## Diagnosis
Query the resolver directly from an affected host. Check resolver health, upstream
forwarders and whether a zone change was published recently.

## Remediation
1. Test resolution of the failing hostname from an affected host.
2. Restart the unhealthy resolver instance. (requires approval)
3. Roll back the most recent zone change. (requires approval)

## Escalation
Page the network on-call if both resolvers are failing.
