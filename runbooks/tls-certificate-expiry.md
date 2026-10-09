# TLS certificate expiry
<!-- fault: none -->

## Symptoms
Clients fail the TLS handshake and report that the certificate has expired or is not yet
valid. Errors start at an exact time, usually midnight UTC, and affect every connection to
the endpoint. Latency is normal for the few requests that still succeed.

## Diagnosis
Inspect the certificate served by the endpoint and compare its validity dates with the
current time. Check whether automatic renewal failed.

## Remediation
1. Confirm the expiry date of the served certificate.
2. Issue and install a renewed certificate. (requires approval)
3. Reload the proxy so it picks up the new certificate. (requires approval)

## Escalation
Escalate to the security team if the issuing authority is unavailable.
