# Error spike after a deployment
<!-- fault: bad_deploy -->

## Symptoms
Error rate steps up right after a rollout completes. The service returns HTTP 500 responses
and logs unhandled exceptions or missing configuration keys that mention the new build or
release version. Latency and CPU look normal: the service is fast, it is just failing.

## Diagnosis
Line the first errors up with the deployment timeline. Read the exception and check whether
a configuration key or feature flag changed in the release. Confirm the previous version was
healthy on the same traffic.

## Remediation
1. Confirm the errors started with the latest rollout.
2. Roll back to the previous version. (requires approval)
3. Disable the feature flag introduced by the release. (requires approval)
4. Attach the stack trace and release identifier to the defect for the owning team.

## Escalation
Escalate to the release manager if the rollback does not restore the error rate within ten minutes.
