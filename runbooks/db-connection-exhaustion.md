# Database connection pool exhaustion
<!-- fault: db_connection_exhaustion -->

## Symptoms
Applications cannot acquire a connection from the pool and report checkout timeouts. The
database rejects new sessions because too many clients are already connected and the
remaining connection slots are reserved. Error rate jumps sharply, callers wait, and
throughput on the database drops while its CPU stays normal or even falls.

## Diagnosis
Compare active connections with the configured maximum. Look for idle-in-transaction
sessions and for one client holding most of the connections. Check whether a caller recently
raised its pool size or lost its connection pooler.

## Remediation
1. List sessions by client and state to find the connection hog.
2. Terminate idle-in-transaction sessions older than five minutes. (requires approval)
3. Restart the connection pooler in front of the database. (requires approval)
4. Lower the pool size of the offending client and redeploy it. (requires approval)

## Escalation
Page the database on-call if the pool is still exhausted ten minutes after sessions were cleared.
