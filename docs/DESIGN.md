# Design notes

## Requirement

Given per-minute metrics and logs for a set of services and their dependency graph, tell the
on-call engineer, for each incident: which service is the root cause, what kind of fault it
is, and which runbook applies. Never change the system. Say how sure the answer is.

Non-functional requirements: every stage must be measurable against ground truth with a
baseline; results must be reproducible from a seed; features must be causal so the same code
works on a stream; the agent must be bounded and auditable.

## Data flow and contracts

| Step | Input | Output |
|---|---|---|
| `simulate(cfg)` | `Config` | `Dataset`: `metrics (services, minutes, 4)`, `labels (services, minutes)`, `incidents`, `logs` |
| `Detector.fit(clean).score(metrics)` | metric cube | anomaly score per service-minute, higher is worse |
| `best_f1_threshold(scores, truth)` | history scores and labels | one threshold |
| `build_alarms(scores, threshold)` | scores | `Alarm(id, service, start, end, peak)` |
| `correlate` then `split_by_root` | alarms, graph | `IncidentGroup(id, alarm_ids, services, start, end)` |
| `rank_*(group, alarms, graph)` | one group | ranked `Candidate(service, score, first_alarm, peak)` |
| `signature(z, group, alarms, service)` | residual z-scores | 19 features for the fault classifier |
| `Retriever.search(query, k)` | text | ranked `Hit(runbook, title, section, score)` |
| `Agent.investigate(incident_id)` | workspace, planner | `Report` with trace |

The timeline is split three ways and the splits never overlap:

1. **Clean** (7 days, no incidents): fit seasonal baselines and the unsupervised detectors.
2. **History** (14 days, labelled): tune thresholds, choose the detector, correlation method,
   root-cause method and retriever, and train the fault classifier.
3. **Test** (14 days, labelled): produce every reported number.

## Decisions and trade-offs

**One detector interface, four implementations.** `fit(clean)` and `score(metrics)` are all a
detector needs. That made it cheap to keep a naive baseline, a statistical method, a tree
ensemble and a neural model in the same comparison, and to let the history period choose.

**Exact threshold search.** The threshold is the cut with the best point-wise F1, found by
sorting scores once. An earlier version tried a fixed grid of quantiles and could not land on
a clean separating value; a unit test with two well-separated classes caught it.

**Correlation in two steps.** Joining alarms that are close in time and adjacent in the graph
is simple and explains most storms. It fails when two faults overlap on neighbouring
services, because a shared caller such as the gateway glues them together. The second step
looks for *root candidates* in a group: alarmed services that do not depend, directly or
transitively, on any other alarmed service. One fault gives one root candidate. Two give two,
and each alarm is handed to the candidate it depends on that started closest in time.
Transitive dependency is used rather than direct edges so a missing alarm on an intermediate
service does not cause a false split.

**Random walk for root cause.** The walk moves from a service to the alarmed dependencies it
calls, weighted by their anomaly score, and stays put at a service with no alarmed
dependency. Mass collects at the most upstream alarmed service. It needs no training data.

**Classifier at the predicted root, not the true root.** The fault classifier is trained on
history incidents at the true root and tested at whatever the ranking predicted. That makes
its reported accuracy an end-to-end figure that includes ranking mistakes.

**Retrieval indexes symptoms, returns remediation.** Only the Symptoms and Diagnosis sections
are searched. Remediation text is fetched by runbook name afterwards, so the search is not
polluted by generic action words and the steps shown are always the runbook's own.

**Planner as a plug-in.** The loop, the tools, validation, the step limit and the trace are
shared. A planner only decides the next call. The deterministic planner makes the system
testable and gives a reproducible benchmark; the LLM planner reuses everything else.

**Approval flags live in the runbook.** The final report reads remediation steps from the
runbook after the planner has finished. A planner can pick the wrong runbook, but it cannot
invent a step or mark a risky one as safe.

## Defects found and fixed

| Found by | Problem | Fix | Effect |
|---|---|---|---|
| Failure analysis on development seeds | Overlapping incidents merged into one group; accuracy on merged groups was about half that of clean ones | `split_by_root` | Correlation ARI 82.1% to 92.3%; root-cause top-1 76.0% to 91.0%; agent fully correct 75.2% to 90.5% |
| Unit test | Rolling z-score returned values in the thousands during its first minutes, when the trailing spread was computed from one or two samples | Warm-up period scored as zero | No change to reported numbers (the warm-up falls in the clean period) |
| Unit test | Grid search over score quantiles could not find a threshold between two separated classes | Exact search over sorted scores | Thresholds are now optimal for the history period |
| Reading agent traces | Every log line of a bad deploy named a different build number | One build id per incident | More realistic logs |
| Failure analysis | The planner trusted a top-ranked service whose logs only blamed a dependency | Read the next candidates' logs and switch when one shows its own fault | Root changed in 18 of 531 investigations |
| Calibration check | Confidence was "high" for almost every report and so carried no information | Confidence now counts three independent signals | "Low" reports are right 81.5% of the time against about 92% for the rest |

Development seeds (101 to 103) were used for failure analysis so the five reported seeds
stayed untouched while the planner and confidence rule were being changed.

## Known failure modes

- Two overlapping faults where one root depends on the other stay merged.
- A false alarm on a dependency during a real incident upstream is ranked as the root.
- A root too mild to alarm leaves its callers as apparent roots.
- Lexical retrieval misses paraphrases that share no vocabulary with the runbook.

## Next steps

1. Replay a public incident dataset through the same interfaces instead of the simulator.
2. Add a dense embedding retriever behind `search` and compare it on the paraphrased set.
3. Benchmark `ClaudePlanner` against the rule planner on the same incidents, with cost per
   investigation from the token counts it already records.
4. Weight graph edges by how well a caller's anomaly follows its dependency's in time, to
   resist false alarms on dependencies.
