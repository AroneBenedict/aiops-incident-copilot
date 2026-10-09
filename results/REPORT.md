# Evaluation report

Seeds: 7, 11, 13, 17, 19. Values are mean ± standard deviation across seeds, measured on the held-out test period (551 test incidents in total). Thresholds and model choices were made on the earlier history period only.

## Anomaly detection (point-wise F1 over service-minutes)

| Detector | F1 |
|---|---|
| Rolling z-score (baseline) | 43.8% ± 2.2% |
| Isolation Forest | 77.2% ± 2.0% |
| LSTM autoencoder (PyTorch) | 75.5% ± 3.0% |
| Seasonal residual z-score | 82.5% ± 1.1% |

Selected on history: seasonal_residual. Incidents that raised at least one alarm of three or more minutes: 96.3% ± 1.6%. By the looser test of any single flagged minute on an affected service, recall is 100.0% ± 0.0% with a median delay of 1.00 ± 0.00 minutes; that test is easy to pass by chance and is reported only for completeness.

## Alarm correlation (adjusted Rand index against true incidents)

| Method | ARI |
|---|---|
| Time window only (baseline) | 78.2% ± 3.4% |
| Time window + topology adjacency | 82.1% ± 2.9% |
| Time + topology + root split | 92.3% ± 1.8% |

Selected on history: time_topology_root_split. Alarms collapsed into incidents: 54.9% ± 2.9% fewer items to triage.

## Root-cause ranking (top-1 accuracy on detected incidents)

| Method | Top-1 |
|---|---|
| Earliest alarm (baseline) | 86.6% ± 2.8% |
| Highest anomaly score (baseline) | 89.2% ± 2.9% |
| Random walk on dependency graph | 91.0% ± 2.1% |

Selected on history: peak_score, random_walk. Random-walk top-3 accuracy: 98.7% ± 1.2%.

## Fault-type classification (8 classes, at the predicted root)

| Model | Accuracy | Macro F1 |
|---|---|---|
| Service-kind lookup (baseline) | 60.2% ± 4.4% | n/a |
| Random Forest on metric signature | 87.9% ± 1.7% | 86.6% ± 2.3% |

## Runbook retrieval (12 runbooks)

| Query source | Retriever | Hit@1 |
|---|---|---|
| Incident logs + metric symptoms | TF-IDF | 80.6% ± 4.3% |
| Incident logs + metric symptoms | BM25 | 80.8% ± 3.9% |
| 32 paraphrased operator queries | TF-IDF | 81.2% ± 0.0% |
| 32 paraphrased operator queries | BM25 | 81.2% ± 0.0% |

Paraphrased-query Hit@3 (TF-IDF): 96.9% ± 0.0%. The paraphrased set is fixed, so its spread across seeds is zero.

## Agent, end to end (rule planner)

| Measure | Value |
|---|---|
| Root cause correct | 90.9% ± 1.3% |
| Fault type correct | 90.9% ± 1.3% |
| Runbook correct | 90.9% ± 1.3% |
| All three correct (detected incidents) | 90.5% ± 1.1% |
| All three correct (all test incidents, including missed ones) | 87.2% ± 1.8% |
| All three correct when the root logged its fault | 93.0% ± 1.8% |
| All three correct when the root logged nothing useful | 82.6% ± 4.0% |
| Tool calls per investigation | 7.41 ± 0.13 |

### Is the agent's confidence label informative?

| Confidence | Reports | All three correct |
|---|---|---|
| high | 270 | 91.8% |
| medium | 207 | 91.3% |
| low | 54 | 81.5% |

