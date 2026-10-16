# PEX-Mesh prototype + SLR search test

## 1. SLR search-string test (quasi-gold standard, 15 known papers)
Abstracts of the 15 papers were fetched from OpenAlex / Semantic Scholar (`results/qgs_abstracts.json`)
and each string was matched against title + abstract (`slr_search_test.py`). Target: >= 12/15 (80%).

| String | Found | Missed |
|---|---|---|
| Old string from the proposal (AIoT only) | 1/15 | 14 papers |
| Protocol v1.0 string | 9/15 | 3, 4, 6, 7, 11, 13 |
| **Fixed string** (+ `AI`, `framework*`, `federated`, `"on-device"`) | **12/15** | 4, 6, 11 (left to snowballing) |

Limits: only title + abstract were tested (Scopus also searches keywords, so real recall should be equal or higher).
Total hit counts were not measured: the database search API could not be reached from here, so run the string in Scopus.

## 2. PEX-Mesh prototype (v2, issues fixed)
Seven microservices, each its own process (FastAPI): edge inference, cloud inference (mutual TLS only),
privacy sidecar, XAI sidecar (SHAP), observability collector, gateway with the Placement Controller
(rule-based), and a monolith baseline. Data: synthetic CNC sensor data. Edge model = depth-2 decision tree,
cloud model = random forest. Edge-to-cloud network simulated as a 40 ms round trip. Machine: 2 CPU cores.

Run: `pip install fastapi uvicorn httpx shap scikit-learn pytest && ./gen_certs.sh && python3 train.py && ./run_all.sh`
then `pytest tests -q`, `python3 bench.py`, `python3 bench_light.py`. `gen_certs.sh` creates test certificates (private keys are not included in the zip).

### What changed in v2
| Issue in v1 | Fix in v2 | Result |
|---|---|---|
| DP noise (epsilon=1) on each reading made the cloud model near-random; PEX accuracy 0.877, below edge-only 0.897 | Inference requests: pseudonymised device id + mutual TLS, no noise. DP moved to window averages shared for analytics, with a per-device epsilon budget | Accuracy 0.901 (same as monolith). DP window means: 2.8% median error at epsilon=1, 200 readings |
| Async SHAP made p95 worse (jobs competed for CPU) | "auto" explanation mode: cheap edge-model explanations (0.1 ms) returned inline; costly cloud ones (19 ms) queued to one low-priority worker | p95 under load: 384 ms auto vs 467 sync vs 596 async; throughput 86 req/s |
| Cloud reached over plain HTTP | Cloud accepts only clients with a certificate from the PEX-Mesh CA | Tested: call without certificate is rejected |

### Functional tests: 11/11 pass (`tests/test_journeys.py`)
J1 confident reading at edge + explanation + full trace · J2 unsure reading escalated to cloud via privacy
sidecar · J3 sensitive data never leaves the edge · J4 tight latency budget stays at edge · J5 cloud outage falls
back to edge · J6 privacy budget exhausted blocks data sharing · J7 async explanation does not block ·
J8 cloud used again after the outage · J9 cloud rejects clients without a certificate · J10 DP window stats are
accurate and only carry a pseudonym · J11 auto explanation mode (edge inline, cloud in background).

### Benchmarks (`results/results.json`, `results/light_load.json`)
| Measure | PEX-Mesh v2 | Monolith |
|---|---|---|
| Latency, one request at a time (p50 / p95) | 7.6 / 10.1 ms at edge; 62.6 / 67.9 ms via cloud | 70.3 / 80.3 ms |
| Throughput, 50 concurrent clients | 61.5 req/s | 36.5 req/s |
| Requests that succeeded during a 4 s cloud outage | 1200 / 1200 | 44 / 1200 |
| Accuracy / AUC (2,000 requests) | 0.902 / 0.931 | 0.901 / 0.949 |
| Served at edge | 87% | 0% |

AUC is a bit lower for PEX-Mesh because sensitive and time-critical readings are always answered by the small edge
model; accuracy is the same. Under 50 concurrent clients both systems queue on this 2-core machine, so absolute
latency under load is not meaningful; the comparison is. Results vary a few percent between runs.

### Still open
1. Placement Controller is rule-based, not reinforcement learning yet.
2. Privacy model to explain in the thesis: inference data is protected by pseudonymisation + mTLS (not DP);
   DP protects what is stored in the cloud (window statistics). Training with DP (e.g. DP-SGD / federated) not built.
3. Not on Azure, synthetic data (not NASA C-MAPSS), single machine, simulated WAN.
# pexmesh
