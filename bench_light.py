"""Light-load latency per path (one request at a time) + raw SHAP cost per model."""
import json, time, httpx, joblib, numpy as np, shap
from common import URL, generate
c = httpx.Client(); X, _ = generate(300, seed=5)
c.post(f"{URL['privacy']}/reset"); c.post(f"{URL['gateway']}/controller/reset")
def run(url, body):
    ts = []
    for x in X:
        t = time.perf_counter(); c.post(url, json=body(x)); ts.append((time.perf_counter() - t) * 1000)
    return {"p50": round(float(np.median(ts)), 1), "p95": round(float(np.percentile(ts, 95)), 1)}
out = {
 "pex_edge_path": run(f"{URL['gateway']}/infer", lambda x: {"features": x.tolist(), "sensitivity": "high"}),
 "pex_cloud_path_forced": None,
 "monolith": run(f"{URL['monolith']}/infer", lambda x: {"features": x.tolist()}),
}
import os
conf = [c.post(f"{URL['edge']}/predict", json={"trace_id": "p", "features": x.tolist()}).json()["confidence"] for x in X]
ts = []
for i in np.argsort(conf)[:60]:
    t = time.perf_counter(); r = c.post(f"{URL['gateway']}/infer", json={"features": X[i].tolist()}).json()
    ts.append(((time.perf_counter() - t) * 1000, r["served_by"]))
cl = [t for t, w in ts if w == "cloud"]
out["pex_cloud_path_forced"] = {"n_cloud": len(cl), "p50": round(float(np.median(cl)), 1) if cl else None,
                                "p95": round(float(np.percentile(cl, 95)), 1) if cl else None}
for m in ("edge", "cloud"):
    ex = shap.TreeExplainer(joblib.load(f"models/{m}.joblib")); ts = []
    for x in X[:100]:
        t = time.perf_counter(); ex.shap_values(x.reshape(1, -1)); ts.append((time.perf_counter() - t) * 1000)
    out[f"shap_ms_{m}_model"] = round(float(np.median(ts)), 2)
print(json.dumps(out, indent=2)); json.dump(out, open("results/light_load.json", "w"), indent=2)
