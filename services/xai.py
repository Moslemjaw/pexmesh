"""Explainability sidecar (v2): SHAP explanations.
Lesson from v1: background jobs stole CPU from the request path and made p95 worse.
v2: one low-priority worker drains a bounded queue, so explanation work never competes with predictions."""
import asyncio, os, time, joblib, numpy as np, shap
from fastapi import FastAPI
from common import FEATURES
from services.base import span
os.nice(10)                                  # run below the inference services
# Give the sidecar its own CPU core when the machine has spare cores (stands in for a separate node)
if os.cpu_count() and os.cpu_count() >= 4 and os.environ.get("XAI_PIN", "1") == "1":
    os.sched_setaffinity(0, {os.cpu_count() - 1})
app = FastAPI()
explainers = {m: shap.TreeExplainer(joblib.load(f"models/{m}.joblib")) for m in ("edge", "cloud")}
results, queue = {}, asyncio.Queue(maxsize=5000)
stats = {"dropped": 0, "done": 0}

def _explain(model, x):
    sv = explainers[model].shap_values(np.array([x]))
    sv = sv[..., 1] if np.ndim(sv) == 3 else (sv[1] if isinstance(sv, list) else sv)
    vals = dict(zip(FEATURES, np.round(np.ravel(sv), 4).tolist()))
    return {"shap": vals, "top_reasons": sorted(vals, key=lambda k: -abs(vals[k]))[:3]}

async def worker():
    while True:
        req = await queue.get(); t = time.perf_counter()
        results[req["trace_id"]] = await asyncio.to_thread(_explain, req["model"], req["features"])
        stats["done"] += 1; span(req["trace_id"], "xai-sidecar", "explain", t, model=req["model"])

@app.on_event("startup")
async def start():
    asyncio.create_task(worker())

@app.post("/explain_async")
async def explain_async(req: dict):
    try:
        queue.put_nowait(req)
    except asyncio.QueueFull:
        stats["dropped"] += 1                # back-pressure: never slow the caller
    return {"status": "queued"}

@app.post("/explain_sync")
async def explain_sync(req: dict):
    return await asyncio.to_thread(_explain, req["model"], req["features"])

@app.get("/explanation/{tid}")
async def get(tid: str):
    return results.get(tid, {"status": "pending"})

@app.get("/stats")
async def get_stats():
    return {**stats, "queued": queue.qsize()}
