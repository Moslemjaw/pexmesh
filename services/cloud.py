"""Cloud inference service: bigger model, reached over a simulated WAN link. Has a chaos switch."""
import asyncio, time, joblib, numpy as np
from fastapi import FastAPI, HTTPException
from common import WAN_RTT_MS
from services.base import span
app = FastAPI(); model = joblib.load("models/cloud.joblib"); state = {"down": False}

@app.post("/chaos")
async def chaos(req: dict):
    state["down"] = bool(req["down"]); return state

fleet_stats = []

@app.post("/ingest_stats")
async def ingest_stats(req: dict):
    """Receives only DP-protected window statistics, keyed by pseudonym."""
    fleet_stats.append(req); return {"stored": len(fleet_stats)}

@app.get("/fleet_stats")
async def get_stats():
    return {"windows": fleet_stats[-20:], "count": len(fleet_stats)}

@app.post("/predict")
async def predict(req: dict):
    t = time.perf_counter()
    if state["down"]:
        raise HTTPException(503, "cloud region unavailable")
    await asyncio.sleep(WAN_RTT_MS / 1000)          # simulated edge->cloud network round trip
    p = float(model.predict_proba(np.array([req["features"]]))[0, 1])
    span(req["trace_id"], "cloud-inference", "predict", t, p=p)
    return {"p_fail": p, "confidence": abs(p - 0.5) * 2, "model": "cloud"}
