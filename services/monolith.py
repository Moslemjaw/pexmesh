"""Baseline: a monolith in the cloud. Device sends raw data over the WAN, one process runs the big model
and computes the SHAP explanation inline. No privacy layer, no edge path."""
import asyncio, time, joblib, numpy as np, shap
from fastapi import FastAPI, HTTPException
from common import FEATURES, WAN_RTT_MS
app = FastAPI(); model = joblib.load("models/cloud.joblib"); ex = shap.TreeExplainer(model); state = {"down": False}

@app.post("/chaos")
async def chaos(req: dict):
    state["down"] = bool(req["down"]); return state

@app.post("/infer")
async def infer(req: dict):
    if state["down"]:
        raise HTTPException(503, "cloud region unavailable")
    await asyncio.sleep(WAN_RTT_MS / 1000)
    x = np.array([req["features"]]); p = float(model.predict_proba(x)[0, 1])
    sv = ex.shap_values(x); sv = sv[..., 1] if np.ndim(sv) == 3 else sv
    return {"p_fail": p, "alert": p > 0.5, "served_by": "cloud",
            "explanation": dict(zip(FEATURES, np.round(np.ravel(sv), 4).tolist()))}
