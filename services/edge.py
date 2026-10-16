"""Edge inference service: small model, runs at the factory gateway."""
import time, joblib, numpy as np
from fastapi import FastAPI
from services.base import span
app = FastAPI(); model = joblib.load("models/edge.joblib")

@app.post("/predict")
async def predict(req: dict):
    t = time.perf_counter()
    p = float(model.predict_proba(np.array([req["features"]]))[0, 1])
    conf = abs(p - 0.5) * 2              # 0 = unsure, 1 = very sure
    span(req["trace_id"], "edge-inference", "predict", t, p=p)
    return {"p_fail": p, "confidence": conf, "model": "edge"}
