"""Privacy sidecar (v2).
Lesson from v1: Laplace noise on single readings at epsilon=1 made the cloud model near-random (AUC 0.53).
v2 applies each protection where it works:
  * inference requests -> pseudonymised device id + data minimisation; the payload then travels over mTLS
  * data shared with the cloud for fleet analytics / retraining -> differential privacy on AGGREGATES
    (window means), where noise is divided by the window size, under a per-device epsilon budget."""
import hmac, hashlib, os, time, numpy as np
from fastapi import FastAPI
from common import FEATURES, RANGES
from services.base import span
app = FastAPI(); rng = np.random.default_rng()
SALT = os.urandom(16)                        # secret kept at the edge; cloud never sees real device ids
BUDGET_PER_DEVICE = 10.0
spent = {}
LO = np.array([RANGES[f][0] for f in FEATURES]); HI = np.array([RANGES[f][1] for f in FEATURES])

def pseudonym(dev):
    return hmac.new(SALT, dev.encode(), hashlib.sha256).hexdigest()[:16]

@app.post("/protect_inference")
async def protect_inference(req: dict):
    t = time.perf_counter()
    out = {"device": pseudonym(req["device_id"]), "features": req["features"]}   # only model inputs leave the site
    span(req["trace_id"], "privacy-sidecar", "pseudonymise", t)
    return out

@app.post("/aggregate")
async def aggregate(req: dict):
    """Epsilon-DP mean of each feature over a window of readings (bounded by the clipping ranges)."""
    t = time.perf_counter(); dev, eps = req["device_id"], float(req["epsilon"])
    if spent.get(dev, 0) + eps > BUDGET_PER_DEVICE:
        span(req["trace_id"], "privacy-sidecar", "refused", t)
        return {"allowed": False, "reason": "privacy budget exhausted", "spent": spent.get(dev, 0)}
    spent[dev] = spent.get(dev, 0) + eps
    W = np.clip(np.array(req["readings"], dtype=float), LO, HI); n = len(W)
    eps_f = eps / len(FEATURES)                                   # split epsilon across the features
    noisy_mean = W.mean(0) + rng.laplace(0, (HI - LO) / (n * eps_f))  # sensitivity of a mean = range / n
    span(req["trace_id"], "privacy-sidecar", "dp-aggregate", t, epsilon=eps, n=n)
    return {"allowed": True, "device": pseudonym(dev), "n": n, "dp_mean": np.clip(noisy_mean, LO, HI).tolist(),
            "epsilon": eps, "spent": spent[dev]}

@app.post("/reset")
async def reset():
    spent.clear(); return {"ok": True}

@app.get("/budget/{dev}")
async def budget(dev: str):
    return {"device": dev, "spent": spent.get(dev, 0), "limit": BUDGET_PER_DEVICE}
