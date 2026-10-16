"""PEX-Mesh edge gateway with the Placement Controller.
Every request: edge model first -> controller decides whether to escalate to the cloud
(through the Privacy sidecar) -> XAI sidecar explains asynchronously -> spans go to observability."""
import time, os
from fastapi import FastAPI
from common import URL, new_id, WAN_RTT_MS
import asyncio
from services.base import client, mtls_client, span
_bg = set()
app = FastAPI()
CONF_THRESHOLD = float(os.environ.get("CONF_THRESHOLD", "0.6"))
COOLDOWN_S = 5.0
ctl = {"cloud_ms_ewma": WAN_RTT_MS + 10, "circuit_open_until": 0.0, "stats": {}}

def decide(sensitivity, latency_budget_ms, edge_conf):
    """Placement Controller (rule-based v1). Returns (where, reason)."""
    if sensitivity == "high":
        return "edge", "sensitive data stays at the edge"
    if latency_budget_ms is not None and latency_budget_ms < ctl["cloud_ms_ewma"]:
        return "edge", "latency budget below expected cloud latency"
    if edge_conf >= CONF_THRESHOLD:
        return "edge", "edge model confident"
    if time.time() < ctl["circuit_open_until"]:
        return "edge", "cloud unavailable (circuit open) - fallback"
    return "cloud", "edge unsure - escalate"

@app.post("/infer")
async def infer(req: dict):
    t0 = time.perf_counter(); tid = new_id(); c = client()
    feats, dev = req["features"], req.get("device_id", "cnc-1")
    sens, budget, xai_mode = req.get("sensitivity", "normal"), req.get("latency_budget_ms"), req.get("xai", "auto")
    edge = (await c.post(f"{URL['edge']}/predict", json={"trace_id": tid, "features": feats})).json()
    where, reason = decide(sens, budget, edge["confidence"])
    result, used_feats, degraded = edge, feats, False
    if where == "cloud":
        prot = (await c.post(f"{URL['privacy']}/protect_inference", json={"trace_id": tid, "device_id": dev,
                "features": feats})).json()
        tc = time.perf_counter()
        try:
            r = await mtls_client().post(f"{URL['cloud']}/predict", json={"trace_id": tid, "device": prot["device"],
                                         "features": prot["features"]}, timeout=2)
            r.raise_for_status(); result = r.json()
            ms = (time.perf_counter() - tc) * 1000
            ctl["cloud_ms_ewma"] = 0.9 * ctl["cloud_ms_ewma"] + 0.1 * ms
        except Exception:
            ctl["circuit_open_until"] = time.time() + COOLDOWN_S
            where, reason, degraded = "edge", "cloud call failed - fallback to edge", True
    xreq = {"trace_id": tid, "model": result["model"] if where == "cloud" else "edge", "features": used_feats}
    explanation = None
    if xai_mode == "auto":     # v2: cheap edge-model explanations inline (~0.06 ms), costly cloud ones queued
        xai_mode = "async" if xreq["model"] == "cloud" else "sync"
    if xai_mode == "async":
        task = asyncio.create_task(c.post(f"{URL['xai']}/explain_async", json=xreq))   # fire and forget
        _bg.add(task); task.add_done_callback(_bg.discard)
    elif xai_mode == "sync":
        explanation = (await c.post(f"{URL['xai']}/explain_sync", json=xreq)).json()
    ctl["stats"][reason] = ctl["stats"].get(reason, 0) + 1
    span(tid, "gateway", "infer", t0, where=where, reason=reason, degraded=degraded)
    return {"trace_id": tid, "p_fail": result["p_fail"], "alert": result["p_fail"] > 0.5, "served_by": where,
            "reason": reason, "degraded": degraded, "explanation": explanation,
            "explanation_url": f"{URL['xai']}/explanation/{tid}"}

@app.get("/controller")
async def controller():
    return ctl

@app.post("/controller/reset")
async def reset():
    ctl.update({"cloud_ms_ewma": WAN_RTT_MS + 10, "circuit_open_until": 0.0, "stats": {}}); return ctl

@app.post("/share_window")
async def share_window(req: dict):
    """Send a window of readings to the cloud for fleet analytics, protected by DP on the aggregate."""
    tid = new_id(); c = client()
    agg = (await c.post(f"{URL['privacy']}/aggregate", json={"trace_id": tid, "device_id": req["device_id"],
           "readings": req["readings"], "epsilon": req.get("epsilon", 1.0)})).json()
    if not agg["allowed"]:
        return {"shared": False, "reason": agg["reason"]}
    await mtls_client().post(f"{URL['cloud']}/ingest_stats", json={k: agg[k] for k in ("device", "n", "dp_mean", "epsilon")})
    return {"shared": True, "dp_mean": agg["dp_mean"], "spent": agg["spent"]}
