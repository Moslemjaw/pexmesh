"""Observability collector: stores spans per trace (stand-in for OpenTelemetry + Azure Monitor)."""
from collections import defaultdict
from fastapi import FastAPI
app = FastAPI(); traces = defaultdict(list)

@app.post("/span")
async def span(s: dict):
    traces[s["trace_id"]].append(s); return {"ok": True}

@app.get("/trace/{tid}")
async def trace(tid: str):
    return {"trace_id": tid, "spans": traces.get(tid, [])}

@app.get("/stats")
async def stats():
    return {"traces": len(traces), "spans": sum(len(v) for v in traces.values())}
