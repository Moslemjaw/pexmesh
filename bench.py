"""Benchmarks: PEX-Mesh vs monolith (latency, throughput, accuracy), DP epsilon sweep, chaos test, XAI cost."""
import asyncio, json, time, httpx, joblib, numpy as np
from common import URL, MTLS, generate, FEATURES, RANGES
from sklearn.metrics import accuracy_score, roc_auc_score
R = {}
X, y = generate(2000, seed=11)
rng = np.random.default_rng(3)
sens = rng.choice(["normal", "high"], len(X), p=[0.8, 0.2])            # 20% of readings tagged sensitive
budget = rng.choice([None, 10], len(X), p=[0.9, 0.1])                  # 10% need an answer within 10 ms

async def load(url, body_fn, n, conc=50):
    lat, out, errs = [], [None] * n, 0
    sem = asyncio.Semaphore(conc)
    async with httpx.AsyncClient(timeout=10, limits=httpx.Limits(max_connections=conc)) as c:
        async def one(i):
            nonlocal errs
            async with sem:
                t = time.perf_counter()
                try:
                    r = await c.post(url, json=body_fn(i)); r.raise_for_status(); out[i] = r.json()
                except Exception:
                    errs += 1
                lat.append((time.perf_counter() - t) * 1000)
        t0 = time.perf_counter(); await asyncio.gather(*[one(i) for i in range(n)]); wall = time.perf_counter() - t0
    return np.array(lat), out, errs, wall

def summary(lat, out, errs, wall, n):
    ok = [i for i in range(n) if out[i]]
    p = np.array([out[i]["p_fail"] for i in ok])
    return {"requests": n, "errors": errs, "throughput_rps": round(n / wall, 1),
            "p50_ms": round(float(np.percentile(lat, 50)), 1), "p95_ms": round(float(np.percentile(lat, 95)), 1),
            "p99_ms": round(float(np.percentile(lat, 99)), 1),
            "accuracy": round(accuracy_score(y[ok], p > .5), 3), "auc": round(roc_auc_score(y[ok], p), 3)}

async def main():
    sync = httpx.Client()
    cm = httpx.Client(**MTLS)
    ctl_post = lambda tgt, down: (cm if tgt == 'cloud' else sync).post(f"{URL[tgt]}/chaos", json={'down': down})
    for s in ("cloud", "monolith"): ctl_post(s, False)
    sync.post(f"{URL['privacy']}/reset"); sync.post(f"{URL['gateway']}/controller/reset")
    n = len(X)
    pex_body = lambda i: {"features": X[i].tolist(), "device_id": f"cnc-{i % 3}", "sensitivity": sens[i],
                          "latency_budget_ms": budget[i]}
    mono_body = lambda i: {"features": X[i].tolist()}
    await load(f"{URL['gateway']}/infer", pex_body, 100); await load(f"{URL['monolith']}/infer", mono_body, 100)  # warm-up
    sync.post(f"{URL['privacy']}/reset"); sync.post(f"{URL['gateway']}/controller/reset")
    lat, out, errs, wall = await load(f"{URL['gateway']}/infer", pex_body, n)
    R["pex_mesh"] = summary(lat, out, errs, wall, n)
    R["pex_mesh"]["served_by"] = {k: sum(1 for o in out if o and o["served_by"] == k) for k in ("edge", "cloud")}
    R["pex_mesh"]["decision_reasons"] = sync.get(f"{URL['gateway']}/controller").json()["stats"]
    lat, out, errs, wall = await load(f"{URL['monolith']}/infer", mono_body, n)
    R["monolith"] = summary(lat, out, errs, wall, n)

    # Chaos: cloud region goes down for 4 s in the middle of a 6 s run, for both systems
    async def chaos(target):
        await asyncio.sleep(1.0); ctl_post(target, True)
        await asyncio.sleep(4.0); ctl_post(target, False)
    for name, url, body, tgt in [("pex_mesh", f"{URL['gateway']}/infer", pex_body, "cloud"),
                                 ("monolith", f"{URL['monolith']}/infer", mono_body, "monolith")]:
        sync.post(f"{URL['privacy']}/reset"); sync.post(f"{URL['gateway']}/controller/reset")
        task = asyncio.create_task(chaos(tgt))
        lat, out, errs, wall = await load(url, body, 1200, conc=10)
        await task
        R.setdefault("chaos", {})[name] = {"requests": 1200, "failed": errs, "success_rate": round(1 - errs / 1200, 3)}

    # XAI: cost of sync vs async explanation on the response time (sequential, 200 calls each)
    for mode in ("auto", "async", "sync"):
        ts = []
        for i in range(200):
            t = time.perf_counter(); sync.post(f"{URL['gateway']}/infer", json={"features": X[i].tolist(), "xai": mode})
            ts.append((time.perf_counter() - t) * 1000)
        R.setdefault("xai_response_ms", {})[mode] = {"p50": round(float(np.median(ts)), 2), "p95": round(float(np.percentile(ts, 95)), 2)}
    for mode in ("auto", "async", "sync"):
        lat, out, errs, wall = await load(f"{URL['gateway']}/infer", lambda i: {"features": X[i].tolist(), "xai": mode}, 600, conc=20)
        R.setdefault("xai_under_load_ms", {})[mode] = {"p50": round(float(np.percentile(lat, 50)), 1),
            "p95": round(float(np.percentile(lat, 95)), 1), "throughput_rps": round(600 / wall, 1)}
    sync.post(f"{URL['privacy']}/reset")

def dp_sweep():
    """Offline comparison of where differential privacy is applied.
    v1: Laplace noise on every single reading sent for inference.  v2: no noise on inference (pseudonym + mTLS),
    DP on window means shared for analytics. Reports model accuracy and the error of the DP window means."""
    cloud, edge = joblib.load("models/cloud.joblib"), joblib.load("models/edge.joblib")
    Xte, yte = np.load("models/Xte.npy"), np.load("models/yte.npy")
    sensv = np.array([RANGES[f][1] - RANGES[f][0] for f in FEATURES])
    lo = np.array([RANGES[f][0] for f in FEATURES]); hi = np.array([RANGES[f][1] for f in FEATURES])
    r = np.random.default_rng(0); res = {"v1_per_reading_dp": {}, "v2_dp_on_window_means": {}}
    pe = edge.predict_proba(Xte)[:, 1]; unsure = np.abs(pe - .5) * 2 < 0.6; pc_clean = cloud.predict_proba(Xte)[:, 1]
    for eps in [0.5, 1, 2, 5, 10]:
        Xn = np.clip(Xte + r.laplace(0, sensv / (eps / len(FEATURES)), Xte.shape), lo, hi)
        res["v1_per_reading_dp"][eps] = {"pex_accuracy": round(accuracy_score(yte, np.where(unsure, cloud.predict_proba(Xn)[:, 1], pe) > .5), 3)}
        for n in (50, 200, 1000):
            errs = []
            for k in range(200):
                W = Xte[r.choice(len(Xte), n, replace=False)]; m = W.mean(0)
                noisy = np.clip(m + r.laplace(0, sensv / (n * eps / len(FEATURES))), lo, hi)
                errs.append(np.median(np.abs(noisy - m) / m))
            res["v2_dp_on_window_means"].setdefault(f"eps={eps}", {})[f"window={n}"] = f"{100 * np.median(errs):.1f}% median error"
    res["v2_pex_accuracy (no noise on inference)"] = round(accuracy_score(yte, np.where(unsure, pc_clean, pe) > .5), 3)
    res["edge_only_accuracy"] = round(accuracy_score(yte, pe > .5), 3)
    res["cloud_only_accuracy"] = round(accuracy_score(yte, pc_clean > .5), 3)
    res["share_escalated"] = round(float(unsure.mean()), 3)
    return res

asyncio.run(main())
R["dp_sweep"] = dp_sweep()
json.dump(R, open("results/results.json", "w"), indent=2, default=str)
print(json.dumps(R, indent=2, default=str))
