"""Functional test journeys for PEX-Mesh (services must be running: ./run_all.sh)."""
import time, httpx, numpy as np, pytest
import ssl
from common import URL, MTLS, CA_FILE, generate, FEATURES
c = httpx.Client(timeout=10)
cm = httpx.Client(timeout=10, **MTLS)        # mTLS client for the cloud service
SURE = [2.4, 105, 24, 2600, 1800]      # clearly failing machine -> edge is confident
X, _ = generate(3000, seed=7)
p_edge = [c.post(f"{URL['edge']}/predict", json={"trace_id": "probe", "features": x.tolist()}).json() for x in X[:400]]
UNSURE = X[int(np.argmin([r["confidence"] for r in p_edge]))].tolist()   # edge model is unsure here

def infer(**kw):
    return c.post(f"{URL['gateway']}/infer", json=kw).json()

@pytest.fixture(autouse=True)
def clean():
    cm.post(f"{URL['cloud']}/chaos", json={"down": False}); c.post(f"{URL['privacy']}/reset")
    c.post(f"{URL['gateway']}/controller/reset"); yield

def wait_explanation(tid, timeout=5):
    end = time.time() + timeout
    while time.time() < end:
        r = c.get(f"{URL['xai']}/explanation/{tid}").json()
        if "shap" in r: return r
        time.sleep(0.05)
    return None

def test_J1_confident_reading_served_at_edge_with_explanation_and_trace():
    r = infer(features=SURE, xai="async")
    assert r["served_by"] == "edge" and r["alert"] is True
    e = wait_explanation(r["trace_id"]); assert e and len(e["top_reasons"]) == 3
    time.sleep(0.2)
    services = {s["service"] for s in c.get(f"{URL['obs']}/trace/{r['trace_id']}").json()["spans"]}
    assert {"gateway", "edge-inference", "xai-sidecar"} <= services

def test_J2_unsure_reading_escalated_to_cloud_through_privacy_sidecar():
    r = infer(features=UNSURE, device_id="cnc-2")
    assert r["served_by"] == "cloud", r
    time.sleep(0.2)
    services = {s["service"] for s in c.get(f"{URL['obs']}/trace/{r['trace_id']}").json()["spans"]}
    assert {"privacy-sidecar", "cloud-inference"} <= services       # pseudonymised before leaving the edge

def test_J3_sensitive_data_never_leaves_the_edge():
    r = infer(features=UNSURE, sensitivity="high", device_id="cnc-3")
    assert r["served_by"] == "edge" and "sensitive" in r["reason"]
    time.sleep(0.2)
    services = {s["service"] for s in c.get(f"{URL['obs']}/trace/{r['trace_id']}").json()["spans"]}
    assert "cloud-inference" not in services and "privacy-sidecar" not in services

def test_J4_tight_latency_budget_keeps_request_at_edge():
    r = infer(features=UNSURE, latency_budget_ms=10)
    assert r["served_by"] == "edge" and "latency" in r["reason"]

def test_J5_cloud_outage_falls_back_to_edge_without_errors():
    cm.post(f"{URL['cloud']}/chaos", json={"down": True})
    r1 = infer(features=UNSURE); r2 = infer(features=UNSURE)
    assert r1["served_by"] == "edge" and r1["degraded"] is True        # first call detects the failure
    assert "circuit open" in r2["reason"]                               # next calls skip the cloud at once

def test_J6_privacy_budget_exhausted_blocks_sharing():
    W = X[:200].tolist()
    for _ in range(10):                                                 # budget 10, epsilon 1 per window
        assert c.post(f"{URL['gateway']}/share_window", json={"device_id": "cnc-6", "readings": W}).json()["shared"]
    r = c.post(f"{URL['gateway']}/share_window", json={"device_id": "cnc-6", "readings": W}).json()
    assert r["shared"] is False and "budget" in r["reason"]

def test_J7_async_explanation_does_not_block_response():
    t = time.perf_counter(); infer(features=SURE, xai="async"); fast = time.perf_counter() - t
    t = time.perf_counter(); r = infer(features=SURE, xai="sync"); slow = time.perf_counter() - t
    assert r["explanation"]["top_reasons"] and fast < slow

def test_J8_cloud_recovers_after_cooldown():
    cm.post(f"{URL['cloud']}/chaos", json={"down": True}); infer(features=UNSURE)
    cm.post(f"{URL['cloud']}/chaos", json={"down": False}); time.sleep(5.2)
    assert infer(features=UNSURE)["served_by"] == "cloud"

def test_J9_cloud_rejects_clients_without_certificate():
    with pytest.raises(Exception):
        httpx.get(f"{URL['cloud']}/fleet_stats", verify=ssl.create_default_context(cafile=CA_FILE))          # no client cert -> handshake fails
    assert cm.get(f"{URL['cloud']}/fleet_stats").status_code == 200            # gateway cert -> accepted

def test_J10_dp_window_stats_are_useful_and_pseudonymous():
    W = X[:200]
    r = c.post(f"{URL['gateway']}/share_window", json={"device_id": "cnc-10", "readings": W.tolist()}).json()
    rel_err = np.abs(np.array(r["dp_mean"]) - W.mean(0)) / W.mean(0)
    assert np.median(rel_err) < 0.25                                            # epsilon=1, 200 readings
    last = cm.get(f"{URL['cloud']}/fleet_stats").json()["windows"][-1]
    assert last["device"] != "cnc-10" and len(last["device"]) == 16           # cloud only sees a pseudonym

def test_J11_auto_xai_explains_edge_inline_and_cloud_in_background():
    r = infer(features=SURE)                               # edge -> explanation already in the response
    assert r["served_by"] == "edge" and r["explanation"]["top_reasons"]
    r = infer(features=UNSURE)                             # cloud -> explanation fetched later
    assert r["served_by"] == "cloud" and r["explanation"] is None and wait_explanation(r["trace_id"])
