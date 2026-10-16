"""Shared pieces: synthetic predictive-maintenance data, feature ranges, ports, tracing helper."""
import os, time, uuid, numpy as np, httpx
# local services must never go through an outbound proxy
os.environ["NO_PROXY"] = os.environ["no_proxy"] = "127.0.0.1,localhost," + os.environ.get("NO_PROXY", "")

FEATURES = ["vibration_rms", "temperature_c", "current_a", "rpm", "hours_since_maint"]
# Clipping ranges (needed for differential privacy sensitivity)
RANGES = {"vibration_rms": (0, 4), "temperature_c": (40, 120), "current_a": (0, 30),
          "rpm": (500, 3000), "hours_since_maint": (0, 2000)}

PORTS = dict(gateway=8100, edge=8101, cloud=8102, privacy=8103, xai=8104, obs=8105, monolith=8200)
URL = {k: f"http://127.0.0.1:{v}" for k, v in PORTS.items()}
URL["cloud"] = f"https://127.0.0.1:{PORTS['cloud']}"          # cloud is only reachable over mutual TLS
CERTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "certs")
CA_FILE = os.path.join(CERTS, "ca.crt")
CLIENT_CERT = (os.path.join(CERTS, "gateway.crt"), os.path.join(CERTS, "gateway.key"))
def mtls_context():
    import ssl
    ctx = ssl.create_default_context(cafile=CA_FILE); ctx.load_cert_chain(*CLIENT_CERT); return ctx
MTLS = dict(verify=mtls_context())          # pass as httpx.Client(**MTLS): trusts only our CA, presents the gateway cert

# Simulated one-way-trip cost of going from the factory edge to the cloud (round trip added once per call)
WAN_RTT_MS = float(os.environ.get("WAN_RTT_MS", "40"))

def generate(n, seed=0):
    """Synthetic CNC machine telemetry. Label = machine fails within the next 48 h."""
    r = np.random.default_rng(seed)
    hours = r.uniform(0, 2000, n)
    load = r.uniform(0, 1, n)
    rpm = 1000 + 1500 * load + r.normal(0, 80, n)
    vib = 0.4 + 0.0009 * hours + 0.8 * load ** 2 + r.normal(0, 0.15, n)
    temp = 55 + 0.012 * hours + 18 * load + r.normal(0, 3, n)
    current = 8 + 12 * load + 0.002 * hours + r.normal(0, 1, n)
    # non-linear failure risk (interaction between vibration and heat)
    z = -9 + 2.2 * vib + 0.06 * (temp - 60) + 1.6 * (vib > 1.6) * (temp > 80) + 0.0015 * hours
    p = 1 / (1 + np.exp(-z))
    y = (r.uniform(0, 1, n) < p).astype(int)
    X = np.column_stack([vib, temp, current, rpm, hours])
    lo = np.array([RANGES[f][0] for f in FEATURES]); hi = np.array([RANGES[f][1] for f in FEATURES])
    return np.clip(X, lo, hi), y

def send_span(client, trace_id, service, op, start, **attrs):
    """Observability: every service reports a span (fire-and-forget style)."""
    try:
        client.post(f"{URL['obs']}/span", json=dict(trace_id=trace_id, service=service, op=op,
                    start=start, dur_ms=(time.perf_counter() - start) * 1000, attrs=attrs), timeout=1)
    except Exception:
        pass

def new_id():
    return uuid.uuid4().hex[:12]
