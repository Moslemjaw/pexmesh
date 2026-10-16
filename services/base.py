import asyncio, ssl, time, httpx
from common import URL, MTLS
_client, _mtls = None, None
LIMITS = httpx.Limits(max_connections=200, max_keepalive_connections=100)
def client():
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=5, limits=LIMITS)
    return _client

def mtls_client():
    """Client that presents the gateway certificate and only trusts the PEX-Mesh CA."""
    global _mtls
    if _mtls is None:
        _mtls = httpx.AsyncClient(timeout=5, limits=LIMITS, **MTLS)
    return _mtls

async def _post_span(span):
    try:
        await client().post(f"{URL['obs']}/span", json=span, timeout=1)
    except Exception:
        pass

def span(trace_id, service, op, start, **attrs):
    """Observability sidecar hook: report a span without blocking the request."""
    asyncio.create_task(_post_span(dict(trace_id=trace_id, service=service, op=op,
                        dur_ms=round((time.perf_counter() - start) * 1000, 3), attrs=attrs)))
