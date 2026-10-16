"""Test SLR search strings against the quasi-gold standard (title + abstract only)."""
import json, re, sys
papers = json.load(open(sys.argv[1]))

def term(t):
    # a term like 'architect*' or '"neural network*"' -> regex, case-insensitive, word boundaries
    t = t.strip('"')
    parts = [re.escape(w).replace(r'\*', r'\w*') for w in re.split(r'[\s-]+', t)]
    return re.compile(r'(?<![\w])' + r'[\s\-/]+'.join(parts) + r'(?![\w])', re.I)

def any_of(*terms):
    rx = [term(t) for t in terms]
    return lambda txt: any(r.search(txt) for r in rx)

AI_IOT_COMBINED = any_of('"Artificial Intelligence of Things"', 'AIoT', '"AI-IoT"', 'AIoMT', '"Intelligent IoT"')
AI = ['"artificial intelligence"', '"machine learning"', '"deep learning"', '"neural network*"', '"federated learning"',
      '"edge intelligence"', '"edge AI"', 'TinyML', '"explainable AI"', 'XAI', 'MLOps']
IOT = any_of('IoT', '"Internet of Things"', 'IIoT', 'IoMT', '"cyber-physical system*"')
ARCH_V1 = ['architect*', 'microservice*', '"service-oriented"', 'serverless', '"design pattern*"', '"model-driven"', 'framework']
EDGE_V1 = ['edge', 'fog', '"cloud-edge"', '"edge-cloud"', '"cloud continuum"', '"compute continuum"', 'osmotic']

def make(ai_terms, arch, edge):
    ai, ar, ed = any_of(*ai_terms), any_of(*arch), any_of(*edge)
    return lambda x: (AI_IOT_COMBINED(x) or (ai(x) and IOT(x))) and ar(x) and ed(x)

variants = {
  'V0 old string (AIoT only)': lambda x: any_of('"Artificial Intelligence of Things"','AIoT')(x)
        and any_of('microservice*','"software architecture"','"design pattern"','"reference architecture"')(x)
        and any_of('"edge computing"','"edge-cloud"','"fog computing"')(x)
        and any_of('performance','scalability','privacy','security','validation','evaluation')(x),
  'V1 protocol string':        make(AI, ARCH_V1, EDGE_V1),
  'V2 = V1 + "AI" + system* + federated/on-device as edge': make(AI + ['AI'], ARCH_V1 + ['system*', 'platform*'], EDGE_V1 + ['federated', '"on-device"']),
}
for name, f in variants.items():
    hits = [p['n'] for p in papers if f(p['title'] + ' . ' + p['abstract'])]
    miss = [p['n'] for p in papers if p['n'] not in hits]
    print(f'{name}: {len(hits)}/15 found  missed={miss}')
extra = {
 'V3 = V1 + "AI" + federated/on-device (no system*)': make(AI + ['AI'], ARCH_V1, EDGE_V1 + ['federated', '"on-device"']),
 'V4 = V3 + platform* + infrastructure*': make(AI + ['AI'], ARCH_V1 + ['platform*', 'infrastructure*'], EDGE_V1 + ['federated', '"on-device"']),
}
for name, f in extra.items():
    hits = [p['n'] for p in papers if f(p['title'] + ' . ' + p['abstract'])]
    print(f'{name}: {len(hits)}/15 found  missed={[p["n"] for p in papers if p["n"] not in hits]}')
ARCH_V5 = ['architect*', 'microservice*', '"service-oriented"', 'serverless', '"design pattern*"', '"model-driven"', 'framework*']
f = make(AI + ['AI'], ARCH_V5, EDGE_V1 + ['federated', '"on-device"'])
hits = [p['n'] for p in papers if f(p['title'] + ' . ' + p['abstract'])]
print(f'V5 FINAL (V3 + framework*): {len(hits)}/15 found  missed={[p["n"] for p in papers if p["n"] not in hits]}')
