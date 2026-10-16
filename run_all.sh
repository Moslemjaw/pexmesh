#!/bin/bash
# Start every PEX-Mesh service + the monolith baseline as separate processes (one microservice each).
# The cloud service only accepts mutual-TLS connections signed by the PEX-Mesh CA.
cd "$(dirname "$0")"; mkdir -p logs
TLS="--ssl-keyfile certs/cloud.key --ssl-certfile certs/cloud.crt --ssl-ca-certs certs/ca.crt --ssl-cert-reqs 2"
for s in obs:8105 edge:8101 cloud:8102 privacy:8103 xai:8104 gateway:8100 monolith:8200; do
  n=${s%%:*}; p=${s##*:}; extra=""; [ "$n" = "cloud" ] && extra="$TLS"
  pin=""; [ "$n" != "xai" ] && [ $(nproc) -ge 4 ] && pin="taskset -c 0-$(( $(nproc) - 2 ))"
  nohup $pin python3 -m uvicorn services.$n:app --port $p --log-level warning $extra > logs/$n.log 2>&1 &
done
up() { curl -s -o /dev/null http://127.0.0.1:$1/docs || curl -s -o /dev/null --cacert certs/ca.crt --cert certs/gateway.crt --key certs/gateway.key https://127.0.0.1:$1/docs; }
for p in 8105 8101 8102 8103 8104 8100 8200; do
  for i in $(seq 1 60); do up $p && break; sleep 0.5; done
done
echo "all services up"
