#!/bin/bash
# Create a test CA, a server cert for the cloud service and a client cert for the edge gateway (mutual TLS).
cd "$(dirname "$0")"; mkdir -p certs; cd certs
openssl req -x509 -newkey rsa:2048 -nodes -keyout ca.key -out ca.crt -days 365 -subj "/CN=PEX-Mesh CA" \
 -addext "basicConstraints=critical,CA:TRUE" -addext "keyUsage=critical,keyCertSign,cRLSign" -addext "subjectKeyIdentifier=hash"
printf "basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=serverAuth\nsubjectAltName=IP:127.0.0.1,DNS:localhost\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n" > server.ext
printf "basicConstraints=CA:FALSE\nkeyUsage=critical,digitalSignature,keyEncipherment\nextendedKeyUsage=clientAuth\nsubjectKeyIdentifier=hash\nauthorityKeyIdentifier=keyid\n" > client.ext
openssl req -newkey rsa:2048 -nodes -keyout cloud.key -out cloud.csr -subj "/CN=127.0.0.1"
openssl x509 -req -in cloud.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out cloud.crt -days 365 -extfile server.ext
openssl req -newkey rsa:2048 -nodes -keyout gateway.key -out gateway.csr -subj "/CN=edge-gateway"
openssl x509 -req -in gateway.csr -CA ca.crt -CAkey ca.key -CAcreateserial -out gateway.crt -days 365 -extfile client.ext
