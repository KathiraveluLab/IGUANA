#!/bin/bash
set -e

# Create target directory
SSL_DIR="priv/ssl"
mkdir -p "$SSL_DIR"

echo "Generating self-signed TLS certificates for BEAM distribution..."

# 1. Generate CA key and cert
openssl req -new -x509 -nodes \
    -keyout "$SSL_DIR/ca-key.pem" \
    -out "$SSL_DIR/ca-cert.pem" \
    -days 3650 \
    -subj "/CN=IguanaCA"

# 2. Generate node key and certificate signing request (CSR)
openssl req -new -nodes \
    -keyout "$SSL_DIR/key.pem" \
    -out "$SSL_DIR/req.pem" \
    -subj "/CN=localhost"

# 3. Sign the CSR with the CA
openssl x509 -req \
    -in "$SSL_DIR/req.pem" \
    -CA "$SSL_DIR/ca-cert.pem" \
    -CAkey "$SSL_DIR/ca-key.pem" \
    -CAcreateserial \
    -out "$SSL_DIR/cert.pem" \
    -days 3650

# Clean up CSR
rm -f "$SSL_DIR/req.pem"

echo "Certificates successfully generated in $SSL_DIR"
