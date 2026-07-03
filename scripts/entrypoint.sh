#!/bin/sh
# Container entrypoint. Starts uvicorn, optionally with TLS.
#
# Browsers only allow microphone capture in a "secure context" (https:// or
# localhost), so the live dashboard needs HTTPS when accessed over the LAN.
# Set ENABLE_TLS=1 to have a self-signed certificate generated at startup into
# a writable tmpfs dir (nothing is baked into the image). Leave it unset to run
# plain HTTP (e.g. behind a TLS-terminating gateway).
set -eu

PORT="${PORT:-8000}"

case "${ENABLE_TLS:-0}" in
  1|true|TRUE|yes)
    CERT_DIR="${TLS_DIR:-/tmp/tls}"
    mkdir -p "$CERT_DIR"
    if [ ! -s "$CERT_DIR/key.pem" ] || [ ! -s "$CERT_DIR/cert.pem" ]; then
      echo "[entrypoint] generating self-signed TLS certificate in $CERT_DIR"
      openssl req -x509 -newkey rsa:2048 -nodes \
        -keyout "$CERT_DIR/key.pem" -out "$CERT_DIR/cert.pem" \
        -days 365 -subj "/CN=parakeet-stt" \
        -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" 2>/dev/null
    fi
    echo "[entrypoint] starting uvicorn with TLS on :$PORT"
    exec uvicorn parakeet_service.main:app --host 0.0.0.0 --port "$PORT" \
      --ssl-keyfile "$CERT_DIR/key.pem" --ssl-certfile "$CERT_DIR/cert.pem"
    ;;
  *)
    echo "[entrypoint] starting uvicorn (plain HTTP) on :$PORT"
    exec uvicorn parakeet_service.main:app --host 0.0.0.0 --port "$PORT"
    ;;
esac
