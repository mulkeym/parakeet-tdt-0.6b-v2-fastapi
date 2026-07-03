"""Container healthcheck that works whether the service runs HTTP or HTTPS.

For HTTPS we verify against the self-signed certificate the entrypoint generated
(trusting it as a CA) rather than disabling verification — the cert's SAN
includes 127.0.0.1, so hostname verification of the loopback probe succeeds.
"""
import os
import ssl
import sys
import urllib.request

cert = os.path.join(os.environ.get("TLS_DIR", "/tmp/tls"), "cert.pem")

attempts = []
if os.path.exists(cert):
    attempts.append(("https", ssl.create_default_context(cafile=cert)))
attempts.append(("http", None))

for scheme, ctx in attempts:
    try:
        resp = urllib.request.urlopen(f"{scheme}://127.0.0.1:8000/healthz", context=ctx, timeout=3)
        if resp.status == 200:
            sys.exit(0)
    except Exception:
        continue
sys.exit(1)
