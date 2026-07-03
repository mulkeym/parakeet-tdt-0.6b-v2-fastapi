# Security Posture

This document summarises the security controls in the Parakeet STT service and
the assumptions behind a hardened deployment. It is intended for reviewers
evaluating the service for a restricted / air-gapped environment.

## Controls

### Authentication
- Optional API key (`API_KEY`). When unset the service is open; when set it is
  required on **all** REST endpoints and the `/ws` WebSocket. `/healthz` is
  intentionally left open for liveness/readiness probes.
- Keys are accepted via the `x-api-key` header (name configurable with
  `API_KEY_HEADER`) or `Authorization: Bearer <key>`; the WebSocket also accepts
  a `?api_key=` query parameter for browser clients that cannot set headers.
- Comparison is constant-time (`hmac.compare_digest`).
- See [Authentication](README.md#authentication).

### Network / supply chain
- **Air-gapped runtime.** The ASR model and Silero VAD are pre-baked into the
  image at build time. At runtime `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1`
  are enforced and `torch.hub` loads from the local cache — no calls to Hugging
  Face, NGC, or GitHub at start.
- **Configurable sources.** Models and Python/wheel components can be sourced
  from an internal Artifactory (see
  [Model Sources](README.md#model-sources-artifactory--air-gapped)). A model
  download token is passed via a BuildKit secret and never persisted in image
  layers or history.
- **Pinned dependencies.** `requirements.txt` is version-pinned. For production,
  generate a hash-locked file (`pip-compile --generate-hashes`) against your
  vetted mirror and install with `--require-hashes`.

### Container hardening
- Runs as a non-root user (uid/gid 10001).
- `docker-compose.yaml` sets `read_only` root filesystem, `tmpfs` for `/tmp`,
  `no-new-privileges`, and `cap_drop: ALL`.
- Base image pinned to `python:3.11.9-slim-bookworm`.
- Dockerfile is linted by hadolint in CI (`.gitlab-ci.yml`, `.hadolint.yaml`).

### Denial-of-service limits
- Upload size cap (`MAX_UPLOAD_BYTES`, default 100 MiB) enforced while streaming;
  temp files from rejected uploads are removed immediately.
- Decoded-audio duration ceiling (`MAX_AUDIO_DURATION`, default 3600 s).
- Hard timeout on the ffmpeg transcode step (`PROCESSING_TIMEOUT`, default 60 s).
- WebSocket ingest queue is bounded (`maxsize=64`).

### Information disclosure
- Error responses are generic; underlying exceptions and ffmpeg stderr are logged
  server-side only.
- The `/debug/cfg` model-configuration endpoint has been removed.

### Demo dashboard (browser client)
- Served **same-origin only**: self-hosted fonts, external CSS/JS, no CDN calls
  (consistent with the air-gapped posture).
- Strict **Content-Security-Policy** (`default-src 'self'`, no `unsafe-inline`)
  plus `X-Frame-Options: DENY`, `X-Content-Type-Options: nosniff`,
  `Referrer-Policy: no-referrer`, `Permissions-Policy` (mic self only), and
  `Cross-Origin-Opener-Policy` on every response.
- Transcript text (untrusted model output) is rendered via `textContent`; the UI
  uses no `innerHTML` with dynamic data.
- The dashboard can be disabled entirely with `SERVE_DASHBOARD=0` (returns 404).
- Optional WebSocket `Origin` allowlist via `ALLOWED_WS_ORIGINS` guards against
  cross-site WebSocket hijacking (auth is already token-based, not cookie-based).

## Deployment assumptions / operator responsibilities
- **TLS termination** is expected at an upstream gateway / ingress. The service
  speaks plain HTTP/WS.
- **`API_KEY` should be set** in any shared or exposed deployment and delivered
  via a secret (not committed to the image or VCS).
- **`MAX_AUDIO_DURATION`** defaults to 3600 s; lower it to match expected inputs.
- **Dependency hash-locking** must be completed against your vetted mirror before
  production (see above).
- Set resource limits (CPU/memory/GPU) via your orchestrator; the compose file
  includes a commented example.

## Reporting
Report suspected vulnerabilities to the maintainers via a private channel; do not
open public issues for security reports.
