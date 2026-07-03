# syntax=docker/dockerfile:1

# =============================================================================
# Stage 1: Builder — install dependencies and pre-bake models into a cache.
# Requires network access at BUILD time so the runtime image can be air-gapped.
# =============================================================================
FROM python:3.11.9-slim-bookworm AS builder

# --- Configurable component/model sources (default to public; override to
#     point pip, torch wheels, and the models at your Artifactory) -------------
ARG PIP_INDEX_URL=https://pypi.org/simple
ARG PIP_EXTRA_INDEX_URL=
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128
ARG HF_ENDPOINT=
ARG MODEL_SOURCE=nvidia/parakeet-tdt-0.6b-v2
ARG VAD_SOURCE=snakers4/silero-vad
ARG VAD_SOURCE_TYPE=github

# Model caches are populated here and copied into the runtime stage.
# pip honours PIP_INDEX_URL / PIP_EXTRA_INDEX_URL from the environment, and the
# prefetch step honours MODEL_SOURCE / HF_ENDPOINT / VAD_SOURCE.
ENV HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch \
    MODEL_CACHE_DIR=/opt/models/asr \
    PIP_NO_CACHE_DIR=1 \
    PIP_INDEX_URL=${PIP_INDEX_URL} \
    PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL} \
    HF_ENDPOINT=${HF_ENDPOINT} \
    MODEL_SOURCE=${MODEL_SOURCE} \
    VAD_SOURCE=${VAD_SOURCE} \
    VAD_SOURCE_TYPE=${VAD_SOURCE_TYPE} \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:${PATH}"

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git build-essential ffmpeg && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create the virtual environment first so it is reused across layers.
RUN python -m venv /opt/venv

# Install Python dependencies. Torch comes from the (configurable) wheel index;
# everything else is pinned in requirements.txt for reproducible builds.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade "pip==24.3.1" && \
    pip install --no-cache-dir \
        torch==2.7.0 torchaudio==2.7.0 \
        --index-url "${TORCH_INDEX_URL}" && \
    pip install --no-cache-dir -r requirements.txt

# Pre-download the ASR model and Silero VAD from the configured sources into
# /opt/models so the runtime container never reaches out at start.
# The optional model token is passed as a BuildKit secret so it never lands in
# the image layers or history:
#   docker build --secret id=model_source_token,env=MODEL_SOURCE_TOKEN .
COPY parakeet_service ./parakeet_service
COPY scripts ./scripts
RUN --mount=type=secret,id=model_source_token \
    MODEL_SOURCE_TOKEN="$(cat /run/secrets/model_source_token 2>/dev/null || true)" \
    python scripts/prefetch_models.py && \
    rm -rf /opt/models/hf

# =============================================================================
# Stage 2: Runtime — minimal, non-root, offline.
# =============================================================================
FROM python:3.11.9-slim-bookworm

# Re-declare the source args so the runtime resolves the SAME cached artifacts
# that were pre-baked in the builder stage (e.g. a URL's cached .nemo filename).
ARG HF_ENDPOINT=
ARG MODEL_SOURCE=nvidia/parakeet-tdt-0.6b-v2
ARG VAD_SOURCE=snakers4/silero-vad
ARG VAD_SOURCE_TYPE=github

# Offline enforcement + cache locations must match the builder stage.
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    HF_HOME=/opt/models/hf \
    TORCH_HOME=/opt/models/torch \
    MODEL_CACHE_DIR=/opt/models/asr \
    MODEL_SOURCE=${MODEL_SOURCE} \
    HF_ENDPOINT=${HF_ENDPOINT} \
    VAD_SOURCE=${VAD_SOURCE} \
    VAD_SOURCE_TYPE=${VAD_SOURCE_TYPE} \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    NUMBA_CACHE_DIR=/tmp \
    MPLCONFIGDIR=/tmp \
    XDG_CACHE_HOME=/tmp

# hadolint ignore=DL3008
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg && \
    rm -rf /var/lib/apt/lists/* && \
    groupadd --gid 10001 appuser && \
    useradd --uid 10001 --gid 10001 --no-create-home --shell /usr/sbin/nologin appuser

WORKDIR /app

# Copy the virtualenv, pre-baked model caches, and application code, all owned
# by the unprivileged runtime user.
COPY --from=builder --chown=appuser:appuser /opt/venv /opt/venv
COPY --from=builder --chown=appuser:appuser /opt/models /opt/models
COPY --chown=appuser:appuser parakeet_service ./parakeet_service

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz').status==200 else 1)"

CMD ["uvicorn", "parakeet_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
