from contextlib import asynccontextmanager
import contextlib
import gc
import os
import shutil
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import torch, asyncio
import nemo.collections.asr as nemo_asr
from omegaconf import open_dict

from .config import (
    MODEL_SOURCE,
    MODEL_CACHE_DIR,
    MODEL_BAKED_PATH,
    MODEL_SOURCE_TOKEN,
    MODEL_PRECISION,
    DEVICE,
    HF_ENDPOINT,
    logger,
)

from parakeet_service.batchworker import batch_worker


def _download_nemo(url: str) -> Path:
    """Fetch a .nemo artifact (e.g. from Artifactory) into MODEL_CACHE_DIR once."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported model URL scheme: {parsed.scheme!r}")
    cache_dir = Path(MODEL_CACHE_DIR)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / (Path(parsed.path).name or "model.nemo")
    if dest.exists():
        logger.info("Using cached model artifact %s", dest)
        return dest
    logger.info("Downloading model artifact from %s", url)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url)
    if MODEL_SOURCE_TOKEN:
        req.add_header("Authorization", f"Bearer {MODEL_SOURCE_TOKEN}")
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as out:  # noqa: S310 - scheme validated above
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)
    return dest


def resolve_from_source():
    """Obtain the ASR model from the configured MODEL_SOURCE.

    MODEL_SOURCE may be a repo id (from_pretrained), a local .nemo path, or an
    http(s) URL to a .nemo file (downloaded then restore_from). Used at build
    time by the prefetch step; runtime prefers the baked artifact (see below).
    """
    src = MODEL_SOURCE

    if src.startswith(("http://", "https://")):
        local = _download_nemo(src)
        logger.info("Restoring ASR model from downloaded artifact %s", local)
        return nemo_asr.models.ASRModel.restore_from(str(local), map_location=DEVICE)

    if src.endswith(".nemo") or os.path.exists(src):
        logger.info("Restoring ASR model from local artifact %s", src)
        return nemo_asr.models.ASRModel.restore_from(src, map_location=DEVICE)

    logger.info("Loading ASR model via from_pretrained: %s (HF_ENDPOINT=%s)",
                src, HF_ENDPOINT or "default")
    return nemo_asr.models.ASRModel.from_pretrained(src, map_location=DEVICE)


def _load_asr_model():
    """Load the ASR model, preferring the pre-baked local artifact.

    If MODEL_BAKED_PATH exists (created by the build-time prefetch), it is loaded
    directly via restore_from so the runtime container never reaches Hugging Face
    / NGC and is fully deterministic offline. Otherwise fall back to resolving
    from MODEL_SOURCE (local dev / non-Docker use).
    """
    baked = Path(MODEL_BAKED_PATH)
    if baked.exists():
        logger.info("Restoring ASR model from baked artifact %s", baked)
        return nemo_asr.models.ASRModel.restore_from(str(baked), map_location=DEVICE)
    return resolve_from_source()


def _to_builtin(obj):
    """torch/NumPy → pure-Python (JSON-safe)."""
    import numpy as np
    import torch as th

    if isinstance(obj, (th.Tensor, np.ndarray)):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; free GPU on shutdown."""
    logger.info("Loading ASR model from %s with optimized memory...", MODEL_SOURCE)
    with torch.inference_mode():
        # Determine precision
        dtype = torch.float16 if MODEL_PRECISION == "fp16" else torch.float32

        # Load model from the configured source with device and precision
        model = _load_asr_model().to(dtype=dtype)
        logger.info("Loaded model with %s weights on %s", MODEL_PRECISION.upper(), DEVICE)
        
    # Aggressive cleanup
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Memory cleanup complete")

    # Enable per-word confidence scoring. Word confidence is derived from frame
    # confidence, which NeMo only computes when alignments are preserved — hence
    # preserve_alignments must be True here (otherwise change_decoding_strategy
    # raises "preserve_alignments flag must also be set").
    with open_dict(model.cfg.decoding):
        model.cfg.decoding.preserve_alignments = True
        model.cfg.decoding.confidence_cfg = {
            "preserve_word_confidence": True,
            "preserve_token_confidence": False,
            "preserve_frame_confidence": False,
            "method_cfg": {"name": "max_prob"},
        }
    model.change_decoding_strategy(model.cfg.decoding)
    logger.info("Enabled word-level confidence scoring")

    app.state.asr_model = model
    logger.info("Model ready on %s", next(model.parameters()).device)

    app.state.worker = asyncio.create_task(batch_worker(model), name="batch_worker")
    logger.info("batch_worker scheduled")

    try:
        yield
    finally:
        app.state.worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await app.state.worker

        logger.info("Releasing GPU memory and shutting down worker")
        del app.state.asr_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()  # free cache but keep driver


def reset_fast_path(model):
    """Restore low-latency decoding flags."""
    with open_dict(model.cfg.decoding):
        if getattr(model.cfg.decoding, "compute_timestamps", False):
            model.cfg.decoding.compute_timestamps = False
        if getattr(model.cfg.decoding, "preserve_alignments", False):
            model.cfg.decoding.preserve_alignments = False
    model.change_decoding_strategy(model.cfg.decoding)
