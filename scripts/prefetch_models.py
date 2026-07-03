"""
Download the ASR model and Silero VAD into the local cache at build time.

Run this once during the Docker build (with access to your model source, e.g.
Artifactory) so that the runtime container is fully self-contained: no calls to
Hugging Face, NGC, or GitHub at start.

Sources honour the same env vars used at runtime (MODEL_SOURCE, HF_ENDPOINT,
VAD_SOURCE, VAD_SOURCE_TYPE, MODEL_CACHE_DIR), so pointing the build at
Artifactory pre-bakes exactly what runtime will load.

Caches are written under $HF_HOME / $TORCH_HOME / $MODEL_CACHE_DIR, which must
be copied into (and readable by the non-root user of) the runtime image.
"""
import os
import sys

# Downloads happen on CPU; no GPU needed at build time.
os.environ.setdefault("DEVICE", "cpu")


def prefetch_vad() -> None:
    print("[prefetch] loading Silero VAD from configured source", flush=True)
    # Importing the shared loader triggers the download/cache from VAD_SOURCE.
    import parakeet_service.vad  # noqa: F401
    print("[prefetch] Silero VAD cached", flush=True)


def prefetch_asr() -> None:
    from pathlib import Path

    from parakeet_service.config import MODEL_SOURCE, MODEL_BAKED_PATH
    from parakeet_service.model import resolve_from_source

    print(f"[prefetch] loading ASR model from: {MODEL_SOURCE}", flush=True)
    model = resolve_from_source()
    # Bake a normalised copy to a stable path so runtime loads it directly
    # (restore_from) with no Hugging Face / NGC involvement.
    baked = Path(MODEL_BAKED_PATH)
    baked.parent.mkdir(parents=True, exist_ok=True)
    model.save_to(str(baked))
    print(f"[prefetch] ASR model baked to {baked}", flush=True)


if __name__ == "__main__":
    try:
        prefetch_vad()
        prefetch_asr()
    except Exception as exc:  # noqa: BLE001 - fail the build loudly
        print(f"[prefetch] FAILED: {exc}", file=sys.stderr, flush=True)
        raise
    print("[prefetch] all models cached successfully", flush=True)
