"""
Shared Silero VAD loader.

Loads the VAD model exactly once per process from a configurable source so that
air-gapped / Artifactory deployments can vendor the repo locally instead of
reaching out to GitHub. Both the offline and streaming chunkers import the
resulting ``vad_model`` / ``vad_utils`` from here.
"""
from __future__ import annotations

from torch.hub import load as torch_hub_load

from .config import VAD_SOURCE, VAD_SOURCE_TYPE, logger


def _load_vad():
    if VAD_SOURCE_TYPE == "local":
        # Load a vendored copy of the repo from a local directory (no network).
        logger.info("Loading Silero VAD from local dir: %s", VAD_SOURCE)
        return torch_hub_load(VAD_SOURCE, "silero_vad", source="local", trust_repo=True)
    # Default: torch.hub repo id. trust_repo=True uses the pre-baked cache and
    # skips the interactive trust prompt / network revalidation when cached.
    logger.info("Loading Silero VAD from hub repo: %s", VAD_SOURCE)
    return torch_hub_load(VAD_SOURCE, "silero_vad", trust_repo=True)


vad_model, vad_utils = _load_vad()
