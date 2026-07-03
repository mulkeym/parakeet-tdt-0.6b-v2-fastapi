"""parakeet_service package.

Environment hygiene that MUST run before any submodule imports NeMo /
huggingface_hub (which capture some env vars at their own import time).
"""
import os

# huggingface_hub treats an *empty* HF_ENDPOINT as a broken base URL
# ("No scheme supplied"). Container images set ENV HF_ENDPOINT="" from an unset
# build-arg default, so strip it here — before nemo/huggingface_hub are imported
# by parakeet_service.model — rather than in config.py (which is imported *after*
# `import nemo` inside model.py).
if not os.environ.get("HF_ENDPOINT"):
    os.environ.pop("HF_ENDPOINT", None)
