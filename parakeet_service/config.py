import logging, os, sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"  # default model identifier

# --- Model / component sources (Artifactory-friendly) ------------------------
# ASR model source. Accepts any of:
#   * a Hugging Face / NGC repo id  -> resolved via ASRModel.from_pretrained
#     (route through an Artifactory HF proxy by also setting HF_ENDPOINT)
#   * a local path to a .nemo file  -> loaded via ASRModel.restore_from
#   * an http(s) URL to a .nemo file (e.g. Artifactory generic repo) -> the file
#     is downloaded once into MODEL_CACHE_DIR, then restore_from is used.
MODEL_SOURCE = os.getenv("MODEL_SOURCE", MODEL_NAME)
# Directory that downloaded .nemo artifacts are cached in.
MODEL_CACHE_DIR = os.getenv("MODEL_CACHE_DIR", "/opt/models/asr")
# Stable path the model is baked to at build time. When present it is loaded
# directly (restore_from) at runtime, so the container never touches Hugging
# Face / NGC and behaves deterministically offline regardless of MODEL_SOURCE.
MODEL_BAKED_PATH = os.getenv("MODEL_BAKED_PATH", os.path.join(MODEL_CACHE_DIR, "model.nemo"))
# Optional bearer token for authenticating the MODEL_SOURCE download (e.g. an
# Artifactory access token). Sent as "Authorization: Bearer <token>".
MODEL_SOURCE_TOKEN = os.getenv("MODEL_SOURCE_TOKEN") or None
# Optional Hugging Face endpoint override (huggingface_hub reads HF_ENDPOINT
# directly; surfaced here for logging/validation). Point at an Artifactory HF
# remote to proxy from_pretrained downloads, e.g.
#   https://artifactory.example.com/artifactory/api/huggingfaceml/hf-remote
HF_ENDPOINT = os.getenv("HF_ENDPOINT") or None
# Normalise the process environment: huggingface_hub treats an *empty* HF_ENDPOINT
# as a broken base URL ("No scheme supplied"). A container ENV of "" (from an
# unset build-arg default) must therefore be removed, not left blank.
if HF_ENDPOINT is None:
    os.environ.pop("HF_ENDPOINT", None)
else:
    os.environ["HF_ENDPOINT"] = HF_ENDPOINT

# Silero VAD source. Either a torch.hub repo id (default, VAD_SOURCE_TYPE=github)
# or a local directory containing the vendored repo (VAD_SOURCE_TYPE=local).
VAD_SOURCE = os.getenv("VAD_SOURCE", "snakers4/silero-vad")
VAD_SOURCE_TYPE = os.getenv("VAD_SOURCE_TYPE", "github").lower()  # "github" | "local"

# Configuration from environment variables
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))          # model’s native sample-rate
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "fp16")
DEVICE = os.getenv("DEVICE", "cuda")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
BATCH_WINDOW_MS = int(os.getenv("BATCH_WINDOW_MS", "100"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "3600"))  # seconds; hard ceiling on decoded audio
VAD_THRESHOLD = float(os.getenv("VAD_THRESHOLD", "0.5"))
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "60"))    # seconds; ffmpeg/transcode timeout

# --- Security / DoS limits ---------------------------------------------------
# Optional API key. If unset/empty, the service is open (no auth). If set, every
# REST request and WebSocket connection must present a matching key.
API_KEY = os.getenv("API_KEY") or None
# Header clients present the key in (also accepts "Authorization: Bearer <key>").
API_KEY_HEADER = os.getenv("API_KEY_HEADER", "x-api-key")
# Maximum accepted upload size in bytes (default 100 MiB). Guards disk/GPU DoS.
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("parakeet_service")
