#!/usr/bin/env python3
"""Benchmark tool for the Parakeet STT websocket service."""

import argparse
import asyncio
import csv
import hashlib
import io
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets

logger = logging.getLogger("benchmark")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1600  # 100ms chunks at 16kHz (matches voice agent)
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE  # 0.1 seconds
SILENCE_PADDING_FRAMES = int(1.0 / FRAME_DURATION)  # ~10 frames of trailing silence

# Pool of unique paragraphs -- each concurrent session gets a different one.
# This catches cross-session result leaking or stale-data memory bugs.
TEXT_POOL = [
    (
        "The quick brown fox jumps over the lazy dog near the riverbank. "
        "She sells seashells by the seashore every Sunday morning."
    ),
    (
        "A journey of a thousand miles begins with a single step forward. "
        "The weather forecast predicts sunny skies and mild temperatures today."
    ),
    (
        "Musicians practiced their instruments diligently before the evening concert. "
        "Scientists discovered a new species of butterfly in the tropical rainforest."
    ),
    (
        "The old lighthouse keeper watched the ships sail through the foggy harbor. "
        "Children played in the park while their parents read books on the bench."
    ),
    (
        "The mountain trail wound through ancient pine forests and crystal clear streams. "
        "A gentle breeze carried the scent of wildflowers across the meadow."
    ),
    (
        "The professor explained the complex equation with remarkable clarity and patience. "
        "Students gathered around the laboratory table to observe the chemical reaction."
    ),
    (
        "Autumn leaves drifted slowly from the maple trees lining the quiet boulevard. "
        "The bakery on the corner filled the entire street with the aroma of fresh bread."
    ),
    (
        "The submarine descended into the dark waters of the continental shelf. "
        "Marine biologists documented several previously unknown deep sea creatures."
    ),
    (
        "The architect designed a building that harmonized perfectly with the landscape. "
        "Construction workers began pouring the foundation early Monday morning."
    ),
    (
        "The orchestra conductor raised her baton and the symphony hall fell silent. "
        "Violins and cellos blended together in a hauntingly beautiful melody."
    ),
]


def parse_dmon_line(line: str) -> tuple[int, int] | None:
    """Parse one line of nvidia-smi dmon output.

    Columns (with -s um): gpu sm% mem% enc% dec% jpg% ofa% fb(MB) bar1(MB) ccpm(MB)
    Index:                  0   1    2    3    4    5    6    7       8        9
    Returns (gpu_util_pct, vram_used_mib) or None for headers/blanks.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        sm_pct = int(parts[1])
        fb_mib = int(parts[7])
        return (sm_pct, fb_mib)
    except (ValueError, IndexError):
        return None


def compute_gpu_stats(samples: list[tuple[int, int]]) -> dict:
    """Compute summary stats from a list of (gpu_util, vram_mib) samples."""
    if not samples:
        return {
            "gpu_util_mean": None, "gpu_util_p95": None,
            "vram_peak_mib": None, "vram_mean_mib": None,
        }
    utils = [s[0] for s in samples]
    vrams = [s[1] for s in samples]
    return {
        "gpu_util_mean": round(statistics.mean(utils), 1),
        "gpu_util_p95": round(percentile(utils, 95), 1),
        "vram_peak_mib": max(vrams),
        "vram_mean_mib": round(statistics.mean(vrams), 1),
    }


def detect_gpu_id() -> int:
    """Find the GPU running the parakeet service via nvidia-smi.

    Queries nvidia-smi for compute apps with full process names. Prefers
    processes whose command path contains 'parakeet' or 'nemo', falling
    back to any python process on a GPU. Returns device index or 0.
    """
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        gpu_uuids = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        uuid_to_idx = {}
        for line in gpu_uuids.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                uuid_to_idx[parts[1]] = int(parts[0])

        # Collect all python processes with their GPU index and process name
        candidates: list[tuple[int, str, str]] = []  # (gpu_idx, pid, process_name)
        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                pid_str, gpu_uuid, proc_name = parts[0], parts[1], parts[2]
                if "python" in proc_name.lower():
                    idx = uuid_to_idx.get(gpu_uuid, 0)
                    candidates.append((idx, pid_str, proc_name))

        if not candidates:
            logger.warning("No python GPU processes found, defaulting to device 0")
            return 0

        # Prefer processes whose path suggests parakeet/nemo/uvicorn
        for idx, pid, name in candidates:
            name_lower = name.lower()
            if any(hint in name_lower for hint in ("parakeet", "nemo", "uvicorn")):
                logger.info("Detected parakeet on GPU %d (PID %s, %s)", idx, pid, name)
                return idx

        # Fall back to the python process using the most VRAM (likely the ASR model)
        # by picking the last candidate (nvidia-smi tends to list larger processes later)
        idx, pid, name = candidates[-1]
        logger.info("Best-guess parakeet on GPU %d (PID %s, %s)", idx, pid, name)
        return idx

    except Exception as exc:
        logger.warning("GPU auto-detect failed: %s", exc)
    logger.warning("Could not detect GPU, defaulting to device 0")
    return 0


class GpuMonitor:
    """Collects GPU utilization and VRAM samples via nvidia-smi dmon."""

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.samples: list[tuple[int, int]] = []  # (gpu_util%, vram_mib)
        self._proc: subprocess.Popen | None = None
        self._task: asyncio.Task | None = None

    async def start(self):
        """Start the dmon subprocess and reader task."""
        try:
            self._proc = subprocess.Popen(
                ["nvidia-smi", "dmon", "-s", "um", "-d", "1", "-i", str(self.gpu_id)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
            )
            self._task = asyncio.create_task(self._reader())
            logger.info("GPU monitor started on device %d", self.gpu_id)
        except FileNotFoundError:
            logger.warning("nvidia-smi not found, GPU metrics disabled")

    async def _reader(self):
        """Read dmon stdout in a thread and parse lines."""
        loop = asyncio.get_event_loop()
        while self._proc and self._proc.poll() is None:
            line = await loop.run_in_executor(None, self._proc.stdout.readline)
            if not line:
                break
            parsed = parse_dmon_line(line)
            if parsed is not None:
                self.samples.append(parsed)

    def mark(self) -> int:
        """Return current sample count as a snapshot marker."""
        return len(self.samples)

    def slice_stats(self, start_mark: int) -> dict:
        """Compute stats for samples collected since start_mark."""
        window = self.samples[start_mark:]
        return compute_gpu_stats(window)

    async def stop(self):
        """Kill the dmon subprocess."""
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=3)
            self._proc = None
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
@dataclass
class AudioSample:
    text: str  # reference text used for TTS generation
    pcm_int16: np.ndarray  # 16kHz mono int16 PCM audio
    duration: float  # audio duration in seconds


@dataclass
class SessionResult:
    eos_latency: float      # time from end-of-speech to final transcript (seconds)
    audio_duration: float   # duration of the audio clip (seconds)
    transcript: str         # concatenated transcription text
    reference: str          # the reference text this session was given


# ---------------------------------------------------------------------------
# TTS audio generation
# ---------------------------------------------------------------------------
async def generate_audio(tts_url: str, text: str) -> AudioSample:
    """Call Chatterbox TTS and return an AudioSample."""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            tts_url,
            json={
                "text": text,
                "output_format": "wav",
                "voice_mode": "predefined",
                "predefined_voice_id": "Emily.wav",
            },
        )
        response.raise_for_status()

    audio_data, source_sr = sf.read(io.BytesIO(response.content), dtype="float32")

    # Convert stereo to mono if needed
    if audio_data.ndim == 2:
        audio_data = audio_data.mean(axis=1)

    # Resample to 16 kHz if the TTS returns a different rate
    if source_sr != SAMPLE_RATE:
        original_length = len(audio_data)
        target_length = int(original_length * SAMPLE_RATE / source_sr)
        x_original = np.linspace(0, 1, original_length, endpoint=False)
        x_target = np.linspace(0, 1, target_length, endpoint=False)
        audio_data = np.interp(x_target, x_original, audio_data)

    # float32 [-1, 1] -> int16
    pcm_int16 = np.clip(audio_data * 32768, -32768, 32767).astype(np.int16)
    duration = len(pcm_int16) / SAMPLE_RATE
    return AudioSample(text=text, pcm_int16=pcm_int16, duration=duration)


CACHE_DIR = Path(".benchmark_cache")


def _cache_key(text: str, tts_url: str) -> str:
    """Stable cache key from text content and TTS endpoint."""
    h = hashlib.sha256(f"{tts_url}|{text}".encode()).hexdigest()[:12]
    return h


def _load_cached(text: str, tts_url: str) -> AudioSample | None:
    """Load a cached audio sample if it exists."""
    key = _cache_key(text, tts_url)
    path = CACHE_DIR / f"{key}.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
        pcm_int16 = data["pcm_int16"]
        duration = len(pcm_int16) / SAMPLE_RATE
        return AudioSample(text=text, pcm_int16=pcm_int16, duration=duration)
    except Exception:
        return None


def _save_cached(sample: AudioSample, tts_url: str) -> None:
    """Save an audio sample to the cache."""
    CACHE_DIR.mkdir(exist_ok=True)
    key = _cache_key(sample.text, tts_url)
    np.savez_compressed(CACHE_DIR / f"{key}.npz", pcm_int16=sample.pcm_int16)


async def generate_audio_pool(
    tts_url: str, count: int, max_parallel: int = 5,
) -> list[AudioSample]:
    """Generate *count* unique audio samples, using cache and parallel TTS."""
    texts = [TEXT_POOL[i % len(TEXT_POOL)] for i in range(count)]

    # Check cache first
    samples: list[AudioSample | None] = [None] * count
    cached = 0
    for i, text in enumerate(texts):
        sample = _load_cached(text, tts_url)
        if sample is not None:
            samples[i] = sample
            cached += 1

    if cached == count:
        print(f"  Loaded all {count} samples from cache.")
        return samples  # type: ignore[return-value]

    if cached > 0:
        print(f"  Loaded {cached}/{count} from cache, generating {count - cached}...")
    else:
        print(f"  Generating {count} audio samples ({max_parallel} parallel)...")

    # Generate missing samples in parallel with a semaphore
    sem = asyncio.Semaphore(max_parallel)
    progress = {"done": cached, "total": count}

    async def _gen(idx: int, text: str) -> None:
        async with sem:
            sample = await generate_audio(tts_url, text)
            _save_cached(sample, tts_url)
            samples[idx] = sample
            progress["done"] += 1
            print(f"  [{progress['done']}/{progress['total']}] "
                  f"{sample.duration:.1f}s audio generated", flush=True)

    tasks = []
    for i, text in enumerate(texts):
        if samples[i] is None:
            tasks.append(_gen(i, text))
    await asyncio.gather(*tasks)

    return samples  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# WebSocket streaming client
# ---------------------------------------------------------------------------
async def run_session(
    stt_url: str,
    sample: AudioSample,
    timeout: float,
    session_id: int,
) -> SessionResult:
    """Stream audio over the STT websocket and collect transcription results."""

    pcm_int16 = sample.pcm_int16

    # Pre-split audio into frames (raw int16 bytes)
    total_samples = len(pcm_int16)
    frames: list[bytes] = []
    for offset in range(0, total_samples, FRAME_SAMPLES):
        frame = pcm_int16[offset : offset + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        frames.append(frame.tobytes())

    silence_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
    texts: list[str] = []
    last_text_time: float | None = None
    sending_done = asyncio.Event()

    async with websockets.connect(stt_url, ping_interval=None, ping_timeout=None) as ws:

        async def receiver():
            nonlocal last_text_time
            try:
                while True:
                    if not sending_done.is_set():
                        wait_seconds = timeout
                    elif last_text_time is not None:
                        wait_seconds = 5.0
                    else:
                        wait_seconds = timeout
                    raw = await asyncio.wait_for(ws.recv(), timeout=wait_seconds)
                    msg = json.loads(raw)

                    if "text" in msg and msg["text"].strip():
                        last_text_time = time.monotonic()
                        texts.append(msg["text"])
            except asyncio.TimeoutError:
                pass
            except websockets.exceptions.ConnectionClosed:
                pass

        recv_task = asyncio.create_task(receiver())

        # Send audio frames at realtime pace
        for frame in frames:
            await ws.send(frame)
            await asyncio.sleep(FRAME_DURATION)

        # Record EOS time right after last audio frame (before silence padding)
        eos_time = time.monotonic()

        # Send trailing silence to flush VAD
        for _ in range(SILENCE_PADDING_FRAMES):
            await ws.send(silence_frame)
            await asyncio.sleep(FRAME_DURATION)

        sending_done.set()
        await recv_task

    eos_latency = (last_text_time - eos_time) if last_text_time is not None else timeout
    transcript = " ".join(texts)

    return SessionResult(
        eos_latency=eos_latency,
        audio_duration=sample.duration,
        transcript=transcript,
        reference=sample.text,
    )


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------
async def run_benchmark(
    stt_url: str,
    samples: list[AudioSample],
    concurrency: int,
    timeout: float,
) -> list[SessionResult]:
    """Run *concurrency* parallel STT sessions, each with a unique audio sample."""
    print(f"\n  Concurrency={concurrency}...", end=" ", flush=True)

    tasks = [
        run_session(stt_url, samples[i % len(samples)], timeout, session_id=i)
        for i in range(concurrency)
    ]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    valid: list[SessionResult] = []
    errors = 0
    for outcome in outcomes:
        if isinstance(outcome, BaseException):
            errors += 1
            print(f"[session error: {outcome}]", file=sys.stderr)
        else:
            valid.append(outcome)

    print(f"done ({len(valid)} ok, {errors} errors)")
    return valid


# ---------------------------------------------------------------------------
# Metrics calculation and reporting
# ---------------------------------------------------------------------------
def percentile(data: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile from a sorted copy of *data*."""
    sorted_data = sorted(data)
    k = (pct / 100) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def normalize_text(text: str) -> str:
    """Normalize text for WER: lowercase, collapse whitespace, strip punctuation."""
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # remove punctuation
    text = re.sub(r"\s+", " ", text).strip()  # collapse whitespace
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    """Compute order-independent WER: match hypothesis sentences to reference
    sentences by best WER, then compute overall WER on the reordered text.
    This handles batched inference returning results out of order."""
    try:
        import jiwer
        ref_norm = normalize_text(reference)
        hyp_norm = normalize_text(hypothesis)
        if not ref_norm:
            return -1.0

        # Split into sentences (reference uses ". " as delimiter after normalization)
        ref_sentences = [s.strip() for s in reference.split(".") if s.strip()]
        hyp_sentences = [s.strip() for s in hypothesis.split(".") if s.strip()]

        # If sentence counts match, reorder hypothesis to best-match reference
        if len(ref_sentences) > 1 and len(hyp_sentences) == len(ref_sentences):
            ref_norms = [normalize_text(s) for s in ref_sentences]
            hyp_norms = [normalize_text(s) for s in hyp_sentences]
            # Greedy matching: for each ref sentence, find closest hyp sentence
            used = set()
            ordered_hyp = []
            for rn in ref_norms:
                best_idx, best_wer = 0, float("inf")
                for j, hn in enumerate(hyp_norms):
                    if j in used:
                        continue
                    w = jiwer.wer(rn, hn) if rn and hn else 1.0
                    if w < best_wer:
                        best_wer = w
                        best_idx = j
                used.add(best_idx)
                ordered_hyp.append(hyp_norms[best_idx])
            hyp_norm = " ".join(ordered_hyp)
            ref_norm = " ".join(ref_norms)

        return jiwer.wer(ref_norm, hyp_norm)
    except Exception:
        return -1.0


REPORT_HEADER = (
    f"{'Conc':>4}  "
    f"{'EOS p50':>8} {'EOS p95':>8} {'EOS mean':>9}  "
    f"{'Audio Dur':>9}  {'WER%':>6}  "
    f"{'GPU%p95':>7} {'GPU%mean':>8}  "
    f"{'VRAM peak':>9} {'VRAM mean':>9}"
)


def compute_entry_row(entry: dict) -> dict | None:
    """Compute a single result row dict from a benchmark entry."""
    sessions: list[SessionResult] = entry["sessions"]
    gpu_stats: dict = entry.get("gpu_stats", {})
    if not sessions:
        return None

    eos_lats = [s.eos_latency for s in sessions]
    audio_dur = statistics.mean(s.audio_duration for s in sessions)

    eos_p50 = percentile(eos_lats, 50)
    eos_p95 = percentile(eos_lats, 95)
    eos_mean = statistics.mean(eos_lats)

    wers = [compute_wer(s.reference, s.transcript) for s in sessions]
    valid_wers = [w for w in wers if w >= 0]
    wer = statistics.mean(valid_wers) if valid_wers else -1.0
    wer_pct = wer * 100 if wer >= 0 else -1.0

    return {
        "concurrency": entry["concurrency"],
        "eos_p50": round(eos_p50, 4),
        "eos_p95": round(eos_p95, 4),
        "eos_mean": round(eos_mean, 4),
        "audio_duration_s": round(audio_dur, 2),
        "wer_pct": round(wer_pct, 2),
        "gpu_util_p95": gpu_stats.get("gpu_util_p95"),
        "gpu_util_mean": gpu_stats.get("gpu_util_mean"),
        "vram_peak_mib": gpu_stats.get("vram_peak_mib"),
        "vram_mean_mib": gpu_stats.get("vram_mean_mib"),
    }


def format_row(row: dict) -> str:
    """Format a single result row as a table line."""
    gpu_util_p95 = row["gpu_util_p95"]
    gpu_util_mean = row["gpu_util_mean"]
    vram_peak = row["vram_peak_mib"]
    vram_mean = row["vram_mean_mib"]
    wer_pct = row["wer_pct"]

    gpu_p95_str = f"{gpu_util_p95:.0f}%" if gpu_util_p95 is not None else "n/a"
    gpu_mean_str = f"{gpu_util_mean:.0f}%" if gpu_util_mean is not None else "n/a"
    vram_peak_str = f"{vram_peak} MiB" if vram_peak is not None else "n/a"
    vram_mean_str = f"{vram_mean:.0f} MiB" if vram_mean is not None else "n/a"
    wer_str = f"{wer_pct:.2f}%" if wer_pct >= 0 else "n/a"

    return (
        f"{row['concurrency']:>4}  "
        f"{row['eos_p50']:>8.4f} {row['eos_p95']:>8.4f} {row['eos_mean']:>9.4f}  "
        f"{row['audio_duration_s']:>8.2f}s  {wer_str:>6}  "
        f"{gpu_p95_str:>7} {gpu_mean_str:>8}  "
        f"{vram_peak_str:>9} {vram_mean_str:>9}"
    )


def print_report(all_results: list[dict]) -> list[dict]:
    """Print a formatted results table and return row dicts for CSV output."""
    print("\n" + "=" * len(REPORT_HEADER))
    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))

    rows: list[dict] = []
    for entry in all_results:
        row = compute_entry_row(entry)
        if row is None:
            continue
        rows.append(row)
        print(format_row(row))

    print("=" * len(REPORT_HEADER))
    return rows


def write_csv(rows: list[dict], output_path: str) -> None:
    """Write benchmark row dicts to a CSV file."""
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(output_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {output_path}")


# ---------------------------------------------------------------------------
# Probe / diagnostic
# ---------------------------------------------------------------------------
async def probe_session(stt_url: str, sample: AudioSample, timeout: float):
    """Single verbose session for diagnosing connectivity issues."""
    pcm_int16 = sample.pcm_int16
    frames: list[bytes] = []
    for offset in range(0, len(pcm_int16), FRAME_SAMPLES):
        frame = pcm_int16[offset : offset + FRAME_SAMPLES]
        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(frame, (0, FRAME_SAMPLES - len(frame)))
        frames.append(frame.tobytes())

    silence_frame = np.zeros(FRAME_SAMPLES, dtype=np.int16).tobytes()
    print(f"  Reference: {sample.text[:80]}...")
    print(f"  Frames to send: {len(frames)} audio + {SILENCE_PADDING_FRAMES} silence")
    print(f"  Frame size: {FRAME_SAMPLES} samples ({FRAME_DURATION*1000:.0f}ms)")
    print(f"  Connecting to {stt_url} ...")

    got_text = False

    async with websockets.connect(stt_url, ping_interval=None, ping_timeout=None) as ws:
        print("  Connected. Streaming at realtime pace...")
        start = time.monotonic()
        sending_done = asyncio.Event()
        msg_count = 0

        async def receiver():
            nonlocal msg_count, got_text
            try:
                while True:
                    # Full timeout while still sending; after sending, 5s after last text
                    if not sending_done.is_set():
                        wait = timeout
                    elif got_text:
                        wait = 5.0
                    else:
                        wait = timeout
                    raw = await asyncio.wait_for(ws.recv(), timeout=wait)
                    msg = json.loads(raw)
                    elapsed = time.monotonic() - start
                    msg_count += 1
                    kind = "TEXT" if "text" in msg else "status"
                    if "text" in msg:
                        got_text = True
                    print(f"  [{elapsed:7.3f}s] #{msg_count} [{kind}]: {msg}")
            except asyncio.TimeoutError:
                elapsed = time.monotonic() - start
                print(f"  [{elapsed:7.3f}s] receiver timeout ({msg_count} msgs)")
            except websockets.exceptions.ConnectionClosed as e:
                print(f"  Connection closed: {e}")

        recv_task = asyncio.create_task(receiver())

        for frame in frames:
            await ws.send(frame)
            await asyncio.sleep(FRAME_DURATION)

        send_elapsed = time.monotonic() - start
        print(f"  [{send_elapsed:7.3f}s] Audio sent. Sending silence to flush VAD...")

        for _ in range(SILENCE_PADDING_FRAMES):
            await ws.send(silence_frame)
            await asyncio.sleep(FRAME_DURATION)

        silence_elapsed = time.monotonic() - start
        print(f"  [{silence_elapsed:7.3f}s] Silence sent. Waiting for results...")
        sending_done.set()

        await recv_task

    total = time.monotonic() - start
    print(f"\n  Probe complete in {total:.2f}s. Messages received: {msg_count}")
    if not got_text:
        print("  WARNING: No transcription text received.")
        print("  If you see 'status: queued' but no 'text' messages,")
        print("  the STT model may be failing. Check server logs.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
async def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Parakeet STT service")
    parser.add_argument(
        "--stt-url", default="ws://localhost:8000/ws",
        help="WebSocket URL for the STT service",
    )
    parser.add_argument(
        "--tts-url", default="http://localhost:8004/tts",
        help="HTTP URL for the Chatterbox TTS service",
    )
    parser.add_argument(
        "--concurrency", default="1,5,10,20,30,40",
        help="Comma-separated concurrency levels to test",
    )
    parser.add_argument(
        "--output", default="benchmark_results.csv",
        help="Output CSV file path",
    )
    parser.add_argument(
        "--timeout", type=float, default=30.0,
        help="Timeout in seconds for waiting on transcription results",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Run a single diagnostic session with verbose output, then exit",
    )
    parser.add_argument(
        "--gpu-id", type=int, default=None,
        help="GPU device index for monitoring (auto-detected if not set)",
    )
    args = parser.parse_args()

    concurrency_levels = [int(c) for c in args.concurrency.split(",")]
    max_concurrency = max(concurrency_levels)

    # Generate unique audio for each concurrent session
    print(f"Generating {max_concurrency} unique audio samples via TTS...")
    samples = await generate_audio_pool(args.tts_url, max_concurrency)
    avg_duration = statistics.mean(s.duration for s in samples)

    print(f"\nAudio pool ready: {len(samples)} samples, "
          f"avg duration {avg_duration:.2f}s")
    print(f"Concurrency levels: {concurrency_levels}")
    print(f"Output: {args.output}")

    # Probe mode: single diagnostic session with verbose output
    if args.probe:
        print("\n--- PROBE: single session, verbose ---")
        await probe_session(args.stt_url, samples[0], args.timeout)
        return

    # Start GPU monitoring
    gpu_id = args.gpu_id if args.gpu_id is not None else detect_gpu_id()
    gpu_monitor = GpuMonitor(gpu_id)
    await gpu_monitor.start()

    # Print table header once
    print("\n" + "=" * len(REPORT_HEADER))
    print(REPORT_HEADER)
    print("-" * len(REPORT_HEADER))

    all_results: list[dict] = []
    rows: list[dict] = []
    for i, concurrency in enumerate(concurrency_levels):
        gpu_mark = gpu_monitor.mark()
        sessions = await run_benchmark(
            stt_url=args.stt_url,
            samples=samples,
            concurrency=concurrency,
            timeout=args.timeout,
        )
        gpu_stats = gpu_monitor.slice_stats(gpu_mark)
        entry = {
            "concurrency": concurrency,
            "sessions": sessions,
            "gpu_stats": gpu_stats,
        }
        all_results.append(entry)

        # Print this level's results immediately
        row = compute_entry_row(entry)
        if row is not None:
            rows.append(row)
            print(format_row(row), flush=True)

        if i < len(concurrency_levels) - 1:
            await asyncio.sleep(2.0)  # brief settle between levels

    print("=" * len(REPORT_HEADER))

    await gpu_monitor.stop()

    write_csv(rows, args.output)


if __name__ == "__main__":
    asyncio.run(main())
