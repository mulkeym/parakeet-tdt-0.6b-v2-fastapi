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
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import numpy as np
import soundfile as sf
import websockets
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

logger = logging.getLogger("benchmark")
console = Console()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SAMPLE_RATE = 16_000
FRAME_SAMPLES = 1600  # 100ms chunks at 16kHz (matches voice agent)
FRAME_DURATION = FRAME_SAMPLES / SAMPLE_RATE  # 0.1 seconds
SILENCE_PADDING_FRAMES = int(1.0 / FRAME_DURATION)  # ~10 frames of trailing silence
POST_SEND_WAIT = 30.0  # max seconds to wait for results after all audio is sent

# Pool of unique paragraphs -- each concurrent session gets a different one.
# This catches cross-session result leaking or stale-data memory bugs.
# Lengths vary from ~5s (short) to ~15s (long) for realistic mixed workloads.
TEXT_POOL = [
    # --- Short (~5-6 s, one to two short sentences) ---
    "The quick brown fox jumps over the lazy dog near the riverbank on a warm afternoon.",
    "She sells seashells by the seashore every Sunday morning before the crowds arrive.",
    "A gentle breeze carried the scent of wildflowers across the open meadow at sunset.",
    "Children played in the park while their parents watched quietly from the wooden bench.",
    "The bakery on the corner filled the entire street with the wonderful aroma of fresh bread.",
    "Autumn leaves drifted slowly from the tall maple trees lining the quiet boulevard.",
    "Construction workers began pouring the new concrete foundation early on Monday morning.",
    "The weather forecast predicts sunny skies and mild temperatures throughout the day today.",
    "Marine biologists documented several previously unknown species of deep sea creatures.",
    "The submarine descended quietly into the dark and freezing waters of the continental shelf.",
    "Violins and cellos blended together beautifully in a hauntingly emotional melody.",
    "The architect carefully designed a building that harmonized perfectly with the surrounding landscape.",
    "Students gathered eagerly around the laboratory table to observe the dramatic chemical reaction.",
    # --- Medium (~8-10 s, two sentences) ---
    (
        "A journey of a thousand miles begins with a single step forward. "
        "The mountain trail wound through ancient pine forests and crystal clear streams."
    ),
    (
        "Musicians practiced their instruments diligently before the evening concert. "
        "Scientists discovered a new species of butterfly in the tropical rainforest."
    ),
    (
        "The old lighthouse keeper watched the ships sail through the foggy harbor. "
        "Fishermen returned at dawn with their nets heavy from the overnight catch."
    ),
    (
        "The professor explained the complex equation with remarkable clarity and patience. "
        "Her students took careful notes and asked thoughtful questions after the lecture."
    ),
    (
        "The orchestra conductor raised her baton and the symphony hall fell silent. "
        "Every musician held their breath waiting for the downbeat to begin the overture."
    ),
    (
        "Heavy rain pounded against the windows of the old farmhouse all through the night. "
        "By morning the creek had risen three feet and flooded the lower pasture."
    ),
    (
        "The detective examined the crime scene carefully looking for any overlooked evidence. "
        "A single fingerprint on the doorframe would eventually crack the entire case."
    ),
    (
        "The astronaut gazed out the window of the space station at the Earth below. "
        "Clouds swirled over the Pacific Ocean creating mesmerizing spiral patterns."
    ),
    (
        "The chef prepared an elaborate seven course meal for the visiting dignitaries. "
        "Each dish showcased locally sourced ingredients and traditional cooking techniques."
    ),
    (
        "Volunteers spent the entire weekend cleaning up the riverbank after the flood. "
        "They collected over two thousand pounds of debris and recycled what they could."
    ),
    (
        "The librarian organized a summer reading program for children of all ages. "
        "Hundreds of families signed up within the first week of the announcement."
    ),
    (
        "The pilot navigated through turbulent weather over the Rocky Mountains. "
        "Passengers gripped their armrests as the plane bounced through heavy crosswinds."
    ),
    (
        "The marathon runner pushed through the final miles despite the blistering heat. "
        "Spectators along the course offered water and encouragement to every participant."
    ),
    # --- Long (~12-15 s, three to four sentences) ---
    (
        "The research team spent six months in the Arctic studying polar bear migration patterns. "
        "They deployed satellite tracking collars on twelve bears across three different regions. "
        "The data revealed surprising shifts in feeding behavior linked to sea ice reduction."
    ),
    (
        "The city council approved a new public transit plan that would add fifteen bus routes. "
        "Construction of the first dedicated bus lane is scheduled to begin next spring. "
        "Residents in underserved neighborhoods expressed strong support for the expansion."
    ),
    (
        "The photographer waited three hours in the freezing rain for the perfect shot. "
        "When the sun finally broke through the clouds it illuminated the valley below. "
        "The resulting image won first place in the national landscape photography competition."
    ),
    (
        "The software engineer debugged a critical issue that had been crashing the server for weeks. "
        "After tracing the problem through thousands of lines of code she found a subtle race condition. "
        "The fix was only two lines long but it required deep understanding of the entire system."
    ),
    (
        "The ancient Roman aqueduct stretched for miles across the dry Spanish countryside. "
        "Engineers built it over two thousand years ago using precisely cut granite blocks. "
        "Remarkably the structure still stands today as a testament to Roman engineering skill."
    ),
    (
        "The veterinarian treated a young elephant that had been injured by poachers in the reserve. "
        "After weeks of careful rehabilitation the animal was strong enough to rejoin its herd. "
        "Park rangers monitored its recovery using GPS trackers attached to a lightweight collar."
    ),
    (
        "The film director spent three years developing a screenplay based on a true story. "
        "She interviewed dozens of people who had lived through the events depicted in the film. "
        "Critics praised the movie for its authenticity and emotional depth at the premiere."
    ),
    (
        "The farmer rotated crops across four different fields to maintain healthy soil throughout the year. "
        "Corn and soybeans alternated with cover crops of clover and winter rye in a careful sequence. "
        "This sustainable approach doubled the farm's yield within just five growing seasons."
    ),
    (
        "The museum curator assembled a traveling exhibition featuring impressionist paintings from private collections. "
        "Works by Monet Renoir and Degas were displayed alongside lesser known artists of the same period. "
        "The exhibition drew record attendance in every city during its eighteen month international tour."
    ),
    (
        "The emergency response team arrived at the earthquake site within four hours of the initial tremor. "
        "They set up a field hospital and began treating survivors pulled from the collapsed buildings. "
        "International aid organizations coordinated supply drops of food water and medical equipment."
    ),
    (
        "The deep sea exploration vessel descended to nearly four thousand meters below the ocean surface. "
        "Scientists aboard discovered hydrothermal vents surrounded by ecosystems never before documented. "
        "Tube worms and eyeless shrimp thrived in the superheated mineral rich water near the vents."
    ),
    (
        "The school principal introduced a new program that paired older students with younger reading buddies. "
        "Every Tuesday and Thursday afternoon the pairs met in the library for thirty minute sessions. "
        "Reading scores among the younger students improved by twenty percent over the course of the semester."
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
    session_id: int = 0
    cancelled: bool = False # True if the receiver was force-cancelled before completion
    log_lines: list[str] = field(default_factory=list, repr=False)


@dataclass
class SessionState:
    """Mutable shared state updated by run_session() in real-time for live display."""
    session_id: int
    status: str = "pending"       # pending/connecting/streaming/flushing/waiting/done/error
    frames_sent: int = 0
    total_frames: int = 0
    msgs_received: int = 0
    texts_received: int = 0
    elapsed: float = 0.0
    transcript_preview: str = ""
    eos_latency: float | None = None
    error: str = ""


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
    verbose: bool = False,
    post_wait: float = POST_SEND_WAIT,
    state: SessionState | None = None,
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
    session_start = time.monotonic()
    log: list[str] = []  # buffered verbose output, printed sorted after all sessions

    tag = f"[S{session_id:>02d}]"

    if state:
        state.total_frames = len(frames)
        state.status = "connecting"

    if verbose:
        log.append(f"  {tag} connecting to {stt_url} ...")

    try:
        async with websockets.connect(stt_url, ping_interval=None, ping_timeout=None) as ws:

            if verbose:
                elapsed = time.monotonic() - session_start
                log.append(f"  {tag} connected ({elapsed:.3f}s). "
                           f"Sending {len(frames)} audio + {SILENCE_PADDING_FRAMES} silence frames")

            if state:
                state.status = "streaming"
                state.elapsed = time.monotonic() - session_start

            sending_done = asyncio.Event()

            async def receiver():
                nonlocal last_text_time
                msg_count = 0
                queued_count = 0
                results_received = 0

                def _all_results_in() -> bool:
                    return (sending_done.is_set()
                            and results_received >= queued_count > 0)

                try:
                    while True:
                        # Race: either a ws message arrives or sending completes.
                        # If all results arrived before sending_done, we need to
                        # re-check once the event fires rather than blocking on
                        # ws.recv() for the full timeout.
                        recv_fut = asyncio.ensure_future(ws.recv())
                        done_fut = asyncio.ensure_future(sending_done.wait())
                        done, pending = await asyncio.wait(
                            [recv_fut, done_fut],
                            timeout=timeout,
                            return_when=asyncio.FIRST_COMPLETED,
                        )
                        for fut in pending:
                            fut.cancel()

                        if not done:
                            break  # timeout — same as old TimeoutError exit

                        if recv_fut in done:
                            raw = recv_fut.result()
                            msg = json.loads(raw)
                            msg_count += 1

                            if state:
                                state.msgs_received = msg_count
                                state.elapsed = time.monotonic() - session_start

                            if verbose:
                                elapsed = time.monotonic() - session_start
                                kind = "TEXT" if "text" in msg else "status"
                                preview = msg.get("text", msg.get("status", ""))
                                if isinstance(preview, str) and len(preview) > 80:
                                    preview = preview[:80] + "..."
                                log.append(f"  {tag} RECV #{msg_count} [{kind}] "
                                           f"@ {elapsed:.3f}s: {preview}")

                            if msg.get("status") == "queued":
                                queued_count += 1

                            if "text" in msg:
                                results_received += 1
                                if msg["text"].strip():
                                    last_text_time = time.monotonic()
                                    texts.append(msg["text"])
                                    if state:
                                        state.texts_received = len(texts)
                                        state.transcript_preview = msg["text"].strip()[:60]

                        # Check after processing a message OR after sending_done fires
                        if _all_results_in():
                            if verbose:
                                elapsed = time.monotonic() - session_start
                                log.append(
                                    f"  {tag} all {results_received} result(s) "
                                    f"received @ {elapsed:.3f}s — exiting early")
                            break
                except websockets.exceptions.ConnectionClosed:
                    if verbose:
                        log.append(f"  {tag} connection closed ({msg_count} msgs)")
                except asyncio.CancelledError:
                    pass  # normal exit — post-wait cap reached

            recv_task = asyncio.create_task(receiver())

            # Stream frames paced to wall-clock time (like a real microphone).
            t_stream_start = time.monotonic()
            for i, frame in enumerate(frames):
                await ws.send(frame)
                if state:
                    state.frames_sent = i + 1
                    state.elapsed = time.monotonic() - session_start
                target = t_stream_start + (i + 1) * FRAME_DURATION
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            # Record EOS time right after last audio frame (before silence padding)
            eos_time = time.monotonic()

            if state:
                state.status = "flushing"
                state.elapsed = time.monotonic() - session_start

            if verbose:
                elapsed = eos_time - session_start
                log.append(f"  {tag} audio sent ({len(frames)} frames, {elapsed:.3f}s). "
                           f"Flushing VAD with silence...")

            # Send trailing silence to flush VAD, continuing wall-clock pacing
            for j in range(SILENCE_PADDING_FRAMES):
                await ws.send(silence_frame)
                target = t_stream_start + (len(frames) + j + 1) * FRAME_DURATION
                delay = target - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)

            sending_done.set()

            if state:
                state.status = "waiting"
                state.elapsed = time.monotonic() - session_start

            if verbose:
                elapsed = time.monotonic() - session_start
                log.append(f"  {tag} all frames sent ({elapsed:.3f}s). "
                           f"Waiting for results...")

            # Bound how long we wait after sending finishes.
            cancelled = False
            try:
                await asyncio.wait_for(recv_task, timeout=post_wait)
            except asyncio.TimeoutError:
                cancelled = True
                recv_task.cancel()
                try:
                    await recv_task
                except asyncio.CancelledError:
                    pass

    except Exception as exc:
        if state:
            state.status = "error"
            state.error = str(exc)[:80]
            state.elapsed = time.monotonic() - session_start
        raise

    eos_latency = (last_text_time - eos_time) if last_text_time is not None else timeout
    transcript = " ".join(texts)
    failed = cancelled and not texts  # only a failure if we got no transcription

    if state:
        state.status = "error" if failed else "done"
        state.eos_latency = eos_latency
        state.elapsed = time.monotonic() - session_start

    if verbose:
        elapsed = time.monotonic() - session_start
        status = "FAIL (no transcription)" if failed else "OK"
        log.append(f"  {tag} {status} ({elapsed:.3f}s) eos_latency={eos_latency:.3f}s "
                   f"texts={len(texts)} transcript={transcript[:60]}...")

    return SessionResult(
        eos_latency=eos_latency,
        audio_duration=sample.duration,
        transcript=transcript,
        reference=sample.text,
        session_id=session_id,
        cancelled=failed,
        log_lines=log,
    )


STATUS_STYLES = {
    "pending": "dim",
    "connecting": "cyan",
    "streaming": "yellow",
    "flushing": "yellow",
    "waiting": "blue",
    "done": "green",
    "error": "red bold",
}


def render_live_panel(
    states: list[SessionState],
    concurrency: int,
    title: str = "",
) -> Table:
    """Build a Rich Table showing live session states."""
    done_count = sum(1 for s in states if s.status in ("done", "error"))
    header = f"Concurrency: {concurrency}  [{done_count}/{len(states)} done]"
    if title:
        header = f"{title}  {header}"

    table = Table(title=header, expand=False, padding=(0, 1))
    table.add_column("Session", style="bold", width=7)
    table.add_column("Status", width=12)
    table.add_column("Progress", width=9)
    table.add_column("Msgs", justify="right", width=5)
    table.add_column("Texts", justify="right", width=5)
    table.add_column("Elapsed", justify="right", width=8)
    table.add_column("Preview", max_width=40, no_wrap=True)

    for s in states:
        style = STATUS_STYLES.get(s.status, "")
        status_text = Text(s.status, style=style)
        progress = f"{s.frames_sent}/{s.total_frames}" if s.total_frames else "-"
        elapsed_str = f"{s.elapsed:.1f}s" if s.elapsed > 0 else "-"
        preview = s.error if s.status == "error" else s.transcript_preview
        table.add_row(
            f"S{s.session_id:02d}",
            status_text,
            progress,
            str(s.msgs_received),
            str(s.texts_received),
            elapsed_str,
            preview or "",
        )

    return table


# ---------------------------------------------------------------------------
# Benchmark orchestrator
# ---------------------------------------------------------------------------
async def run_benchmark(
    stt_url: str,
    samples: list[AudioSample],
    concurrency: int,
    timeout: float,
    verbose: bool = False,
    post_wait: float = POST_SEND_WAIT,
) -> list[SessionResult]:
    """Run *concurrency* parallel STT sessions, each with a unique audio sample."""
    states = [SessionState(session_id=i) for i in range(concurrency)]

    tasks = [
        run_session(stt_url, samples[i % len(samples)], timeout, session_id=i,
                    verbose=verbose, post_wait=post_wait, state=states[i])
        for i in range(concurrency)
    ]

    async def _refresh(live: Live):
        """Background task: refresh the live display every 0.5s."""
        try:
            while True:
                live.update(render_live_panel(states, concurrency))
                await asyncio.sleep(0.5)
        except asyncio.CancelledError:
            pass

    with Live(render_live_panel(states, concurrency), console=console,
              refresh_per_second=4, transient=True) as live:
        refresh_task = asyncio.create_task(_refresh(live))
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        refresh_task.cancel()
        try:
            await refresh_task
        except asyncio.CancelledError:
            pass
        # Final snapshot before the live display disappears
        live.update(render_live_panel(states, concurrency))

    # Print a static final status line after the live display clears
    valid: list[SessionResult] = []
    errors = 0
    cancelled = 0
    for i, outcome in enumerate(outcomes):
        if isinstance(outcome, BaseException):
            errors += 1
            states[i].status = "error"
            states[i].error = str(outcome)[:80]
            console.print(f"  [red][session {i} error: {outcome}][/red]", highlight=False)
        else:
            valid.append(outcome)
            if outcome.cancelled:
                cancelled += 1

    # Print the final panel (non-transient) so it persists in scrollback
    console.print(render_live_panel(states, concurrency))

    parts = [f"{len(valid) - cancelled} ok"]
    if cancelled:
        parts.append(f"{cancelled} cancelled")
    if errors:
        parts.append(f"{errors} errors")
    console.print(f"  Done: {', '.join(parts)}")
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
    except Exception as exc:
        logger.warning("WER computation failed: %s (ref=%r, hyp=%r)", exc,
                       reference[:80], hypothesis[:80])
        return -1.0


REPORT_HEADER = (
    f"{'Conc':>4}  "
    f"{'EOS p50':>8} {'EOS p95':>8} {'EOS mean':>9}  "
    f"{'Audio Dur':>9}  {'WER%':>6}  {'Fail':>6}  "
    f"{'GPU%p95':>7} {'GPU%mean':>8}  "
    f"{'VRAM peak':>9} {'VRAM mean':>9}"
)


def compute_entry_row(entry: dict) -> dict | None:
    """Compute a single result row dict from a benchmark entry."""
    sessions: list[SessionResult] = entry["sessions"]
    gpu_stats: dict = entry.get("gpu_stats", {})
    if not sessions:
        return None

    total = len(sessions)
    cancelled = sum(1 for s in sessions if s.cancelled)
    good = [s for s in sessions if not s.cancelled]

    # Use only non-cancelled sessions for latency/WER metrics
    if good:
        eos_lats = [s.eos_latency for s in good]
        audio_dur = statistics.mean(s.audio_duration for s in good)
        eos_p50 = percentile(eos_lats, 50)
        eos_p95 = percentile(eos_lats, 95)
        eos_mean = statistics.mean(eos_lats)
        wers = [compute_wer(s.reference, s.transcript) for s in good]
        valid_wers = [w for w in wers if w >= 0]
        wer = statistics.mean(valid_wers) if valid_wers else -1.0
    else:
        audio_dur = statistics.mean(s.audio_duration for s in sessions)
        eos_p50 = eos_p95 = eos_mean = -1.0
        wer = -1.0

    wer_pct = wer * 100 if wer >= 0 else -1.0

    return {
        "concurrency": entry["concurrency"],
        "eos_p50": round(eos_p50, 4),
        "eos_p95": round(eos_p95, 4),
        "eos_mean": round(eos_mean, 4),
        "audio_duration_s": round(audio_dur, 2),
        "wer_pct": round(wer_pct, 2),
        "cancelled": cancelled,
        "total": total,
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
    cancelled = row.get("cancelled", 0)
    total = row.get("total", row["concurrency"])

    gpu_p95_str = f"{gpu_util_p95:.0f}%" if gpu_util_p95 is not None else "n/a"
    gpu_mean_str = f"{gpu_util_mean:.0f}%" if gpu_util_mean is not None else "n/a"
    vram_peak_str = f"{vram_peak} MiB" if vram_peak is not None else "n/a"
    vram_mean_str = f"{vram_mean:.0f} MiB" if vram_mean is not None else "n/a"
    wer_str = f"{wer_pct:.2f}%" if wer_pct >= 0 else "n/a"
    fail_str = f"{cancelled}/{total}" if cancelled else f"0/{total}"

    return (
        f"{row['concurrency']:>4}  "
        f"{row['eos_p50']:>8.4f} {row['eos_p95']:>8.4f} {row['eos_mean']:>9.4f}  "
        f"{row['audio_duration_s']:>8.2f}s  {wer_str:>6}  {fail_str:>6}  "
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
async def probe_session(stt_url: str, sample: AudioSample, timeout: float,
                        post_wait: float = POST_SEND_WAIT):
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
        print("  Connected. Streaming at realtime pace (wall-clock paced)...")
        start = time.monotonic()
        msg_count = 0
        sending_done = asyncio.Event()

        async def receiver():
            nonlocal msg_count, got_text
            queued_count = 0
            results_received = 0

            def _all_results_in() -> bool:
                return (sending_done.is_set()
                        and results_received >= queued_count > 0)

            try:
                while True:
                    recv_fut = asyncio.ensure_future(ws.recv())
                    done_fut = asyncio.ensure_future(sending_done.wait())
                    done, pending = await asyncio.wait(
                        [recv_fut, done_fut],
                        timeout=timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for fut in pending:
                        fut.cancel()

                    if not done:
                        elapsed = time.monotonic() - start
                        print(f"  [{elapsed:7.3f}s] receiver timeout ({msg_count} msgs)")
                        break

                    if recv_fut in done:
                        raw = recv_fut.result()
                        msg = json.loads(raw)
                        elapsed = time.monotonic() - start
                        msg_count += 1
                        kind = "TEXT" if "text" in msg else "status"
                        if msg.get("status") == "queued":
                            queued_count += 1
                        if "text" in msg:
                            got_text = True
                            results_received += 1
                        print(f"  [{elapsed:7.3f}s] #{msg_count} [{kind}]: {msg}")

                    if _all_results_in():
                        elapsed = time.monotonic() - start
                        print(f"  [{elapsed:7.3f}s] all {results_received} result(s) "
                              f"received — exiting early")
                        break
            except websockets.exceptions.ConnectionClosed as e:
                print(f"  Connection closed: {e}")
            except asyncio.CancelledError:
                elapsed = time.monotonic() - start
                print(f"  [{elapsed:7.3f}s] receiver cancelled ({msg_count} msgs)")

        recv_task = asyncio.create_task(receiver())

        # Wall-clock paced sending
        t_stream_start = time.monotonic()
        for i, frame in enumerate(frames):
            await ws.send(frame)
            target = t_stream_start + (i + 1) * FRAME_DURATION
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        send_elapsed = time.monotonic() - start
        print(f"  [{send_elapsed:7.3f}s] Audio sent. Sending silence to flush VAD...")

        for j in range(SILENCE_PADDING_FRAMES):
            await ws.send(silence_frame)
            target = t_stream_start + (len(frames) + j + 1) * FRAME_DURATION
            delay = target - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)

        sending_done.set()
        silence_elapsed = time.monotonic() - start
        print(f"  [{silence_elapsed:7.3f}s] Silence sent. "
              f"Waiting for results...")

        try:
            await asyncio.wait_for(recv_task, timeout=post_wait)
        except asyncio.TimeoutError:
            recv_task.cancel()
            try:
                await recv_task
            except asyncio.CancelledError:
                pass

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
    parser.add_argument(
        "--post-wait", type=float, default=POST_SEND_WAIT,
        help=f"Max seconds to wait for results after all audio is sent (default: {POST_SEND_WAIT:.0f})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print per-session send/receive details (frames sent, messages received, timings)",
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
        await probe_session(args.stt_url, samples[0], args.timeout,
                            post_wait=args.post_wait)
        return

    # Start GPU monitoring
    gpu_id = args.gpu_id if args.gpu_id is not None else detect_gpu_id()
    gpu_monitor = GpuMonitor(gpu_id)
    await gpu_monitor.start()

    all_results: list[dict] = []
    for i, concurrency in enumerate(concurrency_levels):
        gpu_mark = gpu_monitor.mark()
        sessions = await run_benchmark(
            stt_url=args.stt_url,
            samples=samples,
            concurrency=concurrency,
            timeout=args.timeout,
            verbose=args.verbose,
            post_wait=args.post_wait,
        )
        gpu_stats = gpu_monitor.slice_stats(gpu_mark)
        entry = {
            "concurrency": concurrency,
            "sessions": sessions,
            "gpu_stats": gpu_stats,
        }
        all_results.append(entry)

        if i < len(concurrency_levels) - 1:
            await asyncio.sleep(2.0)  # brief settle between levels

    await gpu_monitor.stop()

    # Dump verbose session logs before summary table (if -v)
    if args.verbose:
        console.print()
        for entry in all_results:
            for result in sorted(entry["sessions"], key=lambda r: r.session_id):
                for line in result.log_lines:
                    console.print(line, highlight=False)

    # Print clean summary table at the end
    rows = print_report(all_results)
    write_csv(rows, args.output)


if __name__ == "__main__":
    asyncio.run(main())
