# Benchmark Redesign Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Rewrite `benchmark_stt.py` to focus on realtime streaming concurrency with EOS latency, GPU monitoring via `nvidia-smi dmon`, and WER as percentage.

**Architecture:** Single-file rewrite of `benchmark_stt.py`. A `GpuMonitor` class manages the `nvidia-smi dmon` subprocess and sample collection. Sessions measure EOS latency (last silence frame sent to last transcript received). The main loop runs step-wise concurrency levels (1,5,10,20,30,40), taking GPU snapshots per level. Pure helper functions (dmon parsing, stats, WER) are unit-tested in a separate test file.

**Tech Stack:** Python 3.10+, asyncio, websockets, httpx, soundfile, numpy, jiwer, nvidia-smi (CLI)

**Design doc:** `docs/plans/2026-02-24-benchmark-redesign-design.md`

---

### Task 1: Add test file and test GPU dmon output parsing

**Files:**
- Create: `tests/test_benchmark.py`

**Step 1: Write the parsing tests**

```python
"""Tests for benchmark_stt helper functions."""

import pytest


class TestParseDmonLine:
    """Test nvidia-smi dmon output parsing."""

    def test_parses_valid_line(self):
        from benchmark_stt import parse_dmon_line
        # dmon output columns: gpu idx, sm%, mem%, fb_used_MiB
        result = parse_dmon_line("    0   45   32   1842")
        assert result == (45, 1842)

    def test_skips_header_line(self):
        from benchmark_stt import parse_dmon_line
        assert parse_dmon_line("# gpu   sm   mem    fb") is None

    def test_skips_blank_line(self):
        from benchmark_stt import parse_dmon_line
        assert parse_dmon_line("") is None
        assert parse_dmon_line("   ") is None

    def test_skips_dash_separator(self):
        from benchmark_stt import parse_dmon_line
        assert parse_dmon_line("# ---  ---  ---  ---") is None

    def test_handles_high_values(self):
        from benchmark_stt import parse_dmon_line
        result = parse_dmon_line("    0   99   95  24564")
        assert result == (99, 24564)

    def test_handles_zero_values(self):
        from benchmark_stt import parse_dmon_line
        result = parse_dmon_line("    0    0    0     0")
        assert result == (0, 0)
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark.py -v`
Expected: FAIL with `cannot import name 'parse_dmon_line'`

**Step 3: Write `parse_dmon_line` in benchmark_stt.py**

Add this function near the top of `benchmark_stt.py`, after the constants section:

```python
def parse_dmon_line(line: str) -> tuple[int, int] | None:
    """Parse one line of nvidia-smi dmon output.

    Expected columns (with -s um): gpu_idx, sm%, mem%, fb_used_MiB
    Returns (gpu_util_pct, vram_used_mib) or None for headers/blanks.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 4:
        return None
    try:
        sm_pct = int(parts[1])
        fb_mib = int(parts[3])
        return (sm_pct, fb_mib)
    except (ValueError, IndexError):
        return None
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

```bash
git add tests/test_benchmark.py benchmark_stt.py
git commit -m "feat(benchmark): add parse_dmon_line with tests"
```

---

### Task 2: Add GpuMonitor class

**Files:**
- Modify: `benchmark_stt.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write tests for GpuMonitor sample slicing and stats**

Append to `tests/test_benchmark.py`:

```python
class TestGpuStats:
    """Test GPU stats computation from sample lists."""

    def test_compute_gpu_stats_basic(self):
        from benchmark_stt import compute_gpu_stats
        # samples: list of (gpu_util, vram_mib)
        samples = [(40, 1800), (50, 1900), (60, 2000), (80, 2200), (90, 2400)]
        stats = compute_gpu_stats(samples)
        assert stats["gpu_util_mean"] == 64.0
        assert stats["gpu_util_p95"] >= 86.0  # between 80 and 90
        assert stats["vram_peak_mib"] == 2400
        assert stats["vram_mean_mib"] == 2060.0

    def test_compute_gpu_stats_empty(self):
        from benchmark_stt import compute_gpu_stats
        stats = compute_gpu_stats([])
        assert stats["gpu_util_mean"] is None
        assert stats["vram_peak_mib"] is None

    def test_compute_gpu_stats_single(self):
        from benchmark_stt import compute_gpu_stats
        stats = compute_gpu_stats([(75, 3000)])
        assert stats["gpu_util_mean"] == 75.0
        assert stats["gpu_util_p95"] == 75.0
        assert stats["vram_peak_mib"] == 3000
        assert stats["vram_mean_mib"] == 3000.0
```

**Step 2: Run tests to verify they fail**

Run: `pytest tests/test_benchmark.py::TestGpuStats -v`
Expected: FAIL with `cannot import name 'compute_gpu_stats'`

**Step 3: Write `compute_gpu_stats` and `GpuMonitor` class**

Add to `benchmark_stt.py`:

```python
import subprocess
import logging

logger = logging.getLogger("benchmark")


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
    """Find the GPU running a python process via nvidia-smi. Returns device index or 0."""
    try:
        apps = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid", "--format=csv,noheader"],
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

        for line in apps.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2:
                pid_str, gpu_uuid = parts
                # Check if this PID is a python process
                try:
                    cmd = subprocess.run(
                        ["ps", "-p", pid_str, "-o", "comm="],
                        capture_output=True, text=True, timeout=2,
                    )
                    if "python" in cmd.stdout.lower():
                        idx = uuid_to_idx.get(gpu_uuid, 0)
                        logger.info("Detected parakeet on GPU %d (PID %s)", idx, pid_str)
                        return idx
                except Exception:
                    pass
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark.py -v`
Expected: All 9 tests PASS

**Step 5: Commit**

```bash
git add benchmark_stt.py tests/test_benchmark.py
git commit -m "feat(benchmark): add GpuMonitor with nvidia-smi dmon and auto-detect"
```

---

### Task 3: Refactor SessionResult and run_session for EOS latency

**Files:**
- Modify: `benchmark_stt.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write test for EOS latency dataclass**

Append to `tests/test_benchmark.py`:

```python
class TestSessionResult:
    def test_fields_exist(self):
        from benchmark_stt import SessionResult
        r = SessionResult(eos_latency=0.5, audio_duration=6.0, transcript="hello", reference="hello")
        assert r.eos_latency == 0.5
        assert r.audio_duration == 6.0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_benchmark.py::TestSessionResult -v`
Expected: FAIL (SessionResult has old fields)

**Step 3: Rewrite SessionResult and run_session**

Replace the `SessionResult` dataclass in `benchmark_stt.py`:

```python
@dataclass
class SessionResult:
    eos_latency: float      # time from end-of-speech to final transcript (seconds)
    audio_duration: float   # duration of the audio clip (seconds)
    transcript: str         # concatenated transcription text
    reference: str          # the reference text this session was given
```

Replace `run_session` in `benchmark_stt.py`:

```python
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

        # Send trailing silence to flush VAD
        for _ in range(SILENCE_PADDING_FRAMES):
            await ws.send(silence_frame)
            await asyncio.sleep(FRAME_DURATION)

        eos_time = time.monotonic()
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
```

**Step 4: Run tests to verify they pass**

Run: `pytest tests/test_benchmark.py -v`
Expected: All 10 tests PASS

**Step 5: Commit**

```bash
git add benchmark_stt.py tests/test_benchmark.py
git commit -m "feat(benchmark): replace TTFB/total with EOS latency in session"
```

---

### Task 4: Update run_benchmark to remove mode parameter

**Files:**
- Modify: `benchmark_stt.py`

**Step 1: Simplify run_benchmark**

Replace the `run_benchmark` function:

```python
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
```

**Step 2: Commit**

```bash
git add benchmark_stt.py
git commit -m "refactor(benchmark): simplify run_benchmark, remove mode param"
```

---

### Task 5: Rewrite reporting with new metrics and WER as percentage

**Files:**
- Modify: `benchmark_stt.py`
- Modify: `tests/test_benchmark.py`

**Step 1: Write tests for WER percentage formatting**

Append to `tests/test_benchmark.py`:

```python
class TestWerFormatting:
    def test_wer_as_percentage(self):
        from benchmark_stt import compute_wer
        # "hello world" vs "hello world" should be 0%
        wer = compute_wer("hello world", "hello world")
        assert wer >= 0
        assert wer < 0.01  # essentially 0

    def test_wer_returns_negative_on_empty_ref(self):
        from benchmark_stt import compute_wer
        assert compute_wer("", "hello") == -1.0
```

**Step 2: Run tests**

Run: `pytest tests/test_benchmark.py::TestWerFormatting -v`
Expected: PASS (compute_wer already exists, this validates it still works)

**Step 3: Replace print_report and write_csv**

Replace `print_report` in `benchmark_stt.py`:

```python
def print_report(all_results: list[dict]) -> list[dict]:
    """Print a formatted results table and return row dicts for CSV output."""
    header = (
        f"{'Conc':>4}  "
        f"{'EOS p50':>8} {'EOS p95':>8} {'EOS mean':>9}  "
        f"{'Audio Dur':>9}  {'WER%':>6}  "
        f"{'GPU%p95':>7} {'GPU%mean':>8}  "
        f"{'VRAM peak':>9} {'VRAM mean':>9}"
    )
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))

    rows: list[dict] = []
    for entry in all_results:
        sessions: list[SessionResult] = entry["sessions"]
        gpu_stats: dict = entry.get("gpu_stats", {})
        if not sessions:
            continue

        eos_lats = [s.eos_latency for s in sessions]
        audio_dur = statistics.mean(s.audio_duration for s in sessions)

        eos_p50 = percentile(eos_lats, 50)
        eos_p95 = percentile(eos_lats, 95)
        eos_mean = statistics.mean(eos_lats)

        wers = [compute_wer(s.reference, s.transcript) for s in sessions]
        valid_wers = [w for w in wers if w >= 0]
        wer = statistics.mean(valid_wers) if valid_wers else -1.0
        wer_pct = wer * 100 if wer >= 0 else -1.0

        gpu_util_p95 = gpu_stats.get("gpu_util_p95")
        gpu_util_mean = gpu_stats.get("gpu_util_mean")
        vram_peak = gpu_stats.get("vram_peak_mib")
        vram_mean = gpu_stats.get("vram_mean_mib")

        row = {
            "concurrency": entry["concurrency"],
            "eos_p50": round(eos_p50, 4),
            "eos_p95": round(eos_p95, 4),
            "eos_mean": round(eos_mean, 4),
            "audio_duration_s": round(audio_dur, 2),
            "wer_pct": round(wer_pct, 2),
            "gpu_util_p95": gpu_util_p95,
            "gpu_util_mean": gpu_util_mean,
            "vram_peak_mib": vram_peak,
            "vram_mean_mib": vram_mean,
        }
        rows.append(row)

        gpu_p95_str = f"{gpu_util_p95:.0f}%" if gpu_util_p95 is not None else "n/a"
        gpu_mean_str = f"{gpu_util_mean:.0f}%" if gpu_util_mean is not None else "n/a"
        vram_peak_str = f"{vram_peak} MiB" if vram_peak is not None else "n/a"
        vram_mean_str = f"{vram_mean:.0f} MiB" if vram_mean is not None else "n/a"
        wer_str = f"{wer_pct:.2f}%" if wer_pct >= 0 else "n/a"

        print(
            f"{row['concurrency']:>4}  "
            f"{row['eos_p50']:>8.4f} {row['eos_p95']:>8.4f} {row['eos_mean']:>9.4f}  "
            f"{row['audio_duration_s']:>8.2f}s  {wer_str:>6}  "
            f"{gpu_p95_str:>7} {gpu_mean_str:>8}  "
            f"{vram_peak_str:>9} {vram_mean_str:>9}"
        )

    print("=" * len(header))
    return rows
```

Replace `write_csv` to match the new columns:

```python
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
```

**Step 4: Run all tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: All tests PASS

**Step 5: Commit**

```bash
git add benchmark_stt.py tests/test_benchmark.py
git commit -m "feat(benchmark): new report format with EOS latency, WER%, GPU stats"
```

---

### Task 6: Rewrite main() and CLI

**Files:**
- Modify: `benchmark_stt.py`

**Step 1: Replace main() and CLI argument parsing**

Replace the `main` function in `benchmark_stt.py`:

```python
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
    gpu_id = detect_gpu_id()
    gpu_monitor = GpuMonitor(gpu_id)
    await gpu_monitor.start()

    all_results: list[dict] = []
    for concurrency in concurrency_levels:
        gpu_mark = gpu_monitor.mark()
        sessions = await run_benchmark(
            stt_url=args.stt_url,
            samples=samples,
            concurrency=concurrency,
            timeout=args.timeout,
        )
        gpu_stats = gpu_monitor.slice_stats(gpu_mark)
        all_results.append({
            "concurrency": concurrency,
            "sessions": sessions,
            "gpu_stats": gpu_stats,
        })
        await asyncio.sleep(3.0)  # let GPU settle between levels

    await gpu_monitor.stop()

    rows = print_report(all_results)
    write_csv(rows, args.output)
```

**Step 2: Clean up imports at top of file**

Ensure these imports are present at the top of `benchmark_stt.py`:

```python
import argparse
import asyncio
import csv
import io
import json
import logging
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass

import httpx
import numpy as np
import soundfile as sf
import websockets

logger = logging.getLogger("benchmark")
```

**Step 3: Remove dead code**

Delete the old `AudioSample` reference to `mode` parameter. Remove the `mode` field from `all_results` entries. The `probe_session` function stays unchanged since it's still useful for diagnostics.

**Step 4: Run all tests**

Run: `pytest tests/test_benchmark.py -v`
Expected: All tests PASS

**Step 5: Manual smoke test**

Run: `python benchmark_stt.py --help`
Expected: Shows help with new flags (no `--modes`), default concurrency `1,5,10,20,30,40`

**Step 6: Commit**

```bash
git add benchmark_stt.py
git commit -m "feat(benchmark): rewrite main loop with GPU monitor and new CLI defaults"
```

---

### Task 7: Update README benchmark section

**Files:**
- Modify: `README.md`

**Step 1: Update the Benchmarking section in README.md**

Replace the Benchmarking section to reflect:
- Removed `--modes` flag
- New default concurrency levels (1,5,10,20,30,40)
- New default `--stt-url` (localhost)
- EOS latency as primary metric
- GPU metrics in output
- WER as percentage
- Updated metrics table
- Note about nvidia-smi requirement for GPU metrics

**Step 2: Commit**

```bash
git add README.md
git commit -m "docs: update README benchmark section for redesigned tool"
```

---

### Task 8: Final integration test

**Step 1: Run full test suite**

Run: `pytest tests/test_benchmark.py -v`
Expected: All tests PASS

**Step 2: Verify CLI**

Run: `python benchmark_stt.py --help`
Expected: Clean help output with new flags

**Step 3: Commit and push**

```bash
git push origin main
```
