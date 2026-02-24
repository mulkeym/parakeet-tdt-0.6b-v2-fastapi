# Benchmark Redesign: Realtime Streaming Concurrency with GPU Metrics

## Problem

The current benchmark tool (`benchmark_stt.py`) measures TTFB, total time, RTF, and WER across concurrency levels with both `realtime` and `fast` send modes. It lacks GPU metrics entirely, and the primary latency metric (RTF) conflates streaming time with processing time. A colleague observed heavy VRAM utilization on a 4090 under concurrent load, and we need visibility into how the service scales.

## Scope

- **Drop `fast` send mode.** Realtime-paced streaming only -- the pattern that matters for voice agent workloads.
- **Drop TTFB as a primary metric.** Replace with EOS latency (end-of-speech latency).
- **Add GPU monitoring** via `nvidia-smi dmon` background process.
- **Default concurrency ramp**: 1, 5, 10, 20, 30, 40.

## Core Metrics

| Metric | Description |
|--------|-------------|
| EOS Latency (p50/p95/mean) | Time from last audio frame sent to last transcript received |
| Audio Duration | Length of the speech clip (seconds) |
| WER | Word error rate vs known reference text, displayed as % |
| GPU Util (p95/mean) | SM utilization % during that concurrency level |
| VRAM Used (peak/mean) | Memory in MiB during that concurrency level |

## GPU Monitoring

### Auto-detection

1. Run `nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader`
2. Search for a python process (the parakeet service)
3. Map `gpu_uuid` to device index via `nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader`
4. Fall back to device 0 with a warning if detection fails

### Background Monitor

- Spawn `nvidia-smi dmon -s um -d 1 -i <gpu_id>` as a subprocess at benchmark start
- An asyncio task reads stdout, parsing columnar output into `(timestamp, gpu_util%, mem_used_MiB)` samples
- Samples accumulate into a list for the duration of the benchmark

### Per-level Snapshots

- Before each concurrency level: record current sample index
- After level completes: slice samples from that window
- Compute GPU util p95/mean and VRAM peak/mean for the level

### Lifecycle

- `dmon` subprocess runs for entire benchmark, killed at end
- If `nvidia-smi` is unavailable, skip GPU metrics with a warning

## Session Flow

### Per Session

1. Connect WebSocket
2. Stream audio frames at realtime pace (100ms per chunk)
3. Send 1 second of trailing silence to trigger VAD flush
4. Record `eos_time` = timestamp after last silence frame sent
5. Receive WebSocket messages until 5s timeout after last result
6. Record `final_result_time` = timestamp of last `text` message
7. **EOS Latency** = `final_result_time - eos_time`

### Per Concurrency Level

1. Mark GPU monitor start index
2. Launch N sessions via `asyncio.gather`
3. Wait for all to complete
4. Mark GPU monitor end index
5. Slice GPU samples, compute stats
6. Compute EOS latency p50/p95/mean, WER (as %) across N sessions
7. Sleep 2-3s between levels to let GPU settle

## Audio Generation

Same as current: TTS pool with unique texts per session (up to 40 samples for max concurrency). Each concurrent session gets a different text to catch cross-session result leakage.

## CLI Interface

| Flag | Default | Description |
|------|---------|-------------|
| `--stt-url` | `ws://localhost:8000/ws` | WebSocket URL of the STT service |
| `--tts-url` | `http://localhost:8004/tts` | HTTP URL of the TTS service |
| `--concurrency` | `1,5,10,20,30,40` | Comma-separated concurrency levels |
| `--output` | `benchmark_results.csv` | Output CSV path |
| `--timeout` | `30.0` | Seconds to wait for results |
| `--probe` | off | Single verbose diagnostic session |

## Output

### Console Table

```
============================================================================
 Conc  EOS p50  EOS p95  EOS mean  Audio Dur   WER%   GPU%p95  GPU%mean  VRAM peak  VRAM mean
    1   0.3214   0.3214    0.3214     6.42s   3.12%     45%      38%      1842 MiB   1790 MiB
    5   0.4102   0.5891    0.4530     6.38s   2.98%     78%      65%      2104 MiB   1980 MiB
   10   0.6234   1.2010    0.7812     6.45s   3.41%     92%      84%      2890 MiB   2650 MiB
============================================================================
```

### CSV

Same columns, one row per concurrency level, written to `--output` path.

### Probe Mode

Unchanged from current tool -- single verbose session for diagnostics.
