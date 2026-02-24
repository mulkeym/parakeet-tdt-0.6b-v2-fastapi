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


class TestSessionResult:
    def test_fields_exist(self):
        from benchmark_stt import SessionResult
        r = SessionResult(eos_latency=0.5, audio_duration=6.0, transcript="hello", reference="hello")
        assert r.eos_latency == 0.5
        assert r.audio_duration == 6.0
