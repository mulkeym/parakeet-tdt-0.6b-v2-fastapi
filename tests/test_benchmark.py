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
