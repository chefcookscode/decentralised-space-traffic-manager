"""Tests for iscp.handshake (Task 1.3 — Handshake & Synchronisation)."""

import pytest

from iscp.handshake import (
    CloseReason,
    HandshakeState,
    ISCPSession,
    compute_clock_offset,
    is_clock_in_sync,
    is_high_speed_crossing,
    relative_speed_ms,
    unix_to_gps,
    gps_to_unix,
    HIGH_RELATIVE_SPEED_THRESHOLD_MS,
    MAX_CLOCK_OFFSET_S,
)


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def test_unix_gps_round_trip():
    unix_ts = 1_700_000_000.0
    assert gps_to_unix(unix_to_gps(unix_ts)) == pytest.approx(unix_ts)


def test_clock_offset_zero_for_identical_clocks():
    t = 800_000.0
    assert compute_clock_offset(t, t) == pytest.approx(0.0)


def test_is_clock_in_sync_within_tolerance():
    assert is_clock_in_sync(0.0005) is True  # 0.5 ms < 1 ms threshold


def test_is_clock_in_sync_exceeds_tolerance():
    assert is_clock_in_sync(0.002) is False  # 2 ms > 1 ms threshold
