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


# ---------------------------------------------------------------------------
# Relative speed
# ---------------------------------------------------------------------------

def test_relative_speed_zero_for_identical_velocities():
    v = (7_500.0, 0.0, 0.0)
    assert relative_speed_ms(v, v) == pytest.approx(0.0)


def test_is_high_speed_crossing_above_threshold():
    v_a = (7_784.0, 0.0, 0.0)
    v_b = (-7_784.0, 0.0, 0.0)   # head-on ≈ 15.6 km/s
    assert is_high_speed_crossing(v_a, v_b) is True


def test_is_high_speed_crossing_below_threshold():
    v_a = (100.0, 0.0, 0.0)
    v_b = (50.0, 0.0, 0.0)
    assert is_high_speed_crossing(v_a, v_b) is False


# ---------------------------------------------------------------------------
# Full handshake FSM — happy path
# ---------------------------------------------------------------------------

def test_full_handshake_reaches_established():
    gps_now = 800_000.0
    vel_a = (7_784.0, 0.0, 0.0)
    vel_b = (0.0, 7_784.0, 0.0)

    initiator = ISCPSession(local_id="SAT-A", peer_id="SAT-B")
    responder = ISCPSession(local_id="SAT-B", peer_id="SAT-A")

    hello = initiator.initiate(gps_now, (6_778_000.0, 0.0, 0.0), vel_a)
    assert initiator.state == HandshakeState.INIT_SENT

    hello_ack = responder.receive_hello(hello, gps_now, vel_b)
    assert hello_ack.accepted is True

    challenge = initiator.receive_hello_ack(hello_ack, gps_now)
    assert challenge is not None
    assert initiator.state == HandshakeState.CHALLENGE_SENT

    challenge_ack = responder.receive_challenge(challenge, gps_now)
    assert challenge_ack.accepted is True
    assert responder.state == HandshakeState.ESTABLISHED

    result = initiator.receive_challenge_ack(challenge_ack)
    assert result is True
    assert initiator.state == HandshakeState.ESTABLISHED
