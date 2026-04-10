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
