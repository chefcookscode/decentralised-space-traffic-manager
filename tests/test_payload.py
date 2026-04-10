"""Tests for iscp.payload (Task 1.1 — Data Payload Schema)."""

import struct
import pytest

from iscp.payload import (
    ISCPPayload,
    ManeuverIntent,
    ManeuverType,
    PropulsionType,
    PAYLOAD_SIZE,
    MAX_PAYLOAD_BYTES,
    SATELLITE_ID_LENGTH,
)


@pytest.fixture
def sample_payload() -> ISCPPayload:
    """Return a fully populated ISCPPayload for reuse across tests."""
    return ISCPPayload(
        satellite_id="SAT-0001",
        timestamp=800_000.0,
        position=(6_778_000.0, 0.0, 0.0),
        velocity=(0.0, 7_784.0, 0.0),
        covariance=[0.1 * i for i in range(21)],
        mass=250.0,
        propulsion_type=PropulsionType.ELECTRIC_ION,
        maneuver_intent=ManeuverIntent(
            intent_type=ManeuverType.COLLISION_AVOIDANCE,
            delta_v=(0.05, -0.02, 0.0),
            burn_start_epoch=800_100.0,
            burn_duration=30.0,
            confidence=0.95,
        ),
    )


# ---------------------------------------------------------------------------
# Payload size
# ---------------------------------------------------------------------------

def test_payload_size_under_1kb():
    assert PAYLOAD_SIZE <= MAX_PAYLOAD_BYTES
