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


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------

def test_pack_returns_correct_length(sample_payload):
    data = sample_payload.pack()
    assert len(data) == PAYLOAD_SIZE


def test_unpack_restores_satellite_id(sample_payload):
    data = sample_payload.pack()
    restored = ISCPPayload.unpack(data)
    assert restored.satellite_id == sample_payload.satellite_id


def test_unpack_restores_position(sample_payload):
    restored = ISCPPayload.unpack(sample_payload.pack())
    assert restored.position == pytest.approx(sample_payload.position)


def test_unpack_restores_velocity(sample_payload):
    restored = ISCPPayload.unpack(sample_payload.pack())
    assert restored.velocity == pytest.approx(sample_payload.velocity)


def test_unpack_restores_covariance(sample_payload):
    restored = ISCPPayload.unpack(sample_payload.pack())
    assert restored.covariance == pytest.approx(sample_payload.covariance)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------

def test_validate_rejects_long_satellite_id():
    payload = ISCPPayload(
        satellite_id="TOOLONGID",  # 9 chars > 8-byte limit
        timestamp=0.0,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        covariance=[0.0] * 21,
        mass=100.0,
    )
    with pytest.raises(ValueError, match="satellite_id"):
        payload.validate()


def test_validate_rejects_wrong_covariance_length():
    payload = ISCPPayload(
        satellite_id="SAT-0001",
        timestamp=0.0,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        covariance=[0.0] * 10,  # wrong length
        mass=100.0,
    )
    with pytest.raises(ValueError, match="covariance"):
        payload.validate()


def test_validate_rejects_negative_mass():
    payload = ISCPPayload(
        satellite_id="SAT-0001",
        timestamp=0.0,
        position=(0.0, 0.0, 0.0),
        velocity=(0.0, 0.0, 0.0),
        covariance=[0.0] * 21,
        mass=-1.0,
    )
    with pytest.raises(ValueError, match="mass"):
        payload.validate()


def test_validate_rejects_invalid_confidence():
    intent = ManeuverIntent(confidence=1.5)
    with pytest.raises(ValueError, match="confidence"):
        intent.validate()


def test_summary_returns_string(sample_payload):
    assert isinstance(sample_payload.summary(), str)
