"""Tests for iscp.crypto (Task 3.2 — Low-Compute ECC Cryptography)."""

import pytest

from cryptography.hazmat.primitives.asymmetric import ec

from iscp.identity import generate_satellite_keypair
from iscp.crypto import (
    ECDSA_SIGNATURE_BYTES,
    sign_bytes,
    verify_bytes,
    create_signed_packet,
    verify_signed_packet,
    private_key_to_pem,
    public_key_to_pem,
    private_key_from_pem,
    public_key_from_pem,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def keypair():
    return generate_satellite_keypair()


@pytest.fixture
def sample_data() -> bytes:
    return b"ISCP-payload-data-for-testing-1234567890"


# ---------------------------------------------------------------------------
# Sign and verify
# ---------------------------------------------------------------------------

def test_sign_returns_64_bytes(keypair, sample_data):
    priv, _ = keypair
    sig = sign_bytes(sample_data, priv)
    assert len(sig) == ECDSA_SIGNATURE_BYTES


def test_verify_valid_signature(keypair, sample_data):
    priv, pub = keypair
    sig = sign_bytes(sample_data, priv)
    assert verify_bytes(sample_data, sig, pub) is True


def test_verify_rejects_wrong_data(keypair, sample_data):
    priv, pub = keypair
    sig = sign_bytes(sample_data, priv)
    assert verify_bytes(b"tampered-data", sig, pub) is False


def test_verify_rejects_wrong_key(sample_data):
    priv1, _ = generate_satellite_keypair()
    _, pub2 = generate_satellite_keypair()
    sig = sign_bytes(sample_data, priv1)
    assert verify_bytes(sample_data, sig, pub2) is False


def test_verify_rejects_short_signature(keypair, sample_data):
    _, pub = keypair
    assert verify_bytes(sample_data, b"\x00" * 10, pub) is False


def test_verify_rejects_tampered_signature(keypair, sample_data):
    priv, pub = keypair
    sig = bytearray(sign_bytes(sample_data, priv))
    sig[0] ^= 0xFF  # flip a bit
    assert verify_bytes(sample_data, bytes(sig), pub) is False


# ---------------------------------------------------------------------------
# Signed packet helpers
# ---------------------------------------------------------------------------

def test_signed_packet_length(keypair, sample_data):
    priv, _ = keypair
    pkt = create_signed_packet(sample_data, priv)
    assert len(pkt) == len(sample_data) + ECDSA_SIGNATURE_BYTES


def test_verify_signed_packet_valid(keypair, sample_data):
    priv, pub = keypair
    pkt = create_signed_packet(sample_data, priv)
    valid, payload = verify_signed_packet(pkt, pub, len(sample_data))
    assert valid is True
    assert payload == sample_data


def test_verify_signed_packet_rejects_tampered_payload(keypair, sample_data):
    priv, pub = keypair
    pkt = bytearray(create_signed_packet(sample_data, priv))
    pkt[0] ^= 0xFF  # tamper with first payload byte
    valid, _ = verify_signed_packet(bytes(pkt), pub, len(sample_data))
    assert valid is False


def test_verify_signed_packet_raises_on_short_input(keypair):
    _, pub = keypair
    with pytest.raises(ValueError, match="too short"):
        verify_signed_packet(b"\x00" * 10, pub, 100)


# ---------------------------------------------------------------------------
# Key serialisation round-trips
# ---------------------------------------------------------------------------

def test_private_key_pem_round_trip(keypair):
    priv, _ = keypair
    pem = private_key_to_pem(priv)
    restored = private_key_from_pem(pem)
    # Verify the restored key produces valid signatures
    data = b"round-trip-test"
    sig = sign_bytes(data, restored)
    assert verify_bytes(data, sig, restored.public_key()) is True


def test_public_key_pem_round_trip(keypair):
    priv, pub = keypair
    pem = public_key_to_pem(pub)
    restored_pub = public_key_from_pem(pem)
    data = b"round-trip-test"
    sig = sign_bytes(data, priv)
    assert verify_bytes(data, sig, restored_pub) is True
