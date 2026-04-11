"""
ISCP Task 3.2 — Low-Compute ECC Cryptography
=============================================
Provides ECDSA P-256 digital signing and verification for ISCP data packets.

Algorithm selection rationale
------------------------------
* **Elliptic Curve P-256 (SECP256R1)** — equivalent security to RSA-3072
  with keys/signatures that are ~10× smaller, which is critical for
  bandwidth-constrained inter-satellite links.
* **ECDSA with SHA-256** — hardware-accelerated on most radiation-hardened
  microcontrollers (e.g. LEON4, RAD5545), keeping end-to-end sign+verify
  latency well under 10 ms.
* No padding modes, no ASN.1 certificate chains at the packet level —
  only the 64-byte raw (r, s) signature is appended to each ISCP packet,
  minimising transmission overhead.

Signed-packet format
---------------------
``[ ISCP payload bytes (variable) ][ 64-byte raw ECDSA signature ]``

The signature covers the payload bytes only (the identity of the signer is
established separately through the certificate registry).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import (
    decode_dss_signature,
    encode_dss_signature,
)
from cryptography.exceptions import InvalidSignature


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Length (bytes) of a raw (r, s) P-256 ECDSA signature.
ECDSA_SIGNATURE_BYTES: int = 64

#: Each scalar r or s is 32 bytes for P-256.
_SCALAR_BYTES: int = 32


# ---------------------------------------------------------------------------
# Low-level sign / verify helpers
# ---------------------------------------------------------------------------

def sign_bytes(
    data: bytes,
    private_key: ec.EllipticCurvePrivateKey,
) -> bytes:
    """
    Sign *data* with *private_key* using ECDSA-SHA-256 (P-256).

    Parameters
    ----------
    data:
        The byte string to sign (e.g. a packed ISCP payload).
    private_key:
        The satellite's ECC P-256 private key.

    Returns
    -------
    bytes
        64-byte raw ECDSA signature (r ‖ s), each 32 bytes big-endian.
    """
    der_sig = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
    r, s = decode_dss_signature(der_sig)
    return r.to_bytes(_SCALAR_BYTES, "big") + s.to_bytes(_SCALAR_BYTES, "big")


def verify_bytes(
    data: bytes,
    raw_signature: bytes,
    public_key: ec.EllipticCurvePublicKey,
) -> bool:
    """
    Verify the ECDSA-SHA-256 *raw_signature* over *data*.

    Parameters
    ----------
    data:
        The byte string that was signed.
    raw_signature:
        64-byte raw ECDSA signature (r ‖ s) as returned by :func:`sign_bytes`.
    public_key:
        The sender's ECC P-256 public key (retrieved from the identity
        registry before calling this function).

    Returns
    -------
    bool
        True if the signature is valid; False otherwise.
    """
    if len(raw_signature) != ECDSA_SIGNATURE_BYTES:
        return False
    r = int.from_bytes(raw_signature[:_SCALAR_BYTES], "big")
    s = int.from_bytes(raw_signature[_SCALAR_BYTES:], "big")
    der_sig = encode_dss_signature(r, s)
    try:
        public_key.verify(der_sig, data, ec.ECDSA(hashes.SHA256()))
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# Signed packet helpers
# ---------------------------------------------------------------------------

def create_signed_packet(
    payload_bytes: bytes,
    private_key: ec.EllipticCurvePrivateKey,
) -> bytes:
    """
    Append a 64-byte ECDSA signature to *payload_bytes*.

    The resulting signed packet is::

        payload_bytes  ‖  raw_signature (64 bytes)

    Parameters
    ----------
    payload_bytes:
        Serialised ISCP payload (e.g. from ``ISCPPayload.pack()``).
    private_key:
        The sending satellite's ECC private key.

    Returns
    -------
    bytes
        Signed packet: payload bytes followed by the 64-byte signature.
    """
    signature = sign_bytes(payload_bytes, private_key)
    return payload_bytes + signature


def verify_signed_packet(
    signed_packet: bytes,
    public_key: ec.EllipticCurvePublicKey,
    payload_size: int,
) -> Tuple[bool, bytes]:
    """
    Verify the signature on a signed ISCP packet and extract the payload.

    Parameters
    ----------
    signed_packet:
        Bytes received from the peer: payload ‖ signature.
    public_key:
        The sender's registered ECC public key.
    payload_size:
        Expected length (bytes) of the payload portion.

    Returns
    -------
    (valid, payload_bytes)
        *valid* is True if the signature is authentic.
        *payload_bytes* contains only the payload portion (without sig).

    Raises
    ------
    ValueError
        If *signed_packet* is shorter than ``payload_size + ECDSA_SIGNATURE_BYTES``.
    """
    expected_len = payload_size + ECDSA_SIGNATURE_BYTES
    if len(signed_packet) < expected_len:
        raise ValueError(
            f"signed_packet too short: expected ≥ {expected_len} bytes, "
            f"got {len(signed_packet)}."
        )
    payload_bytes = signed_packet[:payload_size]
    raw_sig = signed_packet[payload_size:payload_size + ECDSA_SIGNATURE_BYTES]
    valid = verify_bytes(payload_bytes, raw_sig, public_key)
    return valid, payload_bytes


# ---------------------------------------------------------------------------
# Key serialisation helpers
# ---------------------------------------------------------------------------

def private_key_to_pem(private_key: ec.EllipticCurvePrivateKey) -> bytes:
    """Return the PEM-encoded PKCS#8 representation of *private_key*."""
    return private_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )


def public_key_to_pem(public_key: ec.EllipticCurvePublicKey) -> bytes:
    """Return the PEM-encoded SubjectPublicKeyInfo representation."""
    return public_key.public_bytes(
        serialization.Encoding.PEM,
        serialization.PublicFormat.SubjectPublicKeyInfo,
    )


def private_key_from_pem(pem: bytes) -> ec.EllipticCurvePrivateKey:
    """Deserialise an ECC private key from PEM bytes."""
    return serialization.load_pem_private_key(pem, password=None)  # type: ignore[return-value]


def public_key_from_pem(pem: bytes) -> ec.EllipticCurvePublicKey:
    """Deserialise an ECC public key from PEM bytes."""
    return serialization.load_pem_public_key(pem)  # type: ignore[return-value]
