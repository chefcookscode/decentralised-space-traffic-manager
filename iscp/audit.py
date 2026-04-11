"""
ISCP Task 3.4 — Immutable Cryptographic Audit Log ("Black Box")
================================================================
Provides an append-only, hash-chained event log that creates a tamper-evident
record of all significant ISCP security events.

Legal and liability context
---------------------------
Under international space law (Outer Space Treaty, Liability Convention 1972,
and national licensing regimes), fault determination after a collision or
near-miss requires contemporaneous, verifiable records.  This module provides
a *flight-recorder* equivalent:

* Every event is individually signed by the satellite's on-board key
  (ECDSA P-256 via ``iscp.crypto``).
* Each event carries the SHA-256 hash of the *previous* event's signed
  record, forming a hash chain.  Altering any past event invalidates all
  subsequent hashes — detected immediately on verification.
* The full log can be exported to JSON for transmission to ground stations,
  insurance underwriters, or the UNOOSA incident database.

Event taxonomy
--------------
``PACKET_RECEIVED``     — an ISCP state-vector packet was received.
``SIGNATURE_VALID``     — packet signature verified successfully.
``SIGNATURE_INVALID``   — packet signature failed; possible spoofing.
``SANITY_CHECK_PASSED`` — position/velocity plausibility check passed.
``SANITY_CHECK_FAILED`` — plausibility check failed; data flagged.
``GROUND_TRUTH_MATCH``  — claimed position matches ground-truth ledger.
``GROUND_TRUTH_MISMATCH`` — claimed position inconsistent with ground truth.
``MANEUVER_ORDERED``    — a collision-avoidance maneuver was commanded.
``CERT_VERIFIED``       — satellite certificate verified in registry.
``CERT_REVOKED``        — satellite certificate has been revoked.
``SESSION_ESTABLISHED`` — ISCP handshake completed successfully.
``SESSION_CLOSED``      — ISCP session closed (with reason).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional

from cryptography.hazmat.primitives.asymmetric import ec

from iscp.crypto import sign_bytes, verify_bytes


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: First block's "previous hash" sentinel (genesis block).
GENESIS_HASH: str = "0" * 64

#: GPS J2000 epoch offset from Unix epoch (seconds).
_GPS_J2000_OFFSET: float = 946_728_000.0


def _gps_now() -> float:
    """Return approximate current GPS seconds since J2000."""
    return time.time() - _GPS_J2000_OFFSET


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AuditEventType(IntEnum):
    """Classification of recorded audit events."""
    PACKET_RECEIVED = 0
    SIGNATURE_VALID = 1
    SIGNATURE_INVALID = 2
    SANITY_CHECK_PASSED = 3
    SANITY_CHECK_FAILED = 4
    GROUND_TRUTH_MATCH = 5
    GROUND_TRUTH_MISMATCH = 6
    MANEUVER_ORDERED = 7
    CERT_VERIFIED = 8
    CERT_REVOKED = 9
    SESSION_ESTABLISHED = 10
    SESSION_CLOSED = 11


# ---------------------------------------------------------------------------
# Audit record
# ---------------------------------------------------------------------------

@dataclass
class AuditRecord:
    """
    A single entry in the immutable audit log.

    Attributes
    ----------
    sequence:
        Monotonically increasing sequence number within this log.
    event_type:
        Categorisation of the event.
    logger_id:
        Identifier of the satellite that created this record.
    subject_id:
        Identifier of the peer satellite or entity the event concerns.
    timestamp_gps_s:
        GPS seconds since J2000 when the event was recorded.
    detail:
        Free-text human-readable description (kept short for bandwidth).
    prev_hash:
        SHA-256 hex digest of the previous record's *signed_bytes*.
        For the first record this is :data:`GENESIS_HASH`.
    record_hash:
        SHA-256 hex digest of this record's canonical bytes (set after
        the record is finalised).
    signature:
        64-byte raw ECDSA signature over the canonical bytes, created by
        the logging satellite's private key.
    """
    sequence: int
    event_type: AuditEventType
    logger_id: str
    subject_id: str
    timestamp_gps_s: float
    detail: str
    prev_hash: str
    record_hash: str = ""
    signature: bytes = b""

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def canonical_dict(self) -> dict:
        """
        Return a deterministic dictionary of core fields, used as input
        for hashing and signing.  Excludes ``record_hash`` and
        ``signature`` which are derived.
        """
        return {
            "sequence": self.sequence,
            "event_type": int(self.event_type),
            "logger_id": self.logger_id,
            "subject_id": self.subject_id,
            "timestamp_gps_s": self.timestamp_gps_s,
            "detail": self.detail,
            "prev_hash": self.prev_hash,
        }

    def canonical_bytes(self) -> bytes:
        """Return UTF-8 encoded JSON of :meth:`canonical_dict`."""
        return json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")

    def compute_hash(self) -> str:
        """Return the SHA-256 hex digest of :meth:`canonical_bytes`."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def to_dict(self) -> dict:
        """Return a full serialisable dictionary including hash and signature."""
        d = self.canonical_dict()
        d["record_hash"] = self.record_hash
        d["signature"] = self.signature.hex()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "AuditRecord":
        """Reconstruct an AuditRecord from a dict produced by :meth:`to_dict`."""
        return cls(
            sequence=d["sequence"],
            event_type=AuditEventType(d["event_type"]),
            logger_id=d["logger_id"],
            subject_id=d["subject_id"],
            timestamp_gps_s=d["timestamp_gps_s"],
            detail=d["detail"],
            prev_hash=d["prev_hash"],
            record_hash=d["record_hash"],
            signature=bytes.fromhex(d["signature"]),
        )

    def summary(self) -> str:
        """One-line summary for logging / debugging."""
        return (
            f"[{self.sequence:06d}] {self.event_type.name} "
            f"logger={self.logger_id!r} subject={self.subject_id!r} "
            f"t={self.timestamp_gps_s:.3f} hash={self.record_hash[:12]}…"
        )


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------

class AuditLog:
    """
    Append-only, hash-chained cryptographic audit log.

    Each appended :class:`AuditRecord` is:

    1. Assigned the hash of the previous record as its ``prev_hash``.
    2. Hashed to produce its own ``record_hash``.
    3. Signed with the logging satellite's ECC private key.

    Tampering with any record breaks the hash chain and the signature,
    providing a dual-layer tamper-evidence guarantee.

    Parameters
    ----------
    logger_id:
        Satellite identifier that owns and writes to this log.
    private_key:
        The satellite's ECC private key for signing records.
    """

    def __init__(
        self,
        logger_id: str,
        private_key: ec.EllipticCurvePrivateKey,
    ) -> None:
        self._logger_id = logger_id
        self._private_key = private_key
        self._public_key = private_key.public_key()
        self._records: List[AuditRecord] = []

    # ------------------------------------------------------------------
    # Append
    # ------------------------------------------------------------------

    def append(
        self,
        event_type: AuditEventType,
        subject_id: str,
        detail: str,
        timestamp_gps_s: Optional[float] = None,
    ) -> AuditRecord:
        """
        Append a new event record to the log.

        Parameters
        ----------
        event_type:
            The type of event being recorded.
        subject_id:
            The peer satellite or entity this event concerns.
        detail:
            Free-text description (keep concise for ISL bandwidth).
        timestamp_gps_s:
            GPS time of the event; defaults to the current time.

        Returns
        -------
        AuditRecord
            The finalised, signed record that was appended.
        """
        ts = timestamp_gps_s if timestamp_gps_s is not None else _gps_now()
        prev_hash = (
            self._records[-1].record_hash
            if self._records
            else GENESIS_HASH
        )
        record = AuditRecord(
            sequence=len(self._records),
            event_type=event_type,
            logger_id=self._logger_id,
            subject_id=subject_id,
            timestamp_gps_s=ts,
            detail=detail,
            prev_hash=prev_hash,
        )
        record.record_hash = record.compute_hash()
        record.signature = sign_bytes(record.canonical_bytes(), self._private_key)
        self._records.append(record)
        return record

    # ------------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------------

    def verify_chain(
        self,
        public_key: Optional[ec.EllipticCurvePublicKey] = None,
    ) -> bool:
        """
        Verify the integrity of the entire hash chain and all signatures.

        Parameters
        ----------
        public_key:
            The ECC public key to verify signatures against.  Defaults to
            the key derived from the log's own private key.

        Returns
        -------
        bool
            True only if every record's hash and signature are valid and
            the hash chain is unbroken.
        """
        pk = public_key if public_key is not None else self._public_key
        expected_prev = GENESIS_HASH
        for record in self._records:
            # Check hash chain linkage
            if record.prev_hash != expected_prev:
                return False
            # Check record_hash integrity
            if record.record_hash != record.compute_hash():
                return False
            # Check ECDSA signature
            if not verify_bytes(record.canonical_bytes(), record.signature, pk):
                return False
            expected_prev = record.record_hash
        return True

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def length(self) -> int:
        """Number of records in the log."""
        return len(self._records)

    def get_record(self, sequence: int) -> AuditRecord:
        """Return the record at *sequence* number."""
        return self._records[sequence]

    def records_for_subject(self, subject_id: str) -> List[AuditRecord]:
        """Return all records concerning *subject_id*."""
        return [r for r in self._records if r.subject_id == subject_id]

    def records_by_type(self, event_type: AuditEventType) -> List[AuditRecord]:
        """Return all records of *event_type*."""
        return [r for r in self._records if r.event_type == event_type]

    # ------------------------------------------------------------------
    # Export / import
    # ------------------------------------------------------------------

    def export_json(self) -> str:
        """
        Export the full log to a JSON string.

        Suitable for transmission to a ground station or archival storage.
        """
        return json.dumps(
            [r.to_dict() for r in self._records],
            separators=(",", ":"),
        )

    @classmethod
    def import_json(
        cls,
        json_str: str,
        logger_id: str,
        private_key: ec.EllipticCurvePrivateKey,
    ) -> "AuditLog":
        """
        Reconstruct an AuditLog from a previously exported JSON string.

        The reconstructed log can be verified with :meth:`verify_chain`
        using the original logger's public key.
        """
        log = cls(logger_id=logger_id, private_key=private_key)
        for d in json.loads(json_str):
            log._records.append(AuditRecord.from_dict(d))
        return log

    def summary(self) -> str:
        """Return a one-line summary of the log state."""
        tail_hash = (
            self._records[-1].record_hash[:16] + "…"
            if self._records
            else "empty"
        )
        return (
            f"AuditLog(logger={self._logger_id!r}, "
            f"records={self.length}, tail={tail_hash})"
        )
