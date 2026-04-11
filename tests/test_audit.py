"""Tests for iscp.audit (Task 3.4 — Immutable Cryptographic Audit Log)."""

import json
import pytest

from iscp.identity import generate_satellite_keypair
from iscp.audit import (
    AuditEventType,
    AuditLog,
    AuditRecord,
    GENESIS_HASH,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def keypair():
    return generate_satellite_keypair()


@pytest.fixture
def audit_log(keypair):
    priv, _ = keypair
    return AuditLog(logger_id="SAT-A", private_key=priv)


# ---------------------------------------------------------------------------
# Append and basic properties
# ---------------------------------------------------------------------------

def test_empty_log_has_length_zero(audit_log: AuditLog):
    assert audit_log.length == 0


def test_append_increments_length(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert audit_log.length == 1


def test_appended_record_has_correct_sequence(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert r.sequence == 0


def test_second_record_has_sequence_one(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "first")
    r2 = audit_log.append(AuditEventType.SIGNATURE_VALID, "SAT-B", "second")
    assert r2.sequence == 1


def test_first_record_prev_hash_is_genesis(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert r.prev_hash == GENESIS_HASH


def test_second_record_prev_hash_links_to_first(audit_log: AuditLog):
    r1 = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "first")
    r2 = audit_log.append(AuditEventType.SIGNATURE_VALID, "SAT-B", "second")
    assert r2.prev_hash == r1.record_hash


def test_record_has_signature(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert len(r.signature) == 64


def test_record_has_non_empty_hash(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert len(r.record_hash) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# Chain verification
# ---------------------------------------------------------------------------

def test_empty_log_verifies(audit_log: AuditLog):
    assert audit_log.verify_chain() is True


def test_single_record_log_verifies(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert audit_log.verify_chain() is True


def test_multi_record_log_verifies(audit_log: AuditLog):
    for i in range(5):
        audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", f"pkt-{i}")
    assert audit_log.verify_chain() is True


def test_tampered_record_fails_verification(audit_log: AuditLog, keypair):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    # Tamper: overwrite detail directly
    audit_log._records[0].detail = "tampered"
    assert audit_log.verify_chain() is False


def test_tampered_prev_hash_fails_verification(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "first")
    audit_log.append(AuditEventType.SIGNATURE_VALID, "SAT-B", "second")
    # Break the chain
    audit_log._records[1].prev_hash = "a" * 64
    assert audit_log.verify_chain() is False


def test_wrong_public_key_fails_verification(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    _, wrong_pub = generate_satellite_keypair()
    assert audit_log.verify_chain(public_key=wrong_pub) is False


# ---------------------------------------------------------------------------
# Querying
# ---------------------------------------------------------------------------

def test_get_record_by_sequence(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    assert audit_log.get_record(0) is r


def test_records_for_subject(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "pkt-b")
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-C", "pkt-c")
    audit_log.append(AuditEventType.SIGNATURE_VALID, "SAT-B", "sig-b")
    records = audit_log.records_for_subject("SAT-B")
    assert len(records) == 2
    assert all(r.subject_id == "SAT-B" for r in records)


def test_records_by_type(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "pkt")
    audit_log.append(AuditEventType.SIGNATURE_INVALID, "SAT-B", "bad sig")
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-C", "pkt2")
    records = audit_log.records_by_type(AuditEventType.PACKET_RECEIVED)
    assert len(records) == 2


# ---------------------------------------------------------------------------
# Export / import round-trip
# ---------------------------------------------------------------------------

def test_export_json_returns_string(audit_log: AuditLog):
    audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", "test")
    s = audit_log.export_json()
    assert isinstance(s, str)


def test_export_import_round_trip_verifies(audit_log: AuditLog, keypair):
    priv, pub = keypair
    for i in range(3):
        audit_log.append(AuditEventType.PACKET_RECEIVED, "SAT-B", f"pkt-{i}")

    exported = audit_log.export_json()
    restored = AuditLog.import_json(exported, "SAT-A", priv)

    assert restored.length == audit_log.length
    assert restored.verify_chain(public_key=pub) is True


def test_export_json_is_parseable(audit_log: AuditLog):
    audit_log.append(AuditEventType.CERT_REVOKED, "SAT-X", "revocation")
    data = json.loads(audit_log.export_json())
    assert isinstance(data, list)
    assert len(data) == 1
    assert data[0]["event_type"] == int(AuditEventType.CERT_REVOKED)


# ---------------------------------------------------------------------------
# AuditRecord helpers
# ---------------------------------------------------------------------------

def test_record_summary_returns_string(audit_log: AuditLog):
    r = audit_log.append(AuditEventType.MANEUVER_ORDERED, "SAT-B", "avoid")
    assert isinstance(r.summary(), str)


def test_audit_log_summary_returns_string(audit_log: AuditLog):
    assert isinstance(audit_log.summary(), str)
