"""Tests for iscp.identity (Task 3.1 — Lightweight Identity Registry)."""

import pytest

from iscp.identity import (
    CertificateAuthority,
    CertificateStatus,
    IdentityRegistry,
    SatelliteCertificate,
    generate_satellite_keypair,
    DEFAULT_CERT_VALIDITY_S,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def ca() -> CertificateAuthority:
    return CertificateAuthority(ca_id="TEST-CA", organisation="Test Authority")


@pytest.fixture
def registry(ca: CertificateAuthority) -> IdentityRegistry:
    reg = IdentityRegistry(registry_id="TEST-REGISTRY")
    reg.register_ca(ca)
    return reg


@pytest.fixture
def sat_keys():
    return generate_satellite_keypair()


@pytest.fixture
def cert(ca: CertificateAuthority, sat_keys) -> SatelliteCertificate:
    _, pub = sat_keys
    return ca.issue_certificate(
        satellite_id="SAT-0001",
        operator="TestCorp",
        satellite_public_key=pub,
        issued_at=0.0,
    )


# ---------------------------------------------------------------------------
# Key generation
# ---------------------------------------------------------------------------

def test_generate_keypair_returns_two_keys():
    priv, pub = generate_satellite_keypair()
    assert priv is not None
    assert pub is not None


# ---------------------------------------------------------------------------
# Certificate issuance
# ---------------------------------------------------------------------------

def test_issued_certificate_is_valid(cert: SatelliteCertificate):
    assert cert.status == CertificateStatus.VALID


def test_issued_certificate_has_serial(cert: SatelliteCertificate):
    assert len(cert.serial) == 64  # SHA-256 hex


def test_issued_certificate_has_signature(cert: SatelliteCertificate):
    assert len(cert.ca_signature) > 0


def test_certificate_satellite_id(cert: SatelliteCertificate):
    assert cert.satellite_id == "SAT-0001"


def test_certificate_validity_window(cert: SatelliteCertificate):
    # issued_at=0, so valid from 0 to DEFAULT_CERT_VALIDITY_S
    assert cert.issued_at == 0.0
    assert cert.expires_at == pytest.approx(DEFAULT_CERT_VALIDITY_S)


def test_issue_rejects_long_satellite_id(ca: CertificateAuthority, sat_keys):
    _, pub = sat_keys
    with pytest.raises(ValueError, match="satellite_id"):
        ca.issue_certificate(
            satellite_id="TOOLONGID",  # 9 chars > 8-byte limit
            operator="TestCorp",
            satellite_public_key=pub,
        )


# ---------------------------------------------------------------------------
# CA signature verification
# ---------------------------------------------------------------------------

def test_ca_verifies_its_own_certificate(
    ca: CertificateAuthority, cert: SatelliteCertificate
):
    assert ca.verify_certificate(cert) is True


def test_different_ca_cannot_verify_certificate(cert: SatelliteCertificate):
    other_ca = CertificateAuthority(ca_id="OTHER-CA", organisation="Other")
    # Wrong CA id — verify returns False
    assert other_ca.verify_certificate(cert) is False


def test_tampered_certificate_fails_verification(
    ca: CertificateAuthority, cert: SatelliteCertificate
):
    cert.operator = "HackerCorp"  # tamper with a signed field
    assert ca.verify_certificate(cert) is False


# ---------------------------------------------------------------------------
# Certificate lifecycle
# ---------------------------------------------------------------------------

def test_expired_certificate_is_not_valid(cert: SatelliteCertificate):
    # Set current time far in the future
    future_time = cert.expires_at + 1.0
    assert cert.is_valid(current_gps_s=future_time) is False


def test_valid_certificate_within_window(cert: SatelliteCertificate):
    assert cert.is_valid(current_gps_s=1.0) is True


def test_revoked_certificate_is_not_valid(cert: SatelliteCertificate):
    cert.status = CertificateStatus.REVOKED
    assert cert.is_valid(current_gps_s=1.0) is False


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry_registers_certificate(
    registry: IdentityRegistry, cert: SatelliteCertificate
):
    registry.register_certificate(cert)
    assert registry.get_certificate("SAT-0001") is cert


def test_registry_rejects_untrusted_ca_certificate(sat_keys):
    reg = IdentityRegistry()
    priv, pub = sat_keys
    foreign_ca = CertificateAuthority(ca_id="FOREIGN-CA", organisation="Foreign")
    cert = foreign_ca.issue_certificate("SAT-X", "ForeignCorp", pub, issued_at=0.0)
    with pytest.raises(ValueError, match="not trusted"):
        reg.register_certificate(cert)


def test_registry_rejects_duplicate_ca(registry: IdentityRegistry, ca):
    with pytest.raises(ValueError, match="already registered"):
        registry.register_ca(ca)


def test_satellite_is_trusted_after_registration(
    registry: IdentityRegistry, cert: SatelliteCertificate
):
    registry.register_certificate(cert)
    assert registry.is_satellite_trusted("SAT-0001", current_gps_s=1.0) is True


def test_unregistered_satellite_is_not_trusted(registry: IdentityRegistry):
    assert registry.is_satellite_trusted("UNKNOWN") is False


def test_registry_revoke_marks_certificate(
    registry: IdentityRegistry, cert: SatelliteCertificate
):
    registry.register_certificate(cert)
    registry.revoke_certificate("SAT-0001")
    assert cert.status == CertificateStatus.REVOKED


def test_registry_lookup_public_key(
    registry: IdentityRegistry, cert: SatelliteCertificate
):
    registry.register_certificate(cert)
    pk = registry.lookup_public_key("SAT-0001")
    assert pk is not None


def test_registry_summary_returns_string(registry: IdentityRegistry):
    assert isinstance(registry.summary(), str)


def test_certificate_summary_returns_string(cert: SatelliteCertificate):
    assert isinstance(cert.summary(), str)
