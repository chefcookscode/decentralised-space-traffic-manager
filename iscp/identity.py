"""
ISCP Task 3.1 — Lightweight Identity Registry
==============================================
Implements a federated Public Key Infrastructure (PKI) for space.

Every licensed satellite is issued a cryptographic certificate by a
ground-based Certificate Authority (CA).  The certificate binds the
satellite's human-readable identifier to an ECC public key and carries
an expiry timestamp (GPS seconds since J2000) plus an optional operator
organisation name.

Design goals
------------
* Lightweight — no X.509 overhead; a custom binary/dict format sized for
  bandwidth-constrained ISL links.
* Federated — multiple CAs can be registered (national / operator CAs)
  so that no single authority controls the whole constellation.
* Deterministic — all certificates are signed by the issuing CA's private
  key using ECDSA P-256, and the signature is verified before a certificate
  is accepted into the registry.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional

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

#: GPS epoch offset from Unix epoch (seconds).
GPS_EPOCH_OFFSET_S: float = 315_964_800.0

#: Default certificate validity period: 5 years in seconds.
DEFAULT_CERT_VALIDITY_S: float = 5 * 365.25 * 86_400.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unix_to_gps(unix_ts: float) -> float:
    """Convert a Unix timestamp to GPS seconds since J2000 (approx)."""
    # J2000 epoch = 2000-01-01T12:00:00 TT ≈ Unix 946_728_000
    return unix_ts - 946_728_000.0


def _gps_now() -> float:
    """Return the current time as GPS seconds since J2000 (approx)."""
    return _unix_to_gps(time.time())


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CertificateStatus(IntEnum):
    """Lifecycle status of a satellite certificate."""
    VALID = 0
    EXPIRED = 1
    REVOKED = 2
    PENDING = 3


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SatelliteCertificate:
    """
    Cryptographic certificate binding a satellite ID to an ECC public key.

    Parameters
    ----------
    satellite_id:
        Unique satellite identifier (≤ 8 ASCII chars, matching ISCP payload).
    operator:
        Human-readable operator / licence-holder name.
    public_key_pem:
        PEM-encoded ECC public key (P-256).
    issued_at:
        Issuance time in GPS seconds since J2000.
    expires_at:
        Expiry time in GPS seconds since J2000.
    issuer_ca_id:
        Identifier of the Certificate Authority that issued this certificate.
    serial:
        Hex string certificate serial number (SHA-256 of core fields).
    ca_signature:
        DER-encoded ECDSA signature by the issuing CA over the certificate
        canonical bytes.
    status:
        Current lifecycle status.
    """
    satellite_id: str
    operator: str
    public_key_pem: bytes
    issued_at: float
    expires_at: float
    issuer_ca_id: str
    serial: str = ""
    ca_signature: bytes = b""
    status: CertificateStatus = CertificateStatus.PENDING

    # ------------------------------------------------------------------
    # Canonical form for signing / hashing
    # ------------------------------------------------------------------

    def canonical_bytes(self) -> bytes:
        """
        Return a deterministic byte representation of the certificate's
        core fields, used as input for the CA signature.
        """
        doc = {
            "satellite_id": self.satellite_id,
            "operator": self.operator,
            "public_key_pem": self.public_key_pem.decode("ascii"),
            "issued_at": self.issued_at,
            "expires_at": self.expires_at,
            "issuer_ca_id": self.issuer_ca_id,
        }
        return json.dumps(doc, sort_keys=True, separators=(",", ":")).encode()

    def compute_serial(self) -> str:
        """Compute and return the SHA-256 hex digest of canonical bytes."""
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def is_valid(self, current_gps_s: Optional[float] = None) -> bool:
        """Return True if the certificate is VALID and not expired."""
        if self.status == CertificateStatus.REVOKED:
            return False
        t = current_gps_s if current_gps_s is not None else _gps_now()
        if t > self.expires_at:
            return False
        return self.status == CertificateStatus.VALID

    def public_key(self) -> ec.EllipticCurvePublicKey:
        """Deserialise and return the ECC public key object."""
        return serialization.load_pem_public_key(self.public_key_pem)  # type: ignore[return-value]

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        return (
            f"Cert(id={self.satellite_id!r}, op={self.operator!r}, "
            f"ca={self.issuer_ca_id!r}, serial={self.serial[:12]}…, "
            f"status={self.status.name})"
        )


@dataclass
class CertificateAuthority:
    """
    A ground-based Certificate Authority that issues and signs satellite
    certificates.

    Parameters
    ----------
    ca_id:
        Unique short identifier for this CA (e.g. ``'NASA-CA'``).
    organisation:
        Human-readable organisation name.
    """
    ca_id: str
    organisation: str
    _private_key: ec.EllipticCurvePrivateKey = field(
        default=None, init=False, repr=False  # type: ignore[assignment]
    )
    _public_key: ec.EllipticCurvePublicKey = field(
        default=None, init=False, repr=False  # type: ignore[assignment]
    )

    def __post_init__(self) -> None:
        self._private_key = ec.generate_private_key(ec.SECP256R1())
        self._public_key = self._private_key.public_key()

    @property
    def public_key(self) -> ec.EllipticCurvePublicKey:
        """Return the CA's ECC public key."""
        return self._public_key

    @property
    def public_key_pem(self) -> bytes:
        """Return the PEM-encoded CA public key."""
        return self._public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def issue_certificate(
        self,
        satellite_id: str,
        operator: str,
        satellite_public_key: ec.EllipticCurvePublicKey,
        issued_at: Optional[float] = None,
        validity_s: float = DEFAULT_CERT_VALIDITY_S,
    ) -> SatelliteCertificate:
        """
        Issue a signed certificate for *satellite_id*.

        Parameters
        ----------
        satellite_id:
            Satellite identifier (≤ 8 ASCII chars).
        operator:
            Operator / licence-holder name.
        satellite_public_key:
            The satellite's ECC P-256 public key.
        issued_at:
            Issuance GPS time (defaults to now).
        validity_s:
            Certificate validity period in seconds.

        Returns
        -------
        SatelliteCertificate
            Signed, VALID certificate ready for registry insertion.
        """
        if len(satellite_id.encode("ascii", errors="replace")) > 8:
            raise ValueError(
                f"satellite_id '{satellite_id}' exceeds 8 ASCII bytes"
            )

        t0 = issued_at if issued_at is not None else _gps_now()
        pub_pem = satellite_public_key.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        cert = SatelliteCertificate(
            satellite_id=satellite_id,
            operator=operator,
            public_key_pem=pub_pem,
            issued_at=t0,
            expires_at=t0 + validity_s,
            issuer_ca_id=self.ca_id,
        )
        cert.serial = cert.compute_serial()

        # Sign the canonical bytes with the CA private key
        signature = self._private_key.sign(
            cert.canonical_bytes(),
            ec.ECDSA(hashes.SHA256()),
        )
        cert.ca_signature = signature
        cert.status = CertificateStatus.VALID
        return cert

    def verify_certificate(self, cert: SatelliteCertificate) -> bool:
        """
        Verify the CA signature on *cert*.

        Returns True if the signature is valid; False otherwise.
        """
        if cert.issuer_ca_id != self.ca_id:
            return False
        try:
            self._public_key.verify(
                cert.ca_signature,
                cert.canonical_bytes(),
                ec.ECDSA(hashes.SHA256()),
            )
            return True
        except InvalidSignature:
            return False


@dataclass
class IdentityRegistry:
    """
    Federated identity registry for the ISCP constellation.

    Maintains a set of trusted CAs and a certificate store indexed by
    satellite ID.  Any CA in the trusted set may issue certificates; a
    satellite certificate is accepted only if it passes CA signature
    verification.

    Parameters
    ----------
    registry_id:
        Human-readable name for this registry node.
    """
    registry_id: str = "ISCP-GlobalRegistry"
    _trusted_cas: Dict[str, CertificateAuthority] = field(
        default_factory=dict, init=False, repr=False
    )
    _certificates: Dict[str, SatelliteCertificate] = field(
        default_factory=dict, init=False, repr=False
    )
    _revocation_list: List[str] = field(
        default_factory=list, init=False, repr=False
    )

    # ------------------------------------------------------------------
    # CA management
    # ------------------------------------------------------------------

    def register_ca(self, ca: CertificateAuthority) -> None:
        """Add *ca* to the set of trusted Certificate Authorities."""
        if ca.ca_id in self._trusted_cas:
            raise ValueError(f"CA '{ca.ca_id}' is already registered.")
        self._trusted_cas[ca.ca_id] = ca

    def get_ca(self, ca_id: str) -> Optional[CertificateAuthority]:
        """Return the CA with *ca_id*, or None."""
        return self._trusted_cas.get(ca_id)

    @property
    def trusted_ca_ids(self) -> List[str]:
        """Return list of registered CA identifiers."""
        return list(self._trusted_cas.keys())

    # ------------------------------------------------------------------
    # Certificate management
    # ------------------------------------------------------------------

    def register_certificate(self, cert: SatelliteCertificate) -> None:
        """
        Accept and store *cert* after verifying the CA signature.

        Raises
        ------
        ValueError
            If the issuing CA is not trusted, the signature is invalid,
            or the certificate serial is already revoked.
        """
        ca = self._trusted_cas.get(cert.issuer_ca_id)
        if ca is None:
            raise ValueError(
                f"Issuing CA '{cert.issuer_ca_id}' is not trusted "
                f"by registry '{self.registry_id}'."
            )
        if not ca.verify_certificate(cert):
            raise ValueError(
                f"Certificate signature verification failed for "
                f"satellite '{cert.satellite_id}'."
            )
        if cert.serial in self._revocation_list:
            raise ValueError(
                f"Certificate serial {cert.serial[:12]}… is revoked."
            )
        self._certificates[cert.satellite_id] = cert

    def get_certificate(
        self, satellite_id: str
    ) -> Optional[SatelliteCertificate]:
        """Return the certificate for *satellite_id*, or None."""
        return self._certificates.get(satellite_id)

    def revoke_certificate(self, satellite_id: str) -> None:
        """
        Revoke the certificate for *satellite_id*.

        The serial is added to the revocation list so re-registration of the
        same certificate is prevented.

        Raises
        ------
        KeyError
            If no certificate for *satellite_id* exists.
        """
        cert = self._certificates[satellite_id]
        cert.status = CertificateStatus.REVOKED
        self._revocation_list.append(cert.serial)

    # ------------------------------------------------------------------
    # Verification helpers
    # ------------------------------------------------------------------

    def is_satellite_trusted(
        self,
        satellite_id: str,
        current_gps_s: Optional[float] = None,
    ) -> bool:
        """
        Return True if *satellite_id* has a valid, non-expired certificate
        registered with a trusted CA.
        """
        cert = self._certificates.get(satellite_id)
        if cert is None:
            return False
        return cert.is_valid(current_gps_s)

    def lookup_public_key(
        self, satellite_id: str
    ) -> Optional[ec.EllipticCurvePublicKey]:
        """
        Return the ECC public key for *satellite_id* if a valid certificate
        exists, otherwise None.
        """
        cert = self._certificates.get(satellite_id)
        if cert is None or cert.status != CertificateStatus.VALID:
            return None
        return cert.public_key()

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a registry status summary string."""
        total = len(self._certificates)
        valid = sum(
            1 for c in self._certificates.values()
            if c.status == CertificateStatus.VALID
        )
        revoked = len(self._revocation_list)
        return (
            f"IdentityRegistry({self.registry_id!r}): "
            f"{len(self._trusted_cas)} CAs, "
            f"{total} certs ({valid} valid, {revoked} revoked)"
        )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

def generate_satellite_keypair() -> (
    tuple[ec.EllipticCurvePrivateKey, ec.EllipticCurvePublicKey]
):
    """
    Generate a fresh ECC P-256 key pair for a satellite.

    Returns
    -------
    (private_key, public_key)
    """
    priv = ec.generate_private_key(ec.SECP256R1())
    return priv, priv.public_key()
