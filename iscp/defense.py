"""
ISCP Task 3.3 — Sybil and Spoofing Defense Mechanisms
======================================================
Provides algorithmic sanity-checks and a ground-truth ledger to detect and
reject fraudulent or compromised trajectory broadcasts.

Two-layer defense model
-----------------------
1. **Real-time sanity checks** (on-board, zero-uplink required)

   When Satellite A claims position P_A, the receiving satellite B runs
   the following checks *before* trusting the data:

   a. *Cryptographic authenticity* — the ISCP packet signature must verify
      against A's registered public key (handled by ``iscp.crypto``).
   b. *Physics plausibility* — the claimed orbit must be consistent with
      LEO physics (altitude ≥ 160 km, speed within LEO envelope).
   c. *Sensor cross-check* — if B has onboard ranging data (radar/lidar)
      for A, the claimed range must match the measured range within a
      configurable tolerance.  A mismatch flags the packet as compromised.
   d. *Velocity continuity* — the claimed velocity change between two
      consecutive packets must not exceed the satellite's maximum ΔV
      capability.

2. **Periodic ground-truth uplinks** (requires ground contact)

   Ground stations transmit a compressed *ground-truth ledger* containing
   independently-verified positions for all known objects in the neighbourhood.
   On-board logic cross-references received state vectors against this ledger
   and flags inconsistencies.

Threat taxonomy
---------------
* **Spoofing** — a legitimate satellite transmits false state data.
* **Sybil attack** — an attacker injects packets that claim to be multiple
  distinct satellites.  Detected by cross-referencing claimed IDs against
  the certificate registry (handled upstream) and by detecting two
  "satellites" that report mutually inconsistent positions.
* **Replay attack** — an attacker retransmits an old valid packet.  Detected
  by enforcing strictly increasing timestamps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth's mean radius in metres.
EARTH_RADIUS_M: float = 6_371_000.0

#: Minimum LEO altitude above Earth's surface (metres) — Kármán line ≈ 100 km;
#: practically the lower bound for a stable orbit is ~160 km.
MIN_LEO_ALTITUDE_M: float = 160_000.0

#: Maximum LEO altitude above Earth's surface (metres).
MAX_LEO_ALTITUDE_M: float = 2_000_000.0

#: Minimum plausible orbital speed at MAX_LEO_ALTITUDE (m/s).
MIN_ORBITAL_SPEED_MS: float = 6_900.0

#: Maximum plausible orbital speed at MIN_LEO_ALTITUDE (m/s).
MAX_ORBITAL_SPEED_MS: float = 8_200.0

#: Default sensor range-measurement tolerance (metres).
DEFAULT_RANGE_TOLERANCE_M: float = 500.0

#: Default maximum ΔV per timestep (m/s) — generous for chemical propulsion.
DEFAULT_MAX_DELTA_V_MS: float = 100.0

#: Maximum gap between consecutive claim timestamps before a replay/stale
#: data warning is raised (seconds).
MAX_TIMESTAMP_GAP_S: float = 60.0


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ThreatLevel(IntEnum):
    """Severity classification for detected anomalies."""
    CLEAN = 0           # no anomaly detected
    WARNING = 1         # suspicious; additional verification needed
    COMPROMISED = 2     # strong evidence of spoofing / Sybil attack
    REPLAY = 3          # timestamp violation (replay or stale packet)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SanityCheckResult:
    """
    Outcome of a single-packet sanity check.

    Attributes
    ----------
    satellite_id:
        The satellite whose claim was evaluated.
    threat_level:
        Overall threat assessment.
    reasons:
        Human-readable list of the reasons behind the threat level.
    """
    satellite_id: str
    threat_level: ThreatLevel = ThreatLevel.CLEAN
    reasons: List[str] = field(default_factory=list)

    @property
    def is_trusted(self) -> bool:
        """Return True if the check passed without raising an alarm."""
        return self.threat_level == ThreatLevel.CLEAN

    def summary(self) -> str:
        """One-line summary string."""
        tag = self.threat_level.name
        detail = "; ".join(self.reasons) if self.reasons else "OK"
        return f"SanityCheck({self.satellite_id!r}): {tag} — {detail}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _euclidean_distance(
    p1: Tuple[float, float, float],
    p2: Tuple[float, float, float],
) -> float:
    """Return the Euclidean distance between two 3-D points."""
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + (p1[2] - p2[2]) ** 2
    )


def _vector_magnitude(v: Tuple[float, float, float]) -> float:
    """Return the magnitude of 3-D vector *v*."""
    return math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)


# ---------------------------------------------------------------------------
# Real-time sanity checker
# ---------------------------------------------------------------------------

class PositionSanityChecker:
    """
    Per-satellite real-time sanity checker.

    Maintains a rolling history of the last accepted state vector for
    each peer so that velocity-continuity checks can be applied.

    Parameters
    ----------
    range_tolerance_m:
        Acceptable difference (metres) between a claimed range and a
        sensor-measured range before a COMPROMISED flag is raised.
    max_delta_v_ms:
        Maximum physically plausible speed change between two consecutive
        packets (m/s).  Exceeding this triggers a COMPROMISED flag.
    """

    def __init__(
        self,
        range_tolerance_m: float = DEFAULT_RANGE_TOLERANCE_M,
        max_delta_v_ms: float = DEFAULT_MAX_DELTA_V_MS,
    ) -> None:
        self._range_tolerance = range_tolerance_m
        self._max_delta_v = max_delta_v_ms
        # satellite_id -> (timestamp, position, velocity)
        self._last_state: Dict[
            str, Tuple[float, Tuple[float, float, float], Tuple[float, float, float]]
        ] = {}

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def check(
        self,
        satellite_id: str,
        timestamp: float,
        position: Tuple[float, float, float],
        velocity: Tuple[float, float, float],
        observer_position: Optional[Tuple[float, float, float]] = None,
        sensor_measured_range_m: Optional[float] = None,
    ) -> SanityCheckResult:
        """
        Evaluate a received state vector for authenticity.

        Parameters
        ----------
        satellite_id:
            ID of the broadcasting satellite.
        timestamp:
            GPS seconds since J2000 in the received packet.
        position:
            Claimed ECI position (x, y, z) in metres.
        velocity:
            Claimed ECI velocity (vx, vy, vz) in m/s.
        observer_position:
            ECI position of the receiving satellite (needed for range check).
        sensor_measured_range_m:
            Range to *satellite_id* as measured by onboard radar/lidar.
            Supply together with *observer_position* to enable sensor
            cross-check.

        Returns
        -------
        SanityCheckResult
        """
        result = SanityCheckResult(satellite_id=satellite_id)

        # 1. Physics plausibility — altitude and speed
        self._check_altitude(position, result)
        self._check_speed(velocity, result)

        # 2. Sensor cross-check (if ranging data available)
        if observer_position is not None and sensor_measured_range_m is not None:
            self._check_sensor_range(
                position, observer_position, sensor_measured_range_m, result
            )

        # 3. Timestamp and velocity continuity
        if satellite_id in self._last_state:
            self._check_continuity(
                satellite_id, timestamp, velocity, result
            )

        # 4. Update history only if no serious threat (not COMPROMISED or REPLAY)
        if result.threat_level not in (ThreatLevel.COMPROMISED, ThreatLevel.REPLAY):
            self._last_state[satellite_id] = (timestamp, position, velocity)

        return result

    # ------------------------------------------------------------------
    # Internal checks
    # ------------------------------------------------------------------

    def _check_altitude(
        self,
        position: Tuple[float, float, float],
        result: SanityCheckResult,
    ) -> None:
        r = _vector_magnitude(position)
        altitude = r - EARTH_RADIUS_M
        if altitude < MIN_LEO_ALTITUDE_M:
            result.threat_level = max(result.threat_level, ThreatLevel.COMPROMISED)
            result.reasons.append(
                f"altitude {altitude/1000:.1f} km below LEO minimum "
                f"({MIN_LEO_ALTITUDE_M/1000:.0f} km)"
            )
        elif altitude > MAX_LEO_ALTITUDE_M:
            result.threat_level = max(result.threat_level, ThreatLevel.WARNING)
            result.reasons.append(
                f"altitude {altitude/1000:.1f} km exceeds LEO maximum "
                f"({MAX_LEO_ALTITUDE_M/1000:.0f} km)"
            )

    def _check_speed(
        self,
        velocity: Tuple[float, float, float],
        result: SanityCheckResult,
    ) -> None:
        speed = _vector_magnitude(velocity)
        if speed < MIN_ORBITAL_SPEED_MS or speed > MAX_ORBITAL_SPEED_MS:
            result.threat_level = max(result.threat_level, ThreatLevel.COMPROMISED)
            result.reasons.append(
                f"orbital speed {speed:.1f} m/s outside LEO envelope "
                f"[{MIN_ORBITAL_SPEED_MS:.0f}, {MAX_ORBITAL_SPEED_MS:.0f}] m/s"
            )

    def _check_sensor_range(
        self,
        claimed_position: Tuple[float, float, float],
        observer_position: Tuple[float, float, float],
        measured_range_m: float,
        result: SanityCheckResult,
    ) -> None:
        claimed_range = _euclidean_distance(claimed_position, observer_position)
        discrepancy = abs(claimed_range - measured_range_m)
        if discrepancy > self._range_tolerance:
            result.threat_level = max(result.threat_level, ThreatLevel.COMPROMISED)
            result.reasons.append(
                f"sensor range mismatch: claimed {claimed_range/1000:.3f} km, "
                f"measured {measured_range_m/1000:.3f} km "
                f"(Δ={discrepancy/1000:.3f} km > "
                f"tolerance {self._range_tolerance/1000:.3f} km)"
            )

    def _check_continuity(
        self,
        satellite_id: str,
        timestamp: float,
        velocity: Tuple[float, float, float],
        result: SanityCheckResult,
    ) -> None:
        last_ts, _last_pos, last_vel = self._last_state[satellite_id]

        # Replay / stale-data check
        if timestamp <= last_ts:
            result.threat_level = max(result.threat_level, ThreatLevel.REPLAY)
            result.reasons.append(
                f"timestamp {timestamp:.3f} not strictly greater than "
                f"previous {last_ts:.3f} (possible replay)"
            )
            return

        # Velocity continuity: Δv between consecutive reports
        dv = (
            velocity[0] - last_vel[0],
            velocity[1] - last_vel[1],
            velocity[2] - last_vel[2],
        )
        delta_v = _vector_magnitude(dv)
        if delta_v > self._max_delta_v:
            result.threat_level = max(result.threat_level, ThreatLevel.COMPROMISED)
            result.reasons.append(
                f"velocity jump {delta_v:.2f} m/s exceeds max ΔV "
                f"{self._max_delta_v:.2f} m/s"
            )


# ---------------------------------------------------------------------------
# Ground-truth ledger
# ---------------------------------------------------------------------------

@dataclass
class GroundTruthEntry:
    """
    A single entry in the ground-truth ledger downloaded from a ground station.

    Parameters
    ----------
    satellite_id:
        Satellite identifier.
    position:
        Independently verified ECI position (metres).
    velocity:
        Independently verified ECI velocity (m/s).
    epoch_gps_s:
        Epoch of the ground-truth measurement (GPS seconds since J2000).
    source:
        Ground station or tracking network identifier.
    """
    satellite_id: str
    position: Tuple[float, float, float]
    velocity: Tuple[float, float, float]
    epoch_gps_s: float
    source: str = "GROUND"


class GroundTruthLedger:
    """
    Compressed ledger of ground-station-verified satellite positions.

    Ground stations periodically upload a snapshot of independently
    verified positions to each satellite.  The satellite uses this as an
    authoritative reference to cross-check what it hears from peers.

    Parameters
    ----------
    max_age_s:
        Entries older than this many seconds are considered stale and
        excluded from cross-checks.
    position_tolerance_m:
        Maximum acceptable difference (metres) between a peer's claimed
        position and the ground-truth entry before raising a WARNING.
    """

    def __init__(
        self,
        max_age_s: float = 300.0,
        position_tolerance_m: float = 5_000.0,
    ) -> None:
        self._entries: Dict[str, GroundTruthEntry] = {}
        self._max_age = max_age_s
        self._position_tolerance = position_tolerance_m

    def update(self, entry: GroundTruthEntry) -> None:
        """Insert or replace the ground-truth entry for a satellite."""
        self._entries[entry.satellite_id] = entry

    def bulk_update(self, entries: List[GroundTruthEntry]) -> None:
        """Insert or replace multiple ground-truth entries at once."""
        for entry in entries:
            self.update(entry)

    def cross_check(
        self,
        satellite_id: str,
        claimed_position: Tuple[float, float, float],
        current_gps_s: float,
    ) -> SanityCheckResult:
        """
        Cross-check *claimed_position* against the ground-truth ledger.

        Parameters
        ----------
        satellite_id:
            The satellite making the position claim.
        claimed_position:
            The position claimed by the satellite (metres, ECI).
        current_gps_s:
            Current GPS time (seconds since J2000), used to assess entry age.

        Returns
        -------
        SanityCheckResult
        """
        result = SanityCheckResult(satellite_id=satellite_id)
        entry = self._entries.get(satellite_id)

        if entry is None:
            result.threat_level = ThreatLevel.WARNING
            result.reasons.append("no ground-truth entry available for satellite")
            return result

        age = current_gps_s - entry.epoch_gps_s
        if age > self._max_age:
            result.threat_level = ThreatLevel.WARNING
            result.reasons.append(
                f"ground-truth entry is stale ({age:.1f} s old, "
                f"max {self._max_age:.1f} s)"
            )
            return result

        discrepancy = _euclidean_distance(claimed_position, entry.position)
        if discrepancy > self._position_tolerance:
            result.threat_level = ThreatLevel.COMPROMISED
            result.reasons.append(
                f"position discrepancy {discrepancy/1000:.3f} km vs ground "
                f"truth (tolerance {self._position_tolerance/1000:.3f} km)"
            )
        return result

    @property
    def entry_count(self) -> int:
        """Return the number of entries currently in the ledger."""
        return len(self._entries)
