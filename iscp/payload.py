"""
ISCP Task 1.1 — Data Payload Schema
====================================
Defines the binary payload format for inter-satellite state and intent
broadcasts.  The packed representation must fit within 1 024 bytes
(sub-kilobyte constraint) so it can be carried over bandwidth-constrained
links without fragmentation.

Payload layout (big-endian):
  satellite_id       8s      8  bytes   null-padded ASCII identifier
  timestamp          d       8  bytes   GPS seconds since J2000 (float64)
  position_x/y/z     3d     24  bytes   ECI position in metres  (float64 × 3)
  velocity_x/y/z     3d     24  bytes   ECI velocity in m/s     (float64 × 3)
  covariance         21d   168  bytes   upper-triangle of 6×6 pos-vel
                                        covariance matrix (float64 × 21)
  mass               d       8  bytes   current wet mass in kg  (float64)
  propulsion_type    B       1  byte    PropulsionType enum value (uint8)
  maneuver_intent    —      49  bytes   ManeuverIntent sub-struct (see below)
                                ———
  Total             292  bytes  ✓ well under 1 024 bytes

ManeuverIntent sub-layout:
  intent_type        B       1  byte    ManeuverType enum value (uint8)
  delta_v_x/y/z      3d     24  bytes   burn delta-v vector in m/s (float64 × 3)
  burn_start_epoch   d       8  bytes   GPS seconds since J2000 (float64)
  burn_duration      d       8  bytes   burn duration in seconds  (float64)
  confidence         d       8  bytes   probability [0, 1]        (float64)
                                ———
  Sub-total          49  bytes
"""

import struct
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class PropulsionType(IntEnum):
    """Propulsion system classification."""
    UNKNOWN = 0
    CHEMICAL_MONOPROPELLANT = 1
    CHEMICAL_BIPROPELLANT = 2
    ELECTRIC_ION = 3
    ELECTRIC_HALL_EFFECT = 4
    COLD_GAS = 5
    SOLAR_SAIL = 6
    RESISTOJET = 7


class ManeuverType(IntEnum):
    """Intended orbital maneuver category."""
    NONE = 0             # no planned maneuver
    COLLISION_AVOIDANCE = 1
    STATION_KEEPING = 2
    ORBIT_RAISING = 3
    ORBIT_LOWERING = 4
    PLANE_CHANGE = 5
    DEORBIT = 6
    CONTINGENCY = 7


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_PAYLOAD_BYTES: int = 1024
SATELLITE_ID_LENGTH: int = 8  # bytes

# struct formats (big-endian '>')
_INTENT_FMT: str = ">B3ddd"          # 1 + 24 + 8 + 8 = 41 + 8 confidence
_INTENT_FMT_FULL: str = ">Bdddddd"   # 1 + 6 × 8 = 49 bytes
_PAYLOAD_FMT: str = ">8sdddddddd21dBBdddddd"
# Breakdown of _PAYLOAD_FMT:
#   8s   satellite_id            8
#   d    timestamp               8
#   ddd  position                24
#   ddd  velocity                24
#   21d  covariance              168
#   d    mass                    8
#   B    propulsion_type         1
#   B    intent_type             1
#   ddd  delta_v                 24
#   d    burn_start_epoch        8
#   d    burn_duration           8
#   d    confidence              8
#                              ———
#                              292 bytes

PAYLOAD_SIZE: int = struct.calcsize(_PAYLOAD_FMT)
assert PAYLOAD_SIZE <= MAX_PAYLOAD_BYTES, (
    f"Payload size {PAYLOAD_SIZE} exceeds {MAX_PAYLOAD_BYTES} bytes"
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ManeuverIntent:
    """Planned maneuver description broadcast by a satellite."""
    intent_type: ManeuverType = ManeuverType.NONE
    delta_v: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # m/s (ECI)
    burn_start_epoch: float = 0.0   # GPS seconds since J2000
    burn_duration: float = 0.0      # seconds
    confidence: float = 0.0         # probability [0.0 – 1.0]

    def validate(self) -> None:
        """Raise ValueError if fields are outside acceptable ranges."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence {self.confidence} is outside [0, 1]"
            )
        if self.burn_duration < 0.0:
            raise ValueError(
                f"burn_duration {self.burn_duration} must be non-negative"
            )


@dataclass
class ISCPPayload:
    """
    Full ISCP broadcast payload carrying satellite state and intent data.

    All position/velocity values use the Earth-Centred Inertial (ECI) frame
    with SI units (metres, m/s, kg).  The covariance matrix covers the 6-D
    state vector [x, y, z, vx, vy, vz] and is stored as the 21 unique
    elements of the upper triangle (row-major order).
    """
    satellite_id: str                              # ≤ 8 ASCII characters
    timestamp: float                               # GPS seconds since J2000
    position: Tuple[float, float, float]           # metres
    velocity: Tuple[float, float, float]           # m/s
    covariance: List[float]                        # 21 upper-triangle elements
    mass: float                                    # kg
    propulsion_type: PropulsionType = PropulsionType.UNKNOWN
    maneuver_intent: ManeuverIntent = field(
        default_factory=ManeuverIntent
    )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """Raise ValueError on invalid field values."""
        sid = self.satellite_id.encode("ascii", errors="replace")
        if len(sid) > SATELLITE_ID_LENGTH:
            raise ValueError(
                f"satellite_id '{self.satellite_id}' exceeds "
                f"{SATELLITE_ID_LENGTH} bytes when ASCII-encoded"
            )
        if len(self.covariance) != 21:
            raise ValueError(
                f"covariance must contain exactly 21 elements "
                f"(upper triangle of 6×6), got {len(self.covariance)}"
            )
        if self.mass < 0.0:
            raise ValueError(f"mass {self.mass} must be non-negative")
        self.maneuver_intent.validate()

    # ------------------------------------------------------------------
    # Serialisation / deserialisation
    # ------------------------------------------------------------------

    def pack(self) -> bytes:
        """
        Serialise the payload to a binary packet.

        Returns
        -------
        bytes
            Packed binary representation, always ≤ MAX_PAYLOAD_BYTES.

        Raises
        ------
        ValueError
            If any field fails validation.
        """
        self.validate()
        sid_bytes = self.satellite_id.encode("ascii", errors="replace")
        sid_bytes = sid_bytes.ljust(SATELLITE_ID_LENGTH, b"\x00")[
            :SATELLITE_ID_LENGTH
        ]
        intent = self.maneuver_intent
        return struct.pack(
            _PAYLOAD_FMT,
            sid_bytes,
            self.timestamp,
            *self.position,
            *self.velocity,
            *self.covariance,
            self.mass,
            int(self.propulsion_type),
            int(intent.intent_type),
            *intent.delta_v,
            intent.burn_start_epoch,
            intent.burn_duration,
            intent.confidence,
        )

    @classmethod
    def unpack(cls, data: bytes) -> "ISCPPayload":
        """
        Deserialise a binary packet produced by :meth:`pack`.

        Parameters
        ----------
        data:
            Raw bytes, exactly :data:`PAYLOAD_SIZE` bytes long.

        Returns
        -------
        ISCPPayload

        Raises
        ------
        ValueError
            If *data* has the wrong length.
        struct.error
            If the binary data cannot be unpacked.
        """
        if len(data) != PAYLOAD_SIZE:
            raise ValueError(
                f"Expected {PAYLOAD_SIZE} bytes, got {len(data)}"
            )
        fields = struct.unpack(_PAYLOAD_FMT, data)
        satellite_id = fields[0].rstrip(b"\x00").decode("ascii", errors="replace")
        timestamp = fields[1]
        position = (fields[2], fields[3], fields[4])
        velocity = (fields[5], fields[6], fields[7])
        covariance = list(fields[8:29])
        mass = fields[29]
        propulsion_type = PropulsionType(fields[30])
        intent = ManeuverIntent(
            intent_type=ManeuverType(fields[31]),
            delta_v=(fields[32], fields[33], fields[34]),
            burn_start_epoch=fields[35],
            burn_duration=fields[36],
            confidence=fields[37],
        )
        return cls(
            satellite_id=satellite_id,
            timestamp=timestamp,
            position=position,
            velocity=velocity,
            covariance=covariance,
            mass=mass,
            propulsion_type=propulsion_type,
            maneuver_intent=intent,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable single-line summary."""
        x, y, z = self.position
        vx, vy, vz = self.velocity
        return (
            f"ISCPPayload(id={self.satellite_id!r}, "
            f"t={self.timestamp:.3f}s, "
            f"pos=({x:.1f},{y:.1f},{z:.1f})m, "
            f"vel=({vx:.3f},{vy:.3f},{vz:.3f})m/s, "
            f"mass={self.mass:.1f}kg, "
            f"prop={self.propulsion_type.name}, "
            f"intent={self.maneuver_intent.intent_type.name})"
        )
