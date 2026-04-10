"""
ISCP Task 1.3 — Handshake and Synchronisation Logic
=====================================================
Implements the rules for establishing an ISCP session between two satellites
crossing orbital planes at relative speeds that may exceed 10 km/s, and the
GPS/GNSS-based time-synchronisation protocol that ensures state vectors can
be accurately compared.

Handshake state machine
-----------------------
   ┌────────┐  HELLO sent   ┌──────────┐  HELLO_ACK rcvd  ┌─────────────┐
   │  IDLE  │ ─────────────►│ INIT_SENT│ ────────────────►│  CHALLENGE  │
   └────────┘               └──────────┘                   │    _SENT    │
        ▲                                                   └──────┬──────┘
        │ CLOSE / error                                            │ CHALLENGE_ACK rcvd
        │                                                          ▼
   ┌────┴───────┐                                         ┌───────────────┐
   │   CLOSED   │◄────────────────────────────────────────│  ESTABLISHED  │
   └────────────┘  CLOSE / timeout / error                └───────────────┘

Time-synchronisation
--------------------
ISCP requires all participants to maintain GPS/GNSS time with millisecond
accuracy.  Each HELLO message carries the sender's current GPS epoch (seconds
since J2000 = 2000-01-01T11:58:55.816 UTC).  On receiving a HELLO, the peer
computes the clock-offset and applies it as a correction before comparing
state vectors.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Satellites crossing at relative speeds above this threshold are treated as
# high-priority: the handshake timeout is shortened and state broadcasts are
# increased in frequency.
HIGH_RELATIVE_SPEED_THRESHOLD_MS: float = 10_000.0      # 10 km/s in m/s

# Maximum one-way clock offset (seconds) before we flag a sync failure.
MAX_CLOCK_OFFSET_S: float = 0.001                        # 1 ms

# Handshake timeouts (seconds)
HANDSHAKE_TIMEOUT_NORMAL_S: float = 5.0
HANDSHAKE_TIMEOUT_HIGH_SPEED_S: float = 1.0             # shortened for fast crossings

# J2000 epoch offset from Unix epoch (seconds)
# J2000 = 2000-01-01T11:58:55.816 UTC
J2000_UNIX_OFFSET_S: float = 946_727_935.816

# Minimum data required to attempt a handshake
ISCP_PROTOCOL_VERSION: int = 1


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class HandshakeState(IntEnum):
    """States of the ISCP connection-establishment finite-state machine."""
    IDLE = 0
    INIT_SENT = 1
    CHALLENGE_SENT = 2
    ESTABLISHED = 3
    CLOSED = 4
    ERROR = 5


class CloseReason(IntEnum):
    """Reason a session was closed or rejected."""
    NORMAL = 0
    TIMEOUT = 1
    VERSION_MISMATCH = 2
    CLOCK_SKEW = 3
    CHALLENGE_FAILED = 4
    INTERNAL_ERROR = 5


# ---------------------------------------------------------------------------
# Message types
# ---------------------------------------------------------------------------

@dataclass
class HelloMessage:
    """
    First message sent by an ISCP node to a newly discovered peer.

    Contains enough information for the peer to decide whether to accept
    the session and to begin clock-offset estimation.
    """
    sender_id: str
    protocol_version: int = ISCP_PROTOCOL_VERSION
    gps_epoch_s: float = field(default_factory=lambda: unix_to_gps(time.time()))
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # ECI metres
    velocity: Tuple[float, float, float] = (0.0, 0.0, 0.0)   # ECI m/s


@dataclass
class HelloAckMessage:
    """
    Acknowledgement sent by the responder after receiving a HelloMessage.

    Echoes the receiver's own GPS epoch so the initiator can also compute
    the clock offset.
    """
    sender_id: str
    receiver_id: str
    accepted: bool
    close_reason: Optional[CloseReason] = None
    gps_epoch_s: float = field(default_factory=lambda: unix_to_gps(time.time()))
    protocol_version: int = ISCP_PROTOCOL_VERSION


@dataclass
class ChallengeMessage:
    """
    Sent by the initiator after receiving HelloAck to prove liveness and
    confirm clock sync.
    """
    sender_id: str
    receiver_id: str
    challenge_token: int = 0    # 32-bit nonce
    gps_epoch_s: float = field(default_factory=lambda: unix_to_gps(time.time()))


@dataclass
class ChallengeAckMessage:
    """Response to a ChallengeMessage; completes the handshake."""
    sender_id: str
    receiver_id: str
    challenge_token: int = 0    # must mirror the received token
    accepted: bool = True
    clock_offset_s: float = 0.0


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

def unix_to_gps(unix_ts: float) -> float:
    """Convert a Unix timestamp (seconds since 1970-01-01) to GPS seconds
    since J2000 (2000-01-01T11:58:55.816 UTC)."""
    return unix_ts - J2000_UNIX_OFFSET_S


def gps_to_unix(gps_ts: float) -> float:
    """Convert GPS seconds since J2000 back to a Unix timestamp."""
    return gps_ts + J2000_UNIX_OFFSET_S


def compute_clock_offset(local_gps_s: float, remote_gps_s: float) -> float:
    """
    Estimate the clock offset (seconds) between the local clock and a remote
    peer's clock.

    A positive result means the remote clock is ahead of the local clock.

    Parameters
    ----------
    local_gps_s:
        Local GPS epoch at the moment the remote message was received.
    remote_gps_s:
        GPS epoch carried inside the remote message.

    Returns
    -------
    float
        Estimated clock offset in seconds.
    """
    return remote_gps_s - local_gps_s


def is_clock_in_sync(offset_s: float) -> bool:
    """Return True if *offset_s* is within the tolerated skew."""
    return abs(offset_s) <= MAX_CLOCK_OFFSET_S


# ---------------------------------------------------------------------------
# Relative-speed helpers
# ---------------------------------------------------------------------------

def relative_speed_ms(
    vel_a: Tuple[float, float, float],
    vel_b: Tuple[float, float, float],
) -> float:
    """
    Compute the relative speed (scalar, m/s) between two ECI velocity vectors.

    Parameters
    ----------
    vel_a, vel_b:
        ECI velocity vectors in m/s.

    Returns
    -------
    float
        Relative speed in m/s.
    """
    dvx = vel_a[0] - vel_b[0]
    dvy = vel_a[1] - vel_b[1]
    dvz = vel_a[2] - vel_b[2]
    return math.sqrt(dvx ** 2 + dvy ** 2 + dvz ** 2)


def is_high_speed_crossing(
    vel_a: Tuple[float, float, float],
    vel_b: Tuple[float, float, float],
) -> bool:
    """Return True if the relative speed exceeds the high-speed threshold."""
    return relative_speed_ms(vel_a, vel_b) > HIGH_RELATIVE_SPEED_THRESHOLD_MS


# ---------------------------------------------------------------------------
# Session object
# ---------------------------------------------------------------------------

@dataclass
class ISCPSession:
    """
    Represents an ISCP communication session between a local satellite and
    one peer satellite.

    Drives the handshake FSM and tracks synchronisation state.
    """
    local_id: str
    peer_id: str
    state: HandshakeState = HandshakeState.IDLE
    clock_offset_s: float = 0.0
    relative_speed_ms: float = 0.0
    _challenge_token: int = field(default=0, init=False, repr=False)
    close_reason: Optional[CloseReason] = None
    established_at: Optional[float] = None  # GPS seconds since J2000

    # ------------------------------------------------------------------
    # Handshake steps
    # ------------------------------------------------------------------

    def initiate(
        self,
        local_gps_s: float,
        local_position: Tuple[float, float, float],
        local_velocity: Tuple[float, float, float],
    ) -> HelloMessage:
        """
        Begin the handshake by constructing a HelloMessage to send to the peer.

        Transitions the FSM from IDLE → INIT_SENT.

        Parameters
        ----------
        local_gps_s:
            Current local GPS epoch (seconds since J2000).
        local_position:
            ECI position in metres.
        local_velocity:
            ECI velocity in m/s.

        Returns
        -------
        HelloMessage
            The message to transmit to the peer.

        Raises
        ------
        RuntimeError
            If the session is not in the IDLE state.
        """
        if self.state != HandshakeState.IDLE:
            raise RuntimeError(
                f"initiate() called in state {self.state.name}; expected IDLE"
            )
        self.state = HandshakeState.INIT_SENT
        return HelloMessage(
            sender_id=self.local_id,
            gps_epoch_s=local_gps_s,
            position=local_position,
            velocity=local_velocity,
        )

    def receive_hello(
        self,
        msg: HelloMessage,
        local_gps_s: float,
        local_velocity: Tuple[float, float, float],
    ) -> HelloAckMessage:
        """
        Process an incoming HelloMessage (responder side).

        Validates the protocol version, estimates clock offset, and checks
        relative speed to determine the appropriate response.

        Parameters
        ----------
        msg:
            The received HelloMessage.
        local_gps_s:
            Local GPS epoch at the time of receipt.
        local_velocity:
            ECI velocity of the local satellite in m/s.

        Returns
        -------
        HelloAckMessage
            Response to send back to the initiator.
        """
        if self.state not in (HandshakeState.IDLE, HandshakeState.INIT_SENT):
            return HelloAckMessage(
                sender_id=self.local_id,
                receiver_id=msg.sender_id,
                accepted=False,
                close_reason=CloseReason.INTERNAL_ERROR,
            )

        if msg.protocol_version != ISCP_PROTOCOL_VERSION:
            self.state = HandshakeState.CLOSED
            self.close_reason = CloseReason.VERSION_MISMATCH
            return HelloAckMessage(
                sender_id=self.local_id,
                receiver_id=msg.sender_id,
                accepted=False,
                close_reason=CloseReason.VERSION_MISMATCH,
            )

        offset = compute_clock_offset(local_gps_s, msg.gps_epoch_s)
        self.clock_offset_s = offset

        if not is_clock_in_sync(offset):
            self.state = HandshakeState.CLOSED
            self.close_reason = CloseReason.CLOCK_SKEW
            return HelloAckMessage(
                sender_id=self.local_id,
                receiver_id=msg.sender_id,
                accepted=False,
                close_reason=CloseReason.CLOCK_SKEW,
            )

        self.relative_speed_ms = relative_speed_ms(local_velocity, msg.velocity)
        self.state = HandshakeState.INIT_SENT
        return HelloAckMessage(
            sender_id=self.local_id,
            receiver_id=msg.sender_id,
            accepted=True,
            gps_epoch_s=local_gps_s,
        )

    def receive_hello_ack(
        self,
        msg: HelloAckMessage,
        local_gps_s: float,
    ) -> Optional[ChallengeMessage]:
        """
        Process an incoming HelloAckMessage (initiator side).

        If the peer accepted, generate a ChallengeMessage to finalise the
        handshake.  If the peer rejected, transition to CLOSED/ERROR.

        Returns
        -------
        ChallengeMessage or None
            ChallengeMessage to transmit, or None if the session was rejected.
        """
        if self.state != HandshakeState.INIT_SENT:
            return None

        if not msg.accepted:
            self.state = HandshakeState.CLOSED
            self.close_reason = msg.close_reason or CloseReason.NORMAL
            return None

        offset = compute_clock_offset(local_gps_s, msg.gps_epoch_s)
        self.clock_offset_s = offset

        import random
        self._challenge_token = random.getrandbits(32)
        self.state = HandshakeState.CHALLENGE_SENT
        return ChallengeMessage(
            sender_id=self.local_id,
            receiver_id=self.peer_id,
            challenge_token=self._challenge_token,
            gps_epoch_s=local_gps_s,
        )

    def receive_challenge(
        self,
        msg: ChallengeMessage,
        local_gps_s: float,
    ) -> ChallengeAckMessage:
        """
        Process an incoming ChallengeMessage (responder side).

        Mirrors the token and finalises clock offset, then transitions to
        ESTABLISHED.

        Returns
        -------
        ChallengeAckMessage
        """
        offset = compute_clock_offset(local_gps_s, msg.gps_epoch_s)
        self.clock_offset_s = (self.clock_offset_s + offset) / 2.0

        if not is_clock_in_sync(self.clock_offset_s):
            self.state = HandshakeState.CLOSED
            self.close_reason = CloseReason.CLOCK_SKEW
            return ChallengeAckMessage(
                sender_id=self.local_id,
                receiver_id=msg.sender_id,
                challenge_token=msg.challenge_token,
                accepted=False,
                clock_offset_s=self.clock_offset_s,
            )

        self.state = HandshakeState.ESTABLISHED
        self.established_at = local_gps_s
        return ChallengeAckMessage(
            sender_id=self.local_id,
            receiver_id=msg.sender_id,
            challenge_token=msg.challenge_token,
            accepted=True,
            clock_offset_s=self.clock_offset_s,
        )

    def receive_challenge_ack(self, msg: ChallengeAckMessage) -> bool:
        """
        Process the final ChallengeAckMessage (initiator side).

        Transitions to ESTABLISHED on success, CLOSED on failure.

        Returns
        -------
        bool
            True if the session is now ESTABLISHED.
        """
        if self.state != HandshakeState.CHALLENGE_SENT:
            return False

        if not msg.accepted or msg.challenge_token != self._challenge_token:
            self.state = HandshakeState.CLOSED
            self.close_reason = CloseReason.CHALLENGE_FAILED
            return False

        self.clock_offset_s = msg.clock_offset_s
        self.state = HandshakeState.ESTABLISHED
        self.established_at = unix_to_gps(time.time())
        return True

    def close(self, reason: CloseReason = CloseReason.NORMAL) -> None:
        """Tear down the session."""
        self.state = HandshakeState.CLOSED
        self.close_reason = reason

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @property
    def handshake_timeout_s(self) -> float:
        """Return the applicable handshake timeout given relative speed."""
        if self.relative_speed_ms > HIGH_RELATIVE_SPEED_THRESHOLD_MS:
            return HANDSHAKE_TIMEOUT_HIGH_SPEED_S
        return HANDSHAKE_TIMEOUT_NORMAL_S

    @property
    def is_established(self) -> bool:
        """Return True if the session is fully established."""
        return self.state == HandshakeState.ESTABLISHED

    def correct_timestamp(self, remote_ts: float) -> float:
        """
        Apply the stored clock offset to convert a remote GPS timestamp to
        the local time-reference frame.

        Parameters
        ----------
        remote_ts:
            GPS epoch received from the peer satellite.

        Returns
        -------
        float
            Corrected GPS epoch in the local time-reference frame.
        """
        return remote_ts - self.clock_offset_s

    def summary(self) -> str:
        speed_flag = (
            " [HIGH-SPEED CROSSING]"
            if self.relative_speed_ms > HIGH_RELATIVE_SPEED_THRESHOLD_MS
            else ""
        )
        return (
            f"ISCPSession({self.local_id!r} ↔ {self.peer_id!r}, "
            f"state={self.state.name}, "
            f"clock_offset={self.clock_offset_s * 1000:.3f} ms, "
            f"rel_speed={self.relative_speed_ms / 1000:.2f} km/s"
            f"{speed_flag})"
        )
