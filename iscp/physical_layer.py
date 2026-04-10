"""
ISCP Task 1.2 — Physical Layer & Frequency Specifications
==========================================================
Defines the hardware-layer specifications for the Inter-Satellite
Communication Protocol (ISCP).

Two communication modes are supported:

1. **Primary – Optical Inter-Satellite Link (OISL)**
   Narrow-beam laser link offering high throughput and inherent
   directionality that limits interference.  Requires pointing, acquisition,
   and tracking (PAT) subsystem.

2. **Fallback – Radio-Frequency (RF)**
   Omnidirectional RF beacon used when the primary laser link cannot be
   established (e.g., relative attitude uncertainty, acquisition failure,
   obscuration by Earth's limb).  Two sub-bands are specified:
     • S-band (2–4 GHz) — preferred RF fallback, moderate data rate
     • VHF (30–300 MHz) — last-resort resilient fallback, low data rate

Broadcast range requirement: satellites must begin broadcasting to
every peer within a **500 km** radius.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, auto
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class CommMode(IntEnum):
    """Active communication mode between two ISCP peers."""
    OFFLINE = 0       # not connected
    OPTICAL = 1       # primary: optical ISL (laser)
    RF_SBAND = 2      # fallback: S-band RF
    RF_VHF = 3        # last-resort fallback: VHF RF


class LinkState(IntEnum):
    """Current state of a physical link."""
    DOWN = 0
    ACQUIRING = 1
    LOCKED = 2
    DEGRADED = 3      # link up but BER above threshold


# ---------------------------------------------------------------------------
# Physical-layer constants
# ---------------------------------------------------------------------------

# Broadcast range ─ satellites within this distance (metres) are peers.
BROADCAST_RANGE_M: float = 500_000.0           # 500 km

# ── Optical ISL ─────────────────────────────────────────────────────────────
OPTICAL_WAVELENGTH_NM: float = 1_550.0          # nm  (telecom C-band)
OPTICAL_DATA_RATE_BPS: int = 10_000_000_000     # 10 Gbps peak
OPTICAL_DIVERGENCE_RAD: float = 10e-6           # 10 µrad beam divergence
OPTICAL_TX_POWER_W: float = 1.0                 # 1 W transmit power
OPTICAL_POINTING_ACCURACY_URAD: float = 2.0     # ≤ 2 µrad pointing error
OPTICAL_MAX_RANGE_M: float = 5_000_000.0        # 5 000 km maximum range

# ── S-band RF ────────────────────────────────────────────────────────────────
SBAND_FREQ_HZ_MIN: float = 2_000e6             # 2 GHz
SBAND_FREQ_HZ_MAX: float = 4_000e6             # 4 GHz
SBAND_CENTER_FREQ_HZ: float = 2_400e6          # 2.4 GHz (ISCP default)
SBAND_CHANNEL_BW_HZ: float = 2e6               # 2 MHz channel bandwidth
SBAND_DATA_RATE_BPS: int = 1_000_000           # 1 Mbps
SBAND_TX_POWER_W: float = 2.0                  # 2 W transmit power
SBAND_MAX_RANGE_M: float = 3_000_000.0         # 3 000 km maximum range

# ── VHF RF ───────────────────────────────────────────────────────────────────
VHF_FREQ_HZ_MIN: float = 30e6                  # 30 MHz
VHF_FREQ_HZ_MAX: float = 300e6                 # 300 MHz
VHF_CENTER_FREQ_HZ: float = 137e6              # 137 MHz (ISCP beacon)
VHF_CHANNEL_BW_HZ: float = 25_000.0            # 25 kHz channel bandwidth
VHF_DATA_RATE_BPS: int = 9_600                 # 9.6 kbps
VHF_TX_POWER_W: float = 5.0                    # 5 W transmit power
VHF_MAX_RANGE_M: float = BROADCAST_RANGE_M     # 500 km (omnidirectional)

# Maximum acceptable bit-error rate on a usable link.
MAX_BER: float = 1e-6


# ---------------------------------------------------------------------------
# Helper: select preferred mode
# ---------------------------------------------------------------------------

def select_mode(
    range_m: float,
    optical_available: bool = True,
    sband_available: bool = True,
) -> CommMode:
    """
    Choose the best communication mode given a peer distance and
    available hardware.

    Parameters
    ----------
    range_m:
        Distance to the peer satellite in metres.
    optical_available:
        Whether the optical PAT subsystem is functional and has acquired
        the peer.
    sband_available:
        Whether the S-band transceiver is functional.

    Returns
    -------
    CommMode
        Best available mode, or :attr:`CommMode.OFFLINE` if the peer is
        beyond all link budgets.

    Raises
    ------
    ValueError
        If *range_m* is negative.
    """
    if range_m < 0:
        raise ValueError(f"range_m must be non-negative, got {range_m}")

    if optical_available and range_m <= OPTICAL_MAX_RANGE_M:
        return CommMode.OPTICAL
    if sband_available and range_m <= SBAND_MAX_RANGE_M:
        return CommMode.RF_SBAND
    if range_m <= VHF_MAX_RANGE_M:
        return CommMode.RF_VHF
    return CommMode.OFFLINE


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class LinkBudget:
    """Estimated link-budget parameters for a given mode and range."""
    mode: CommMode
    range_m: float
    tx_power_w: float
    estimated_data_rate_bps: int
    estimated_ber: float
    usable: bool = field(init=False)

    def __post_init__(self) -> None:
        self.usable = self.estimated_ber <= MAX_BER

    def summary(self) -> str:
        status = "OK" if self.usable else "DEGRADED"
        return (
            f"LinkBudget({self.mode.name}, {self.range_m / 1000:.1f} km, "
            f"{self.estimated_data_rate_bps // 1000} kbps, "
            f"BER={self.estimated_ber:.2e}, {status})"
        )


@dataclass
class PhysicalLayerSpec:
    """
    Consolidated ISCP physical-layer specification.

    Carries the normative values for frequency, power, bandwidth, and range
    for both the optical ISL and the two RF fallback links.  Instances of
    this class should be treated as immutable reference data.
    """
    # Optical ISL
    optical_wavelength_nm: float = OPTICAL_WAVELENGTH_NM
    optical_data_rate_bps: int = OPTICAL_DATA_RATE_BPS
    optical_divergence_rad: float = OPTICAL_DIVERGENCE_RAD
    optical_tx_power_w: float = OPTICAL_TX_POWER_W
    optical_max_range_m: float = OPTICAL_MAX_RANGE_M

    # S-band RF
    sband_center_freq_hz: float = SBAND_CENTER_FREQ_HZ
    sband_channel_bw_hz: float = SBAND_CHANNEL_BW_HZ
    sband_data_rate_bps: int = SBAND_DATA_RATE_BPS
    sband_tx_power_w: float = SBAND_TX_POWER_W
    sband_max_range_m: float = SBAND_MAX_RANGE_M

    # VHF RF
    vhf_center_freq_hz: float = VHF_CENTER_FREQ_HZ
    vhf_channel_bw_hz: float = VHF_CHANNEL_BW_HZ
    vhf_data_rate_bps: int = VHF_DATA_RATE_BPS
    vhf_tx_power_w: float = VHF_TX_POWER_W

    # Broadcast range (applies to all modes; peers within this radius
    # must be contactable via at least one mode)
    broadcast_range_m: float = BROADCAST_RANGE_M

    def data_rate_bps(self, mode: CommMode) -> Optional[int]:
        """Return the nominal data rate for *mode*, or None if offline."""
        return {
            CommMode.OPTICAL: self.optical_data_rate_bps,
            CommMode.RF_SBAND: self.sband_data_rate_bps,
            CommMode.RF_VHF: self.vhf_data_rate_bps,
        }.get(mode)

    def max_range_m(self, mode: CommMode) -> Optional[float]:
        """Return the maximum link range for *mode*, or None if offline."""
        return {
            CommMode.OPTICAL: self.optical_max_range_m,
            CommMode.RF_SBAND: self.sband_max_range_m,
            CommMode.RF_VHF: self.broadcast_range_m,
        }.get(mode)

    def summary(self) -> str:
        lines = [
            "ISCP Physical Layer Specification",
            "─" * 40,
            f"Broadcast range      : {self.broadcast_range_m / 1000:.0f} km",
            "",
            "PRIMARY – Optical ISL",
            f"  Wavelength         : {self.optical_wavelength_nm:.0f} nm",
            f"  Data rate          : {self.optical_data_rate_bps // 1_000_000_000:.0f} Gbps",
            f"  Beam divergence    : {self.optical_divergence_rad * 1e6:.0f} µrad",
            f"  TX power           : {self.optical_tx_power_w:.1f} W",
            f"  Max range          : {self.optical_max_range_m / 1000:.0f} km",
            "",
            "FALLBACK – S-band RF",
            f"  Centre frequency   : {self.sband_center_freq_hz / 1e9:.2f} GHz",
            f"  Channel bandwidth  : {self.sband_channel_bw_hz / 1e6:.1f} MHz",
            f"  Data rate          : {self.sband_data_rate_bps // 1000:.0f} kbps",
            f"  TX power           : {self.sband_tx_power_w:.1f} W",
            f"  Max range          : {self.sband_max_range_m / 1000:.0f} km",
            "",
            "LAST-RESORT FALLBACK – VHF RF",
            f"  Centre frequency   : {self.vhf_center_freq_hz / 1e6:.0f} MHz",
            f"  Channel bandwidth  : {self.vhf_channel_bw_hz / 1000:.1f} kHz",
            f"  Data rate          : {self.vhf_data_rate_bps / 1000:.1f} kbps",
            f"  TX power           : {self.vhf_tx_power_w:.1f} W",
            f"  Max range (omni)   : {self.broadcast_range_m / 1000:.0f} km",
        ]
        return "\n".join(lines)


# Singleton convenience instance — importers may use this directly.
DEFAULT_SPEC = PhysicalLayerSpec()
