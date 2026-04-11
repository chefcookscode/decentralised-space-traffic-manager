"""Tests for iscp.defense (Task 3.3 — Sybil and Spoofing Defense)."""

import pytest

from iscp.defense import (
    ThreatLevel,
    SanityCheckResult,
    PositionSanityChecker,
    GroundTruthEntry,
    GroundTruthLedger,
    EARTH_RADIUS_M,
    MIN_LEO_ALTITUDE_M,
    MAX_LEO_ALTITUDE_M,
    MIN_ORBITAL_SPEED_MS,
    MAX_ORBITAL_SPEED_MS,
    DEFAULT_RANGE_TOLERANCE_M,
)


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

# A standard LEO position: ~400 km altitude directly above equator
_LEO_POS = (EARTH_RADIUS_M + 400_000.0, 0.0, 0.0)
# Circular orbital speed at ~400 km altitude
_LEO_VEL = (0.0, 7_672.0, 0.0)


@pytest.fixture
def checker() -> PositionSanityChecker:
    return PositionSanityChecker()


# ---------------------------------------------------------------------------
# Clean pass
# ---------------------------------------------------------------------------

def test_clean_leo_state_is_trusted(checker: PositionSanityChecker):
    result = checker.check("SAT-A", 1000.0, _LEO_POS, _LEO_VEL)
    assert result.is_trusted is True
    assert result.threat_level == ThreatLevel.CLEAN


# ---------------------------------------------------------------------------
# Altitude checks
# ---------------------------------------------------------------------------

def test_below_minimum_altitude_is_compromised(checker: PositionSanityChecker):
    # Position inside Earth
    low_pos = (EARTH_RADIUS_M - 100_000.0, 0.0, 0.0)
    result = checker.check("SAT-A", 1000.0, low_pos, _LEO_VEL)
    assert result.threat_level == ThreatLevel.COMPROMISED
    assert any("altitude" in r for r in result.reasons)


def test_above_maximum_altitude_is_warning(checker: PositionSanityChecker):
    high_pos = (EARTH_RADIUS_M + MAX_LEO_ALTITUDE_M + 100_000.0, 0.0, 0.0)
    result = checker.check("SAT-A", 1000.0, high_pos, _LEO_VEL)
    assert result.threat_level >= ThreatLevel.WARNING


# ---------------------------------------------------------------------------
# Speed checks
# ---------------------------------------------------------------------------

def test_speed_too_low_is_compromised(checker: PositionSanityChecker):
    slow_vel = (0.0, 1_000.0, 0.0)  # far below orbital speed
    result = checker.check("SAT-A", 1000.0, _LEO_POS, slow_vel)
    assert result.threat_level == ThreatLevel.COMPROMISED


def test_speed_too_high_is_compromised(checker: PositionSanityChecker):
    fast_vel = (0.0, 20_000.0, 0.0)  # above escape velocity
    result = checker.check("SAT-A", 1000.0, _LEO_POS, fast_vel)
    assert result.threat_level == ThreatLevel.COMPROMISED


# ---------------------------------------------------------------------------
# Sensor range check
# ---------------------------------------------------------------------------

def test_range_match_stays_clean(checker: PositionSanityChecker):
    observer = (EARTH_RADIUS_M + 400_000.0, 100_000.0, 0.0)
    # Distance from _LEO_POS to observer
    import math
    true_range = math.sqrt(
        (observer[0] - _LEO_POS[0]) ** 2
        + (observer[1] - _LEO_POS[1]) ** 2
        + (observer[2] - _LEO_POS[2]) ** 2
    )
    result = checker.check(
        "SAT-A", 1000.0, _LEO_POS, _LEO_VEL,
        observer_position=observer,
        sensor_measured_range_m=true_range,
    )
    assert result.threat_level == ThreatLevel.CLEAN


def test_range_mismatch_is_compromised(checker: PositionSanityChecker):
    observer = (EARTH_RADIUS_M + 400_000.0, 0.0, 0.0)
    wrong_range = 999_999_999.0  # clearly wrong
    result = checker.check(
        "SAT-A", 1000.0, _LEO_POS, _LEO_VEL,
        observer_position=observer,
        sensor_measured_range_m=wrong_range,
    )
    assert result.threat_level == ThreatLevel.COMPROMISED
    assert any("sensor range" in r for r in result.reasons)


# ---------------------------------------------------------------------------
# Replay detection
# ---------------------------------------------------------------------------

def test_replay_detected_for_non_increasing_timestamp(checker: PositionSanityChecker):
    checker.check("SAT-A", 1000.0, _LEO_POS, _LEO_VEL)
    # Replay same or older timestamp
    result = checker.check("SAT-A", 1000.0, _LEO_POS, _LEO_VEL)
    assert result.threat_level == ThreatLevel.REPLAY


def test_replay_detected_for_older_timestamp(checker: PositionSanityChecker):
    checker.check("SAT-A", 2000.0, _LEO_POS, _LEO_VEL)
    result = checker.check("SAT-A", 1500.0, _LEO_POS, _LEO_VEL)
    assert result.threat_level == ThreatLevel.REPLAY


def test_increasing_timestamp_passes(checker: PositionSanityChecker):
    checker.check("SAT-A", 1000.0, _LEO_POS, _LEO_VEL)
    result = checker.check("SAT-A", 1001.0, _LEO_POS, _LEO_VEL)
    assert result.threat_level == ThreatLevel.CLEAN


# ---------------------------------------------------------------------------
# Velocity continuity check
# ---------------------------------------------------------------------------

def test_large_velocity_jump_is_compromised(checker: PositionSanityChecker):
    checker.check("SAT-A", 1000.0, _LEO_POS, _LEO_VEL)
    # Claimed velocity jumps by 1000 m/s — beyond default max_delta_v=100 m/s
    big_jump_vel = (_LEO_VEL[0], _LEO_VEL[1] + 1_000.0, _LEO_VEL[2])
    result = checker.check("SAT-A", 1001.0, _LEO_POS, big_jump_vel)
    assert result.threat_level == ThreatLevel.COMPROMISED


# ---------------------------------------------------------------------------
# SanityCheckResult
# ---------------------------------------------------------------------------

def test_sanity_result_summary_returns_string():
    r = SanityCheckResult(satellite_id="SAT-X")
    assert isinstance(r.summary(), str)


def test_clean_result_is_trusted():
    r = SanityCheckResult(satellite_id="SAT-X", threat_level=ThreatLevel.CLEAN)
    assert r.is_trusted is True


def test_compromised_result_is_not_trusted():
    r = SanityCheckResult(satellite_id="SAT-X", threat_level=ThreatLevel.COMPROMISED)
    assert r.is_trusted is False


# ---------------------------------------------------------------------------
# Ground-truth ledger
# ---------------------------------------------------------------------------

@pytest.fixture
def ledger() -> GroundTruthLedger:
    return GroundTruthLedger(max_age_s=300.0, position_tolerance_m=5_000.0)


def test_ledger_cross_check_clean(ledger: GroundTruthLedger):
    entry = GroundTruthEntry(
        satellite_id="SAT-B",
        position=_LEO_POS,
        velocity=_LEO_VEL,
        epoch_gps_s=1000.0,
    )
    ledger.update(entry)
    result = ledger.cross_check("SAT-B", _LEO_POS, current_gps_s=1010.0)
    assert result.threat_level == ThreatLevel.CLEAN


def test_ledger_cross_check_position_mismatch(ledger: GroundTruthLedger):
    entry = GroundTruthEntry(
        satellite_id="SAT-B",
        position=_LEO_POS,
        velocity=_LEO_VEL,
        epoch_gps_s=1000.0,
    )
    ledger.update(entry)
    far_away = (
        _LEO_POS[0] + 100_000.0,
        _LEO_POS[1] + 100_000.0,
        _LEO_POS[2],
    )
    result = ledger.cross_check("SAT-B", far_away, current_gps_s=1010.0)
    assert result.threat_level == ThreatLevel.COMPROMISED


def test_ledger_cross_check_stale_entry(ledger: GroundTruthLedger):
    entry = GroundTruthEntry(
        satellite_id="SAT-B",
        position=_LEO_POS,
        velocity=_LEO_VEL,
        epoch_gps_s=1000.0,
    )
    ledger.update(entry)
    # Check 400 seconds later — beyond max_age_s=300
    result = ledger.cross_check("SAT-B", _LEO_POS, current_gps_s=1400.0)
    assert result.threat_level == ThreatLevel.WARNING


def test_ledger_cross_check_missing_entry(ledger: GroundTruthLedger):
    result = ledger.cross_check("UNKNOWN-SAT", _LEO_POS, current_gps_s=1000.0)
    assert result.threat_level == ThreatLevel.WARNING


def test_ledger_entry_count(ledger: GroundTruthLedger):
    entries = [
        GroundTruthEntry("SAT-1", _LEO_POS, _LEO_VEL, 1000.0),
        GroundTruthEntry("SAT-2", _LEO_POS, _LEO_VEL, 1000.0),
    ]
    ledger.bulk_update(entries)
    assert ledger.entry_count == 2
