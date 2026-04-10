"""Tests for iscp.right_of_way (Task 2 — Right of Way Rules / Decision Matrix)."""

import math
import pytest

from iscp.payload import PropulsionType
from iscp.right_of_way import (
    ManeuverabilityCategory,
    ManeuverabilityProfile,
    ManeuverCostWeights,
    MissionPriority,
    ConjunctionProfile,
    ResolutionReason,
    RightOfWayDecision,
    PROPULSION_MANEUVERABILITY_MAP,
    MANEUVERABILITY_MIN_LEAD_TIME_S,
    compute_maneuver_cost,
    resolve_right_of_way,
)


# ===========================================================================
# Task 2.1 — Mission Priority Tiering
# ===========================================================================

def test_human_spaceflight_has_highest_priority():
    """HUMAN_SPACEFLIGHT must be the numerically lowest (most privileged) tier."""
    assert MissionPriority.HUMAN_SPACEFLIGHT < MissionPriority.UNMANOEUVRABLE
    assert MissionPriority.HUMAN_SPACEFLIGHT < MissionPriority.NATIONAL_SECURITY
    assert MissionPriority.HUMAN_SPACEFLIGHT < MissionPriority.ACTIVE_COMMERCIAL
    assert MissionPriority.HUMAN_SPACEFLIGHT < MissionPriority.END_OF_LIFE


def test_priority_ordering_is_strictly_monotone():
    """Each tier must have a strictly higher numeric value than the tier above it."""
    ordered = [
        MissionPriority.HUMAN_SPACEFLIGHT,
        MissionPriority.UNMANOEUVRABLE,
        MissionPriority.NATIONAL_SECURITY,
        MissionPriority.ACTIVE_COMMERCIAL,
        MissionPriority.END_OF_LIFE,
    ]
    for i in range(len(ordered) - 1):
        assert ordered[i] < ordered[i + 1]


def test_all_five_priority_tiers_defined():
    tiers = set(MissionPriority)
    assert len(tiers) == 5


# ===========================================================================
# Task 2.2 — Propulsion and Manoeuvrability Constraints
# ===========================================================================

def test_chemical_propulsion_maps_to_high_thrust():
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.CHEMICAL_MONOPROPELLANT]
        == ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE
    )
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.CHEMICAL_BIPROPELLANT]
        == ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE
    )


def test_electric_propulsion_maps_to_low_thrust_slow():
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.ELECTRIC_ION]
        == ManeuverabilityCategory.LOW_THRUST_SLOW
    )
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.ELECTRIC_HALL_EFFECT]
        == ManeuverabilityCategory.LOW_THRUST_SLOW
    )


def test_cold_gas_and_resistojet_map_to_medium_thrust():
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.COLD_GAS]
        == ManeuverabilityCategory.MEDIUM_THRUST
    )
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.RESISTOJET]
        == ManeuverabilityCategory.MEDIUM_THRUST
    )


def test_solar_sail_and_unknown_map_to_non_manoeuvrable():
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.SOLAR_SAIL]
        == ManeuverabilityCategory.NON_MANOEUVRABLE
    )
    assert (
        PROPULSION_MANEUVERABILITY_MAP[PropulsionType.UNKNOWN]
        == ManeuverabilityCategory.NON_MANOEUVRABLE
    )


def test_all_propulsion_types_are_covered():
    """Every PropulsionType must have a mapping in PROPULSION_MANEUVERABILITY_MAP."""
    for pt in PropulsionType:
        assert pt in PROPULSION_MANEUVERABILITY_MAP, (
            f"PropulsionType.{pt.name} is missing from PROPULSION_MANEUVERABILITY_MAP"
        )


def test_non_manoeuvrable_lead_time_is_infinite():
    assert math.isinf(
        MANEUVERABILITY_MIN_LEAD_TIME_S[ManeuverabilityCategory.NON_MANOEUVRABLE]
    )


def test_lead_times_increase_with_category():
    """Slower propulsion systems require more lead time."""
    high = MANEUVERABILITY_MIN_LEAD_TIME_S[ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE]
    med = MANEUVERABILITY_MIN_LEAD_TIME_S[ManeuverabilityCategory.MEDIUM_THRUST]
    low = MANEUVERABILITY_MIN_LEAD_TIME_S[ManeuverabilityCategory.LOW_THRUST_SLOW]
    assert high < med < low


def test_maneuverability_profile_defaults_min_lead_time():
    """When min_lead_time_s is not supplied, it is derived from propulsion type."""
    profile = ManeuverabilityProfile(
        propulsion_type=PropulsionType.CHEMICAL_BIPROPELLANT,
        available_delta_v_ms=50.0,
    )
    expected = MANEUVERABILITY_MIN_LEAD_TIME_S[ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE]
    assert profile.min_lead_time_s == expected


def test_maneuverability_profile_can_override_lead_time():
    profile = ManeuverabilityProfile(
        propulsion_type=PropulsionType.ELECTRIC_ION,
        available_delta_v_ms=20.0,
        min_lead_time_s=3600.0,
    )
    assert profile.min_lead_time_s == 3600.0


def test_can_manoeuvre_is_false_for_non_manoeuvrable():
    profile = ManeuverabilityProfile(
        propulsion_type=PropulsionType.UNKNOWN,
        available_delta_v_ms=0.0,
    )
    assert not profile.can_manoeuvre


def test_can_manoeuvre_is_true_for_chemical_propulsion():
    profile = ManeuverabilityProfile(
        propulsion_type=PropulsionType.CHEMICAL_MONOPROPELLANT,
        available_delta_v_ms=100.0,
    )
    assert profile.can_manoeuvre


def test_maneuverability_profile_rejects_negative_delta_v():
    with pytest.raises(ValueError, match="available_delta_v_ms"):
        ManeuverabilityProfile(
            propulsion_type=PropulsionType.COLD_GAS,
            available_delta_v_ms=-1.0,
        )


# ===========================================================================
# Task 2.3 — Cost-of-Manoeuvre Optimisation Function
# ===========================================================================

def test_compute_maneuver_cost_basic():
    weights = ManeuverCostWeights(w1=1.0, w2=0.01, w3=1000.0)
    # J = 1.0 * 2.0 + 0.01 * 60.0 + 1000.0 * 0.001 = 2.0 + 0.6 + 1.0 = 3.6
    cost = compute_maneuver_cost(
        delta_v_ms=2.0, outage_s=60.0, pc_post=0.001, weights=weights
    )
    assert cost == pytest.approx(3.6)


def test_compute_maneuver_cost_zero_inputs():
    cost = compute_maneuver_cost(0.0, 0.0, 0.0)
    assert cost == 0.0


def test_compute_maneuver_cost_rejects_negative_delta_v():
    with pytest.raises(ValueError, match="delta_v_ms"):
        compute_maneuver_cost(-1.0, 0.0, 0.0)


def test_compute_maneuver_cost_rejects_negative_outage():
    with pytest.raises(ValueError, match="outage_s"):
        compute_maneuver_cost(0.0, -10.0, 0.0)


def test_compute_maneuver_cost_rejects_pc_above_one():
    with pytest.raises(ValueError, match="pc_post"):
        compute_maneuver_cost(0.0, 0.0, 1.5)


def test_compute_maneuver_cost_rejects_pc_below_zero():
    with pytest.raises(ValueError, match="pc_post"):
        compute_maneuver_cost(0.0, 0.0, -0.1)


def test_maneuver_cost_weights_rejects_negative_weight():
    with pytest.raises(ValueError, match="w1"):
        ManeuverCostWeights(w1=-1.0, w2=0.01, w3=1000.0)


def test_conjunction_profile_cost_is_inf_for_non_manoeuvrable():
    profile = ConjunctionProfile(
        satellite_id="DEBRIS-1",
        mission_priority=MissionPriority.UNMANOEUVRABLE,
        maneuverability=ManeuverabilityProfile(
            propulsion_type=PropulsionType.UNKNOWN,
        ),
        required_delta_v_ms=0.0,
        service_outage_s=0.0,
        pc_post_maneuver=0.0,
    )
    assert math.isinf(profile.maneuver_cost())


def test_conjunction_profile_rejects_negative_delta_v():
    with pytest.raises(ValueError, match="required_delta_v_ms"):
        ConjunctionProfile(
            satellite_id="SAT-A",
            mission_priority=MissionPriority.ACTIVE_COMMERCIAL,
            maneuverability=ManeuverabilityProfile(PropulsionType.ELECTRIC_ION),
            required_delta_v_ms=-5.0,
        )


def test_conjunction_profile_rejects_invalid_pc():
    with pytest.raises(ValueError, match="pc_post_maneuver"):
        ConjunctionProfile(
            satellite_id="SAT-A",
            mission_priority=MissionPriority.ACTIVE_COMMERCIAL,
            maneuverability=ManeuverabilityProfile(PropulsionType.ELECTRIC_ION),
            pc_post_maneuver=2.0,
        )


# ===========================================================================
# Task 2.4 — Deterministic Resolution Logic
# ===========================================================================

# --- Fixtures -----------------------------------------------------------------

def _make_profile(
    satellite_id: str,
    priority: MissionPriority,
    propulsion: PropulsionType,
    delta_v: float = 1.0,
    outage: float = 60.0,
    pc_post: float = 0.001,
    available_dv: float = 100.0,
) -> ConjunctionProfile:
    return ConjunctionProfile(
        satellite_id=satellite_id,
        mission_priority=priority,
        maneuverability=ManeuverabilityProfile(
            propulsion_type=propulsion,
            available_delta_v_ms=available_dv,
        ),
        required_delta_v_ms=delta_v,
        service_outage_s=outage,
        pc_post_maneuver=pc_post,
    )


# --- Priority-based resolution (Task 2.4, branch 1) --------------------------

def test_lower_priority_satellite_must_manoeuvre():
    """A commercial sat vs an ISS-class sat: the commercial one manoeuvres."""
    iss = _make_profile("ISS", MissionPriority.HUMAN_SPACEFLIGHT, PropulsionType.CHEMICAL_BIPROPELLANT)
    commercial = _make_profile("STAR-1", MissionPriority.ACTIVE_COMMERCIAL, PropulsionType.ELECTRIC_ION)

    decision = resolve_right_of_way(iss, commercial)
    assert decision.obligated_satellite_id == "STAR-1"
    assert decision.right_of_way_satellite_id == "ISS"
    assert decision.reason == ResolutionReason.LOWER_PRIORITY


def test_lower_priority_satellite_must_manoeuvre_reversed_argument_order():
    """Argument order must not affect the outcome."""
    iss = _make_profile("ISS", MissionPriority.HUMAN_SPACEFLIGHT, PropulsionType.CHEMICAL_BIPROPELLANT)
    eol = _make_profile("OLD-SAT", MissionPriority.END_OF_LIFE, PropulsionType.COLD_GAS)

    decision = resolve_right_of_way(eol, iss)
    assert decision.obligated_satellite_id == "OLD-SAT"
    assert decision.reason == ResolutionReason.LOWER_PRIORITY


def test_national_security_outranks_commercial():
    natsec = _make_profile("DSP-1", MissionPriority.NATIONAL_SECURITY, PropulsionType.CHEMICAL_BIPROPELLANT)
    commercial = _make_profile("KBLK-1", MissionPriority.ACTIVE_COMMERCIAL, PropulsionType.ELECTRIC_HALL_EFFECT)

    decision = resolve_right_of_way(natsec, commercial)
    assert decision.obligated_satellite_id == "KBLK-1"
    assert decision.reason == ResolutionReason.LOWER_PRIORITY


def test_unmanoeuvrable_debris_outranks_end_of_life():
    """Debris has priority tier 2 (UNMANOEUVRABLE); EoL has tier 5."""
    debris = _make_profile("DEB-007", MissionPriority.UNMANOEUVRABLE, PropulsionType.UNKNOWN)
    eol = _make_profile("EOL-SAT", MissionPriority.END_OF_LIFE, PropulsionType.COLD_GAS)

    decision = resolve_right_of_way(debris, eol)
    assert decision.obligated_satellite_id == "EOL-SAT"
    assert decision.reason == ResolutionReason.LOWER_PRIORITY


# --- Cost-based resolution (Task 2.4, branch 2) ------------------------------

def test_lower_cost_satellite_must_manoeuvre():
    """Both satellites in the same priority tier; lower cost must manoeuvre."""
    weights = ManeuverCostWeights(w1=1.0, w2=0.01, w3=1000.0)
    # sat_a: cost = 1*5 + 0.01*600 + 1000*0.002 = 5 + 6 + 2 = 13.0
    sat_a = _make_profile(
        "SAT-A", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.ELECTRIC_ION,
        delta_v=5.0, outage=600.0, pc_post=0.002,
    )
    # sat_b: cost = 1*1 + 0.01*60 + 1000*0.001 = 1 + 0.6 + 1 = 2.6
    sat_b = _make_profile(
        "SAT-B", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.ELECTRIC_ION,
        delta_v=1.0, outage=60.0, pc_post=0.001,
    )
    decision = resolve_right_of_way(sat_a, sat_b, weights)
    assert decision.obligated_satellite_id == "SAT-B"
    assert decision.reason == ResolutionReason.LOWER_MANEUVER_COST


def test_cheapest_manoeuvre_satellite_is_obligated():
    """The satellite bearing the *lower* burden manoeuvres (same priority)."""
    weights = ManeuverCostWeights(w1=1.0, w2=0.01, w3=1000.0)
    # cheap: cost = 0.5
    cheap = _make_profile(
        "CHEAP", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.CHEMICAL_BIPROPELLANT,
        delta_v=0.5, outage=0.0, pc_post=0.0,
    )
    # expensive: cost = 10.0
    expensive = _make_profile(
        "EXPENSIVE", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.CHEMICAL_MONOPROPELLANT,
        delta_v=10.0, outage=0.0, pc_post=0.0,
    )
    decision = resolve_right_of_way(cheap, expensive, weights)
    assert decision.obligated_satellite_id == "CHEAP"
    assert decision.reason == ResolutionReason.LOWER_MANEUVER_COST


# --- Tie-break resolution (Task 2.4, branch 3) --------------------------------

def test_tie_break_uses_lexicographically_larger_id():
    """When costs are equal, the lexicographically larger ID manoeuvres."""
    weights = ManeuverCostWeights(w1=1.0, w2=0.0, w3=0.0)
    # Both satellites have identical cost parameters: cost = 1.0 * 2.0 = 2.0
    sat_a = _make_profile(
        "SAT-Z", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.CHEMICAL_BIPROPELLANT,
        delta_v=2.0, outage=0.0, pc_post=0.0,
    )
    sat_b = _make_profile(
        "SAT-A", MissionPriority.ACTIVE_COMMERCIAL,
        PropulsionType.CHEMICAL_BIPROPELLANT,
        delta_v=2.0, outage=0.0, pc_post=0.0,
    )
    decision = resolve_right_of_way(sat_a, sat_b, weights)
    assert decision.obligated_satellite_id == "SAT-Z"
    assert decision.reason == ResolutionReason.TIE_BREAK_SATELLITE_ID


def test_tie_break_is_deterministic_regardless_of_argument_order():
    """The same satellite must be selected regardless of which is passed as A or B."""
    weights = ManeuverCostWeights(w1=1.0, w2=0.0, w3=0.0)
    sat_x = _make_profile(
        "SAT-X", MissionPriority.END_OF_LIFE,
        PropulsionType.COLD_GAS,
        delta_v=1.0, outage=0.0, pc_post=0.0,
    )
    sat_m = _make_profile(
        "SAT-M", MissionPriority.END_OF_LIFE,
        PropulsionType.COLD_GAS,
        delta_v=1.0, outage=0.0, pc_post=0.0,
    )
    result_1 = resolve_right_of_way(sat_x, sat_m, weights)
    result_2 = resolve_right_of_way(sat_m, sat_x, weights)
    assert result_1.obligated_satellite_id == result_2.obligated_satellite_id == "SAT-X"


# --- Non-manoeuvrable handling ------------------------------------------------

def test_non_manoeuvrable_satellite_is_never_obligated_by_cost():
    """A debris object cannot physically manoeuvre; cost must be inf."""
    debris = _make_profile(
        "DEBRIS-1", MissionPriority.UNMANOEUVRABLE,
        PropulsionType.UNKNOWN,
        delta_v=0.0, outage=0.0, pc_post=0.0,
    )
    active = _make_profile(
        "SAT-ACT", MissionPriority.UNMANOEUVRABLE,
        PropulsionType.ELECTRIC_HALL_EFFECT,
        delta_v=1.0, outage=60.0, pc_post=0.001,
    )
    decision = resolve_right_of_way(debris, active)
    assert decision.obligated_satellite_id == "SAT-ACT"


def test_both_non_manoeuvrable_raises_value_error():
    """Two debris objects create an unresolvable deadlock; must raise ValueError."""
    debris_a = _make_profile(
        "DEB-A", MissionPriority.UNMANOEUVRABLE,
        PropulsionType.UNKNOWN,
    )
    debris_b = _make_profile(
        "DEB-B", MissionPriority.UNMANOEUVRABLE,
        PropulsionType.SOLAR_SAIL,
    )
    with pytest.raises(ValueError, match="non-manoeuvrable"):
        resolve_right_of_way(debris_a, debris_b)


# --- Decision summary ---------------------------------------------------------

def test_right_of_way_decision_summary_returns_string():
    iss = _make_profile("ISS", MissionPriority.HUMAN_SPACEFLIGHT, PropulsionType.CHEMICAL_BIPROPELLANT)
    commercial = _make_profile("STAR-1", MissionPriority.ACTIVE_COMMERCIAL, PropulsionType.ELECTRIC_ION)
    decision = resolve_right_of_way(iss, commercial)
    assert isinstance(decision.summary(), str)
    assert "STAR-1" in decision.summary()
