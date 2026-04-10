"""
ISCP Task 2 — Right of Way Rules (The Decision Matrix)
=======================================================
Defines the globally agreed-upon, mathematically deterministic ruleset for
determining which satellite is obligated to manoeuvre during a high-risk
conjunction.

Task 2.1 — Mission Priority Tiering
-------------------------------------
A hardcoded hierarchy of space-asset classes.  Lower numeric value means
higher operational importance and therefore the greater right-of-way (the
asset is *not* obligated to manoeuvre when a higher-priority peer is
involved).

    1  HUMAN_SPACEFLIGHT    ISS, Tiangong, Crew Dragon  (highest priority)
    2  UNMANOEUVRABLE       Debris / defunct objects
    3  NATIONAL_SECURITY    Early-warning / mil-sat
    4  ACTIVE_COMMERCIAL    Active mega-constellation satellites
    5  END_OF_LIFE          Expendable / end-of-life assets   (lowest priority)

Task 2.2 — Propulsion and Manoeuvrability Constraints
------------------------------------------------------
Avoidance capability varies enormously by propulsion technology:

    HIGH_THRUST_IMMEDIATE   Chemical mono/bi-propellant: manoeuvre in minutes
    MEDIUM_THRUST           Cold-gas / resistojet: tens of minutes
    LOW_THRUST_SLOW         Electric ion / Hall-effect: hours to days
    NON_MANOEUVRABLE        No propulsion / debris: cannot manoeuvre

:data:`PROPULSION_MANEUVERABILITY_MAP` maps every :class:`~iscp.payload.PropulsionType`
to a :class:`ManeuverabilityCategory` and provides a canonical minimum
lead-time (``min_lead_time_s``) for each category.

Task 2.3 — Cost-of-Manoeuvre Optimisation Function
----------------------------------------------------
The scalar manoeuvre cost is computed as:

    J = w1 · ΔV  +  w2 · Δt_outage  +  w3 · P_c,post-manoeuvre

where the weights ``(w1, w2, w3)`` are agreed-upon protocol constants
stored in :class:`ManeuverCostWeights`.  The satellite with the *lower* J
is obligated to manoeuvre.

Task 2.4 — Deterministic Resolution Logic
------------------------------------------
:func:`resolve_right_of_way` implements the following flowchart::

    ┌─────────────────────────────────────────────────────────┐
    │  Priority(A) == Priority(B)?                            │
    │  No  →  Higher-priority satellite has right-of-way.     │
    │         Lower-priority satellite MUST manoeuvre.        │
    │                                                         │
    │  Yes →  Compute Cost(A) and Cost(B).                    │
    │         Satellite with lower cost MUST manoeuvre.       │
    │                                                         │
    │         Costs equal?  →  Tie-break on satellite_id:     │
    │                          lexicographically larger ID    │
    │                          MUST manoeuvre (deterministic) │
    └─────────────────────────────────────────────────────────┘

The tie-breaker on satellite_id guarantees zero deadlock scenarios: given
any two distinct satellite identifiers, exactly one lexicographic ordering
exists, so a unique obligation is always produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional

from iscp.payload import PropulsionType


# ---------------------------------------------------------------------------
# Task 2.1 — Mission Priority Tiering
# ---------------------------------------------------------------------------

class MissionPriority(IntEnum):
    """
    Operational priority tier for space assets.

    Lower numeric value indicates higher priority (greater right-of-way).
    An asset at a higher-priority tier is *never* obligated to manoeuvre
    when the other party in the conjunction belongs to a lower-priority tier.
    """
    HUMAN_SPACEFLIGHT = 1   # ISS, Tiangong, Crew Dragon
    UNMANOEUVRABLE    = 2   # Debris / defunct objects
    NATIONAL_SECURITY = 3   # Early-warning / military satellites
    ACTIVE_COMMERCIAL = 4   # Active mega-constellation satellites
    END_OF_LIFE       = 5   # Expendable / end-of-life assets


# ---------------------------------------------------------------------------
# Task 2.2 — Propulsion and Manoeuvrability Constraints
# ---------------------------------------------------------------------------

class ManeuverabilityCategory(IntEnum):
    """
    Coarse classification of a satellite's avoidance capability.

    Categories reflect the minimum time required to execute a meaningful
    collision-avoidance manoeuvre.  Lower numeric value = faster response.
    """
    HIGH_THRUST_IMMEDIATE = 1   # Chemical mono/bi-prop:  minutes
    MEDIUM_THRUST         = 2   # Cold-gas / resistojet:  tens of minutes
    LOW_THRUST_SLOW       = 3   # Electric ion / Hall:    hours to days
    NON_MANOEUVRABLE      = 4   # No propulsion / debris: cannot manoeuvre


# Minimum lead-time (seconds) required to execute a useful CAM, per category.
MANEUVERABILITY_MIN_LEAD_TIME_S: dict[ManeuverabilityCategory, float] = {
    ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE: 300.0,     # 5 minutes
    ManeuverabilityCategory.MEDIUM_THRUST:         1_800.0,   # 30 minutes
    ManeuverabilityCategory.LOW_THRUST_SLOW:       86_400.0,  # 24 hours
    ManeuverabilityCategory.NON_MANOEUVRABLE:      float("inf"),
}

# Canonical mapping from PropulsionType → ManeuverabilityCategory.
PROPULSION_MANEUVERABILITY_MAP: dict[PropulsionType, ManeuverabilityCategory] = {
    PropulsionType.UNKNOWN:                  ManeuverabilityCategory.NON_MANOEUVRABLE,
    PropulsionType.CHEMICAL_MONOPROPELLANT:  ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE,
    PropulsionType.CHEMICAL_BIPROPELLANT:    ManeuverabilityCategory.HIGH_THRUST_IMMEDIATE,
    PropulsionType.ELECTRIC_ION:             ManeuverabilityCategory.LOW_THRUST_SLOW,
    PropulsionType.ELECTRIC_HALL_EFFECT:     ManeuverabilityCategory.LOW_THRUST_SLOW,
    PropulsionType.COLD_GAS:                 ManeuverabilityCategory.MEDIUM_THRUST,
    PropulsionType.SOLAR_SAIL:               ManeuverabilityCategory.NON_MANOEUVRABLE,
    PropulsionType.RESISTOJET:               ManeuverabilityCategory.MEDIUM_THRUST,
}


@dataclass
class ManeuverabilityProfile:
    """
    Physical manoeuvrability characteristics of a single satellite.

    Parameters
    ----------
    propulsion_type:
        The satellite's propulsion system (from :class:`~iscp.payload.PropulsionType`).
    available_delta_v_ms:
        Remaining delta-V budget in m/s (derived from propellant mass fraction).
    min_lead_time_s:
        Minimum time (seconds) needed to plan and execute a meaningful CAM.
        When *None*, the canonical value for the propulsion category is used.
    """
    propulsion_type: PropulsionType
    available_delta_v_ms: float = 0.0   # m/s; remaining fuel budget

    # Populated automatically from propulsion_type when None.
    min_lead_time_s: Optional[float] = field(default=None)

    def __post_init__(self) -> None:
        if self.available_delta_v_ms < 0.0:
            raise ValueError(
                f"available_delta_v_ms must be non-negative, "
                f"got {self.available_delta_v_ms}"
            )
        if self.min_lead_time_s is None:
            cat = PROPULSION_MANEUVERABILITY_MAP[self.propulsion_type]
            self.min_lead_time_s = MANEUVERABILITY_MIN_LEAD_TIME_S[cat]

    @property
    def category(self) -> ManeuverabilityCategory:
        """Return the manoeuvrability category for this propulsion type."""
        return PROPULSION_MANEUVERABILITY_MAP[self.propulsion_type]

    @property
    def can_manoeuvre(self) -> bool:
        """Return True if the satellite is physically capable of manoeuvring."""
        return self.category != ManeuverabilityCategory.NON_MANOEUVRABLE


# ---------------------------------------------------------------------------
# Task 2.3 — Cost-of-Manoeuvre Optimisation Function
# ---------------------------------------------------------------------------

@dataclass
class ManeuverCostWeights:
    """
    Agreed-upon weighting coefficients for the cost-of-manoeuvre function.

    The cost function is:
        J = w1 · ΔV  +  w2 · Δt_outage  +  w3 · P_c,post

    Attributes
    ----------
    w1:
        Weight applied to delta-V cost (m/s).  Penalises propellant expenditure.
    w2:
        Weight applied to service-outage duration (seconds).  Penalises
        mission downtime caused by pointing the spacecraft away from its
        operational attitude.
    w3:
        Weight applied to the post-manoeuvre residual collision probability
        P_c ∈ [0, 1].  Penalises manoeuvres that offer little risk reduction.
    """
    w1: float = 1.0       # ΔV weight          (m/s)
    w2: float = 0.01      # Δt_outage weight   (seconds → normalised)
    w3: float = 1_000.0   # P_c,post weight    (dimensionless probability)

    def __post_init__(self) -> None:
        for name, val in (("w1", self.w1), ("w2", self.w2), ("w3", self.w3)):
            if val < 0.0:
                raise ValueError(
                    f"Weight {name} must be non-negative, got {val}"
                )


def compute_maneuver_cost(
    delta_v_ms: float,
    outage_s: float,
    pc_post: float,
    weights: ManeuverCostWeights = ManeuverCostWeights(),
) -> float:
    """
    Compute the scalar cost of executing a collision-avoidance manoeuvre.

    Implements the agreed ISCP cost function:

        J = w1 · ΔV  +  w2 · Δt_outage  +  w3 · P_c,post

    Parameters
    ----------
    delta_v_ms:
        Magnitude of the required delta-V in m/s (non-negative).
    outage_s:
        Expected service-outage duration in seconds (non-negative).
    pc_post:
        Post-manoeuvre collision probability P_c ∈ [0, 1].
    weights:
        Weighting coefficients (default: :class:`ManeuverCostWeights` defaults).

    Returns
    -------
    float
        The scalar cost J ≥ 0.

    Raises
    ------
    ValueError
        If any input is outside its valid range.
    """
    if delta_v_ms < 0.0:
        raise ValueError(f"delta_v_ms must be non-negative, got {delta_v_ms}")
    if outage_s < 0.0:
        raise ValueError(f"outage_s must be non-negative, got {outage_s}")
    if not 0.0 <= pc_post <= 1.0:
        raise ValueError(f"pc_post must be in [0, 1], got {pc_post}")

    return weights.w1 * delta_v_ms + weights.w2 * outage_s + weights.w3 * pc_post


# ---------------------------------------------------------------------------
# Task 2.4 — Conjunction Profile and Resolution Logic
# ---------------------------------------------------------------------------

@dataclass
class ConjunctionProfile:
    """
    All parameters required to evaluate right-of-way for one participant in
    a high-risk conjunction event.

    Parameters
    ----------
    satellite_id:
        Unique identifier for this satellite (used as tie-breaker).
    mission_priority:
        Operational priority tier (:class:`MissionPriority`).
    maneuverability:
        Physical manoeuvrability profile (:class:`ManeuverabilityProfile`).
    required_delta_v_ms:
        Magnitude of delta-V (m/s) that would be needed to reduce P_c below
        the acceptable threshold if *this* satellite manoeuvres.
    service_outage_s:
        Expected mission downtime (seconds) caused by a CAM executed by
        *this* satellite.
    pc_post_maneuver:
        Residual collision probability after *this* satellite executes the
        avoidance manoeuvre (dimensionless, 0–1).
    """
    satellite_id: str
    mission_priority: MissionPriority
    maneuverability: ManeuverabilityProfile
    required_delta_v_ms: float = 0.0   # m/s
    service_outage_s: float = 0.0      # seconds
    pc_post_maneuver: float = 0.0      # [0, 1]

    def __post_init__(self) -> None:
        if self.required_delta_v_ms < 0.0:
            raise ValueError(
                f"required_delta_v_ms must be non-negative, "
                f"got {self.required_delta_v_ms}"
            )
        if self.service_outage_s < 0.0:
            raise ValueError(
                f"service_outage_s must be non-negative, "
                f"got {self.service_outage_s}"
            )
        if not 0.0 <= self.pc_post_maneuver <= 1.0:
            raise ValueError(
                f"pc_post_maneuver must be in [0, 1], "
                f"got {self.pc_post_maneuver}"
            )

    def maneuver_cost(
        self, weights: ManeuverCostWeights = ManeuverCostWeights()
    ) -> float:
        """
        Compute the cost of *this* satellite performing the avoidance manoeuvre.

        Returns :data:`math.inf` when the satellite is non-manoeuvrable.
        """
        if not self.maneuverability.can_manoeuvre:
            return float("inf")
        return compute_maneuver_cost(
            self.required_delta_v_ms,
            self.service_outage_s,
            self.pc_post_maneuver,
            weights,
        )


class ResolutionReason(IntEnum):
    """Reason code explaining why a particular satellite was selected to manoeuvre."""
    LOWER_PRIORITY         = 1   # Other satellite has higher mission priority
    LOWER_MANEUVER_COST    = 2   # This satellite bears the lowest manoeuvre burden
    TIE_BREAK_SATELLITE_ID = 3   # Costs were equal; resolved by satellite_id


@dataclass
class RightOfWayDecision:
    """
    Outcome of a right-of-way resolution between two conjunction participants.

    Attributes
    ----------
    obligated_satellite_id:
        ID of the satellite that *must* execute the avoidance manoeuvre.
    right_of_way_satellite_id:
        ID of the satellite that is *not* obligated to manoeuvre.
    reason:
        :class:`ResolutionReason` code explaining the decision.
    cost_a:
        Computed cost for satellite A (may be *inf* if non-manoeuvrable).
    cost_b:
        Computed cost for satellite B (may be *inf* if non-manoeuvrable).
    """
    obligated_satellite_id: str
    right_of_way_satellite_id: str
    reason: ResolutionReason
    cost_a: float
    cost_b: float

    def summary(self) -> str:
        """Return a human-readable single-line summary of the decision."""
        return (
            f"RightOfWayDecision: '{self.obligated_satellite_id}' MUST manoeuvre "
            f"(reason={self.reason.name}, "
            f"cost_a={self.cost_a:.4f}, cost_b={self.cost_b:.4f})"
        )


def resolve_right_of_way(
    sat_a: ConjunctionProfile,
    sat_b: ConjunctionProfile,
    weights: ManeuverCostWeights = ManeuverCostWeights(),
) -> RightOfWayDecision:
    """
    Determine which satellite is obligated to execute a collision-avoidance
    manoeuvre for a high-risk conjunction.

    Decision flowchart
    ------------------
    1. **Priority differs** → The satellite with *lower* :class:`MissionPriority`
       (numerically larger value) must manoeuvre.  The higher-priority satellite
       retains full right-of-way.

    2. **Priority equal** → Compute ``cost_a`` and ``cost_b`` using the agreed
       cost function.  The satellite with the *lower* cost must manoeuvre
       (it bears the least burden).

    3. **Costs equal** (tie-break) → The satellite whose ``satellite_id`` is
       *lexicographically larger* must manoeuvre.  This rule is deterministic
       for any pair of distinct identifiers, guaranteeing zero deadlock.

    Parameters
    ----------
    sat_a:
        Conjunction profile for satellite A.
    sat_b:
        Conjunction profile for satellite B.
    weights:
        Cost-function weighting coefficients.

    Returns
    -------
    RightOfWayDecision
        The resolved obligation with full audit information.

    Raises
    ------
    ValueError
        If both satellites are non-manoeuvrable (deadlock is physically
        impossible to resolve — a ground-level escalation is required).
    """
    cost_a = sat_a.maneuver_cost(weights)
    cost_b = sat_b.maneuver_cost(weights)

    # Guard: at least one satellite must be capable of manoeuvring.
    if cost_a == float("inf") and cost_b == float("inf"):
        raise ValueError(
            f"Both satellites '{sat_a.satellite_id}' and "
            f"'{sat_b.satellite_id}' are non-manoeuvrable. "
            "This conjunction cannot be resolved autonomously; "
            "ground-level intervention is required."
        )

    # ------------------------------------------------------------------
    # Step 1: Priority-based resolution
    # ------------------------------------------------------------------
    if sat_a.mission_priority != sat_b.mission_priority:
        # Higher numeric value → lower operational priority → must manoeuvre.
        if sat_a.mission_priority > sat_b.mission_priority:
            obligated, right_of_way = sat_a, sat_b
        else:
            obligated, right_of_way = sat_b, sat_a
        return RightOfWayDecision(
            obligated_satellite_id=obligated.satellite_id,
            right_of_way_satellite_id=right_of_way.satellite_id,
            reason=ResolutionReason.LOWER_PRIORITY,
            cost_a=cost_a,
            cost_b=cost_b,
        )

    # ------------------------------------------------------------------
    # Step 2: Cost-based resolution (same priority tier)
    # ------------------------------------------------------------------
    if cost_a != cost_b:
        # The satellite with the *lower* cost is obligated (least burden).
        if cost_a < cost_b:
            obligated, right_of_way = sat_a, sat_b
        else:
            obligated, right_of_way = sat_b, sat_a
        return RightOfWayDecision(
            obligated_satellite_id=obligated.satellite_id,
            right_of_way_satellite_id=right_of_way.satellite_id,
            reason=ResolutionReason.LOWER_MANEUVER_COST,
            cost_a=cost_a,
            cost_b=cost_b,
        )

    # ------------------------------------------------------------------
    # Step 3: Tie-break on satellite_id (lexicographically larger ID manoeuvres)
    # ------------------------------------------------------------------
    if sat_a.satellite_id >= sat_b.satellite_id:
        obligated, right_of_way = sat_a, sat_b
    else:
        obligated, right_of_way = sat_b, sat_a
    return RightOfWayDecision(
        obligated_satellite_id=obligated.satellite_id,
        right_of_way_satellite_id=right_of_way.satellite_id,
        reason=ResolutionReason.TIE_BREAK_SATELLITE_ID,
        cost_a=cost_a,
        cost_b=cost_b,
    )
