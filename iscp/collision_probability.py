"""
ISCP Task 7 — Edge-Processed Collision Probability (Pc) Calculator
===================================================================
Numerically estimates 2D collision probability in the encounter plane
using a combined Gaussian position uncertainty model.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import math
from typing import Sequence, Tuple


Vector3 = Tuple[float, float, float]
Matrix6 = Sequence[Sequence[float]]


@dataclass(frozen=True)
class SatelliteState:
    """Cartesian state vector for a conjunction participant."""
    position: Vector3
    velocity: Vector3


@dataclass(frozen=True)
class CollisionProbabilityResult:
    """Output bundle for onboard conjunction assessment."""
    probability_of_collision: float
    time_of_closest_approach: datetime
    miss_distance: float
    tca_seconds: float


def _dot(a: Vector3, b: Vector3) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vector3) -> float:
    return math.sqrt(_dot(a, a))


def _sub(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _add(a: Vector3, b: Vector3) -> Vector3:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _scale(a: Vector3, k: float) -> Vector3:
    return (a[0] * k, a[1] * k, a[2] * k)


def _cross(a: Vector3, b: Vector3) -> Vector3:
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def _unit(a: Vector3) -> Vector3:
    n = _norm(a)
    if n == 0.0:
        raise ValueError("Cannot normalise a zero-length vector.")
    return _scale(a, 1.0 / n)


def _validate_covariance(covariance: Matrix6, name: str) -> None:
    if len(covariance) != 6 or any(len(row) != 6 for row in covariance):
        raise ValueError(f"{name} must be a 6x6 matrix.")


def _combined_position_covariance(ego_covariance: Matrix6, target_covariance: Matrix6):
    return [
        [
            ego_covariance[i][j] + target_covariance[i][j]
            for j in range(3)
        ]
        for i in range(3)
    ]


def _project_covariance_2d(covariance_3d, y_hat: Vector3, z_hat: Vector3):
    c_yy = sum(y_hat[i] * covariance_3d[i][j] * y_hat[j] for i in range(3) for j in range(3))
    c_yz = sum(y_hat[i] * covariance_3d[i][j] * z_hat[j] for i in range(3) for j in range(3))
    c_zz = sum(z_hat[i] * covariance_3d[i][j] * z_hat[j] for i in range(3) for j in range(3))
    return ((c_yy, c_yz), (c_yz, c_zz))


def _encounter_frame(relative_velocity: Vector3, relative_position_tca: Vector3):
    x_hat = _unit(relative_velocity)
    in_plane = _sub(relative_position_tca, _scale(x_hat, _dot(relative_position_tca, x_hat)))
    if _norm(in_plane) == 0.0:
        seed = (1.0, 0.0, 0.0) if abs(x_hat[0]) < 0.9 else (0.0, 1.0, 0.0)
        in_plane = _cross(x_hat, seed)
    y_hat = _unit(in_plane)
    z_hat = _unit(_cross(x_hat, y_hat))
    return y_hat, z_hat


def _integrate_gaussian_over_hard_body(
    mean_2d,
    covariance_2d,
    hard_body_radius: float,
    radial_steps: int = 24,
    angular_steps: int = 48,
) -> float:
    det = covariance_2d[0][0] * covariance_2d[1][1] - covariance_2d[0][1] * covariance_2d[1][0]
    if det <= 0.0:
        raise ValueError("Encounter-plane covariance must be positive-definite.")

    inv00 = covariance_2d[1][1] / det
    inv11 = covariance_2d[0][0] / det
    inv01 = -covariance_2d[0][1] / det

    norm_factor = 1.0 / (2.0 * math.pi * math.sqrt(det))
    dr = hard_body_radius / radial_steps
    dtheta = 2.0 * math.pi / angular_steps
    total = 0.0

    for i in range(radial_steps):
        r = (i + 0.5) * dr
        for j in range(angular_steps):
            theta = (j + 0.5) * dtheta
            x = r * math.cos(theta) - mean_2d[0]
            y = r * math.sin(theta) - mean_2d[1]
            quad = inv00 * x * x + 2.0 * inv01 * x * y + inv11 * y * y
            total += math.exp(-0.5 * quad) * r

    probability = norm_factor * total * dr * dtheta
    return min(1.0, max(0.0, probability))


def calculate_probability_of_collision(
    ego_state: SatelliteState,
    ego_covariance: Matrix6,
    target_state: SatelliteState,
    target_covariance: Matrix6,
    hard_body_radius: float,
    reference_time: datetime | None = None,
) -> CollisionProbabilityResult:
    """
    Compute conjunction probability of collision at time of closest approach.
    """
    if hard_body_radius <= 0.0:
        raise ValueError("hard_body_radius must be positive.")
    _validate_covariance(ego_covariance, "ego_covariance")
    _validate_covariance(target_covariance, "target_covariance")

    rel_position = _sub(target_state.position, ego_state.position)
    rel_velocity = _sub(target_state.velocity, ego_state.velocity)
    rel_speed_sq = _dot(rel_velocity, rel_velocity)
    if rel_speed_sq == 0.0:
        raise ValueError("Relative velocity must be non-zero to determine TCA.")

    tca_seconds = -_dot(rel_position, rel_velocity) / rel_speed_sq
    rel_position_tca = _add(rel_position, _scale(rel_velocity, tca_seconds))
    miss_distance = _norm(rel_position_tca)

    y_hat, z_hat = _encounter_frame(rel_velocity, rel_position_tca)
    covariance_3d = _combined_position_covariance(ego_covariance, target_covariance)
    covariance_2d = _project_covariance_2d(covariance_3d, y_hat, z_hat)
    mean_2d = (_dot(rel_position_tca, y_hat), _dot(rel_position_tca, z_hat))

    probability = _integrate_gaussian_over_hard_body(
        mean_2d=mean_2d,
        covariance_2d=covariance_2d,
        hard_body_radius=hard_body_radius,
    )

    if reference_time is None:
        reference_time = datetime.now(timezone.utc)
    time_of_closest_approach = reference_time + timedelta(seconds=tca_seconds)

    return CollisionProbabilityResult(
        probability_of_collision=probability,
        time_of_closest_approach=time_of_closest_approach,
        miss_distance=miss_distance,
        tca_seconds=tca_seconds,
    )
