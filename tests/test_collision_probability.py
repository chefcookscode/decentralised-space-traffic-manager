"""Tests for the edge-processed collision probability calculator."""

from datetime import datetime, timezone
import time

import pytest

from iscp.collision_probability import (
    SatelliteState,
    calculate_probability_of_collision,
)


def _diag_covariance(pos_sigma_m: float, vel_sigma_ms: float):
    p2 = pos_sigma_m * pos_sigma_m
    v2 = vel_sigma_ms * vel_sigma_ms
    return [
        [p2, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, p2, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, p2, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, v2, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, v2, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, v2],
    ]


def test_high_risk_conjunction_returns_pc_above_threshold():
    ref_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    ego = SatelliteState(position=(0.0, 0.0, 0.0), velocity=(7500.0, 0.0, 0.0))
    target = SatelliteState(position=(100.0, 0.0, 0.0), velocity=(7490.0, 0.0, 0.0))
    cov = _diag_covariance(pos_sigma_m=20.0, vel_sigma_ms=1.0)

    result = calculate_probability_of_collision(
        ego_state=ego,
        ego_covariance=cov,
        target_state=target,
        target_covariance=cov,
        hard_body_radius=10.0,
        reference_time=ref_time,
    )

    assert 0.0 <= result.probability_of_collision <= 1.0
    assert result.probability_of_collision > 1e-3
    assert result.tca_seconds == pytest.approx(10.0)
    assert result.miss_distance == pytest.approx(0.0, abs=1e-6)
    assert result.time_of_closest_approach == datetime(2026, 1, 1, 0, 0, 10, tzinfo=timezone.utc)


def test_low_risk_conjunction_returns_very_small_pc():
    ego = SatelliteState(position=(0.0, 0.0, 0.0), velocity=(7500.0, 0.0, 0.0))
    target = SatelliteState(position=(100.0, 500.0, 0.0), velocity=(7490.0, 0.0, 0.0))
    cov = _diag_covariance(pos_sigma_m=20.0, vel_sigma_ms=1.0)

    result = calculate_probability_of_collision(
        ego_state=ego,
        ego_covariance=cov,
        target_state=target,
        target_covariance=cov,
        hard_body_radius=10.0,
    )

    assert 0.0 <= result.probability_of_collision <= 1.0
    assert result.probability_of_collision < 1e-6
    assert result.miss_distance == pytest.approx(500.0, abs=1e-6)


def test_integration_runtime_is_under_50ms():
    ego = SatelliteState(position=(0.0, 0.0, 0.0), velocity=(7500.0, 0.0, 0.0))
    target = SatelliteState(position=(100.0, 30.0, 0.0), velocity=(7490.0, 0.0, 0.0))
    cov = _diag_covariance(pos_sigma_m=20.0, vel_sigma_ms=1.0)

    start = time.perf_counter()
    calculate_probability_of_collision(
        ego_state=ego,
        ego_covariance=cov,
        target_state=target,
        target_covariance=cov,
        hard_body_radius=10.0,
    )
    elapsed = time.perf_counter() - start

    assert elapsed < 0.05
