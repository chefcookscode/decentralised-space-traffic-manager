"""Tests for iscp.physical_layer (Task 1.2 — Physical Layer & Frequencies)."""

import pytest

from iscp.physical_layer import (
    CommMode,
    PhysicalLayerSpec,
    DEFAULT_SPEC,
    BROADCAST_RANGE_M,
    select_mode,
)


def test_broadcast_range_is_500km():
    assert BROADCAST_RANGE_M == 500_000.0


def test_select_mode_optical_within_range():
    assert select_mode(100_000.0) == CommMode.OPTICAL


def test_select_mode_sband_fallback_when_no_optical():
    assert select_mode(1_000_000.0, optical_available=False) == CommMode.RF_SBAND


def test_select_mode_vhf_last_resort():
    mode = select_mode(400_000.0, optical_available=False, sband_available=False)
    assert mode == CommMode.RF_VHF


def test_select_mode_offline_beyond_all_ranges():
    mode = select_mode(600_000.0, optical_available=False, sband_available=False)
    assert mode == CommMode.OFFLINE


def test_select_mode_raises_on_negative_range():
    with pytest.raises(ValueError):
        select_mode(-1.0)


def test_spec_data_rate_optical():
    assert DEFAULT_SPEC.data_rate_bps(CommMode.OPTICAL) == 10_000_000_000


def test_spec_summary_contains_500km(capsys):
    summary = DEFAULT_SPEC.summary()
    assert "500" in summary
