"""Tests for iscp.physical_layer (Task 1.2 — Physical Layer & Frequencies)."""

import pytest

from iscp.physical_layer import (
    CommMode,
    PhysicalLayerSpec,
    DEFAULT_SPEC,
    BROADCAST_RANGE_M,
    select_mode,
)
