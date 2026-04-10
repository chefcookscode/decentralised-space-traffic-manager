"""Tests for iscp.payload (Task 1.1 — Data Payload Schema)."""

import struct
import pytest

from iscp.payload import (
    ISCPPayload,
    ManeuverIntent,
    ManeuverType,
    PropulsionType,
    PAYLOAD_SIZE,
    MAX_PAYLOAD_BYTES,
    SATELLITE_ID_LENGTH,
)
