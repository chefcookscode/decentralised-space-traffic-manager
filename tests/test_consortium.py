"""Tests for iscp.consortium (Task 1.4 — Industry Consortium)."""

import pytest

from iscp.consortium import (
    ConsortiumMember,
    ISCPConsortium,
    MemberCategory,
    RatificationStatus,
    ReviewStatus,
    RATIFICATION_THRESHOLD,
    create_founding_consortium,
)
