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


@pytest.fixture
def founding() -> ISCPConsortium:
    return create_founding_consortium()


def test_founding_consortium_has_eight_members(founding):
    assert len(founding.members) == 8


def test_founding_consortium_includes_spacex(founding):
    assert founding.get_member("SpaceX") is not None


def test_founding_consortium_includes_all_agencies(founding):
    for name in ("NASA", "ESA", "ISRO"):
        assert founding.get_member(name) is not None
