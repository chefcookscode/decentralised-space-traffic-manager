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


# ---------------------------------------------------------------------------
# Review workflow
# ---------------------------------------------------------------------------

def test_open_for_review_resets_statuses(founding):
    founding.open_for_review("0.2.0-draft")
    for member in founding.members:
        assert member.review_status == ReviewStatus.PENDING


def test_attempt_ratification_succeeds_with_unanimous_approvals(founding):
    founding.open_for_review("0.1.0-draft")
    for member in founding.members:
        member.submit_review(ReviewStatus.APPROVED)
    result = founding.attempt_ratification()
    assert result is True
    assert founding.ratification_status == RatificationStatus.RATIFIED


def test_attempt_ratification_fails_below_threshold(founding):
    founding.open_for_review("0.1.0-draft")
    # Only one approval out of eight → 12.5 % < 75 % threshold
    founding.members[0].submit_review(ReviewStatus.APPROVED)
    result = founding.attempt_ratification()
    assert result is False


def test_add_duplicate_member_raises(founding):
    with pytest.raises(ValueError, match="already registered"):
        founding.add_member(
            ConsortiumMember("SpaceX", MemberCategory.LEO_OPERATOR, "US", "x@x.com")
        )


def test_member_summary_returns_string(founding):
    assert isinstance(founding.member_summary(), str)
