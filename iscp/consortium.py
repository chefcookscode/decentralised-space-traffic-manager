"""
ISCP Task 1.4 — Industry Consortium
=====================================
Defines the data structures and workflow for establishing the ISCP Industry
Consortium and managing the peer-review / ratification process for the
Inter-Satellite Communication Protocol standard.

Consortium composition (initial founding members)
--------------------------------------------------
  LEO Operators   : SpaceX, Amazon (Project Kuiper), OneWeb
  Space Agencies  : NASA, ESA, ISRO
  STM Entities    : LeoLabs, US Space Force (Space Delta 2)

The draft ISCP specification is published to the consortium for peer review.
Each member organisation submits a formal review and casts a ratification
vote.  The standard is adopted when the :attr:`RATIFICATION_THRESHOLD`
fraction of votes are in favour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Fraction of total member votes required to ratify the standard.
RATIFICATION_THRESHOLD: float = 0.75   # 75 %

# Maximum number of review rounds before the specification is escalated.
MAX_REVIEW_ROUNDS: int = 3


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class MemberCategory(IntEnum):
    """High-level category of a consortium member organisation."""
    LEO_OPERATOR = 0
    SPACE_AGENCY = 1
    STM_ENTITY = 2
    STANDARDS_BODY = 3
    ACADEMIC = 4
    OTHER = 5


class ReviewStatus(IntEnum):
    """Status of a member's review of the current draft."""
    PENDING = 0
    IN_REVIEW = 1
    APPROVED = 2
    APPROVED_WITH_COMMENTS = 3
    REJECTED = 4
    ABSTAINED = 5


class RatificationStatus(IntEnum):
    """Overall ratification status of the draft specification."""
    DRAFT = 0
    OPEN_FOR_REVIEW = 1
    UNDER_REVISION = 2
    RATIFIED = 3
    WITHDRAWN = 4


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ConsortiumMember:
    """A member organisation participating in the ISCP consortium."""
    name: str
    category: MemberCategory
    country_code: str        # ISO 3166-1 alpha-2 or "INT" for international
    contact_email: str
    review_status: ReviewStatus = ReviewStatus.PENDING
    comments: str = ""

    def submit_review(
        self,
        status: ReviewStatus,
        comments: str = "",
    ) -> None:
        """
        Record this member's review decision.

        Parameters
        ----------
        status:
            The member's formal review outcome.
        comments:
            Optional free-text comments accompanying the review.

        Raises
        ------
        ValueError
            If *status* is PENDING (that is not a valid submission state).
        """
        if status == ReviewStatus.PENDING:
            raise ValueError(
                "PENDING is not a valid submitted review status; "
                "use IN_REVIEW, APPROVED, APPROVED_WITH_COMMENTS, "
                "REJECTED, or ABSTAINED."
            )
        self.review_status = status
        self.comments = comments

    @property
    def has_voted(self) -> bool:
        """Return True if this member has submitted a review."""
        return self.review_status not in (
            ReviewStatus.PENDING,
            ReviewStatus.IN_REVIEW,
        )

    @property
    def vote_is_for(self) -> bool:
        """Return True if the submitted review counts as an approval vote."""
        return self.review_status in (
            ReviewStatus.APPROVED,
            ReviewStatus.APPROVED_WITH_COMMENTS,
        )


@dataclass
class ISCPConsortium:
    """
    The ISCP Industry Consortium.

    Manages the membership list, the current draft specification, and the
    peer-review / ratification workflow.
    """
    name: str = "ISCP Industry Consortium"
    members: List[ConsortiumMember] = field(default_factory=list)
    draft_version: str = "0.1.0-draft"
    ratification_status: RatificationStatus = RatificationStatus.DRAFT
    review_round: int = 0
    ratification_history: List[Dict] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Membership management
    # ------------------------------------------------------------------

    def add_member(self, member: ConsortiumMember) -> None:
        """
        Add a new member to the consortium.

        Raises
        ------
        ValueError
            If a member with the same name is already registered.
        """
        for existing in self.members:
            if existing.name == member.name:
                raise ValueError(
                    f"Member '{member.name}' is already registered."
                )
        self.members.append(member)

    def get_member(self, name: str) -> Optional[ConsortiumMember]:
        """Return the member with *name*, or None if not found."""
        for m in self.members:
            if m.name == name:
                return m
        return None

    # ------------------------------------------------------------------
    # Review workflow
    # ------------------------------------------------------------------

    def open_for_review(self, draft_version: str) -> None:
        """
        Publish a new draft version and open it for member review.

        Resets all member review statuses to PENDING and increments the
        review round counter.

        Parameters
        ----------
        draft_version:
            Version string of the new draft (e.g. ``'0.2.0-draft'``).

        Raises
        ------
        RuntimeError
            If the specification has already been ratified or withdrawn.
        """
        if self.ratification_status in (
            RatificationStatus.RATIFIED,
            RatificationStatus.WITHDRAWN,
        ):
            raise RuntimeError(
                f"Cannot open for review: specification is "
                f"{self.ratification_status.name}."
            )
        self.draft_version = draft_version
        self.review_round += 1
        for member in self.members:
            member.review_status = ReviewStatus.PENDING
            member.comments = ""
        self.ratification_status = RatificationStatus.OPEN_FOR_REVIEW

    def tally_votes(self) -> Dict:
        """
        Count votes from members who have submitted reviews.

        Returns
        -------
        dict
            ``{'total_members': int, 'voted': int, 'for': int,
               'against': int, 'abstained': int,
               'approval_fraction': float, 'quorum_reached': bool}``
        """
        voted = [m for m in self.members if m.has_voted]
        votes_for = sum(1 for m in voted if m.vote_is_for)
        votes_against = sum(
            1 for m in voted
            if m.review_status == ReviewStatus.REJECTED
        )
        votes_abstained = sum(
            1 for m in voted
            if m.review_status == ReviewStatus.ABSTAINED
        )
        total = len(self.members)
        approval_fraction = votes_for / total if total > 0 else 0.0
        return {
            "total_members": total,
            "voted": len(voted),
            "for": votes_for,
            "against": votes_against,
            "abstained": votes_abstained,
            "approval_fraction": approval_fraction,
            "quorum_reached": approval_fraction >= RATIFICATION_THRESHOLD,
        }

    def attempt_ratification(self) -> bool:
        """
        Attempt to ratify the current draft based on the vote tally.

        If the approval fraction meets :data:`RATIFICATION_THRESHOLD`, the
        status is set to RATIFIED and the outcome is recorded in history.
        Otherwise the status reverts to UNDER_REVISION (provided the
        maximum review rounds have not been exceeded).

        Returns
        -------
        bool
            True if ratification succeeded.

        Raises
        ------
        RuntimeError
            If the consortium is not currently in OPEN_FOR_REVIEW status.
        """
        if self.ratification_status != RatificationStatus.OPEN_FOR_REVIEW:
            raise RuntimeError(
                "attempt_ratification() requires OPEN_FOR_REVIEW status; "
                f"current status is {self.ratification_status.name}."
            )

        tally = self.tally_votes()
        self.ratification_history.append(
            {
                "draft_version": self.draft_version,
                "review_round": self.review_round,
                "tally": tally,
            }
        )

        if tally["quorum_reached"]:
            self.ratification_status = RatificationStatus.RATIFIED
            return True

        if self.review_round >= MAX_REVIEW_ROUNDS:
            self.ratification_status = RatificationStatus.WITHDRAWN
        else:
            self.ratification_status = RatificationStatus.UNDER_REVISION
        return False

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def member_summary(self) -> str:
        """Return a formatted membership roster with review statuses."""
        lines = [
            f"{self.name} — Member Roster (draft {self.draft_version})",
            "─" * 60,
        ]
        by_category: Dict[MemberCategory, List[ConsortiumMember]] = {}
        for m in self.members:
            by_category.setdefault(m.category, []).append(m)

        category_labels = {
            MemberCategory.LEO_OPERATOR: "LEO Operators",
            MemberCategory.SPACE_AGENCY: "Space Agencies",
            MemberCategory.STM_ENTITY: "STM Entities",
            MemberCategory.STANDARDS_BODY: "Standards Bodies",
            MemberCategory.ACADEMIC: "Academic Institutions",
            MemberCategory.OTHER: "Other",
        }
        for cat, label in category_labels.items():
            members = by_category.get(cat, [])
            if not members:
                continue
            lines.append(f"\n{label}:")
            for m in members:
                lines.append(
                    f"  • {m.name:<30s} [{m.country_code}]  "
                    f"{m.review_status.name}"
                )
        return "\n".join(lines)

    def ratification_summary(self) -> str:
        """Return a summary of the current ratification state."""
        tally = self.tally_votes()
        pct = tally["approval_fraction"] * 100
        lines = [
            f"Ratification status  : {self.ratification_status.name}",
            f"Draft version        : {self.draft_version}",
            f"Review round         : {self.review_round} / {MAX_REVIEW_ROUNDS}",
            f"Members              : {tally['total_members']}",
            f"Votes cast           : {tally['voted']}",
            f"  For                : {tally['for']}",
            f"  Against            : {tally['against']}",
            f"  Abstained          : {tally['abstained']}",
            f"Approval fraction    : {pct:.1f} %  "
            f"(threshold: {RATIFICATION_THRESHOLD * 100:.0f} %)",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory: create the founding consortium
# ---------------------------------------------------------------------------

def create_founding_consortium() -> ISCPConsortium:
    """
    Instantiate the ISCP Industry Consortium with the initial founding
    member organisations as specified in the ISCP issue.

    Returns
    -------
    ISCPConsortium
    """
    consortium = ISCPConsortium()

    founding_members: List[ConsortiumMember] = [
        # LEO Operators
        ConsortiumMember(
            name="SpaceX",
            category=MemberCategory.LEO_OPERATOR,
            country_code="US",
            contact_email="iscp-liaison@spacex.com",
        ),
        ConsortiumMember(
            name="Amazon (Project Kuiper)",
            category=MemberCategory.LEO_OPERATOR,
            country_code="US",
            contact_email="iscp-liaison@kuiper.amazon.com",
        ),
        ConsortiumMember(
            name="OneWeb",
            category=MemberCategory.LEO_OPERATOR,
            country_code="GB",
            contact_email="iscp-liaison@oneweb.net",
        ),
        # Space Agencies
        ConsortiumMember(
            name="NASA",
            category=MemberCategory.SPACE_AGENCY,
            country_code="US",
            contact_email="iscp-liaison@nasa.gov",
        ),
        ConsortiumMember(
            name="ESA",
            category=MemberCategory.SPACE_AGENCY,
            country_code="INT",
            contact_email="iscp-liaison@esa.int",
        ),
        ConsortiumMember(
            name="ISRO",
            category=MemberCategory.SPACE_AGENCY,
            country_code="IN",
            contact_email="iscp-liaison@isro.gov.in",
        ),
        # STM Entities
        ConsortiumMember(
            name="LeoLabs",
            category=MemberCategory.STM_ENTITY,
            country_code="US",
            contact_email="iscp-liaison@leolabs.space",
        ),
        ConsortiumMember(
            name="US Space Force (Space Delta 2)",
            category=MemberCategory.STM_ENTITY,
            country_code="US",
            contact_email="iscp-liaison@spaceforce.mil",
        ),
    ]

    for member in founding_members:
        consortium.add_member(member)

    return consortium
