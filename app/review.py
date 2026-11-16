"""Review workflow: build the inter-revision diff and record confirmations.

A confirmation is bound to both revisions and to the exact differences the
archivist reviewed (a canonical snapshot plus its SHA-256 fingerprint). If a
new revision is captured before the form is submitted, the store refuses the
stale confirmation inside one ``BEGIN IMMEDIATE`` transaction, so the newest
revision stays pending while earlier confirmations remain visible.
"""
from __future__ import annotations

from .diffing import RevisionDiff
from .domain import (
    ReviewAlreadyConfirmedError,
    ReviewError,
    ReviewMismatchError,
    StaleReviewError,
    VersionRecord,
    normalize_number,
    utc_now_iso,
)
from .store import ArchiveStore

REVIEWER_MAX_CHARS = 100


def _clean_reviewer(value: str | None) -> str:
    reviewer = (value or "").strip()
    if not reviewer:
        return "档案员"
    # Keep names single-line and bounded; no HTML escaping needed at this layer
    # because Jinja autoescapes when rendering.
    reviewer = reviewer.replace("\r", " ").replace("\n", " ").strip()
    return reviewer[:REVIEWER_MAX_CHARS] or "档案员"


class ReviewService:
    def __init__(self, store: ArchiveStore):
        self.store = store

    # ---------------------------------------------------------------- reads

    def build_diff(self, number: str, revision: int) -> tuple[RevisionDiff, VersionRecord, VersionRecord | None]:
        """Return the diff of one archived revision against its predecessor."""
        number = normalize_number(number)
        record = self.store.get_version(number, revision)
        if record is None:
            raise ReviewMismatchError(f"档案中不存在 r{revision}，无法核对。")
        previous = self._predecessor(number, revision)
        return RevisionDiff.from_versions(previous, record), record, previous

    def pending_revision(self, number: str) -> tuple[VersionRecord, VersionRecord | None] | None:
        """The latest revision together with its predecessor, if still pending.

        Returns ``None`` when the latest revision already has a confirmation
        (or the announcement is unknown).
        """
        number = normalize_number(number)
        latest = self.store.latest_version(number)
        if latest is None:
            return None
        if self.store.get_review(number, latest.revision) is not None:
            return None
        previous = self._predecessor(number, latest.revision)
        return latest, previous

    def _predecessor(self, number: str, revision: int) -> VersionRecord | None:
        versions = self.store.list_versions(number)
        previous: VersionRecord | None = None
        for version in reversed(versions):  # ascending revision order
            if version.revision < revision:
                previous = version
            else:
                break
        return previous

    # --------------------------------------------------------------- write

    def confirm(
        self,
        raw_number: str,
        old_revision: int | None,
        new_revision: int,
        expected_fingerprint: str,
        reviewer: str | None,
    ) -> dict:
        """Validate and store a review confirmation.

        The fingerprint submitted by the browser is recomputed from the stored
        revisions inside the write transaction, so tampering with the form or a
        change in either revision makes the confirmation fail.
        """
        number = normalize_number(raw_number)
        new_record = self.store.get_version(number, new_revision)
        if new_record is None:
            raise ReviewMismatchError(
                f"确认被拒：档案中不存在 r{new_revision}，未写入核对记录。"
            )
        old_record = (
            self.store.get_version(number, old_revision)
            if old_revision is not None
            else None
        )
        if old_revision is not None and old_record is None:
            raise ReviewMismatchError(
                f"确认被拒：前一存档版 r{old_revision} 不存在，未写入核对记录。"
            )

        actual_predecessor = self._predecessor(number, new_revision)
        if (actual_predecessor.revision if actual_predecessor else None) != old_revision:
            expected = actual_predecessor.revision if actual_predecessor else None
            raise ReviewMismatchError(
                "确认被拒：提交的前一版与档案中的相邻存档版不一致"
                f"（应为 r{expected}），未写入核对记录。"
            )

        diff = RevisionDiff.from_versions(old_record, new_record)
        if diff.fingerprint != (expected_fingerprint or "").strip():
            raise ReviewMismatchError(
                "确认被拒：提交的差异指纹与档案实际差异不一致，"
                "请重新打开核对页面后再确认；未写入核对记录。"
            )

        # The store re-checks everything inside a write transaction so that a
        # capture landing between these reads cannot sneak past the guard.
        self.store.save_review_confirmation(
            number=number,
            old_revision=old_revision,
            new_revision=new_revision,
            reviewer=_clean_reviewer(reviewer),
            diff_snapshot=diff.to_snapshot_json(),
            diff_fingerprint=diff.fingerprint,
            body_sha256_old=old_record.body_sha256 if old_record else None,
            body_sha256_new=new_record.body_sha256,
            attachment_sha256_old=(
                old_record.attachment_sha256 if old_record else None
            ),
            attachment_sha256_new=new_record.attachment_sha256,
            confirmed_at=utc_now_iso(),
        )
        review = self.store.get_review(number, new_revision)
        assert review is not None
        return review


__all__ = [
    "ReviewService",
    "ReviewError",
    "StaleReviewError",
    "ReviewMismatchError",
    "ReviewAlreadyConfirmedError",
]
