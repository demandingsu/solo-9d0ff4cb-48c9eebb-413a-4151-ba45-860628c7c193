"""Paragraph-level comparison of two archived revisions.

The archivist checks what changed between the latest revision and the previous
archived one. Bodies are split into *natural paragraphs* (blocks separated by
blank lines). Matching is content-based with multiset counts, so a paragraph
that is byte-identical in both revisions — even when duplicated or moved to a
different position — is always reported as unchanged, never as an
addition/deletion.

Paragraphs left over after exact matching are paired as "modified" when they
look like edits of one another (greedy best-similarity pairing); the rest are
reported as pure additions or deletions. Attachment fingerprint changes are
described separately.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Any

from .domain import VersionRecord, format_bytes

# A natural paragraph is a block of text separated from the next block by one
# or more blank lines (optionally containing whitespace).
PARAGRAPH_SPLIT_RE = re.compile(r"\n[ \t\r\f\v]*\n")
# Below this character-similarity ratio two leftover paragraphs are treated as
# unrelated (pure add/delete) rather than as an edit of each other.
MODIFIED_MATCH_THRESHOLD = 0.5
# Two leftover paragraphs are treated as a modification only when they are
# clearly related. Real edits (most characters preserved) score well above the
# threshold, while an unrelated delete-plus-add pair that merely shares
# boilerplate (e.g. a common suffix) stays at or below it. Very short
# paragraphs get a lower threshold so single-character edits still pair.
MODIFIED_MATCH_THRESHOLD = 0.6
MODIFIED_SHORT_THRESHOLD = 0.5
MODIFIED_SHORT_PARAGRAPH_CHARS = 2


@dataclass(frozen=True)
class InlineSegment:
    """One piece of an inline (character-level) paragraph diff."""

    kind: str  # "equal" | "insert" | "delete"
    text: str


@dataclass(frozen=True)
class ParagraphChange:
    kind: str  # "added" | "deleted" | "modified" | "unchanged"
    old_text: str | None
    new_text: str | None
    old_index: int | None
    new_index: int | None
    segments: tuple[InlineSegment, ...] = ()


@dataclass(frozen=True)
class AttachmentChange:
    changed: bool
    is_first: bool
    old_filename: str | None
    new_filename: str | None
    old_content_type: str | None
    new_content_type: str | None
    old_size: int | None
    new_size: int | None
    old_sha256: str | None
    new_sha256: str | None
    details: tuple[str, ...]


@dataclass(frozen=True)
class RevisionDiff:
    number: str
    old_revision: int | None
    new_revision: int
    is_first: bool
    title_changed: bool
    old_title: str | None
    new_title: str
    added: tuple[ParagraphChange, ...]
    deleted: tuple[ParagraphChange, ...]
    modified: tuple[ParagraphChange, ...]
    unchanged: tuple[ParagraphChange, ...]
    attachment: AttachmentChange

    # ------------------------------------------------------------ construct

    @classmethod
    def from_versions(
        cls, old: VersionRecord | None, new: VersionRecord
    ) -> "RevisionDiff":
        if old is not None and old.announcement_number != new.announcement_number:
            raise ValueError("只能比对同一公告的两个版本")
        old_paragraphs = split_paragraphs(old.body) if old is not None else []
        new_paragraphs = split_paragraphs(new.body)
        unchanged, old_left, new_left = _match_identical(
            old_paragraphs, new_paragraphs
        )
        modified, deleted, added = _pair_leftovers(old_left, new_left)

        return cls(
            number=new.announcement_number,
            old_revision=old.revision if old is not None else None,
            new_revision=new.revision,
            is_first=old is None,
            title_changed=old is not None and old.title != new.title,
            old_title=old.title if old is not None else None,
            new_title=new.title,
            added=tuple(sorted(added, key=lambda c: (c.new_index is None, c.new_index))),
            deleted=tuple(
                sorted(deleted, key=lambda c: (c.old_index is None, c.old_index))
            ),
            modified=tuple(
                sorted(modified, key=lambda c: (c.new_index is None, c.new_index))
            ),
            unchanged=tuple(
                sorted(unchanged, key=lambda c: (c.new_index is None, c.new_index))
            ),
            attachment=_diff_attachment(old, new),
        )

    # --------------------------------------------------------------- counts

    @property
    def body_changed(self) -> bool:
        return bool(self.added or self.deleted or self.modified)

    @property
    def counts(self) -> dict[str, int]:
        return {
            "added": len(self.added),
            "deleted": len(self.deleted),
            "modified": len(self.modified),
            "unchanged": len(self.unchanged),
        }

    # ----------------------------------------------------- snapshot / hash

    def to_payload(self) -> dict[str, Any]:
        """Canonical, JSON-serialisable record of what was reviewed.

        Inline segments are deliberately excluded: they are derived from the
        paired paragraph texts, which are fully captured here. The fingerprint
        over this payload binds the confirmation to the actual differences.
        """
        paragraphs = [
            {
                "kind": change.kind,
                "old_index": change.old_index,
                "new_index": change.new_index,
                "old_text": change.old_text,
                "new_text": change.new_text,
            }
            for change in (
                *self.modified,
                *self.added,
                *self.deleted,
                *self.unchanged,
            )
        ]
        attachment = {
            "changed": self.attachment.changed,
            "is_first": self.attachment.is_first,
            "details": list(self.attachment.details),
            "old": None
            if self.attachment.old_sha256 is None
            else {
                "filename": self.attachment.old_filename,
                "content_type": self.attachment.old_content_type,
                "size": self.attachment.old_size,
                "sha256": self.attachment.old_sha256,
            },
            "new": {
                "filename": self.attachment.new_filename,
                "content_type": self.attachment.new_content_type,
                "size": self.attachment.new_size,
                "sha256": self.attachment.new_sha256,
            },
        }
        return {
            "number": self.number,
            "old_revision": self.old_revision,
            "new_revision": self.new_revision,
            "is_first": self.is_first,
            "title_changed": self.title_changed,
            "old_title": self.old_title,
            "new_title": self.new_title,
            "paragraphs": paragraphs,
            "attachment": attachment,
        }

    def to_snapshot_json(self) -> str:
        return json.dumps(
            self.to_payload(), ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(self.to_snapshot_json().encode("utf-8")).hexdigest()

    @classmethod
    def from_snapshot(cls, payload: dict[str, Any]) -> "RevisionDiff":
        """Rebuild a RevisionDiff from a stored confirmation snapshot."""
        groups: dict[str, list[ParagraphChange]] = {
            "added": [],
            "deleted": [],
            "modified": [],
            "unchanged": [],
        }
        for item in payload["paragraphs"]:
            change = ParagraphChange(
                kind=item["kind"],
                old_text=item["old_text"],
                new_text=item["new_text"],
                old_index=item["old_index"],
                new_index=item["new_index"],
            )
            groups[item["kind"]].append(change)
        attachment = payload["attachment"]
        old = attachment["old"] or {}
        new = attachment["new"] or {}
        return cls(
            number=payload["number"],
            old_revision=payload["old_revision"],
            new_revision=payload["new_revision"],
            is_first=payload["is_first"],
            title_changed=payload["title_changed"],
            old_title=payload["old_title"],
            new_title=payload["new_title"],
            added=tuple(groups["added"]),
            deleted=tuple(groups["deleted"]),
            modified=tuple(groups["modified"]),
            unchanged=tuple(groups["unchanged"]),
            attachment=AttachmentChange(
                changed=attachment["changed"],
                is_first=attachment["is_first"],
                old_filename=old.get("filename"),
                new_filename=new.get("filename"),
                old_content_type=old.get("content_type"),
                new_content_type=new.get("content_type"),
                old_size=old.get("size"),
                new_size=new.get("size"),
                old_sha256=old.get("sha256"),
                new_sha256=new.get("sha256"),
                details=tuple(attachment["details"]),
            ),
        )

    def view_paragraphs(self) -> list[dict[str, Any]]:
        """Paragraphs in display order, with inline segments filled in."""
        result: list[dict[str, Any]] = []
        for change in (*self.modified, *self.added, *self.deleted, *self.unchanged):
            segments = change.segments
            if not segments and change.kind == "modified":
                segments = inline_diff(change.old_text or "", change.new_text or "")
            result.append(
                {
                    "kind": change.kind,
                    "old_text": change.old_text,
                    "new_text": change.new_text,
                    "old_index": change.old_index,
                    "new_index": change.new_index,
                    "segments": [
                        {"kind": seg.kind, "text": seg.text} for seg in segments
                    ],
                }
            )
        return result


# ---------------------------------------------------------------- helpers


def split_paragraphs(body: str) -> list[str]:
    """Split body text into natural paragraphs.

    Blocks are separated by blank lines. Whitespace-only lines also count as
    separators, and surrounding whitespace on each block is trimmed.
    """
    normalized = body.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    return [
        block.strip()
        for block in PARAGRAPH_SPLIT_RE.split(normalized)
        if block.strip()
    ]


def inline_diff(old_text: str, new_text: str) -> tuple[InlineSegment, ...]:
    """Character-level diff used to mark edits inside a modified paragraph."""
    matcher = SequenceMatcher(None, old_text, new_text)
    segments: list[InlineSegment] = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            segments.append(InlineSegment("equal", old_text[i1:i2]))
        elif tag == "delete":
            segments.append(InlineSegment("delete", old_text[i1:i2]))
        elif tag == "insert":
            segments.append(InlineSegment("insert", new_text[j1:j2]))
        elif tag == "replace":
            if i1 != i2:
                segments.append(InlineSegment("delete", old_text[i1:i2]))
            if j1 != j2:
                segments.append(InlineSegment("insert", new_text[j1:j2]))
    return tuple(segments)


def _match_identical(
    old_paragraphs: list[str], new_paragraphs: list[str]
) -> tuple[
    list[ParagraphChange],
    list[tuple[int, str]],
    list[tuple[int, str]],
]:
    """Pair identical paragraphs by multiset count.

    Duplicates and re-ordered paragraphs are consumed here so they can never
    surface as additions or deletions.
    """
    remaining = Counter(old_paragraphs) & Counter(new_paragraphs)
    old_used = [False] * len(old_paragraphs)
    new_used = [False] * len(new_paragraphs)
    unchanged: list[ParagraphChange] = []

    for new_index, text in enumerate(new_paragraphs):
        if remaining.get(text, 0) <= 0:
            continue
        for old_index, candidate in enumerate(old_paragraphs):
            if not old_used[old_index] and candidate == text:
                old_used[old_index] = True
                new_used[new_index] = True
                remaining[text] -= 1
                unchanged.append(
                    ParagraphChange(
                        kind="unchanged",
                        old_text=text,
                        new_text=text,
                        old_index=old_index,
                        new_index=new_index,
                    )
                )
                break

    old_left = [
        (index, text)
        for index, text in enumerate(old_paragraphs)
        if not old_used[index]
    ]
    new_left = [
        (index, text)
        for index, text in enumerate(new_paragraphs)
        if not new_used[index]
    ]
    return unchanged, old_left, new_left


def _pair_leftovers(
    old_left: list[tuple[int, str]],
    new_left: list[tuple[int, str]],
) -> tuple[
    list[ParagraphChange],
    list[ParagraphChange],
    list[ParagraphChange],
]:
    """Greedy best-similarity pairing of the non-identical leftovers."""
    candidates: list[tuple[float, int, int]] = []
    for old_index, old_text in old_left:
        for new_index, new_text in new_left:
            matcher = SequenceMatcher(None, old_text, new_text)
            ratio = matcher.ratio()
            shorter = min(len(old_text), len(new_text))
            threshold = (
                MODIFIED_SHORT_THRESHOLD
                if shorter <= MODIFIED_SHORT_PARAGRAPH_CHARS
                else MODIFIED_MATCH_THRESHOLD
            )
            if ratio >= threshold:
                candidates.append((ratio, old_index, new_index))
    # Highest similarity first; indices make the ordering deterministic.
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    paired_old: set[int] = set()
    paired_new: set[int] = set()
    modified: list[ParagraphChange] = []
    for _ratio, old_index, new_index in candidates:
        if old_index in paired_old or new_index in paired_new:
            continue
        paired_old.add(old_index)
        paired_new.add(new_index)
        old_text = dict(old_left)[old_index]
        new_text = dict(new_left)[new_index]
        modified.append(
            ParagraphChange(
                kind="modified",
                old_text=old_text,
                new_text=new_text,
                old_index=old_index,
                new_index=new_index,
                segments=inline_diff(old_text, new_text),
            )
        )

    deleted = [
        ParagraphChange(
            kind="deleted",
            old_text=text,
            new_text=None,
            old_index=index,
            new_index=None,
        )
        for index, text in old_left
        if index not in paired_old
    ]
    added = [
        ParagraphChange(
            kind="added",
            old_text=None,
            new_text=text,
            old_index=None,
            new_index=index,
        )
        for index, text in new_left
        if index not in paired_new
    ]
    return modified, deleted, added


def _diff_attachment(
    old: VersionRecord | None, new: VersionRecord
) -> AttachmentChange:
    if old is None:
        return AttachmentChange(
            changed=False,
            is_first=True,
            old_filename=None,
            new_filename=new.attachment_filename,
            old_content_type=None,
            new_content_type=new.attachment_content_type,
            old_size=None,
            new_size=new.attachment_size,
            old_sha256=None,
            new_sha256=new.attachment_sha256,
            details=("首个存档版本，无前置附件可比对。",),
        )

    details: list[str] = []
    if old.attachment_sha256 != new.attachment_sha256:
        details.append(
            "附件 SHA-256 指纹变化："
            f"{old.attachment_sha256[:16]}… → {new.attachment_sha256[:16]}…"
        )
    if old.attachment_filename != new.attachment_filename:
        details.append(
            f"附件文件名变化：{old.attachment_filename} → {new.attachment_filename}"
        )
    if old.attachment_content_type != new.attachment_content_type:
        details.append(
            "附件类型变化："
            f"{old.attachment_content_type} → {new.attachment_content_type}"
        )
    if old.attachment_size != new.attachment_size:
        details.append(
            "附件大小变化："
            f"{format_bytes(old.attachment_size)} → {format_bytes(new.attachment_size)}"
        )

    changed = bool(details)
    if not changed:
        details.append(
            "附件未变化（文件名、类型、字节大小与 SHA-256 指纹均一致）。"
        )
    return AttachmentChange(
        changed=changed,
        is_first=False,
        old_filename=old.attachment_filename,
        new_filename=new.attachment_filename,
        old_content_type=old.attachment_content_type,
        new_content_type=new.attachment_content_type,
        old_size=old.attachment_size,
        new_size=new.attachment_size,
        old_sha256=old.attachment_sha256,
        new_sha256=new.attachment_sha256,
        details=tuple(details),
    )
