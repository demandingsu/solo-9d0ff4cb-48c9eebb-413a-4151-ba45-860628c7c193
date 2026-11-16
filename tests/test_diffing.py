"""Unit tests for paragraph-level revision diffing."""
from __future__ import annotations

import hashlib

from app.diffing import (
    MODIFIED_MATCH_THRESHOLD,
    RevisionDiff,
    inline_diff,
    split_paragraphs,
)
from app.domain import (
    AttachmentSnapshot,
    SourceSnapshot,
    VersionRecord,
)

NUMBER = "DIFF-0001"


def _record(
    revision: int,
    body: str,
    *,
    title: str = "测试公告",
    blob: bytes | None = None,
    filename: str | None = None,
) -> VersionRecord:
    blob = blob if blob is not None else f"blob-r{revision}".encode()
    return VersionRecord.from_snapshot(
        SourceSnapshot(
            number=NUMBER,
            title=title,
            body=body,
            body_sha256=hashlib.sha256(body.encode()).hexdigest(),
            revision=revision,
            body_revision=revision,
            attachment_revision=revision,
            retracted=False,
            source_url="http://demo.test/api/notices/x",
            attachment=AttachmentSnapshot(
                filename=filename or f"file-r{revision}.txt",
                content_type="text/plain",
                size=len(blob),
                sha256=hashlib.sha256(blob).hexdigest(),
                data=blob,
            ),
        ),
        captured_at="2026-01-01T00:00:00Z",
    )


def test_split_paragraphs_on_blank_lines():
    body = "第一段。\n仍然是第一段。\n\n  \n第二段。\n\n第三段。\n"
    assert split_paragraphs(body) == ["第一段。\n仍然是第一段。", "第二段。", "第三段。"]
    assert split_paragraphs("   \n  ") == []
    assert split_paragraphs("只有一段，没有空行。") == ["只有一段，没有空行。"]


def test_identical_paragraphs_never_reported_as_added_or_deleted():
    body = "一、项目概况\n采购设备一批。\n\n二、资格要求\n具有法人资格。\n\n三、时间安排\n截止 12 月 10 日。"
    diff = RevisionDiff.from_versions(_record(1, body), _record(2, body))
    assert diff.counts == {"added": 0, "deleted": 0, "modified": 0, "unchanged": 3}
    assert diff.body_changed is False


def test_duplicate_identical_paragraphs_are_not_false_changes():
    old_body = "通用条款。\n\n通用条款。\n\n特殊条款 A。"
    new_body = "通用条款。\n\n通用条款。\n\n特殊条款 B。"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    # The two duplicated common paragraphs must stay unchanged.
    assert diff.counts["unchanged"] == 2
    assert diff.counts["added"] == 0
    assert diff.counts["deleted"] == 0
    assert diff.counts["modified"] == 1
    assert diff.modified[0].old_text == "特殊条款 A。"
    assert diff.modified[0].new_text == "特殊条款 B。"


def test_reordered_identical_paragraphs_are_not_changes():
    old_body = "第一段。\n\n第二段。\n\n第三段。"
    new_body = "第三段。\n\n第一段。\n\n第二段。"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    assert diff.counts == {"added": 0, "deleted": 0, "modified": 0, "unchanged": 3}


def test_reorder_plus_one_edit_and_duplicate_counts():
    # Paragraph "B" appears twice; only one copy is edited, the other stays.
    old_body = "A\n\nB\n\nB\n\nC"
    new_body = "C\n\nB\n\nB-改\n\nA\n\nD"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    unchanged_texts = sorted(c.new_text for c in diff.unchanged)
    assert "A" in unchanged_texts and "C" in unchanged_texts
    assert sum(1 for t in unchanged_texts if t == "B") == 1
    assert any(c.new_text == "D" for c in diff.added)
    assert any(c.old_text == "B" and c.new_text == "B-改" for c in diff.modified)
    # No deletions: every old paragraph still has a matching new paragraph.
    assert diff.counts["deleted"] == 0


def test_added_deleted_and_modified_classification():
    old_body = "保留段落。\n\n将被修改的段落：旧表述。\n\n将被删除的段落。"
    new_body = "保留段落。\n\n将被修改的段落：新表述。\n\n全新增加的段落。"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    assert [c.new_text for c in diff.unchanged] == ["保留段落。"]
    assert len(diff.modified) == 1
    assert diff.modified[0].old_text.startswith("将被修改")
    assert len(diff.added) == 1 and diff.added[0].new_text == "全新增加的段落。"
    assert len(diff.deleted) == 1 and diff.deleted[0].old_text == "将被删除的段落。"


def test_inline_diff_marks_insert_and_delete():
    segments = inline_diff("投标截止 12 月 10 日", "投标截止 12 月 11 日")
    joined_old = "".join(s.text for s in segments if s.kind != "insert")
    joined_new = "".join(s.text for s in segments if s.kind != "delete")
    assert joined_old == "投标截止 12 月 10 日"
    assert joined_new == "投标截止 12 月 11 日"
    assert any(s.kind == "delete" and "0" in s.text for s in segments)
    assert any(s.kind == "insert" and "1" in s.text for s in segments)
    # Modified paragraphs carry inline segments in the diff result.
    diff = RevisionDiff.from_versions(
        _record(1, "投标截止 12 月 10 日"), _record(2, "投标截止 12 月 11 日")
    )
    assert diff.modified[0].segments


def test_completely_unrelated_paragraphs_are_add_delete_not_modified():
    old_body = "完全不同的旧内容，讲的是照明设备采购事项。"
    new_body = "毫不相干的新内容，关于绿化养护服务采购。"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    assert diff.counts["modified"] == 0
    assert diff.counts["added"] == 1
    assert diff.counts["deleted"] == 1


def test_attachment_fingerprint_change_marked_separately():
    old_blob = "旧附件".encode()
    new_blob = "新附件内容不同".encode()
    body = "正文完全没有变化。\n\n第二段也没有变化。"
    diff = RevisionDiff.from_versions(
        _record(1, body, blob=old_blob, filename="old.txt"),
        _record(2, body, blob=new_blob, filename="new.txt"),
    )
    assert diff.body_changed is False
    assert diff.counts["unchanged"] == 2
    assert diff.attachment.changed is True
    detail_text = "；".join(diff.attachment.details)
    assert "SHA-256" in detail_text
    assert diff.attachment.old_sha256 == hashlib.sha256(old_blob).hexdigest()
    assert diff.attachment.new_sha256 == hashlib.sha256(new_blob).hexdigest()


def test_unchanged_attachment_reported_as_unchanged():
    blob = "同一份附件".encode()
    body = "同一份正文。"
    diff = RevisionDiff.from_versions(
        _record(1, body, blob=blob, filename="same.txt"),
        _record(2, body, blob=blob, filename="same.txt"),
    )
    assert diff.attachment.changed is False
    assert "未变化" in diff.attachment.details[0]


def test_first_revision_marks_everything_without_predecessor():
    diff = RevisionDiff.from_versions(None, _record(1, "首段。\n\n次段。"))
    assert diff.is_first is True
    assert diff.old_revision is None
    assert diff.counts["added"] == 2
    assert diff.attachment.is_first is True
    assert diff.attachment.changed is False


def test_fingerprint_is_stable_and_snapshot_round_trips():
    old_body = "第一段保留。\n\n第二段旧版。"
    new_body = "第一段保留。\n\n第二段新版。"
    diff = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    assert len(diff.fingerprint) == 64
    # Deterministic: rebuilding the same diff yields the same fingerprint.
    rebuilt = RevisionDiff.from_versions(_record(1, old_body), _record(2, new_body))
    assert rebuilt.fingerprint == diff.fingerprint

    restored = RevisionDiff.from_snapshot(diff.to_payload())
    assert restored.fingerprint == diff.fingerprint
    assert restored.counts == diff.counts
    assert restored.attachment.new_sha256 == diff.attachment.new_sha256
    assert restored.modified[0].old_text == "第二段旧版。"


def test_fingerprint_changes_when_content_differs():
    base = "第一段保留。\n\n第二段旧版。"
    diff_a = RevisionDiff.from_versions(
        _record(1, base), _record(2, "第一段保留。\n\n第二段新版 A。")
    )
    diff_b = RevisionDiff.from_versions(
        _record(1, base), _record(2, "第一段保留。\n\n第二段新版 B。")
    )
    assert diff_a.fingerprint != diff_b.fingerprint


def test_title_change_is_recorded():
    diff = RevisionDiff.from_versions(
        _record(1, "正文。", title="旧标题"),
        _record(2, "正文。", title="新标题"),
    )
    assert diff.title_changed is True
    assert diff.old_title == "旧标题"
    assert diff.new_title == "新标题"
    assert diff.body_changed is False


def test_threshold_constant_is_sane_ratio():
    assert 0 < MODIFIED_MATCH_THRESHOLD < 1


def test_empty_body_revisions_diff_cleanly():
    diff = RevisionDiff.from_versions(_record(1, ""), _record(2, "新出现的正文。"))
    assert diff.counts["added"] == 1
    assert diff.counts["unchanged"] == 0
