"""Review-confirmation workflow: diffs, stale guards, failure handling, UI."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.diffing import RevisionDiff
from app.domain import (
    ArchiveStorageError,
    ReviewAlreadyConfirmedError,
    ReviewMismatchError,
    StaleReviewError,
)
from app.main import create_app
from app.review import ReviewService
from app.store import ArchiveStore

NUMBER = "ZFCG-2026-0001"


def _edit(demo, *, body=None, title=None, attachment=None):
    data = {
        "title": title or "市政道路照明设备采购招标公告",
        "body": body if body is not None else "修订后的正文。",
    }
    files = None
    if attachment is not None:
        files = {"attachment": ("附件.txt", attachment, "text/plain; charset=utf-8")}
    response = demo.post(
        f"/notice/{NUMBER}/edit", data=data, files=files, follow_redirects=False
    )
    assert response.status_code == 303, response.text


@pytest.fixture()
def review_service(store: ArchiveStore) -> ReviewService:
    return ReviewService(store)


def _capture_two_revisions(service, demo):
    first = service.capture(NUMBER)
    _edit(demo, body="第一段保留。\n\n第二段被修改：新表述。\n\n第三段也保留。")
    second = service.capture(NUMBER)
    assert second.status == "created"
    return first, second


# ------------------------------------------------------------- service tests


def test_pending_diff_compares_latest_with_predecessor(service, review_service, store, demo):
    service.capture(NUMBER)
    # r2 establishes a body, r3 edits one paragraph while keeping two verbatim.
    _edit(demo, body="第一段保留。\n\n第二段旧表述。\n\n第三段也保留。")
    service.capture(NUMBER)
    _edit(demo, body="第一段保留。\n\n第二段新表述。\n\n第三段也保留。")
    service.capture(NUMBER)

    pending = review_service.pending_revision(NUMBER)
    assert pending is not None
    latest, previous = pending
    assert latest.revision == 3 and previous.revision == 2

    diff, record, old = review_service.build_diff(NUMBER, 3)
    assert record.revision == 3 and old.revision == 2
    assert diff.old_revision == 2 and diff.new_revision == 3
    # The two paragraphs carried over verbatim stay unchanged; one is modified.
    assert diff.counts["unchanged"] == 2
    assert diff.counts["modified"] == 1
    assert diff.counts["added"] == 0 and diff.counts["deleted"] == 0
    assert diff.modified[0].old_text == "第二段旧表述。"
    assert diff.modified[0].new_text == "第二段新表述。"


def test_confirm_binds_both_revisions_and_actual_diff(
    service, review_service, store, demo
):
    _capture_two_revisions(service, demo)
    diff, new_record, old_record = review_service.build_diff(NUMBER, 2)

    review = review_service.confirm(
        NUMBER, 1, 2, diff.fingerprint, "张档案"
    )
    assert review["announcement_number"] == NUMBER
    assert review["old_revision"] == 1
    assert review["new_revision"] == 2
    assert review["reviewer"] == "张档案"
    assert review["diff_fingerprint"] == diff.fingerprint
    assert review["body_sha256_old"] == old_record.body_sha256
    assert review["body_sha256_new"] == new_record.body_sha256
    assert review["attachment_sha256_old"] == old_record.attachment_sha256
    assert review["attachment_sha256_new"] == new_record.attachment_sha256

    # The stored snapshot reconstructs the exact same diff and fingerprint.
    stored = store.get_review(NUMBER, 2)
    rebuilt = RevisionDiff.from_snapshot(
        __import__("json").loads(stored["diff_snapshot"])
    )
    assert rebuilt.fingerprint == diff.fingerprint
    assert rebuilt.counts == diff.counts

    # The latest revision is no longer pending.
    assert review_service.pending_revision(NUMBER) is None


def test_duplicate_confirmation_is_rejected(
    service, review_service, demo
):
    _capture_two_revisions(service, demo)
    diff, _, _ = review_service.build_diff(NUMBER, 2)
    review_service.confirm(NUMBER, 1, 2, diff.fingerprint, None)
    with pytest.raises(ReviewAlreadyConfirmedError):
        review_service.confirm(NUMBER, 1, 2, diff.fingerprint, "另一人")


def test_stale_confirmation_refused_when_newer_revision_captured(
    service, review_service, store, demo
):
    _capture_two_revisions(service, demo)
    diff_r2, _, _ = review_service.build_diff(NUMBER, 2)

    # The archivist keeps the r1->r2 form open; meanwhile a third revision lands.
    _edit(demo, body="第一段保留。\n\n第二段被修改：再次更新。")
    third = service.capture(NUMBER)
    assert third.status == "created" and third.revision == 3

    with pytest.raises(StaleReviewError):
        review_service.confirm(NUMBER, 1, 2, diff_r2.fingerprint, "张档案")

    # r3 stays pending and no confirmation row for r2 was written.
    pending = review_service.pending_revision(NUMBER)
    assert pending is not None and pending[0].revision == 3
    assert store.get_review(NUMBER, 2) is None
    assert [r["new_revision"] for r in store.list_reviews(NUMBER)] == []


def test_old_confirmation_remains_visible_after_newer_capture(
    service, review_service, store, demo
):
    _capture_two_revisions(service, demo)
    diff_r2, _, _ = review_service.build_diff(NUMBER, 2)
    review_service.confirm(NUMBER, 1, 2, diff_r2.fingerprint, "张档案")

    _edit(demo, body="又一次改稿后的正文。")
    third = service.capture(NUMBER)
    assert third.revision == 3

    # The new revision is pending again, but the old confirmation survives.
    pending = review_service.pending_revision(NUMBER)
    assert pending is not None and pending[0].revision == 3
    old_review = store.get_review(NUMBER, 2)
    assert old_review is not None and old_review["reviewer"] == "张档案"
    assert [r["new_revision"] for r in store.list_reviews(NUMBER)] == [2]


def test_wrong_fingerprint_rejected(service, review_service, demo):
    _capture_two_revisions(service, demo)
    with pytest.raises(ReviewMismatchError):
        review_service.confirm(NUMBER, 1, 2, "0" * 64, "张档案")
    assert review_service.pending_revision(NUMBER) is not None


def test_wrong_predecessor_rejected(service, review_service, demo):
    _capture_two_revisions(service, demo)
    diff, _, _ = review_service.build_diff(NUMBER, 2)
    with pytest.raises(ReviewMismatchError):
        review_service.confirm(NUMBER, 99, 2, diff.fingerprint, "张档案")


def test_first_revision_confirm_without_predecessor(
    service, review_service, store, demo
):
    service.capture(NUMBER)
    diff, _, previous = review_service.build_diff(NUMBER, 1)
    assert previous is None and diff.is_first
    review = review_service.confirm(NUMBER, None, 1, diff.fingerprint, None)
    assert review["old_revision"] is None
    assert review["reviewer"] == "档案员"  # blank name defaults
    assert review_service.pending_revision(NUMBER) is None


def test_repeated_and_reordered_paragraphs_not_false_changes(
    service, review_service, demo
):
    service.capture(NUMBER)
    shared_body = "通用条款。\n\n通用条款。\n\n第一段。\n\n第二段。"
    reordered = "第二段。\n\n通用条款。\n\n第一段。\n\n通用条款。"
    # First publish the shared body as r2, then the re-ordered copy as r3:
    # both revisions contain exactly the same paragraph multiset.
    _edit(demo, body=shared_body)
    service.capture(NUMBER)
    _edit(demo, body=reordered)
    service.capture(NUMBER)
    diff, _, _ = review_service.build_diff(NUMBER, 3)
    assert diff.counts == {"added": 0, "deleted": 0, "modified": 0, "unchanged": 4}


def test_attachment_only_change_is_marked_separately(
    service, review_service, demo
):
    service.capture(NUMBER)
    new_blob = "替换后的附件内容".encode("utf-8")
    response = demo.post(
        f"/notice/{NUMBER}/attachment",
        files={"attachment": ("新版.txt", new_blob, "text/plain; charset=utf-8")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    service.capture(NUMBER)

    diff, record, old = review_service.build_diff(NUMBER, 2)
    assert diff.body_changed is False
    assert diff.attachment.changed is True
    assert diff.attachment.old_sha256 == old.attachment_sha256
    assert diff.attachment.new_sha256 == record.attachment_sha256
    assert any("SHA-256" in d for d in diff.attachment.details)


def test_confirmation_write_failure_keeps_pending_and_versions_usable(
    service, review_service, store, demo, monkeypatch
):
    _capture_two_revisions(service, demo)
    diff_before, _, _ = review_service.build_diff(NUMBER, 2)
    real_connect = store._connect

    class FailingConnection:
        def __init__(self, conn):
            self._conn = conn

        def __getattr__(self, name):
            return getattr(self._conn, name)

        def execute(self, sql, *params):
            if "INSERT INTO review_confirmations" in sql:
                raise sqlite3.OperationalError("disk I/O error")
            return self._conn.execute(sql, *params)

    monkeypatch.setattr(store, "_connect", lambda: FailingConnection(real_connect()))
    with pytest.raises(ArchiveStorageError):
        review_service.confirm(NUMBER, 1, 2, diff_before.fingerprint, "张档案")
    monkeypatch.setattr(store, "_connect", real_connect)

    # Still pending, no review row, and original bodies/attachments still read.
    assert review_service.pending_revision(NUMBER) is not None
    assert store.get_review(NUMBER, 2) is None
    for revision in (1, 2):
        record = store.get_version(NUMBER, revision)
        assert record is not None and record.attachment_blob


def test_interleaved_capture_and_review_flow(service, review_service, demo):
    """Capture r1 -> confirm r1 -> capture r2 -> confirm r2 -> capture r3 (pending)."""
    r1 = service.capture(NUMBER)
    diff1, _, _ = review_service.build_diff(NUMBER, r1.revision)
    review_service.confirm(NUMBER, None, 1, diff1.fingerprint, "甲")

    _edit(demo, body="第二版正文，与第一版不同。")
    r2 = service.capture(NUMBER)
    diff2, _, _ = review_service.build_diff(NUMBER, r2.revision)
    assert diff2.old_revision == 1
    review_service.confirm(NUMBER, 1, 2, diff2.fingerprint, "乙")

    _edit(demo, body="第三版正文，采入后应重新待核。")
    r3 = service.capture(NUMBER)
    assert r3.revision == 3
    pending = review_service.pending_revision(NUMBER)
    assert pending is not None and pending[0].revision == 3

    # Both older confirmations remain queryable in order.
    reviews = review_service.store.list_reviews(NUMBER)
    assert [r["new_revision"] for r in reviews] == [2, 1]
    assert [r["reviewer"] for r in reviews] == ["乙", "甲"]


# ----------------------------------------------------------------- web tests


def _web_app(tmp_path: Path, demo):
    return create_app(
        db_path=str(tmp_path / "review-web.db"),
        source_base_url="http://demo.test",
        source_client=demo,
    )


def test_web_review_happy_path(tmp_path, service, demo):
    # Use the service-backed store through the web app's own store by capturing
    # via the web client itself.
    app = _web_app(tmp_path, demo)
    with TestClient(app) as client:
        client.post("/capture", data={"number": NUMBER})
        _edit(demo, body="第一段保留。\n\n第二段修改后的内容。")
        client.post("/capture", data={"number": NUMBER})

        notice_page = client.get(f"/notice/{NUMBER}")
        assert "待核" in notice_page.text
        diff_page = client.get(f"/notice/{NUMBER}/version/2/diff")
        assert diff_page.status_code == 200
        assert "差异核对" in diff_page.text
        assert "附件指纹核对" in diff_page.text

        # Extract the fingerprint rendered in the hidden form field.
        import re

        match = re.search(
            r'name="fingerprint" value="([0-9a-f]{64})"', diff_page.text
        )
        assert match
        response = client.post(
            f"/notice/{NUMBER}/review",
            data={
                "old_revision": "1",
                "new_revision": "2",
                "fingerprint": match.group(1),
                "reviewer": "李档案",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        location = response.headers["location"]
        assert "/review/" in location

        record_page = client.get(location)
        assert record_page.status_code == 200
        assert "核对确认记录" in record_page.text
        assert "李档案" in record_page.text

        # The notice page now shows the reviewed state.
        notice_after = client.get(f"/notice/{NUMBER}")
        assert "已核对确认" in notice_after.text
        assert "待核" not in notice_after.text


def test_web_stale_confirmation_shows_error_and_keeps_pending(tmp_path, demo):
    app = _web_app(tmp_path, demo)
    with TestClient(app) as client:
        client.post("/capture", data={"number": NUMBER})
        _edit(demo, body="第二版正文。")
        client.post("/capture", data={"number": NUMBER})
        diff_page = client.get(f"/notice/{NUMBER}/version/2/diff")
        import re

        fingerprint = re.search(
            r'name="fingerprint" value="([0-9a-f]{64})"', diff_page.text
        ).group(1)

        # A third revision is captured before the form is submitted.
        _edit(demo, body="第三版正文。")
        client.post("/capture", data={"number": NUMBER})

        response = client.post(
            f"/notice/{NUMBER}/review",
            data={
                "old_revision": "1",
                "new_revision": "2",
                "fingerprint": fingerprint,
                "reviewer": "李档案",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        error_page = client.get(response.headers["location"])
        assert error_page.status_code == 200
        assert "确认已过期" in error_page.text

        notice_page = client.get(f"/notice/{NUMBER}")
        assert "待核" in notice_page.text


def test_web_diff_and_downloads_still_available_after_failed_confirm(
    tmp_path, demo, monkeypatch
):
    app = _web_app(tmp_path, demo)
    with TestClient(app) as client:
        client.post("/capture", data={"number": NUMBER})
        _edit(demo, body="第二版正文。")
        client.post("/capture", data={"number": NUMBER})
        diff_page = client.get(f"/notice/{NUMBER}/version/2/diff")
        import re

        fingerprint = re.search(
            r'name="fingerprint" value="([0-9a-f]{64})"', diff_page.text
        ).group(1)

        real_connect = app.state.store._connect

        class FailingConnection:
            def __init__(self, conn):
                self._conn = conn

            def __getattr__(self, name):
                return getattr(self._conn, name)

            def execute(self, sql, *params):
                if "INSERT INTO review_confirmations" in sql:
                    raise sqlite3.OperationalError("disk I/O error")
                return self._conn.execute(sql, *params)

        monkeypatch.setattr(
            app.state.store,
            "_connect",
            lambda: FailingConnection(real_connect()),
        )
        response = client.post(
            f"/notice/{NUMBER}/review",
            data={
                "old_revision": "1",
                "new_revision": "2",
                "fingerprint": fingerprint,
                "reviewer": "李档案",
            },
            follow_redirects=False,
        )
        assert response.status_code == 303
        error_page = client.get(response.headers["location"])
        assert "写入失败" in error_page.text
        monkeypatch.setattr(app.state.store, "_connect", real_connect)

        # Pending state unchanged; both revisions' bodies and attachments work.
        assert "待核" in client.get(f"/notice/{NUMBER}").text
        for revision in (1, 2):
            download = client.get(
                f"/notice/{NUMBER}/version/{revision}/attachment"
            )
            assert download.status_code == 200 and download.content
            page = client.get(f"/notice/{NUMBER}/version/{revision}")
            assert page.status_code == 200


def test_web_review_record_page_renders_snapshot(tmp_path, demo):
    app = _web_app(tmp_path, demo)
    with TestClient(app) as client:
        client.post("/capture", data={"number": NUMBER})
        diff_page = client.get(f"/notice/{NUMBER}/version/1/diff")
        import re

        fingerprint = re.search(
            r'name="fingerprint" value="([0-9a-f]{64})"', diff_page.text
        ).group(1)
        response = client.post(
            f"/notice/{NUMBER}/review",
            data={
                "old_revision": "",
                "new_revision": "1",
                "fingerprint": fingerprint,
                "reviewer": "",
            },
            follow_redirects=False,
        )
        page = client.get(response.headers["location"])
        assert "首个存档版" in page.text
        assert "差异快照" in page.text


def test_web_duplicate_review_confirmation_rejected(tmp_path, demo):
    app = _web_app(tmp_path, demo)
    with TestClient(app) as client:
        client.post("/capture", data={"number": NUMBER})
        diff_page = client.get(f"/notice/{NUMBER}/version/1/diff")
        import re

        fingerprint = re.search(
            r'name="fingerprint" value="([0-9a-f]{64})"', diff_page.text
        ).group(1)
        payload = {
            "old_revision": "",
            "new_revision": "1",
            "fingerprint": fingerprint,
            "reviewer": "甲",
        }
        first = client.post(f"/notice/{NUMBER}/review", data=payload,
                            follow_redirects=False)
        assert first.status_code == 303 and "/review/" in first.headers["location"]
        second = client.post(f"/notice/{NUMBER}/review", data=payload,
                             follow_redirects=False)
        assert second.status_code == 303
        error_page = client.get(second.headers["location"])
        assert "已有核对确认记录" in error_page.text


def test_store_refuses_confirmation_for_non_latest_revision(
    service, review_service, store, demo
):
    """The store itself rejects a stale revision even if the service check is
    bypassed (e.g. a capture lands between the service read and the write)."""
    service.capture(NUMBER)
    _edit(demo, body="第二版正文。")
    service.capture(NUMBER)
    diff_r2, new2, old1 = review_service.build_diff(NUMBER, 2)

    # A third revision lands after the diff was rendered.
    _edit(demo, body="第三版正文。")
    service.capture(NUMBER)

    with pytest.raises(StaleReviewError):
        store.save_review_confirmation(
            number=NUMBER,
            old_revision=1,
            new_revision=2,
            reviewer="张档案",
            diff_snapshot=diff_r2.to_snapshot_json(),
            diff_fingerprint=diff_r2.fingerprint,
            body_sha256_old=old1.body_sha256,
            body_sha256_new=new2.body_sha256,
            attachment_sha256_old=old1.attachment_sha256,
            attachment_sha256_new=new2.attachment_sha256,
            confirmed_at="2026-02-01T00:00:00Z",
        )
    # Nothing was written and r3 is the pending revision.
    assert store.list_reviews(NUMBER) == []
    pending = review_service.pending_revision(NUMBER)
    assert pending[0].revision == 3


def test_store_rejects_tampered_body_fingerprint(service, review_service, store, demo):
    service.capture(NUMBER)
    _edit(demo, body="第二版正文。")
    service.capture(NUMBER)
    diff, new2, old1 = review_service.build_diff(NUMBER, 2)
    with pytest.raises(ReviewMismatchError):
        store.save_review_confirmation(
            number=NUMBER,
            old_revision=1,
            new_revision=2,
            reviewer="张档案",
            diff_snapshot=diff.to_snapshot_json(),
            diff_fingerprint=diff.fingerprint,
            body_sha256_old=old1.body_sha256,
            body_sha256_new="f" * 64,  # tampered
            attachment_sha256_old=old1.attachment_sha256,
            attachment_sha256_new=new2.attachment_sha256,
            confirmed_at="2026-02-01T00:00:00Z",
        )
    assert store.list_reviews(NUMBER) == []


def test_schema_initialization_is_idempotent(archive_db: Path):
    first = ArchiveStore(archive_db)
    first.initialize()
    second = ArchiveStore(archive_db)  # reopen existing database
    second.initialize()
    # Review table exists and is usable.
    assert second.latest_version("ANY") is None
