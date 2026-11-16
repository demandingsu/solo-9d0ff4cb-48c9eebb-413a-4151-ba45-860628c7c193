"""Environment-driven application wiring."""
from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from .capture import CaptureService
from .diffing import RevisionDiff
from .domain import (
    ArchiveStorageError,
    InvalidNumber,
    ReviewError,
    SourceError,
    format_bytes,
    format_timestamp,
)
from .review import ReviewService
from .source import HttpSourceClient
from .store import ArchiveStore

BASE_DIR = Path(__file__).resolve().parent


def _env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value or default


def create_app(
    *,
    db_path: str | None = None,
    source_base_url: str | None = None,
    source_client=None,
) -> FastAPI:
    store = ArchiveStore(
        db_path or _env("ARCHIVE_DB_PATH", "./data/archive.db")
    )
    source = HttpSourceClient(
        source_base_url or _env("SOURCE_BASE_URL", "http://127.0.0.1:8000"),
        client=source_client,
    )
    service = CaptureService(store, source)
    review_service = ReviewService(store)
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
    templates.env.filters["bytes"] = format_bytes
    templates.env.filters["ts"] = format_timestamp

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        source.close()

    app = FastAPI(
        title="市政采购公告档案",
        docs_url=None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.store = store
    app.state.source = source
    app.state.service = service
    app.state.review_service = review_service

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return templates.TemplateResponse(
            request,
            "index.html",
            {
                "announcements": store.list_announcements(),
                "events": store.recent_events(),
                "captured": request.query_params.get("captured"),
                "error": request.query_params.get("error"),
            },
        )

    @app.post("/capture")
    def capture(number: str = Form(...)):
        try:
            result = service.capture(number)
        except InvalidNumber as exc:
            return _error_redirect("", exc.message)
        except (SourceError, ArchiveStorageError) as exc:
            return _error_redirect(number.strip(), exc.message)
        return RedirectResponse(
            f"/notice/{quote(result.number, safe='')}?status={result.status}",
            status_code=303,
        )

    @app.get("/notice/{number}", response_class=HTMLResponse)
    def notice(request: Request, number: str):
        versions = store.list_versions(number)
        if not versions:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"number": number, "error": request.query_params.get("error")},
                status_code=404,
            )
        status = request.query_params.get("status")
        pending = review_service.pending_revision(number)
        pending_diff = None
        if pending is not None:
            pending_latest, pending_previous = pending
            pending_diff = RevisionDiff.from_versions(
                pending_previous, pending_latest
            )
        return templates.TemplateResponse(
            request,
            "notice.html",
            {
                "number": number,
                "versions": versions,
                "latest": versions[0],
                "status": status,
                "status_message": _STATUS_MESSAGES.get(status),
                "events": store.events_for(number),
                "error": request.query_params.get("error"),
                "announcement": store.get_announcement(number),
                "pending_diff": pending_diff,
                "reviews": store.list_reviews(number),
            },
        )

    @app.get(
        "/notice/{number}/version/{revision}/diff",
        response_class=HTMLResponse,
    )
    def version_diff(request: Request, number: str, revision: int):
        try:
            diff, record, previous = review_service.build_diff(number, revision)
        except ReviewError as exc:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"number": number, "error": exc.message},
                status_code=404,
            )
        review = store.get_review(number, revision)
        return templates.TemplateResponse(
            request,
            "diff.html",
            {
                "number": number,
                "diff": diff,
                "paragraphs": diff.view_paragraphs(),
                "record": record,
                "previous": previous,
                "review": review,
                "is_latest": store.latest_version(number).revision == revision,
                "error": request.query_params.get("error"),
            },
        )

    @app.post("/notice/{number}/review")
    def submit_review(
        number: str,
        old_revision: str = Form(""),
        new_revision: int = Form(...),
        fingerprint: str = Form(...),
        reviewer: str = Form(""),
    ):
        try:
            old_value = int(old_revision) if old_revision.strip() else None
        except ValueError:
            old_value = -1  # fails validation in the service with a clear error
        try:
            review = review_service.confirm(
                number,
                old_value,
                new_revision,
                fingerprint,
                reviewer,
            )
        except ReviewError as exc:
            return _review_error_redirect(number, new_revision, exc.message)
        except ArchiveStorageError as exc:
            return _review_error_redirect(number, new_revision, exc.message)
        return RedirectResponse(
            f"/notice/{quote(number, safe='')}/review/{review['id']}"
            "?confirmed=1",
            status_code=303,
        )

    @app.get(
        "/notice/{number}/review/{review_id}",
        response_class=HTMLResponse,
    )
    def review_detail(request: Request, number: str, review_id: int):
        versions = store.list_versions(number)
        if not versions:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"number": number, "error": "该公告不存在"},
                status_code=404,
            )
        review = next(
            (item for item in store.list_reviews(number) if item["id"] == review_id),
            None,
        )
        if review is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"number": number, "error": "该核对确认记录不存在"},
                status_code=404,
            )
        try:
            snapshot = RevisionDiff.from_snapshot(json.loads(review["diff_snapshot"]))
        except (ValueError, KeyError) as exc:
            raise HTTPException(status_code=500, detail="核对快照无法解析") from exc
        return templates.TemplateResponse(
            request,
            "review.html",
            {
                "number": number,
                "review": review,
                "snapshot": snapshot,
                "paragraphs": snapshot.view_paragraphs(),
                "versions": versions,
                "confirmed": request.query_params.get("confirmed") == "1",
            },
        )

    @app.get("/notice/{number}/version/{revision}", response_class=HTMLResponse)
    def version(request: Request, number: str, revision: int):
        record = store.get_version(number, revision)
        if record is None:
            return templates.TemplateResponse(
                request,
                "not_found.html",
                {"number": number, "error": "该版本不存在"},
                status_code=404,
            )
        all_versions = store.list_versions(number)
        return templates.TemplateResponse(
            request,
            "version.html",
            {"record": record, "versions": all_versions},
        )

    @app.get("/notice/{number}/version/{revision}/attachment")
    def download(number: str, revision: int):
        record = store.get_version(number, revision)
        if record is None:
            raise HTTPException(status_code=404, detail="该版本附件不存在")
        return Response(
            content=record.attachment_blob,
            media_type=record.attachment_content_type,
            headers={
                "Content-Disposition": (
                    f'attachment; filename="{_ascii_filename(record.attachment_filename)}"; '
                    f"filename*=UTF-8''{quote(record.attachment_filename, safe='')}"
                ),
                "X-Content-SHA256": record.attachment_sha256,
                "Cache-Control": "private, max-age=0, must-revalidate",
            },
        )

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    return app


_STATUS_MESSAGES = {
    "created": "采集成功：已在同一事务中保存正文与附件。最新版本进入待核状态，请核对差异后确认。",
    "unchanged": "相同修订号且内容一致：未重复建版。",
    "conflict": "采集被拒：源站冲突，旧版未覆盖。",
    "retracted_update": "已记录源站撤稿状态变化，未新建版本。",
}


def _error_redirect(number: str, message: str) -> RedirectResponse:
    target = "/?error=" + quote(message, safe="")
    if number:
        target = f"/notice/{quote(number, safe='')}?error=" + quote(message, safe="")
    return RedirectResponse(target, status_code=303)


def _review_error_redirect(number: str, revision: int, message: str) -> RedirectResponse:
    target = (
        f"/notice/{quote(number, safe='')}/version/{revision}/diff?error="
        + quote(message, safe="")
    )
    return RedirectResponse(target, status_code=303)


def _ascii_filename(name: str) -> str:
    fallback = "".join(ch if 32 <= ord(ch) < 127 and ch not in '"\\' else "_" for ch in name)
    return fallback or "attachment.bin"

