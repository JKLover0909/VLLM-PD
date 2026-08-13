"""Gmail send-only integration for controlled email actions."""

from __future__ import annotations

import asyncio
import base64
import os
import re
import time
import unicodedata
from collections import OrderedDict
from dataclasses import dataclass, replace
from email.message import EmailMessage
from pathlib import Path
from typing import Any


class GmailSenderError(RuntimeError):
    """Raised when Gmail cannot be authorized or an email cannot be sent."""


SCOPES = ["https://www.googleapis.com/auth/gmail.send"]
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


@dataclass(frozen=True)
class EmailSendCommand:
    to_email: str
    data_question: str
    subject: str
    explicit_body: str = ""

    @property
    def has_explicit_body(self) -> bool:
        return bool(self.explicit_body.strip())


@dataclass(frozen=True)
class GmailSendResult:
    message_id: str
    to_email: str
    subject: str


@dataclass(frozen=True)
class EmailDraft:
    id: str
    session_id: str
    employee_id: str
    to_email: str
    subject: str
    body_text: str
    artifact_id: str = ""
    filename: str = ""
    content: str = ""
    media_type: str = ""
    status: str = "pending"
    message_id: str = ""


class EmailDraftStore:
    def __init__(self, *, max_items: int = 200, ttl_seconds: float = 15 * 60):
        self._items: "OrderedDict[str, tuple[float, EmailDraft]]" = OrderedDict()
        self._lock = asyncio.Lock()
        self.max_items = max(1, max_items)
        self.ttl_seconds = ttl_seconds

    async def put(self, draft: EmailDraft) -> None:
        async with self._lock:
            self._items[draft.session_id] = (
                time.monotonic() + self.ttl_seconds,
                draft,
            )
            self._items.move_to_end(draft.session_id)
            while len(self._items) > self.max_items:
                self._items.popitem(last=False)

    async def get(self, session_id: str) -> EmailDraft | None:
        async with self._lock:
            cached = self._live_item(session_id)
            if cached is None:
                return None
            self._items.move_to_end(session_id)
            return cached[1]

    async def claim_for_send(
        self,
        session_id: str,
        employee_id: str,
    ) -> tuple[EmailDraft | None, bool]:
        """Atomically claim one pending draft and report whether this caller won."""
        async with self._lock:
            cached = self._live_item(session_id)
            if cached is None:
                return None, False
            expiry_at, draft = cached
            if draft.employee_id != employee_id or draft.status != "pending":
                return draft, False
            claimed = replace(draft, status="sending")
            self._items[session_id] = (expiry_at, claimed)
            self._items.move_to_end(session_id)
            return claimed, True

    async def update_status(
        self,
        session_id: str,
        *,
        status: str,
        message_id: str = "",
    ) -> EmailDraft | None:
        async with self._lock:
            cached = self._live_item(session_id)
            if cached is None:
                return None
            _, draft = cached
            updated = replace(draft, status=status, message_id=message_id)
            self._items[session_id] = (
                time.monotonic() + self.ttl_seconds,
                updated,
            )
            self._items.move_to_end(session_id)
            return updated

    async def discard(self, session_id: str) -> EmailDraft | None:
        async with self._lock:
            cached = self._items.pop(session_id, None)
            return cached[1] if cached else None

    def _live_item(self, session_id: str) -> tuple[float, EmailDraft] | None:
        cached = self._items.get(session_id)
        if cached is None:
            return None
        expiry_at, _ = cached
        if time.monotonic() > expiry_at:
            self._items.pop(session_id, None)
            return None
        return cached


def is_email_confirm_request(question: str) -> bool:
    text = _normalize_text(question or "")
    return text in {
        "xac nhan gui email",
        "xac nhan gui mail",
        "xac nhan gui",
        "dong y gui email",
        "dong y gui mail",
        "confirm send email",
        "confirm send",
    } or any(marker in (question or "") for marker in ("メール送信を確定", "送信を確定"))


def is_email_cancel_request(question: str) -> bool:
    text = _normalize_text(question or "")
    return text in {
        "huy gui email",
        "huy gui mail",
        "huy ban nhap email",
        "huy ban nhap mail",
        "huy email",
        "huy mail",
        "cancel send email",
        "cancel email",
    } or any(marker in (question or "") for marker in ("メール送信をキャンセル", "送信をキャンセル"))


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _is_email_context_reference(value: str) -> bool:
    normalized = _normalize_text(value or "")
    reference_markers = (
        "thong tin nay",
        "noi dung nay",
        "ket qua nay",
        "cau tra loi nay",
        "bao cao nay",
        "report nay",
        "ban bao cao nay",
        "phan tren",
        "o tren",
        "vua roi",
        "ben tren",
        "this report",
        "このレポート",
        "この報告書",
        "この内容",
        "上記のレポート",
    )
    return any(marker in normalized for marker in reference_markers)


def parse_email_send_command(question: str) -> EmailSendCommand | None:
    """Parse simple Vietnamese/English email-send commands with explicit address."""

    text = (question or "").strip()
    normalized = _normalize_text(text)
    match = EMAIL_RE.search(text)
    has_send_intent = any(
        marker in normalized
        for marker in (
            "gui email",
            "gui mail",
            "send email",
            "email cho",
            "mail cho",
            "cho email",
            "toi email",
            "den email",
        )
    ) or any(
        marker in text
        for marker in (
            "メールで送信",
            "メールを送信",
            "メール送信",
            "メールで共有",
        )
    ) or (match is not None and "gui" in normalized and "email" in normalized)
    if not has_send_intent:
        return None

    if not match:
        # Một số câu safety/database có thể bị bước dịch hoặc normalize làm lẫn
        # chữ "email" vào nội dung kỹ thuật. Không biến các câu đó thành lỗi
        # Gmail; để guardrail/MES handler xử lý đúng phạm vi.
        if _looks_like_non_email_database_request(text, normalized):
            return None
        raise GmailSenderError(
            "Bạn cần nêu rõ địa chỉ email người nhận, ví dụ: gui email cho a@mkac.vn ..."
        )

    to_email = match.group(0)
    remaining = (text[: match.start()] + " " + text[match.end() :]).strip()
    explicit_body = _extract_explicit_body(remaining)
    explicit_subject = _extract_explicit_subject(remaining)
    if explicit_body:
        subject = explicit_subject or "Meibook - Thông báo"
        return EmailSendCommand(
            to_email=to_email,
            data_question=explicit_body,
            subject=subject,
            explicit_body=explicit_body,
        )

    if _is_email_context_reference(remaining):
        data_question = "báo cáo này" if "bao cao" in _normalize_text(remaining) else "thông tin này"
    else:
        remaining = re.sub(
            r"(?i)\b(send\s+email|email|mail)\b",
            " ",
            remaining,
        )
        remaining = re.sub(
            r"\b(gửi|gui|cho|to|tới|toi|đến|den|về|ve|báo|rằng|rang|nội dung|noi dung)\b",
            " ",
            remaining,
            flags=re.IGNORECASE,
        )
        data_question = re.sub(r"\s+", " ", remaining).strip(" :,-")
        if len(data_question) < 8 and not _is_email_context_reference(data_question):
            raise GmailSenderError(
                "Chưa rõ nội dung cần gửi. Hãy hỏi theo dạng: gửi email cho a@mkac.vn báo mã hàng ... có tổng bao nhiêu lỗi."
            )

    subject = _build_subject(data_question)
    return EmailSendCommand(
        to_email=to_email,
        data_question=data_question,
        subject=subject,
    )


def try_parse_email_send_command(question: str) -> EmailSendCommand | None:
    """Parse email commands for passive checks without turning false positives into errors."""
    try:
        return parse_email_send_command(question)
    except GmailSenderError:
        return None


def _looks_like_non_email_database_request(text: str, normalized: str) -> bool:
    original = text or ""
    safety_or_database_markers = (
        "error_events",
        "drop table",
        "update ",
        "bo qua moi gioi han",
        "bo qua tat ca gioi han",
        "liet ke toan bo",
        "100 000",
        "100.000",
    )
    if any(marker in normalized for marker in safety_or_database_markers):
        return True
    return bool("error_events" in original and ("全" in original or "列挙" in original))


def _extract_explicit_body(text_without_email: str) -> str:
    """Extract literal email content from commands like 'với nội dung: ...'."""
    body_markers = (
        r"nội\s+dung",
        r"noi\s+dung",
        r"body",
        r"message",
        r"tin\s+nhắn",
        r"tin\s+nhan",
    )
    body_marker = "|".join(body_markers)
    body_start = (
        rf"(?:(?:với|voi|có|co)\s*(?:{body_marker})"
        rf"\s*(?:là|la|:|-)?\s+|(?:{body_marker})\s*(?:là|la|:|-)\s*)"
    )
    subject_marker = (
        r"tiêu\s+đề|tieu\s+de|chủ\s+đề|chu\s+de|subject"
    )
    pattern = re.compile(
        rf"{body_start}(?P<body>.+)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text_without_email or "")
    if not match:
        return ""

    body = match.group("body").strip()
    trailing_subject = re.search(
        rf"\s+(?:{subject_marker})\s*(?:là|la|:|-)?\s*.+$",
        body,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if trailing_subject:
        body = body[: trailing_subject.start()].strip()
    return _clean_explicit_email_fragment(body)


def _extract_explicit_subject(text_without_email: str) -> str:
    """Extract an optional literal subject from 'tiêu đề/subject: ...'."""
    subject_marker = r"tiêu\s+đề|tieu\s+de|chủ\s+đề|chu\s+de|subject"
    body_marker = (
        r"nội\s+dung|noi\s+dung|body|message|tin\s+nhắn|tin\s+nhan"
    )
    body_start = (
        rf"(?:(?:với|voi|có|co)\s*(?:{body_marker})"
        rf"\s*(?:là|la|:|-)?\s+|(?:{body_marker})\s*(?:là|la|:|-)\s*)"
    )
    pattern = re.compile(
        rf"(?:{subject_marker})\s*(?:là|la|:|-)?\s*(?P<subject>.+?)"
        rf"(?=(?:\s+(?:và|va)?\s*{body_start})|$)",
        flags=re.IGNORECASE | re.DOTALL,
    )
    match = pattern.search(text_without_email or "")
    if not match:
        return ""
    return _clean_explicit_email_fragment(match.group("subject"))


def _clean_explicit_email_fragment(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", (value or "").strip())
    cleaned = cleaned.strip(" :,-;")
    cleaned = re.sub(r"^(?:là|la)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    return cleaned.strip(" :,-;")


def _build_subject(data_question: str) -> str:
    product_match = re.search(
        r"(?:mã\s+hàng|ma\s+hang|product)\s+([A-Za-z0-9_-]+)",
        data_question,
        flags=re.IGNORECASE,
    )
    lot_match = re.search(r"\b\d{6}(?:-\d{2})?-\d{3}(?:-\d{2})?\b", data_question)
    if product_match:
        return f"Meibook - Báo cáo MES mã hàng {product_match.group(1)}"
    if lot_match:
        return f"Meibook - Báo cáo MES Lot {lot_match.group(0)}"
    return "Meibook - Báo cáo dữ liệu"


class GmailSender:
    """Send plain-text email through the Gmail API using OAuth credentials."""

    def __init__(
        self,
        credentials_path: Path | str,
        token_path: Path | str,
        *,
        sender_email: str = "",
        enabled: bool = False,
        allow_interactive_auth: bool = False,
    ):
        self.credentials_path = Path(credentials_path)
        self.token_path = Path(token_path)
        self.sender_email = sender_email.strip()
        self.enabled = enabled
        self.allow_interactive_auth = allow_interactive_auth

    @classmethod
    def from_env(cls) -> "GmailSender | None":
        enabled = os.getenv("GMAIL_SEND_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        credentials_path = os.getenv("GMAIL_CREDENTIALS_PATH", "").strip()
        token_path = os.getenv("GMAIL_TOKEN_PATH", "data/gmail_token.json").strip()
        if not credentials_path:
            return None
        return cls(
            credentials_path=credentials_path,
            token_path=token_path,
            sender_email=os.getenv("GMAIL_SENDER_EMAIL", ""),
            enabled=enabled,
            allow_interactive_auth=os.getenv(
                "GMAIL_ALLOW_INTERACTIVE_AUTH",
                "false",
            ).lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def available(self) -> bool:
        return self.enabled and self.credentials_path.is_file()

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "credentials_path": str(self.credentials_path),
            "token_path": str(self.token_path),
            "token_exists": self.token_path.is_file(),
            "allow_interactive_auth": self.allow_interactive_auth,
            "sender_email": self.sender_email,
        }

    def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        *,
        attachments: list[dict[str, Any]] | None = None,
    ) -> GmailSendResult:
        if not self.enabled:
            raise GmailSenderError("Tính năng gửi Gmail đang tắt.")
        if not self.credentials_path.is_file():
            raise GmailSenderError(
                f"Không tìm thấy Gmail credentials tại {self.credentials_path}."
            )
        if not EMAIL_RE.fullmatch(to_email.strip()):
            raise GmailSenderError("Địa chỉ email người nhận không hợp lệ.")

        service = self._service()
        message = EmailMessage()
        message["To"] = to_email.strip()
        if self.sender_email:
            message["From"] = self.sender_email
        message["Subject"] = subject.strip() or "Meibook - Báo cáo dữ liệu"
        message.set_content(body, charset="utf-8")

        for item in attachments or []:
            filename = str(item.get("filename") or "report.html")
            content = item.get("content")
            if isinstance(content, str):
                payload_bytes = content.encode("utf-8")
            elif isinstance(content, bytes):
                payload_bytes = content
            else:
                continue
            media_type = str(item.get("media_type") or "text/html; charset=utf-8")
            maintype, _, subtype = media_type.partition(";")[0].partition("/")
            message.add_attachment(
                payload_bytes,
                maintype=maintype or "application",
                subtype=subtype or "octet-stream",
                filename=filename,
            )

        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii")

        try:
            response = (
                service.users()
                .messages()
                .send(userId="me", body={"raw": raw_message})
                .execute()
            )
        except Exception as exc:
            raise GmailSenderError(f"Gmail API gửi email thất bại: {exc}") from exc
        return GmailSendResult(
            message_id=str(response.get("id", "")),
            to_email=to_email.strip(),
            subject=message["Subject"],
        )

    def _service(self):
        try:
            from googleapiclient.discovery import build
        except ImportError as exc:
            raise GmailSenderError(
                "Thiếu thư viện Google Gmail API. Hãy cài requirements.txt hoặc rebuild Docker image."
            ) from exc

        credentials = self._credentials()
        return build("gmail", "v1", credentials=credentials, cache_discovery=False)

    def _credentials(self):
        try:
            from google.auth.transport.requests import Request
            from google.auth.exceptions import RefreshError
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
        except ImportError as exc:
            raise GmailSenderError(
                "Thiếu thư viện Google OAuth. Hãy cài requirements.txt hoặc rebuild Docker image."
            ) from exc

        credentials = None
        if self.token_path.is_file():
            credentials = Credentials.from_authorized_user_file(
                str(self.token_path),
                SCOPES,
            )

        if credentials and credentials.expired and credentials.refresh_token:
            try:
                credentials.refresh(Request())
            except RefreshError as exc:
                raise GmailSenderError(
                    "Gmail OAuth token đã hết hạn hoặc bị Google thu hồi. "
                    f"Hãy tạo lại token OAuth tại {self.token_path} bằng "
                    "scripts/init_gmail_oauth.py rồi restart app."
                ) from exc

        if not credentials or not credentials.valid:
            if not self.allow_interactive_auth:
                raise GmailSenderError(
                    f"Chưa có Gmail OAuth token tại {self.token_path}. "
                    "Hãy tạo token trước hoặc bật GMAIL_ALLOW_INTERACTIVE_AUTH=true "
                    "trong môi trường có thể mở trình duyệt."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_path),
                SCOPES,
            )
            credentials = flow.run_local_server(port=0)

        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(credentials.to_json(), encoding="utf-8")
        os.chmod(self.token_path, 0o600)
        return credentials
