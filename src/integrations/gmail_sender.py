"""Gmail send-only integration for controlled email actions."""

from __future__ import annotations

import base64
import os
import re
import unicodedata
from dataclasses import dataclass
from email.mime.text import MIMEText
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


@dataclass(frozen=True)
class GmailSendResult:
    message_id: str
    to_email: str
    subject: str


def _normalize_text(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    normalized = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return re.sub(r"\s+", " ", normalized).strip()


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
    ) or (match is not None and "gui" in normalized and "email" in normalized)
    if not has_send_intent:
        return None

    if not match:
        raise GmailSenderError(
            "Bạn cần nêu rõ địa chỉ email người nhận, ví dụ: gui email cho a@mkac.vn ..."
        )

    to_email = match.group(0)
    remaining = (text[: match.start()] + " " + text[match.end() :]).strip()
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
    if len(data_question) < 8:
        raise GmailSenderError(
            "Chưa rõ nội dung cần gửi. Hãy hỏi theo dạng: gửi email cho a@mkac.vn báo mã hàng ... có tổng bao nhiêu lỗi."
        )

    subject = _build_subject(data_question)
    return EmailSendCommand(
        to_email=to_email,
        data_question=data_question,
        subject=subject,
    )


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

    def send_email(self, to_email: str, subject: str, body: str) -> GmailSendResult:
        if not self.enabled:
            raise GmailSenderError("Tính năng gửi Gmail đang tắt.")
        if not self.credentials_path.is_file():
            raise GmailSenderError(
                f"Không tìm thấy Gmail credentials tại {self.credentials_path}."
            )
        if not EMAIL_RE.fullmatch(to_email.strip()):
            raise GmailSenderError("Địa chỉ email người nhận không hợp lệ.")

        service = self._service()
        message = MIMEText(body, "plain", "utf-8")
        message["To"] = to_email.strip()
        if self.sender_email:
            message["From"] = self.sender_email
        message["Subject"] = subject.strip() or "Meibook - Báo cáo dữ liệu"
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
            credentials.refresh(Request())

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
