#!/usr/bin/env python3
"""Create a Google Calendar OAuth token for Meibook's Calendar integration."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from google_auth_oauthlib.flow import InstalledAppFlow


SCOPES = ["https://www.googleapis.com/auth/calendar.events"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--credentials",
        default=os.getenv("GMAIL_CREDENTIALS_PATH", "data/gmail_credentials.json"),
        help="OAuth client JSON path.",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("CALENDAR_TOKEN_PATH", "data/calendar_token.json"),
        help="Output token JSON path.",
    )
    parser.add_argument(
        "--host",
        default=os.getenv("GMAIL_OAUTH_HOST", "localhost"),
        help="OAuth redirect host sent to Google. Use localhost or 127.0.0.1.",
    )
    parser.add_argument(
        "--bind-host",
        default=os.getenv("GMAIL_OAUTH_BIND_HOST", "0.0.0.0"),
        help="Local callback server bind host. Use 0.0.0.0 inside Docker.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("GMAIL_OAUTH_PORT", "8080")),
        help="OAuth callback port.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    credentials_path = Path(args.credentials)
    token_path = Path(args.token)
    if args.host in {"0.0.0.0", "::"}:
        raise ValueError(
            "--host là OAuth redirect host gửi cho Google, không được dùng 0.0.0.0. "
            "Hãy dùng --host localhost và nếu chạy Docker thì thêm --bind-host 0.0.0.0."
        )
    if not credentials_path.is_file():
        raise FileNotFoundError(f"Không tìm thấy credentials: {credentials_path}")
    credentials_config = json.loads(credentials_path.read_text(encoding="utf-8"))
    if "installed" not in credentials_config:
        credential_type = "web" if "web" in credentials_config else "không xác định"
        raise ValueError(
            f"File credentials hiện là loại {credential_type}. Flow này cần OAuth Client ID "
            "loại Desktop app, file JSON phải có khóa top-level 'installed'. "
            "Hãy tải lại JSON từ OAuth client Desktop trong Google Cloud Console."
        )

    flow = InstalledAppFlow.from_client_secrets_file(
        str(credentials_path),
        SCOPES,
    )
    credentials = flow.run_local_server(
        host=args.host,
        bind_addr=args.bind_host,
        port=args.port,
        open_browser=False,
        authorization_prompt_message=(
            "\nMở URL này trên trình duyệt để cấp quyền Google Calendar:\n{url}\n\n"
        ),
        success_message="Đã cấp quyền Calendar. Bạn có thể đóng tab này.",
    )
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(credentials.to_json(), encoding="utf-8")
    os.chmod(token_path, 0o600)
    print(f"Đã lưu Calendar OAuth token vào: {token_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
