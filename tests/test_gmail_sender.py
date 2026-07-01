import pytest

from src.integrations.gmail_sender import (
    GmailSender,
    GmailSenderError,
    parse_email_send_command,
)


def test_parse_vietnamese_email_send_command_preserves_data_question():
    command = parse_email_send_command(
        "Gửi email cho test@example.com báo mã hàng 3736-0008 có tổng bao nhiêu lỗi"
    )

    assert command is not None
    assert command.to_email == "test@example.com"
    assert command.data_question == "mã hàng 3736-0008 có tổng bao nhiêu lỗi"
    assert command.subject == "Meibook - Báo cáo MES mã hàng 3736-0008"


def test_parse_non_email_question_returns_none():
    assert parse_email_send_command("Mã hàng 3736-0008 có tổng bao nhiêu lỗi?") is None


def test_parse_contextual_email_send_command():
    command = parse_email_send_command(
        "Gửi thông tin này cho email test@example.com"
    )

    assert command is not None
    assert command.to_email == "test@example.com"
    assert command.data_question == "thông tin này"


def test_parse_explicit_email_body_does_not_become_data_question():
    command = parse_email_send_command(
        "Gửi email test đến test@example.com với nội dung: Đây là email kiểm tra Gmail OAuth."
    )

    assert command is not None
    assert command.to_email == "test@example.com"
    assert command.has_explicit_body is True
    assert command.explicit_body == "Đây là email kiểm tra Gmail OAuth."
    assert command.data_question == "Đây là email kiểm tra Gmail OAuth."
    assert command.subject == "Meibook - Thông báo"


def test_parse_explicit_email_subject_and_body():
    command = parse_email_send_command(
        "Gửi email cho test@example.com tiêu đề: Kiểm tra hệ thống nội dung: Meibook gửi thử email tự do."
    )

    assert command is not None
    assert command.to_email == "test@example.com"
    assert command.has_explicit_body is True
    assert command.subject == "Kiểm tra hệ thống"
    assert command.explicit_body == "Meibook gửi thử email tự do."


def test_email_send_command_requires_explicit_address():
    with pytest.raises(GmailSenderError):
        parse_email_send_command("Gửi email báo mã hàng 3736-0008 có tổng bao nhiêu lỗi")


def test_disabled_sender_is_not_available(tmp_path):
    sender = GmailSender(
        credentials_path=tmp_path / "credentials.json",
        token_path=tmp_path / "token.json",
        enabled=False,
    )

    assert sender.available is False
