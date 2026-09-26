"""Password-reset email: the console dev sender, the SES sender (mocked boto3), and the provider factory."""
from unittest.mock import MagicMock, patch

import pytest

from src.core.config import Settings
from src.core.email import (
    ConsoleEmailSender,
    EmailMessage,
    SesEmailSender,
    get_email_sender,
    password_reset_message,
)
from src.core.errors import ConfigError, EmailError


def test_console_sender_logs_instead_of_sending(caplog):
    caplog.set_level("INFO")
    body = "link: http://x/y"
    ConsoleEmailSender().send(EmailMessage(to="a@example.com", subject="Reset your ClarityFI password", text_body=body))
    assert "a@example.com" in caplog.text and "Reset your ClarityFI password" in caplog.text and body in caplog.text


def test_get_email_sender_defaults_to_console():
    assert isinstance(get_email_sender(Settings(email_provider="console")), ConsoleEmailSender)
    assert isinstance(get_email_sender(Settings(email_provider="anything-else")), ConsoleEmailSender)


def test_get_email_sender_returns_ses_when_selected():
    sender = get_email_sender(Settings(email_provider="ses", ses_sender_email="noreply@example.com", aws_region="us-west-2"))
    assert isinstance(sender, SesEmailSender)
    assert sender.sender == "noreply@example.com" and sender.region == "us-west-2"


def test_ses_sender_requires_a_verified_sender_address():
    with pytest.raises(ConfigError, match="SES_SENDER_EMAIL"):
        SesEmailSender(sender="", region="us-east-1")


def test_ses_sender_calls_boto3_send_email_with_the_message():
    sender = SesEmailSender(sender="noreply@example.com", region="us-east-1")
    fake_client = MagicMock()
    with patch("boto3.client", return_value=fake_client) as make_client:
        sender.send(EmailMessage(to="user@example.com", subject="Subject", text_body="Body text"))
    make_client.assert_called_once_with("ses", region_name="us-east-1")
    fake_client.send_email.assert_called_once_with(
        Source="noreply@example.com",
        Destination={"ToAddresses": ["user@example.com"]},
        Message={"Subject": {"Data": "Subject"}, "Body": {"Text": {"Data": "Body text"}}},
    )


def test_ses_sender_reuses_the_same_client_across_calls():
    sender = SesEmailSender(sender="noreply@example.com", region="us-east-1")
    with patch("boto3.client", return_value=MagicMock()) as make_client:
        sender.send(EmailMessage(to="a@example.com", subject="s", text_body="b"))
        sender.send(EmailMessage(to="b@example.com", subject="s", text_body="b"))
    make_client.assert_called_once()  # the client is built once and cached, not per send


def test_ses_failures_are_wrapped_in_email_error_not_leaked_raw():
    sender = SesEmailSender(sender="noreply@example.com", region="us-east-1")
    fake_client = MagicMock()
    fake_client.send_email.side_effect = RuntimeError("boto3 blew up")
    with patch("boto3.client", return_value=fake_client):
        with pytest.raises(EmailError, match="boto3 blew up"):
            sender.send(EmailMessage(to="user@example.com", subject="s", text_body="b"))


def test_password_reset_message_includes_the_url_and_ttl_and_never_implies_a_forced_change():
    message = password_reset_message("user@example.com", "https://clarityfi.example.com/?reset_token=abc123", ttl_minutes=60)
    assert message.to == "user@example.com"
    assert "https://clarityfi.example.com/?reset_token=abc123" in message.text_body
    assert "60 minutes" in message.text_body
    assert "did not request" in message.text_body.lower()
