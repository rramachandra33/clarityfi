"""Outbound transactional email (currently just the password-reset link).

Two providers, chosen by `EMAIL_PROVIDER`:
- "console" (default): logs the message instead of sending it. Safe for local dev with no external account,
  and is how this feature is exercised end to end without AWS SES credentials configured.
- "ses": sends via Amazon SES. `boto3` is imported lazily (only when this provider is actually selected) so
  local dev never needs it installed. Requires a verified `SES_SENDER_EMAIL` identity and AWS credentials
  resolved the normal way (environment, `~/.aws/credentials`, or an instance role once deployed).

Callers must never let a send failure change their response to the caller (see `EmailError`'s docstring) —
a password-reset request always answers the same way whether or not the account exists or the send worked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from src.core.config import Settings
from src.core.errors import ConfigError, EmailError

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmailMessage:
    to: str
    subject: str
    text_body: str


class EmailSender(Protocol):
    def send(self, message: EmailMessage) -> None: ...


class ConsoleEmailSender:
    """Dev/default sender: writes the message to the app log instead of sending it."""

    def send(self, message: EmailMessage) -> None:
        logger.info("EMAIL (console mode, not actually sent) to=%s subject=%r\n%s", message.to, message.subject, message.text_body)


class SesEmailSender:
    def __init__(self, sender: str, region: str) -> None:
        if not sender:
            raise ConfigError("SES_SENDER_EMAIL is not set. Add it to .env, or set EMAIL_PROVIDER=console for dev.")
        self.sender, self.region = sender, region
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import boto3  # heavy optional dependency; only imported when SES is actually selected

            self._client = boto3.client("ses", region_name=self.region)
        return self._client

    def send(self, message: EmailMessage) -> None:
        try:
            self.client.send_email(
                Source=self.sender,
                Destination={"ToAddresses": [message.to]},
                Message={"Subject": {"Data": message.subject}, "Body": {"Text": {"Data": message.text_body}}},
            )
        except Exception as exc:  # noqa: BLE001 - any SES/boto3 failure maps to one safe, non-leaking error
            raise EmailError(f"Could not send email via SES: {exc}") from exc


def get_email_sender(settings: Settings) -> EmailSender:
    if settings.email_provider == "ses":
        return SesEmailSender(settings.ses_sender_email, settings.aws_region)
    return ConsoleEmailSender()


def password_reset_message(to: str, reset_url: str, ttl_minutes: int) -> EmailMessage:
    return EmailMessage(
        to=to,
        subject="Reset your ClarityFI password",
        text_body=(
            "We received a request to reset your ClarityFI password.\n\n"
            f"Reset it here (valid for {ttl_minutes} minutes): {reset_url}\n\n"
            "If you did not request this, you can safely ignore this email — your password will not change."
        ),
    )
