"""PII layer tests. The `real` group runs Presidio + spaCy en_core_web_lg; the `rules` group uses a
fake analyzer to test thresholds and allow-lists deterministically."""
import pytest
from presidio_analyzer import RecognizerResult
from presidio_anonymizer import AnonymizerEngine

from src.core import pii, tracing
from src.core.config import GuardrailConfig
from src.core.pii import Finding, PiiGuard, RedactionResult


# ---------------------------------------------------------------------------------- real engine
@pytest.fixture(scope="module")
def guard():
    return PiiGuard(GuardrailConfig(
        pii_entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "US_SSN", "CREDIT_CARD", "IBAN_CODE",
                      "US_BANK_NUMBER", "IP_ADDRESS"]))


@pytest.mark.parametrize(
    "text,secret,placeholder",
    [
        ("My name is John Smith and I want to start investing", "John Smith", "<PERSON>"),
        ("my name is priya raman, how do bonds work", "priya raman", "<PERSON>"),
        ("Hi, I'm Maria Gonzalez, can you explain ETFs?", "Maria Gonzalez", "<PERSON>"),
        ("Email me at john.smith@example.com please", "john.smith@example.com", "<EMAIL_ADDRESS>"),
        ("Call me at (415) 555-0199", "555-0199", "<PHONE_NUMBER>"),
        ("My number is 415-555-0199", "415-555-0199", "<PHONE_NUMBER>"),
        ("My SSN is 536-90-4399", "536-90-4399", "<US_SSN>"),
        ("SSN 123-45-6789 (looks fake but shape matches)", "123-45-6789", "<US_SSN>"),
        ("Card 4111 1111 1111 1111 expires soon", "4111 1111 1111 1111", "<CREDIT_CARD>"),
        ("IBAN GB82 WEST 1234 5698 7654 32", "GB82 WEST 1234 5698 7654 32", "<IBAN_CODE>"),
        ("Server 192.168.1.10 sent it", "192.168.1.10", "<IP_ADDRESS>"),
        ("Account number 000123456789 at my bank", "000123456789", "<US_BANK_NUMBER>"),
        ("here is my key sk-proj-Abcdefghijklmnopqrstuvwxyz0123456789", "sk-proj-Abcdefghijklmnopqrstuvwxyz0123456789", "<API_KEY>"),
        ("my api key is hunter2hunter2", "hunter2hunter2", "<API_KEY>"),
        ("token: " + "ab12" * 16, "ab12" * 16, "<API_KEY>"),
    ],
)
def test_sensitive_values_are_removed(guard, text, secret, placeholder):
    result = guard.redact(text)
    assert secret not in result.text
    assert placeholder in result.text
    assert result.redacted


def test_call_me_at_number_does_not_swallow_the_word_at(guard):
    out = guard.redact("call me at 415-555-0199 tomorrow").text
    assert "415-555-0199" not in out and "call me at <PHONE_NUMBER>" in out


@pytest.mark.parametrize(
    "text",
    [
        "I have $50,000 in VTI and $12,500 in my 401k, earning 7.5% a year",
        "My goal is 1,000,000 by 2040, contributing 1500 per month",
        "What did Warren Buffett and John Bogle say about index funds?",
        "Explain Harry Markowitz's modern portfolio theory",
        "Should I buy AAPL, MSFT or TSLA? Compare Vanguard and Fidelity",
        "Is BTC-USD or ETH-USD riskier than the S&P 500 and the Roth IRA?",
        "How does a 60/40 portfolio with BND and VTI compare over 30 years?",
        "What is the difference between a bond's coupon and its yield to maturity?",
    ],
)
def test_normal_finance_questions_are_left_alone(guard, text):
    result = guard.redact(text)
    assert result.text == text and not result.redacted


def test_multiple_entities_and_counts(guard):
    result = guard.redact("I'm John Smith, reach me at john@example.com or 415-555-0199.")
    assert result.counts.get("EMAIL_ADDRESS") == 1 and result.counts.get("PHONE_NUMBER") == 1
    assert "John Smith" not in result.text and "john@example.com" not in result.text


def test_findings_never_store_the_sensitive_text(guard):
    result = guard.redact("Email me at john.smith@example.com")
    assert all(isinstance(f, Finding) for f in result.findings)
    assert set(vars(result.findings[0])) == {"entity_type", "start", "end", "score"}
    assert "john.smith" not in repr(result.findings)


def test_redaction_is_idempotent(guard):
    once = guard.redact("Email me at john.smith@example.com, SSN 536-90-4399").text
    assert guard.redact(once).text == once


def test_redact_structure_handles_nested_data_and_keeps_non_strings(guard):
    data = {"input": "mail a@b.com", "meta": {"n": 3, "flag": True, "none": None, "items": ["ssn 536-90-4399", 7]},
            "tup": ("ok text", "call 415-555-0199")}
    out = guard.redact_structure(data)
    assert "a@b.com" not in out["input"] and out["meta"]["n"] == 3 and out["meta"]["none"] is None
    assert "536-90-4399" not in out["meta"]["items"][0] and out["meta"]["items"][1] == 7
    assert isinstance(out["tup"], tuple) and out["tup"][0] == "ok text" and "555-0199" not in out["tup"][1]


@pytest.mark.parametrize("value", ["", "   ", None, 42])
def test_empty_or_non_string_input(guard, value):
    result = guard.redact(value)
    assert isinstance(result, RedactionResult) and not result.redacted


def test_long_text_is_processed(guard):
    text = ("Index funds spread risk across many companies. " * 80) + "Contact john@example.com"
    result = guard.redact(text)
    assert "john@example.com" not in result.text and len(result.text) > 3000


def test_singleton_accessor_and_module_helper():
    assert pii.get_pii_guard() is pii.get_pii_guard()
    assert "a@b.com" not in pii.redact_text("write to a@b.com").text


def test_langfuse_mask_redacts_payloads():
    masked = tracing._mask_for_langfuse(data={"input": "my email is a@b.com", "n": 1})
    assert "a@b.com" not in masked["input"] and masked["n"] == 1


def test_langfuse_mask_fails_closed(monkeypatch):
    def boom():
        raise RuntimeError("model missing")

    monkeypatch.setattr(pii, "get_pii_guard", boom)
    assert tracing._mask_for_langfuse(data={"input": "secret a@b.com"}) == "[REDACTED: masking error]"


# ------------------------------------------------------------------------------------ rules only
class FakeAnalyzer:
    def __init__(self, results):
        self.results = results

    def analyze(self, text, language, entities, score_threshold):
        return list(self.results)


def rules_guard(results, **cfg):
    return PiiGuard(GuardrailConfig(**cfg), analyzer=FakeAnalyzer(results), anonymizer=AnonymizerEngine())


def rr(etype, start, end, score):
    return RecognizerResult(entity_type=etype, start=start, end=end, score=score)


def test_default_threshold_drops_low_confidence_but_phone_uses_lower_threshold():
    text = "abc 4155550199 def"
    low_person = rules_guard([rr("PERSON", 0, 3, 0.45)]).redact(text)
    assert not low_person.redacted  # 0.45 < 0.5 default
    phone = rules_guard([rr("PHONE_NUMBER", 4, 14, 0.4)]).redact(text)
    assert phone.text == "abc <PHONE_NUMBER> def"


def test_thresholds_are_configurable():
    text = "abc 4155550199 def"
    strict = rules_guard([rr("PHONE_NUMBER", 4, 14, 0.4)], pii_entity_thresholds={"PHONE_NUMBER": 0.9})
    assert not strict.redact(text).redacted


def test_ticker_flagged_as_person_is_kept_but_unknown_uppercase_word_is_not_trusted():
    text = "Buy TSLA or XYZQ?"
    g = rules_guard([rr("PERSON", 4, 8, 0.85), rr("PERSON", 12, 16, 0.85)])
    out = g.redact(text)
    assert out.text == "Buy TSLA or <PERSON>?"
    allowed = g.redact(text, allow=["xyzq", "$XYZQ"])
    assert allowed.text == text  # portfolio tickers can be passed in via allow


def test_public_figures_kept_including_possessive_and_surname():
    text = "Buffett and Harry Markowitz's theory"
    g = rules_guard([rr("PERSON", 0, 7, 0.85), rr("PERSON", 12, 30, 0.85)])
    assert g.redact(text).text == text


def test_private_person_with_same_first_name_as_a_figure_is_still_redacted():
    text = "Warren Patel asked"
    assert rules_guard([rr("PERSON", 0, 12, 0.85)]).redact(text).text == "<PERSON> asked"


def test_non_person_entities_are_never_allow_listed():
    text = "mail TSLA@x.com"
    out = rules_guard([rr("EMAIL_ADDRESS", 5, 15, 1.0)]).redact(text, allow=["TSLA@X.COM"])
    assert "<EMAIL_ADDRESS>" in out.text


def test_no_results_returns_original_object_text():
    assert rules_guard([]).redact("nothing here").text == "nothing here"


@pytest.mark.parametrize("text", [
    "What is the difference between a Roth and a traditional IRA?",
    "Is a Roth IRA better for young savers?",
    "How is the Sharpe ratio calculated?",
    "What is a Ponzi scheme?",
])
def test_names_that_are_finance_terms_are_not_redacted(text):
    assert pii.redact_text(text).text == text
