"""PII detection and anonymization (Microsoft Presidio + spaCy `en_core_web_lg`).

Policy: only REDACTED text may reach the LLM, the database, caches, logs, or Langfuse traces.
Redaction is one-way (no de-anonymization). Detected values are never stored: a `Finding` keeps
only the entity type, span, and score.

Finance-specific tuning (so education questions stay readable):
- Ticker symbols (TSLA, VTI, BTC-USD) and well-known finance figures (Buffett, Bogle, Markowitz)
  are not treated as private persons.
- Looser SSN pattern than Presidio's default (which ignores "invalid-looking" numbers).
- API keys / passwords pasted into chat are redacted as API_KEY.
- "my name is <name>" is caught even in lowercase, where NER is weak.

Known limits: street addresses, dates of birth and free-form account nicknames are not detected.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Iterable

from src.core.config import GuardrailConfig, get_config
from src.core.portfolio_calc import BROAD_FUNDS

logger = logging.getLogger(__name__)
for _noisy in ("presidio-analyzer", "presidio-anonymizer"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

_TICKER_SHAPE = re.compile(r"^\$?[A-Z]{1,5}([.\-][A-Z]{1,4})?$")

# Popular tickers seen in beginner questions (portfolio tickers are passed in dynamically).
COMMON_TICKERS: frozenset[str] = frozenset(
    """AAPL MSFT GOOGL GOOG AMZN META NVDA TSLA NFLX AMD INTC ORCL CRM ADBE PYPL SQ SHOP UBER LYFT ABNB DIS NKE
    SBUX MCD KO PEP WMT TGT COST HD LOW JPM BAC WFC GS MS V MA AXP JNJ PFE MRK ABBV LLY UNH CVS XOM CVX COP VZ TMUS
    BA GE GM CAT MMM IBM CSCO QCOM TXN AVGO PLTR COIN HOOD SOFI RIVN LCID NIO BABA TSM ASML BRK.B BRK-B
    ARKK XLK XLF XLE XLV VYM VIG VB VO GLD SLV USO TIP LQD HYG IEF SHY TLT IBIT FBTC BTC ETH DOGE SOL ADA XRP
    BTC-USD ETH-USD""".split()
) | BROAD_FUNDS

# Public figures who appear in investing education (full or surname-only forms, lowercase).
PUBLIC_FIGURES: frozenset[str] = frozenset(
    {
        "warren buffett", "buffett", "charlie munger", "munger", "john bogle", "jack bogle", "bogle",
        "harry markowitz", "markowitz", "william sharpe", "eugene fama", "fama", "burton malkiel", "malkiel",
        "benjamin graham", "peter lynch", "robert shiller", "shiller", "ray dalio", "dalio", "daniel kahneman",
        "kahneman", "richard thaler", "thaler", "jerome powell", "janet yellen", "ben bernanke", "bernanke",
        "john maynard keynes", "keynes", "adam smith", "milton friedman", "friedman", "eugene f. fama",
        "george soros", "carl icahn", "cathie wood", "michael burry", "jim cramer",
        # people whose names are now finance terms ("Roth IRA", "Sharpe ratio", "Ponzi scheme")
        "roth", "roth ira", "roth 401k", "roth 401(k)", "keogh", "ponzi", "sharpe", "sortino", "treynor", "jensen", "black-scholes",
        "bollinger", "fibonacci", "modigliani", "miller", "tobin", "dow", "dow jones", "gordon", "graham",
    }
)

_NAME_STOPWORDS = {"at", "on", "when", "if", "later", "maybe", "back", "tomorrow", "today", "now", "asap", "anytime"}


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class RedactionResult:
    text: str
    findings: tuple[Finding, ...] = field(default_factory=tuple)

    @property
    def redacted(self) -> bool:
        return bool(self.findings)

    @property
    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for f in self.findings:
            out[f.entity_type] = out.get(f.entity_type, 0) + 1
        return out


def _build_engines(model: str):
    from presidio_analyzer import AnalyzerEngine, EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_anonymizer import AnonymizerEngine

    nlp_engine = NlpEngineProvider(
        nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]}
    ).create_engine()
    analyzer = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])

    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="US_SSN",
            name="LooseSsnRecognizer",
            patterns=[Pattern("ssn-shaped", r"\b\d{3}[- ]\d{2}[- ]\d{4}\b", 0.6)],
            context=["ssn", "social", "security"],
        )
    )
    analyzer.registry.add_recognizer(
        PatternRecognizer(
            supported_entity="API_KEY",
            name="ApiKeyRecognizer",
            patterns=[
                Pattern("openai-like", r"\b(?:sk|pk|rk)-(?:proj-|lf-|ant-)?[A-Za-z0-9_\-]{16,}\b", 0.95),
                Pattern("long-hex", r"\b[a-fA-F0-9]{40,}\b", 0.7),
                Pattern("labelled-secret", r"(?i)\b(?:api[_ -]?key|secret|token|password|passcode)\s*(?:is|=|:)\s*\S{6,}", 0.85),
            ],
        )
    )

    class ContextNameRecognizer(EntityRecognizer):
        """'my name is <name>' / 'call me <name>' even in lowercase, where NER is unreliable."""

        _rx = re.compile(r"(?i)\b(?:my name is|my name's|call me)\s+([A-Za-z][A-Za-z'\-]+(?:\s+[A-Za-z][A-Za-z'\-]+){0,2})")

        def __init__(self) -> None:
            super().__init__(supported_entities=["PERSON"], name="ContextNameRecognizer", supported_language="en")

        def load(self) -> None:  # required by the base class
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            results = []
            for m in self._rx.finditer(text):
                words = m.group(1).split()
                kept = []
                for w in words:
                    if w.lower() in _NAME_STOPWORDS:
                        break
                    kept.append(w)
                if not kept:
                    continue
                start = m.start(1)
                end = start + len(" ".join(kept))
                results.append(RecognizerResult(entity_type="PERSON", start=start, end=end, score=0.8))
            return results

    analyzer.registry.add_recognizer(ContextNameRecognizer())
    return analyzer, AnonymizerEngine()


class PiiGuard:
    """Detects and replaces PII with `<ENTITY_TYPE>` placeholders."""

    def __init__(self, config: GuardrailConfig | None = None, *, analyzer: Any = None, anonymizer: Any = None,
                 model: str = "en_core_web_lg") -> None:
        self.config = config or get_config().guardrails
        self._model = model
        self._analyzer = analyzer
        self._anonymizer = anonymizer

    def _engines(self):
        if self._analyzer is None or self._anonymizer is None:
            analyzer, anonymizer = _build_engines(self._model)
            self._analyzer = self._analyzer or analyzer
            self._anonymizer = self._anonymizer or anonymizer
        return self._analyzer, self._anonymizer

    # -- filtering ----------------------------------------------------------------------------
    @staticmethod
    def _is_finance_name(span: str, allow: frozenset[str]) -> bool:
        cleaned = re.sub(r"(?:'s|’s)$", "", span.strip())
        if cleaned.lower() in PUBLIC_FIGURES:
            return True
        upper = cleaned.upper()
        return bool(_TICKER_SHAPE.match(cleaned)) and (upper.lstrip("$") in COMMON_TICKERS or upper.lstrip("$") in allow)

    def _keep(self, entity_type: str, score: float, span: str, allow: frozenset[str]) -> bool:
        threshold = self.config.pii_entity_thresholds.get(entity_type, self.config.pii_score_threshold)
        if score < threshold:
            return False
        if entity_type == "PERSON" and self._is_finance_name(span, allow):
            return False
        return True

    # -- public API ---------------------------------------------------------------------------
    def redact(self, text: str, *, allow: Iterable[str] = (), entities: Iterable[str] | None = None) -> RedactionResult:
        """Redact PII in `text`. `allow` are extra tickers/terms never to treat as a person.
        `entities` overrides the configured entity list (used for model OUTPUT, see `pii_output_entities`)."""
        if not isinstance(text, str) or not text.strip():
            return RedactionResult(text if isinstance(text, str) else "")
        allow_set = frozenset(a.upper().lstrip("$") for a in allow)
        analyzer, anonymizer = self._engines()
        entities = list(dict.fromkeys([*(self.config.pii_entities if entities is None else entities), "API_KEY"]))
        raw = analyzer.analyze(text=text, language="en", entities=entities, score_threshold=0.0)
        kept = [r for r in raw if self._keep(r.entity_type, r.score, text[r.start:r.end], allow_set)]
        if not kept:
            return RedactionResult(text)

        from presidio_anonymizer.entities import OperatorConfig

        operators = {
            etype: OperatorConfig("replace", {"new_value": f"<{etype}>"}) for etype in {r.entity_type for r in kept}
        }
        result = anonymizer.anonymize(text=text, analyzer_results=kept, operators=operators)
        findings = tuple(Finding(r.entity_type, r.start, r.end, round(float(r.score), 2)) for r in kept)
        return RedactionResult(result.text, findings)

    def redact_structure(self, data: Any, *, allow: Iterable[str] = ()) -> Any:
        """Redact every string inside nested dict/list/tuple data (for traces and stored payloads)."""
        if isinstance(data, str):
            return self.redact(data, allow=allow).text
        if isinstance(data, dict):
            return {k: self.redact_structure(v, allow=allow) for k, v in data.items()}
        if isinstance(data, list):
            return [self.redact_structure(v, allow=allow) for v in data]
        if isinstance(data, tuple):
            return tuple(self.redact_structure(v, allow=allow) for v in data)
        return data


@lru_cache(maxsize=1)
def get_pii_guard() -> PiiGuard:
    cfg = get_config().guardrails
    return PiiGuard(cfg, model=cfg.pii_model)


def redact_text(text: str, *, allow: Iterable[str] = (), entities: Iterable[str] | None = None) -> RedactionResult:
    return get_pii_guard().redact(text, allow=allow, entities=entities)


def redact_output(text: str) -> RedactionResult:
    """Redact model/agent OUTPUT with the high-precision detectors only (no name detection).

    Names are deliberately not redacted on the way out: the text is built from already-redacted input,
    trusted data and the knowledge base, and NER wrongly labels numbers and tickers as people
    (it turned "+0.41% vs the previous close" into "<PERSON>"). Emails, phones, SSNs, cards, IPs and API
    keys are still removed.
    """
    return redact_text(text, entities=get_config().guardrails.pii_output_entities)
