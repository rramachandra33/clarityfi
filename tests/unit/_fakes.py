"""Shared test doubles for the RAG tests."""
from src.rag.loaders import Chunk
from src.rag.sparse import tokenize, token_id

DIM = 64


class BagEmbedder:
    """Deterministic 'semantic-ish' embedder: hashed bag-of-words. Similar wording => high cosine."""

    embed_model = "fake-bag-64"

    def __init__(self):
        self.calls = 0  # number of embed() invocations
        self.texts = 0  # total texts embedded

    def embed(self, texts):
        self.calls += 1
        self.texts += len(texts)
        out = []
        for text in texts:
            v = [0.0] * DIM
            for tok in tokenize(text):
                v[token_id(tok) % DIM] += 1.0
            if not any(v):
                v[0] = 1.0
            out.append(v)
        return out


def mk_chunk(title, text, category="basics", *, source_name=None, url=None, idx=0, **extra):
    meta = {"title": title, "category": category, "source": "kb.json", "json_path": f"$.documents[{idx}]",
            "chunk_index": 0, "document_index": idx}
    if source_name:
        meta["source_name"] = source_name
    if url:
        meta["source_url"] = url
    meta.update(extra)
    return Chunk(text=text, title=title, metadata=meta)


CORPUS = [
    mk_chunk("Diversification", "Spreading money across many investments reduces the impact of any single loss.", "portfolio_management",
             source_name="Investopedia", url="https://ex.com/div", idx=0),
    mk_chunk("Sharpe Ratio", "The Sharpe ratio measures return per unit of risk, using excess return over volatility.", "risk_measurement",
             source_name="Bogleheads Wiki", url="https://ex.com/sharpe", idx=1),
    mk_chunk("Roth IRA", "A Roth IRA is a retirement account funded with after-tax money that grows tax free.", "tax_education",
             source_name="SEC Investor.gov", idx=2),
    mk_chunk("Capital Gains", "Capital gains tax applies to profit when you sell an investment held in a taxable account.", "taxes_and_investing",
             source_name="Investopedia", url="https://ex.com/cg", idx=3),
    mk_chunk("Bond Basics", "A bond is a loan to a government or company that pays interest and returns principal at maturity.", "bonds",
             source_name="Investopedia", idx=4),
    mk_chunk("Emergency Fund", "An emergency fund keeps cash available for unexpected expenses before you invest.", "financial_basics",
             source_name="Starter educational content; verify current rules before production use.", idx=5),
]
