# ClarityFI MCP server

Exposes four **read-only** educational tools to any [Model Context Protocol](https://modelcontextprotocol.io) client (Claude Desktop, IDE assistants, agent frameworks). It uses the same services and safeguards as the web app; nothing here writes data.

| Tool | Arguments | Returns |
|---|---|---|
| `get_quote` | `symbols` (1–25 tickers, e.g. `["AAPL","VTI","BTC-USD"]`) | Price, daily change, `source`, `data_status` (live / cached / stale / sample) and a `notice` for anything not live; unknown symbols in `errors` |
| `analyze_portfolio` | `user_id` (demo ids `learner-101` … `learner-110`, or any id registered in the app) | Total value, allocation, diversification score, weighted fees, risk label, notes, per-holding table with price provenance |
| `search_knowledge_base` | `query`, optional `category` (`tax`, `retirement`, `bonds`, `funds`, `portfolio`, `risk`, `basics`, `markets`, `protection`, or a raw category), `top_k` 1–8 | Passages with title, publisher, link when known, category, relevance score |
| `project_goal` | `target_amount`, `years`, optional `current_savings`, `monthly_contribution`, `risk_tolerance` (conservative / moderate / aggressive), `goal_name` | Below / expected / above scenarios (also in today's dollars), total monthly amount needed, yearly path, assumptions |

Every result includes an education-only `disclaimer`. Search queries are validated and PII-redacted before they reach the embedding provider (the call is refused if the privacy filter is unavailable). Expected failures (unknown user, bad arguments) come back as readable tool errors; unexpected crashes show only a generic message.

## Run it

```bash
cd /path/to/clarityFI
.venv/bin/python -m src.mcp_server.server        # speaks MCP over stdio; logs go to stderr
```

Keys are read from the project's git-ignored `.env`, so **no secrets go in the client configuration**. Prices need no key (Yahoo Finance fallback); knowledge-base search needs `OPENAI_API_KEY` for query embeddings.

## Claude Desktop

Add this to `~/Library/Application Support/Claude/claude_desktop_config.json` (use your real absolute path), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "clarityfi": {
      "command": "/Users/rramachandra/raju/CAPSTONE2/clarityFI/.venv/bin/python",
      "args": ["-m", "src.mcp_server.server"],
      "cwd": "/Users/rramachandra/raju/CAPSTONE2/clarityFI"
    }
  }
}
```

Try: "What is AAPL trading at?", "Analyze the portfolio for learner-105", "Search the ClarityFI knowledge base for Roth vs traditional IRA", "If I have $5,000 and add $300 a month, can I reach $50,000 in 8 years?"

## Notes
- The MCP server is a local process launched by your own Claude Desktop over stdio: it does not use web sign-in and can read any demo or registered user's portfolio by id. Run it only on a machine you trust.
- To add it to Claude Desktop safely, run `python3 scripts/install_claude_desktop.py --dry-run`, then without `--dry-run`: it backs up the existing config, keeps every other server and setting, and adds only the `clarityfi` entry.
- The embedded Qdrant index allows one process. If the web API is running, the MCP server automatically searches the FAISS copy of the same articles instead.
- Cached, last-known and SAMPLE data are always labelled in the result; clients should relay `notice` to the user.
- Tested with an in-memory MCP client (22 tests) and live over stdio against the real services.
