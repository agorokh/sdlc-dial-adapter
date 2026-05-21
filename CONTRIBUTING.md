# Contributing

Thanks for taking an interest. This is a small, scoped adapter — the contribution surface is intentionally narrow.

## What contributions are welcome

- **Bug reports** with a minimal reproduction (one `curl` command + the gateway you tested against).
- **Compatibility fixes** for additional OpenAI-shape gateways beyond AI DIAL (vLLM, LiteLLM, Together, Fireworks, etc.). Add a new gateway to `PORTABILITY.md` rather than branching the core translation logic.
- **Edge-case translation improvements** for the Anthropic ↔ OpenAI shape mapping, with tests under `tests/`.
- **Observability improvements** under `observability/` (Grafana dashboards, Prometheus rules, structured log enhancements).
- **Documentation clarifications** to README / PORTABILITY / inline docstrings.

## What contributions need a discussion first (open an issue)

- A new vendor binding that touches request/response shape semantics (e.g., a new tool-use protocol).
- A structural change to `app.py` beyond the existing modular split.
- Anything that adds a runtime dependency.

## Workflow

1. Open an issue describing the change before non-trivial work, so we can agree on scope.
2. Fork + branch off `main`. Branch naming: `fix/<slug>`, `feat/<slug>`, `chore/<slug>`, `docs/<slug>`.
3. Run the test suite locally:
   ```bash
   pip install -e ".[dev]"
   pytest
   ```
4. Run the linter:
   ```bash
   ruff check .
   ```
5. Open the PR with a clear description. PR bot reviews run automatically; address actionable feedback.

## Coding conventions

- Python 3.11+, type hints everywhere, prefer typed models over loose dicts at API boundaries.
- Async / `aiohttp` is the only HTTP client; no `requests`.
- Tests under `tests/` use `pytest-asyncio`; mirror the source layout.
- No global mutable state; pass config explicitly.
- Log via the standard `logging` module; structured JSON via the existing GFLog helper in `app.py`.

## License

This project is licensed under [Apache-2.0](LICENSE). By contributing, you agree your contribution is under the same license.

## Companion projects

- [agorokh/agentic-memory-mcp](https://github.com/agorokh/agentic-memory-mcp) — Read-path MCP server bridging MCP clients to LightRAG backends (Apache-2.0)
- [agorokh/applied-ai-research](https://github.com/agorokh/applied-ai-research) — Practitioner notes that cite this adapter in their methodology
