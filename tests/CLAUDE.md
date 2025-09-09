# CLAUDE.md

Scope: Test suite.

Guidelines
- Tests live under tests/ and assume Settings defaults
- Initialize logging if needed; avoid hitting network
- Use AsyncMock/MagicMock; prefer deterministic data
- Tokenizer tests must be async due to async locks implementation
- Run via: pytest -q
