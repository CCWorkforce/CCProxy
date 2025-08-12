# CLAUDE.md

Scope: Test suite.

Guidelines
- Tests live under tests/ and assume Settings defaults
- Initialize logging if needed; avoid hitting network
- Use AsyncMock/MagicMock; prefer deterministic data
- Run via: pytest -q
