# CLAUDE.md

Scope: Application layer (conversion, caching, tokenization).

Guidelines
- Keep pure logic here; no FastAPI or I/O side effects
- Maintain OpenAI↔Anthropic schema parity in converters
- Use response_cache for non-stream flows; validate JSON/UTF‑8
- Avoid logging secrets; use ccproxy.logging
