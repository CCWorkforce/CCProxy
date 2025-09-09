# CLAUDE.md

Scope: Application layer (conversion, caching, tokenization).

Guidelines
- Keep pure logic here; no FastAPI or I/O side effects
- Maintain OpenAI↔Anthropic schema parity in converters
- Use response_cache for non-stream flows; validate JSON/UTF‑8
- Tokenizer functions are async-aware; use await for count_tokens_for_anthropic_request and truncate_request
- Avoid logging secrets; use ccproxy.logging
