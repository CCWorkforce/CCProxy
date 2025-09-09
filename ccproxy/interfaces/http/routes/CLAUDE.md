# CLAUDE.md

Scope: Route handlers.

Guidelines
- messages.py should proxy to provider, use converters and async tokenizer
- Return Anthropic-compatible responses; support streaming via SSE
- Validate inputs; rely on request_validator where applicable
- Await tokenizer functions: count_tokens_for_anthropic_request and truncate_request
- No secrets in logs; include request_id in LogRecord
