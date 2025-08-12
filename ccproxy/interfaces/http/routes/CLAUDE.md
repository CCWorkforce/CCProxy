# CLAUDE.md

Scope: Route handlers.

Guidelines
- messages.py should proxy to provider, use converters and tokenizer
- Return Anthropic-compatible responses; support streaming via SSE
- Validate inputs; rely on request_validator where applicable
- No secrets in logs; include request_id in LogRecord
