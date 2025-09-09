# CLAUDE.md

Scope: HTTP specifics (app, errors, middleware, streaming, routes).

Guidelines
- Keep endpoints Anthropic-compatible; map OpenAI finish reasons
- Use StreamProcessor for SSE with race condition protection in subscriber management
- Exception handlers must capture and log to error.jsonl via ccproxy.logging
- Respect Settings: CORS, rate limits, security headers
