# CLAUDE.md

Scope: HTTP specifics (app, errors, middleware, streaming, routes).

Guidelines
- Keep endpoints Anthropic-compatible; map OpenAI finish reasons
- Use StreamProcessor for SSE; finalize blocks correctly
- Exception handlers must capture and log to error.jsonl via ccproxy.logging
- Respect Settings: CORS, rate limits, security headers
