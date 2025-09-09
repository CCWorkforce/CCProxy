# CLAUDE.md

Scope: Infrastructure adapters (providers, clients).

Guidelines
- OpenAIProvider must wrap errors into openai.APIError
- Use shared httpx client patterns (HTTP/2, timeouts, pooling)
- Providers implement proper resource cleanup via async context managers
- Do not leak secrets in logs; rely on ccproxy.logging redaction
- Keep adapters async and resilient (retries per Settings)
