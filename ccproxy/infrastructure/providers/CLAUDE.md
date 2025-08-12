# CLAUDE.md

Scope: Provider implementations.

Guidelines
- Implement ChatProvider protocol contracts
- Normalize upstream errors to Anthropic-style via errors helpers
- Ensure UTF‑8 integrity on request/response bodies
- Keep provider-specific config under Settings
