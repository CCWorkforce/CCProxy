# CLAUDE.md

Scope: Python package root for CCProxy.

Guidelines:

- Import via relative package paths; avoid global singletons
- Construct apps with ccproxy.interfaces.http.app.create_app(Settings)
- Keep modules small and dependency direction inward (application <- interfaces)
- Do not log secrets; use ccproxy.logging helpers
- Preserve UTF-8 when handling bytes/strings
- JSON logging automatically omits null values for cleaner output
