# CLAUDE.md

Scope: Python package root for CCProxy.

Guidelines:

- Import via relative package paths; avoid global singletons
- Construct apps with ccproxy.interfaces.http.app.create_app(Settings)
- Keep modules small and dependency direction inward (application <- interfaces)
- Do not log secrets; use ccproxy.logging helpers
- Preserve UTF-8 when handling bytes/strings
- JSON logging automatically omits null values for cleaner output

## New Modules Added

### ccproxy/constants.py

Contains all constant values used throughout the application:

- UTF-8 enforcement message for models that support developer role
- Model capability sets (reasoning effort, temperature, developer message support)
- Top-tier models for OpenAI and Anthropic
- Input and output token limits for all supported models

### ccproxy/enums.py

Contains all enumeration classes used throughout the application:

- MessageRoles enum (Developer, System, User)
- ReasoningEfforts enum (High, Medium, Low)
- TruncationStrategy and TruncationConfig for message truncation handling
