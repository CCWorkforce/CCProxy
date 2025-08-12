# CLAUDE.md

Scope: Interface layer (HTTP, routers, middleware).

Guidelines
- Build app via create_app(Settings)
- Use structured logging (LogEvent) and request_id from middleware
- Stream SSE using interfaces/http/streaming.py
- All exception paths must go through errors.log_and_return_error_response
