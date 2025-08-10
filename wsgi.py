"""
WSGI entry point for Gunicorn.
This module provides the application factory for production deployment.
"""

from dotenv import load_dotenv
from ccproxy.config import Settings
from ccproxy.interfaces.http.app import create_app

# Load environment variables
load_dotenv()

# Create application instance
def create_application():
    """Application factory for Gunicorn."""
    try:
        settings = Settings()
    except Exception as e:
        # Fail fast on invalid config
        raise SystemExit(f"Configuration error: {e}")

    return create_app(settings)

# Create the app instance for Gunicorn
app = create_application()

# For development/testing
if __name__ == "__main__":
    import uvicorn
    settings = Settings()
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_config=None,
        access_log=False,
    )
