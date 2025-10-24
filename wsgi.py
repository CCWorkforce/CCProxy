"""
ASGI entry point for Uvicorn.
This module provides the application factory for production deployment.
"""

import sys
from typing import Any
from dotenv import load_dotenv
from ccproxy.config import Settings, ConfigurationError
from ccproxy.interfaces.http.app import create_app

# Load environment variables
load_dotenv()


# Create application instance
def create_application() -> Any:
    """Application factory for Uvicorn."""
    try:
        settings = Settings()
    except ConfigurationError as e:
        # Handle configuration errors gracefully
        print(f"\nConfiguration Error:\n{e}", file=sys.stderr)
    #         sys.exit(1)  # type: ignore[unreachable]
    except Exception as e:
        # Handle other initialization errors
        print(f"\nUnexpected error during configuration: {e}", file=sys.stderr)
    #         sys.exit(1)  # type: ignore[unreachable]
    try:
        return create_app(settings)
    except Exception as e:
        print(f"\nFailed to initialize application: {str(e)}")
        print("Please check your configuration and provider connectivity.")
        raise SystemExit(1)


# Create the app instance for Uvicorn
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
