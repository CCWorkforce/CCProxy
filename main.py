import sys
from fastapi import FastAPI
import uvicorn
from dotenv import load_dotenv
from ccproxy.config import Settings, ConfigurationError
from ccproxy.interfaces.http.app import create_app

load_dotenv()

try:
    settings = Settings()
except ConfigurationError as e:
    # Handle configuration errors gracefully
    print(f"\nConfiguration Error:\n{e}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    # Handle other initialization errors
    print(f"\nUnexpected error during configuration: {e}", file=sys.stderr)
    sys.exit(1)

try:
    app: FastAPI = create_app(settings)
except Exception as e:
    print(f"\nFailed to initialize application: {str(e)}")
    print("Please check your configuration and provider connectivity.")
    raise SystemExit(1)

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,
        access_log=False,
    )
