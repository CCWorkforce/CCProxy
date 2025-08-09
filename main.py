import uvicorn
from dotenv import load_dotenv
from ccproxy.config import Settings
from ccproxy.interfaces.http.app import create_app

load_dotenv()

try:
    settings = Settings()
except Exception as e:
    # Fail fast on invalid config
    raise SystemExit(f"Configuration error: {e}")

app = create_app(settings)

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.reload,
        log_config=None,
        access_log=False,
    )
