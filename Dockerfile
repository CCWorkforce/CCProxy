# Production-grade Dockerfile with Gunicorn and optimizations
# Multi-stage build for minimal size and maximum performance

# Stage 1: Python dependencies builder
FROM python:3.13-slim AS python-builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir gunicorn

# Stage 2: Production runtime
FROM python:3.13-slim AS production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && apt-get upgrade -y \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Security: Create non-root user
RUN groupadd -r ccproxy && \
    useradd -r -g ccproxy -d /app -s /sbin/nologin ccproxy

# Copy Python environment from builder
COPY --from=python-builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Gunicorn settings
    WEB_CONCURRENCY=4 \
    WORKER_CLASS=uvicorn.workers.UvicornWorker \
    WORKER_CONNECTIONS=1000 \
    MAX_REQUESTS=1000 \
    MAX_REQUESTS_JITTER=50 \
    TIMEOUT=120 \
    GRACEFUL_TIMEOUT=30 \
    KEEPALIVE=5 \
    # App settings
    HOST=0.0.0.0 \
    PORT=8082 \
    LOG_LEVEL=INFO

WORKDIR /app

# Copy application code with correct ownership
COPY --chown=ccproxy:ccproxy ccproxy/ ./ccproxy/
COPY --chown=ccproxy:ccproxy *.py ./
COPY --chown=ccproxy:ccproxy gunicorn.conf.py .

# Create necessary directories
RUN mkdir -p /app/logs /app/.cache && \
    chown -R ccproxy:ccproxy /app/logs /app/.cache

# Security hardening
RUN chmod -R 755 /app && \
    chmod -R 700 /app/logs

# Switch to non-root user
USER ccproxy

# Expose port
EXPOSE 8082

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8082/health', timeout=5).read()" || exit 1

# Production entrypoint with signal handling
ENTRYPOINT ["gunicorn", "--config", "gunicorn.conf.py"]
CMD ["wsgi:app"]

# Stage 3: Production with Supervisor (optional, for advanced process management)
FROM production AS production-supervisor

USER root

# Install supervisor
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy supervisor configuration
COPY --chown=root:root supervisor.conf /etc/supervisor/conf.d/ccproxy.conf

# Create supervisor directories
RUN mkdir -p /var/log/supervisor && \
    chown -R ccproxy:ccproxy /var/log/supervisor

USER ccproxy

# Use supervisor as entrypoint
ENTRYPOINT ["/usr/bin/supervisord", "-c", "/etc/supervisor/supervisord.conf"]