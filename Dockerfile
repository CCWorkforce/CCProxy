# Production-grade Dockerfile with Gunicorn and optimizations
# Multi-stage build for minimal size and maximum performance
# Supports both Alpine (smallest) and Debian slim (better compatibility)

# ============================================================================
# Stage 1: Python dependencies builder (Alpine-based for smaller build)
# ============================================================================
FROM python:3.13.5-alpine AS python-builder-alpine

# Install build dependencies required for Python packages
RUN apk add --no-cache \
    gcc \
    musl-dev \
    linux-headers \
    libffi-dev \
    openssl-dev \
    # Required for some Python packages that need Rust
    cargo \
    rust

WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================================================
# Stage 2: Python dependencies builder (Debian-based for compatibility)
# ============================================================================
FROM python:3.13.5-slim AS python-builder-debian

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
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# ============================================================================
# Stage 3: Alpine runtime (SMALLEST ~80-120MB)
# ============================================================================
FROM python:3.13.5-alpine AS production-alpine

# Install only essential runtime dependencies
RUN apk add --no-cache \
    ca-certificates \
    libffi \
    openssl \
    curl \
    tzdata \
    && rm -rf /var/cache/apk/*

# Security: Create non-root user
RUN addgroup -g 1000 -S ccproxy && \
    adduser -u 1000 -S ccproxy -G ccproxy -h /app -s /bin/false

# Copy Python environment from Alpine builder
COPY --from=python-builder-alpine --chown=ccproxy:ccproxy /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Uvicorn settings (CPU-based concurrency)
    WEB_CONCURRENCY=${WEB_CONCURRENCY:-$(($(nproc) * 2 + 1))} \
    # App settings
    HOST=0.0.0.0 \
    PORT=11434 \
    LOG_LEVEL=INFO

WORKDIR /app

# Copy application code with correct ownership
COPY --chown=ccproxy:ccproxy ccproxy/ ./ccproxy/
COPY --chown=ccproxy:ccproxy *.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/.cache && \
    chown -R ccproxy:ccproxy /app

# Switch to non-root user
USER ccproxy

# Expose port
EXPOSE 11434

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/health', timeout=5).read()" || exit 1

# Production entrypoint with signal handling
# Calculate worker count at runtime using shell form
CMD ["sh", "-c", "exec uvicorn wsgi:app --host 0.0.0.0 --port ${PORT:-11434} --workers ${WEB_CONCURRENCY:-$(($(nproc) * 2 + 1))} --loop uvloop --http httptools --timeout-keep-alive 5 --timeout-graceful-shutdown 30"]

# ============================================================================
# Stage 4: Debian slim runtime (RECOMMENDED ~150-200MB, better compatibility)
# ============================================================================
FROM python:3.13.5-slim AS production

# Install runtime dependencies and security updates
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# Security: Create non-root user
RUN groupadd -r ccproxy && \
    useradd -r -g ccproxy -d /app -s /sbin/nologin ccproxy

# Copy Python environment from Debian builder
COPY --from=python-builder-debian --chown=ccproxy:ccproxy /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Uvicorn settings (CPU-based concurrency)
    WEB_CONCURRENCY=${WEB_CONCURRENCY:-$(($(nproc) * 2 + 1))} \
    # App settings
    HOST=0.0.0.0 \
    PORT=11434 \
    LOG_LEVEL=INFO

WORKDIR /app

# Copy application code with correct ownership
COPY --chown=ccproxy:ccproxy ccproxy/ ./ccproxy/
COPY --chown=ccproxy:ccproxy *.py ./

# Create necessary directories
RUN mkdir -p /app/logs /app/.cache && \
    chown -R ccproxy:ccproxy /app

# Security hardening
RUN chmod -R 755 /app && \
    chmod -R 700 /app/logs

# Switch to non-root user
USER ccproxy

# Expose port
EXPOSE 11434

# Health check with timeout
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:11434/health', timeout=5).read()" || exit 1

# Production entrypoint with signal handling
# Calculate worker count at runtime using shell form
CMD ["sh", "-c", "exec uvicorn wsgi:app --host 0.0.0.0 --port ${PORT:-11434} --workers ${WEB_CONCURRENCY:-$(($(nproc) * 2 + 1))} --loop uvloop --http httptools --timeout-keep-alive 5 --timeout-graceful-shutdown 30"]

# ============================================================================
# Stage 5: Production with Supervisor (optional, for advanced process management)
# ============================================================================
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