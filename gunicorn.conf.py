"""
Gunicorn configuration file for production deployment.
Optimized for FastAPI/async applications.
"""

import multiprocessing
import os

# Server socket
bind = f"{os.getenv('HOST', '0.0.0.0')}:{os.getenv('PORT', '8082')}"
backlog = 2048

# Worker processes
# Use single worker for local deployment, otherwise use CPU-based default
if os.getenv("IS_LOCAL_DEPLOYMENT", "False").lower() == "true":
    workers = 1
else:
    workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "uvicorn.workers.UvicornWorker"  # Required for FastAPI
worker_connections = 1000
max_requests = 1000  # Restart workers after this many requests
max_requests_jitter = 50  # Randomize worker restart to avoid all restarting at once
timeout = 120  # Worker timeout for synchronous workers
graceful_timeout = 30  # Time to wait for workers to finish during restart
keepalive = 5  # Keep connections alive for 5 seconds

# Process naming
proc_name = "ccproxy"

# Logging
accesslog = os.getenv("ACCESS_LOG", "-")  # '-' means stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'
errorlog = os.getenv("ERROR_LOG", "-")  # '-' means stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()
capture_output = True
enable_stdio_inheritance = True

# Server mechanics
daemon = False  # Don't daemonize (better for containers)
pidfile = None
user = None  # Run as current user (container will handle user)
group = None
tmp_upload_dir = None

# SSL (optional, uncomment if needed)
# keyfile = os.getenv('SSL_KEYFILE')
# certfile = os.getenv('SSL_CERTFILE')

# Stats
statsd_host = os.getenv("STATSD_HOST")
statsd_prefix = "ccproxy"


# Server hooks for lifecycle management
def when_ready(server):
    """Called just after the master process is initialized."""
    server.log.info("Server is ready. Spawning workers")


def worker_int(worker):
    """Called just after a worker exited on SIGINT or SIGQUIT."""
    worker.log.info(f"Worker {worker.pid} received INT or QUIT signal")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    server.log.info(f"Forking worker {worker}")


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    server.log.info(f"Worker spawned (pid: {worker.pid})")


def worker_abort(worker):
    """Called when a worker received the SIGABRT signal."""
    worker.log.info(f"Worker {worker.pid} received SIGABRT signal")


def pre_exec(server):
    """Called just before a new master process is forked."""
    server.log.info("Forking new master process")


def on_starting(server):
    """Called just before the master process is initialized."""
    server.log.info("Starting Gunicorn server")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    server.log.info("Reloading workers")


def on_exit(server):
    """Called just before exiting."""
    server.log.info("Shutting down Gunicorn server")
