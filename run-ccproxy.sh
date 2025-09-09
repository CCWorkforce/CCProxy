#!/bin/bash

# CCProxy Runner Script
# This script loads environment variables from .env file,
# validates required configurations, and starts the CCProxy application.

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if uv is installed
if ! command_exists uv; then
    print_error "uv is not installed. Please install uv first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Check if Python 3 is available through uv
print_info "Checking Python availability via uv..."
if ! uv python list | grep -q "python"; then
    print_error "No Python installation found via uv. Installing Python..."
    uv python install
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Determine env file (prefer .env, fallback to .env.local)
ENV_FILE=".env"
if [ ! -f "$ENV_FILE" ]; then
    if [ -f ".env.local" ]; then
        ENV_FILE=".env.local"
        print_info "Using $ENV_FILE as configuration source (override by creating .env)."
    else
        print_error "No .env or .env.local file found in $SCRIPT_DIR"
        print_info "Please create a .env file with at least the following required variables:"
    echo "  OPENAI_API_KEY=your_api_key_here"
    echo "  BIG_MODEL_NAME=your_big_model_name"
    echo "  SMALL_MODEL_NAME=your_small_model_name"
    echo ""
    echo "Optional variables:"
    echo "  OPENAI_BASE_URL=https://api.openai.com/v1"
    echo "  HOST=127.0.0.1"
    echo "  PORT=11434"
    echo "  LOG_LEVEL=INFO"
    echo "  IS_LOCAL_DEPLOYMENT=False (use True for single worker local deployment)"
    exit 1
    fi
fi

# Load environment variables from $ENV_FILE
print_info "Loading environment variables from $ENV_FILE..."

# Export variables from .env file while preserving existing environment variables
set -a  # Mark all new variables for export
while IFS='=' read -r key value; do
    # Skip empty lines and comments
    if [[ -z "$key" || "$key" =~ ^[[:space:]]*# ]]; then
        continue
    fi

    # Trim whitespace from key
    key=$(echo "$key" | xargs)

    # Skip if key is empty after trimming
    if [[ -z "$key" ]]; then
        continue
    fi

    # Remove quotes from value if present
    value=$(echo "$value" | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'$//")

    # Export the variable
    export "$key=$value"
done < "$ENV_FILE"
set +a  # Stop marking variables for export

# Function to check required environment variable
check_required_var() {
    local var_name=$1
    local var_value=${!var_name}

    if [ -z "$var_value" ]; then
        print_error "Required environment variable '$var_name' is not set!"
        return 1
    else
        # Mask sensitive information in output
        if [[ "$var_name" == *"API_KEY"* || "$var_name" == *"KEY"* ]]; then
            local masked_value="${var_value:0:6}...${var_value: -4}"
            print_success "$var_name is set (${masked_value})"
        else
            print_success "$var_name is set: $var_value"
        fi
        return 0
    fi
}

print_info "Validating required environment variables..."

# Check required environment variables
MISSING_VARS=0

# Check OPENAI_API_KEY (also accepts OPENROUTER_API_KEY as alternative)
if [ -z "$OPENAI_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    print_error "Required environment variable 'OPENAI_API_KEY' or 'OPENROUTER_API_KEY' is not set!"
    print_info "Please add one of the following to your .env file:"
    echo "  OPENAI_API_KEY=sk-..."
    echo "  or"
    echo "  OPENROUTER_API_KEY=sk-or-..."
    MISSING_VARS=$((MISSING_VARS + 1))
else
    if [ -n "$OPENROUTER_API_KEY" ] && [ -z "$OPENAI_API_KEY" ]; then
        export OPENAI_API_KEY="$OPENROUTER_API_KEY"
        print_success "Using OPENROUTER_API_KEY as OPENAI_API_KEY"
    fi
    check_required_var "OPENAI_API_KEY"
fi

# Check BIG_MODEL_NAME
if ! check_required_var "BIG_MODEL_NAME"; then
    print_info "Please add to your .env file:"
    echo "  BIG_MODEL_NAME=gpt-4"
    MISSING_VARS=$((MISSING_VARS + 1))
fi

# Check SMALL_MODEL_NAME
if ! check_required_var "SMALL_MODEL_NAME"; then
    print_info "Please add to your .env file:"
    echo "  SMALL_MODEL_NAME=gpt-3.5-turbo"
    MISSING_VARS=$((MISSING_VARS + 1))
fi

# Exit if any required variables are missing
if [ $MISSING_VARS -gt 0 ]; then
    echo ""
    print_error "Missing $MISSING_VARS required environment variable(s). Please update your .env file and try again."
    exit 1
fi

# Display optional configuration
print_info "Optional configuration:"
echo "  OPENAI_BASE_URL: ${OPENAI_BASE_URL:-https://api.openai.com/v1 (default)}"
echo "  HOST: ${HOST:-127.0.0.1 (default)}"
echo "  PORT: ${PORT:-11434 (default)}"
echo "  LOG_LEVEL: ${LOG_LEVEL:-INFO (default)}"
echo "  LOG_FILE_PATH: ${LOG_FILE_PATH:-log.jsonl (default)}"
echo "  ERROR_LOG_FILE_PATH: ${ERROR_LOG_FILE_PATH:-error.jsonl (default)}"
echo "  LOG_PRETTY_CONSOLE: ${LOG_PRETTY_CONSOLE:-True (default)}"
echo "  IS_LOCAL_DEPLOYMENT: ${IS_LOCAL_DEPLOYMENT:-False (default)}"
echo ""
print_info "Security/Guardrail configuration:"
echo "  RATE_LIMIT_ENABLED: ${RATE_LIMIT_ENABLED:-unset}"
echo "  RATE_LIMIT_PER_MINUTE: ${RATE_LIMIT_PER_MINUTE:-unset}"
echo "  RATE_LIMIT_BURST: ${RATE_LIMIT_BURST:-unset}"
echo "  SECURITY_HEADERS_ENABLED: ${SECURITY_HEADERS_ENABLED:-unset}"
echo "  ENABLE_HSTS: ${ENABLE_HSTS:-unset}"
echo "  ENABLE_CORS: ${ENABLE_CORS:-unset}"
echo "  CORS_ALLOW_ORIGINS: ${CORS_ALLOW_ORIGINS:-unset}"
echo "  CORS_ALLOW_METHODS: ${CORS_ALLOW_METHODS:-unset}"
echo "  CORS_ALLOW_HEADERS: ${CORS_ALLOW_HEADERS:-unset}"
echo "  ALLOWED_HOSTS: ${ALLOWED_HOSTS:-unset}"
echo "  RESTRICT_BASE_URL: ${RESTRICT_BASE_URL:-unset}"
echo "  ALLOWED_BASE_URL_HOSTS: ${ALLOWED_BASE_URL_HOSTS:-unset}"
echo "  REDACT_LOG_FIELDS: ${REDACT_LOG_FIELDS:-unset}"
echo "  MAX_STREAM_SECONDS: ${MAX_STREAM_SECONDS:-unset}"

print_info "Caching/Performance configuration:"
echo "  CACHE_TOKEN_COUNTS_ENABLED: ${CACHE_TOKEN_COUNTS_ENABLED:-unset}"
echo "  CACHE_TOKEN_COUNTS_TTL_S: ${CACHE_TOKEN_COUNTS_TTL_S:-unset}"
echo "  CACHE_TOKEN_COUNTS_MAX: ${CACHE_TOKEN_COUNTS_MAX:-unset}"
echo "  CACHE_CONVERTERS_ENABLED: ${CACHE_CONVERTERS_ENABLED:-unset}"
echo "  CACHE_CONVERTERS_MAX: ${CACHE_CONVERTERS_MAX:-unset}"
echo "  STREAM_DEDUPE_ENABLED: ${STREAM_DEDUPE_ENABLED:-unset}"
echo "  METRICS_CACHE_ENABLED: ${METRICS_CACHE_ENABLED:-unset}"

echo "  TRUNCATE_LONG_REQUESTS: ${TRUNCATE_LONG_REQUESTS:-unset}"
echo "  TRUNCATION_CONFIG: ${TRUNCATION_CONFIG:-unset}"

print_info "Provider retry configuration:"
echo "  PROVIDER_MAX_RETRIES: ${PROVIDER_MAX_RETRIES:-unset}"
echo "  PROVIDER_RETRY_BASE_DELAY: ${PROVIDER_RETRY_BASE_DELAY:-unset}"
echo "  PROVIDER_RETRY_JITTER: ${PROVIDER_RETRY_JITTER:-unset}"

# Check required files
if [ ! -f "wsgi.py" ]; then
    print_error "wsgi.py not found in $SCRIPT_DIR"
    print_info "Please ensure you're running this script from the CCProxy root directory."
    exit 1
fi
if [ ! -f "gunicorn.conf.py" ]; then
    print_error "gunicorn.conf.py not found in $SCRIPT_DIR"
    print_info "Please ensure gunicorn.conf.py is present in the project root."
    exit 1
fi

# Check Python dependencies via uv
print_info "Checking Python dependencies via uv..."
if [ -f "requirements.txt" ] || [ -f "pyproject.toml" ]; then
    print_info "Installing/syncing dependencies via uv..."
    uv sync --dev
    print_success "Dependencies synchronized successfully"
else
    print_info "No requirements.txt or pyproject.toml found - will install minimal dependencies"
    # Ensure basic dependencies are available
    uv add fastapi uvicorn openai pydantic tiktoken httpx gunicorn --dev
fi

# Function to extract version from pyproject.toml
get_version() {
    if [ -f "pyproject.toml" ]; then
        grep -E '^version = ' pyproject.toml | sed -E 's/version = "(.*)"/\1/' | tr -d ' '
    else
        echo "unknown"
    fi
}

# Display startup information
echo ""
print_success "Environment validation complete!"
print_info "Starting CCProxy..."
echo "==================================================="
echo "  CCProxy - Anthropic to OpenAI-compatible Bridge"
echo "  Version: $(get_version)"
echo "  Host: ${HOST:-127.0.0.1}"
echo "  Port: ${PORT:-11434}"
echo "  API URL: http://${HOST:-127.0.0.1}:${PORT:-11434}"
echo "==================================================="
echo ""

# Run the application
print_info "Launching Gunicorn server..."

# Check if this is a local deployment (single worker)
if [ "${IS_LOCAL_DEPLOYMENT}" = "True" ] || [ "${IS_LOCAL_DEPLOYMENT}" = "true" ]; then
    print_info "Local deployment detected - using single worker process"
    exec uv run gunicorn --config gunicorn.conf.py --workers 1 wsgi:app
else
    exec uv run gunicorn --config gunicorn.conf.py wsgi:app
fi
