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

# Check if Python 3 is installed
if ! command_exists python3; then
    print_error "Python 3 is not installed. Please install Python 3.13 or higher."
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found in $SCRIPT_DIR"
    print_info "Please create a .env file with the following required variables:"
    echo "  OPENAI_API_KEY=your_api_key_here"
    echo "  BIG_MODEL_NAME=your_big_model_name"
    echo "  SMALL_MODEL_NAME=your_small_model_name"
    echo ""
    echo "Optional variables:"
    echo "  BASE_URL=https://api.openai.com/v1"
    echo "  HOST=127.0.0.1"
    echo "  PORT=8082"
    echo "  LOG_LEVEL=INFO"
    exit 1
fi

# Load environment variables from .env file
print_info "Loading environment variables from .env file..."

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
done < .env
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
echo "  BASE_URL: ${BASE_URL:-https://api.openai.com/v1 (default)}"
echo "  HOST: ${HOST:-127.0.0.1 (default)}"
echo "  PORT: ${PORT:-8082 (default)}"
echo "  LOG_LEVEL: ${LOG_LEVEL:-INFO (default)}"
echo "  LOG_FILE_PATH: ${LOG_FILE_PATH:-log.jsonl (default)}"

# Check if main.py exists
if [ ! -f "main.py" ]; then
    print_error "main.py not found in $SCRIPT_DIR"
    print_info "Please ensure you're running this script from the CCProxy root directory."
    exit 1
fi

# Check Python dependencies
print_info "Checking Python dependencies..."
if command_exists pip3; then
    # Check if required packages are installed
    MISSING_PACKAGES=""

    for package in fastapi uvicorn openai pydantic tiktoken httpx; do
        if ! python3 -c "import $package" 2>/dev/null; then
            MISSING_PACKAGES="$MISSING_PACKAGES $package"
        fi
    done

    if [ -n "$MISSING_PACKAGES" ]; then
        print_info "Missing Python packages detected:$MISSING_PACKAGES"
        print_info "Consider running: pip3 install -r requirements.txt"
    fi
fi

# Display startup information
echo ""
print_success "Environment validation complete!"
print_info "Starting CCProxy..."
echo "=========================================="
echo "  CCProxy - Anthropic to OpenAI-compatible Bridge"
echo "  Host: ${HOST:-127.0.0.1}"
echo "  Port: ${PORT:-8082}"
echo "  API URL: http://${HOST:-127.0.0.1}:${PORT:-8082}"
echo "=========================================="
echo ""

# Run the application
print_info "Launching Python application..."
exec python3 main.py
