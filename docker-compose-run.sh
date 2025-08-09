#!/bin/bash

# Docker Compose wrapper script that ensures .env file is loaded
# Provides easy management of the CCProxy service

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Script configuration
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default action
ACTION="${1:-up}"

# Show help
show_help() {
    echo "Usage: $0 [COMMAND] [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  up        Start the service (default)"
    echo "  down      Stop and remove the service"
    echo "  restart   Restart the service"
    echo "  logs      View service logs"
    echo "  build     Build/rebuild the image"
    echo "  status    Show service status"
    echo "  shell     Open a shell in the container"
    echo "  clean     Stop service and clean volumes"
    echo ""
    echo "Options:"
    echo "  -d        Run in detached mode (for 'up' command)"
    echo "  -f        Follow logs (for 'logs' command)"
    echo ""
    echo "Examples:"
    echo "  $0              # Start service (detached)"
    echo "  $0 up -d        # Start service detached"
    echo "  $0 logs -f      # View and follow logs"
    echo "  $0 restart      # Restart the service"
    exit 0
}

# Check for help flag
if [[ "$1" == "--help" ]] || [[ "$1" == "-h" ]]; then
    show_help
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null 2>&1; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Determine docker-compose command (v1 or v2)
if docker compose version &> /dev/null 2>&1; then
    COMPOSE_CMD="docker compose"
else
    COMPOSE_CMD="docker-compose"
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    print_error ".env file not found!"
    print_info "Creating example .env file..."
    cat > .env.example << 'EOF'
# Required environment variables
OPENAI_API_KEY=sk-your-api-key-here
BIG_MODEL_NAME=gpt-4
SMALL_MODEL_NAME=gpt-3.5-turbo

# Optional configuration
BASE_URL=https://api.openai.com/v1
PORT=8082
LOG_LEVEL=INFO
WEB_CONCURRENCY=4

# Docker configuration
CONTAINER_NAME=ccproxy
EOF
    print_info "Please create a .env file with your configuration:"
    echo "  cp .env.example .env"
    echo "  # Then edit .env with your API keys and settings"
    exit 1
fi

# Load and validate environment variables
print_info "Loading environment variables from .env file..."
set -a
source .env
set +a

# Validate required variables
ERRORS=0
if [ -z "$OPENAI_API_KEY" ] && [ -z "$OPENROUTER_API_KEY" ]; then
    print_error "OPENAI_API_KEY or OPENROUTER_API_KEY must be set in .env!"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$BIG_MODEL_NAME" ]; then
    print_error "BIG_MODEL_NAME must be set in .env!"
    ERRORS=$((ERRORS + 1))
fi

if [ -z "$SMALL_MODEL_NAME" ]; then
    print_error "SMALL_MODEL_NAME must be set in .env!"
    ERRORS=$((ERRORS + 1))
fi

if [ $ERRORS -gt 0 ]; then
    print_error "Missing required environment variables. Please update your .env file."
    exit 1
fi

# Execute command based on action
case "$ACTION" in
    up|start)
        print_info "Starting CCProxy service..."
        if [[ "$2" == "-d" ]] || [[ -z "$2" ]]; then
            $COMPOSE_CMD up -d
            print_success "Service started successfully!"
            echo ""
            print_info "Service Information:"
            echo "  URL: http://localhost:${PORT:-8082}"
            echo "  Health: http://localhost:${PORT:-8082}/health"
            echo "  Metrics: http://localhost:${PORT:-8082}/v1/metrics"
            echo ""
            print_info "View logs with: $0 logs -f"
        else
            $COMPOSE_CMD up
        fi
        ;;
        
    down|stop)
        print_info "Stopping CCProxy service..."
        $COMPOSE_CMD down
        print_success "Service stopped"
        ;;
        
    restart)
        print_info "Restarting CCProxy service..."
        $COMPOSE_CMD restart
        print_success "Service restarted"
        ;;
        
    logs)
        if [[ "$2" == "-f" ]]; then
            $COMPOSE_CMD logs -f
        else
            $COMPOSE_CMD logs --tail=100
        fi
        ;;
        
    build|rebuild)
        print_info "Building CCProxy image..."
        $COMPOSE_CMD build
        print_success "Build complete"
        ;;
        
    status|ps)
        print_info "Service status:"
        $COMPOSE_CMD ps
        echo ""
        # Check health endpoint
        if curl -f -s "http://localhost:${PORT:-8082}/health" > /dev/null 2>&1; then
            print_success "Health check: API is responding"
        else
            print_warning "Health check: API is not responding"
        fi
        ;;
        
    shell|exec)
        print_info "Opening shell in container..."
        $COMPOSE_CMD exec ccproxy /bin/bash
        ;;
        
    clean)
        print_warning "This will stop the service and remove all volumes!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            $COMPOSE_CMD down -v
            print_success "Service stopped and volumes removed"
        else
            print_info "Cancelled"
        fi
        ;;
        
    *)
        print_error "Unknown command: $ACTION"
        echo ""
        show_help
        ;;
esac