# CCProxy Run Instructions

## Quick Start

### Step 1: Create `.env` File

Create a `.env` file by copying the example:

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY, BIG_MODEL_NAME, SMALL_MODEL_NAME
```

```bash
# Required variables
OPENAI_API_KEY=sk-your-api-key-here  # Or use OPENROUTER_API_KEY
BIG_MODEL_NAME=gpt-4
SMALL_MODEL_NAME=gpt-3.5-turbo

# Optional variables
OPENAI_BASE_URL=https://api.openai.com/v1
PORT=8082
LOG_LEVEL=INFO
WEB_CONCURRENCY=4
```

### Step 2: Start the Service

```bash
./docker-compose-run.sh up
```

### Step 3: View Logs

```bash
./docker-compose-run.sh logs -f
```

That's it! The service is now running at `http://localhost:8082`

## Environment Variables

### Required

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` or `OPENROUTER_API_KEY` | API key for OpenAI or OpenRouter | `sk-...` or `sk-or-...` |
| `BIG_MODEL_NAME` | Primary model for complex requests | `gpt-4` |
| `SMALL_MODEL_NAME` | Secondary model for simple requests | `gpt-3.5-turbo` |

### Optional

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_BASE_URL` | OpenAI-compatible API endpoint | `https://api.openai.com/v1` |
| `PORT` | Server port | `8082` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `WEB_CONCURRENCY` | Number of Gunicorn workers | `4` |

## Common Commands

```bash
# Start the service
./docker-compose-run.sh up

# View logs (follow mode)
./docker-compose-run.sh logs -f

# Stop the service
./docker-compose-run.sh down

# Restart the service
./docker-compose-run.sh restart

# Check service status
./docker-compose-run.sh status

# Rebuild the image
./docker-compose-run.sh build
```

## Validation and Testing

### Health Check
```bash
curl http://localhost:8082/health
```

### View Metrics
```bash
curl http://localhost:8082/v1/metrics | jq
```

### Test API
```bash
curl -X POST http://localhost:8082/v1/messages \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -H "anthropic-version: 2023-06-01" \
  -d '{
    "model": "claude-3-opus-20240229",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

## Troubleshooting

### Missing `.env` File
If you see "`.env` file not found!", create one:
```bash
cp .env.example .env
# Edit .env with your API keys
```

### Port Already in Use
Change the port in `.env`:
```bash
PORT=8083  # Use a different port
```

### View Container Logs
```bash
./docker-compose-run.sh logs -f
```

### Container Won't Start
Check the status and logs:
```bash
./docker-compose-run.sh status
./docker-compose-run.sh logs
```

## Performance Monitoring

### Real-time Metrics
```bash
# Performance metrics
curl http://localhost:8082/v1/metrics | jq

# Cache statistics
curl http://localhost:8082/v1/cache/stats | jq

# Clear cache if needed
curl -X POST http://localhost:8082/v1/cache/clear
```

### Monitor Resources
```bash
docker stats ccproxy
```

## Security Notes

1. **Never commit `.env` file** to version control
2. **Keep API keys secure** - use environment variables
3. **Container runs as non-root user** for security
4. **Use HTTPS** when exposing to the internet
5. **Set up rate limiting** for public deployments

## Advanced Options

### Use Alpine Image (Smaller)
Edit `docker-compose.yml` to use Alpine:
```yaml
services:
  ccproxy:
    build:
      target: production-alpine  # Use Alpine variant
```

### Adjust Workers
Set in `.env`:
```bash
WEB_CONCURRENCY=8  # Increase workers for higher load
```

### Custom Configuration
All settings can be adjusted in `.env` file. The service will automatically reload with:
```bash
./docker-compose-run.sh restart
```

## Quick Reference

| Task | Command |
|------|---------|
| Start service | `./docker-compose-run.sh up` |
| View logs | `./docker-compose-run.sh logs -f` |
| Stop service | `./docker-compose-run.sh down` |
| Restart | `./docker-compose-run.sh restart` |
| Check health | `curl http://localhost:8082/health` |
| View metrics | `curl http://localhost:8082/v1/metrics` |

---

**That's all you need!** The Docker Compose setup handles everything automatically:
- Builds the image if needed
- Loads your `.env` file
- Manages the container lifecycle
- Provides health checks
- Handles logging

For more details, see [DOCKER_QUICK_START.md](DOCKER_QUICK_START.md)