# CCProxy

🌾 🥳 🌋 🏰 🌅 🌕 Claude Code Proxy 🌖 🌔 🌈 🏆 👑

## Motivation

Recent analytics show a large cost gap between major models: OpenAI GPT‑5 is far more cost‑efficient than Anthropic Claude Opus 4.1 (≈$11.25 vs ≈$90 per 1M input+output tokens). CCProxy helps teams control AI spend and latency by minimizing duplicate work, maximizing transport efficiency, and serving as a drop‑in proxy for OpenAI‑compatible APIs. This allows organizations to standardize on one integration while selecting the most cost‑effective model per workload.

### Pricing Overview

| Model               | Input Tokens (\$/1M) | Output Tokens (\$/1M) |
| ------------------- | -------------------- | --------------------- |
| **OpenAI GPT‑5**    | \$1.25               | \$10.00               |
| **Claude Opus 4.1** | \$15.00              | \$75.00               |

* **GPT‑5** input and output rates are confirmed via Wired, OpenAI’s own API pricing page, and TechCrunch
* **Claude Opus 4.1** pricing is stated directly on Anthropic’s API pricing page.

## ⚡ Performance Optimizations

CCProxy includes high-performance HTTP client optimizations for faster OpenAI API communication:

* **HTTP/2 Support**: Enabled by default for request multiplexing
* **Enhanced Connection Pooling**: 50 keepalive connections, 500 max connections
* **Compression**: Supports gzip, deflate, and Brotli
* **Smart Retries**: Automatic retry with exponential backoff
* **Response Caching**: Prevents duplicate API calls and handles timeouts

### Performance Improvements

* 30-50% faster single request latency
* 2-3x better throughput for concurrent requests
* Reduced connection overhead with persistent connections

See [HTTP_OPTIMIZATION.md](HTTP_OPTIMIZATION.md) for details.

## Quickstart (uv + .env + Gunicorn)

### Caching and performance settings

Environment variables (with defaults) you can tune:

- CACHE_TOKEN_COUNTS_ENABLED=true
- CACHE_TOKEN_COUNTS_TTL_S=300
- CACHE_TOKEN_COUNTS_MAX=2048
- CACHE_CONVERTERS_ENABLED=true
- CACHE_CONVERTERS_MAX=256
- STREAM_DEDUPE_ENABLED=true

Notes:
- Token count cache hashes request shape; no caching on exceptions
- Converter caches are small and safe; they only memoize deterministic mappings
- Streaming de-duplication enables **fan-out**: the first identical streaming request opens a single upstream connection; subsequent identical requests attach as subscribers and receive the same SSE events.
    - Back-pressure: each subscriber queue is bounded (1000 lines). Slow subscribers are dropped to protect the primary stream.
    - All subscribers are finalized on normal completion **or any error path** ensuring no goroutines leak.
- Provider retries: rate-limit (429) responses are retried with exponential back-off (configurable via `PROVIDER_MAX_RETRIES`, `PROVIDER_RETRY_BASE_DELAY`, `PROVIDER_RETRY_JITTER`). `Retry-After` headers are forwarded to the client and recorded in logs.

Metrics:
- GET /v1/metrics exposes performance, response_cache, request_validator_cache, and token_count_cache stats

1. Create your environment file from the template:

```bash
cp .env.example .env
# edit .env to set OPENAI_API_KEY, BIG_MODEL_NAME, SMALL_MODEL_NAME
```

2. Install Python dependencies into an isolated environment using uv:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv
. .venv/bin/activate
uv pip install -r requirements.txt
```

3. Start the server (pure Python with Gunicorn):

```bash
./run-ccproxy.sh
```

4. Point your Anthropic client at the proxy:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

Then start your coding session with Claude Code:

```bash
claude
```
