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

To set up and run CCProxy, follow the steps in [RUN_INSTRUCTIONS.md](RUN_INSTRUCTIONS.md).

Before starting Claude Code, run:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8082
```

Then start your coding session with Claude Code:

```bash
claude
```
