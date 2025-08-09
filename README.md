# CCProxy

🌾 🥳 🌋 🏰 🌅 🌕 Claude Code Proxy 🌖 🌔 🌈 🏆 👑

## ⚡ Performance Optimizations

CCProxy includes high-performance HTTP client optimizations for faster OpenAI API communication:

- **HTTP/2 Support**: Enabled by default for request multiplexing
- **Enhanced Connection Pooling**: 50 keepalive connections, 500 max connections
- **Compression**: Supports gzip, deflate, and Brotli
- **Smart Retries**: Automatic retry with exponential backoff
- **Response Caching**: Prevents duplicate API calls and handles timeouts

### Performance Improvements

- 30-50% faster single request latency
- 2-3x better throughput for concurrent requests
- Reduced connection overhead with persistent connections

See [HTTP_OPTIMIZATION.md](HTTP_OPTIMIZATION.md) for details.
