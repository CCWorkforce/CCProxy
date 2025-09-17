# Application Converters Module - CLAUDE.md

**Scope**: Modular message conversion between Anthropic and OpenAI API formats

## Files in this module:
- `__init__.py`: Converter module exports and factory functions
- `main.py`: Main converter orchestration and high-level conversion logic
- `base.py`: Base converter classes and common conversion utilities
- `anthropic_to_openai.py`: Converter for Anthropic format to OpenAI format
- `openai_to_anthropic.py`: Converter for OpenAI format to Anthropic format
- `content_converter.py`: Content-specific conversion (text, images, tool use, tool results)
- `tool_converter.py`: Tool choice mapping and tool result serialization

## Guidelines:
- **API parity**: Maintain strict compatibility between Anthropic and OpenAI schemas
- **Multi-modal support**: Handle text, images, tool use, and tool results correctly
- **UTF-8 enforcement**: Advanced UTF-8 enforcement with recovery mechanisms
- **Tool choice mapping**: Sophisticated tool choice mapping between API formats
- **Caching**: Use @lru_cache decorators for performance optimization
- **Streaming support**: Ensure converters work with both streaming and non-streaming requests
- **Error handling**: Graceful handling of conversion errors with fallback mechanisms
- **Serialization**: Complex tool result serialization with proper type checking