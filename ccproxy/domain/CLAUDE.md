# Domain Layer - CLAUDE.md

**Scope**: Core business logic, domain models, and domain-specific exceptions

## Files in this layer:
- `models.py`: Core domain entities and data structures for CCProxy
- `exceptions.py`: Domain-specific exceptions and error types

## Guidelines:
- **Pure domain logic**: No dependencies on external frameworks or infrastructure
- **Pydantic models**: Define API shapes and data contracts
- **UTF-8 safe**: Keep models serialization-safe and UTF-8 clean
- **Dependency direction**: Only depends on standard library, not on application/infrastructure layers
- **Domain exceptions**: Use custom exceptions for business rule violations
- **Immutable where possible**: Prefer immutable data structures for domain entities
