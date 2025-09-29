import inspect
from unittest.mock import MagicMock, patch

import pytest
import pytest_asyncio


_original_pytest_fixture = pytest.fixture


def _async_aware_fixture(*args, **kwargs):
    def decorator(func):
        if inspect.iscoroutinefunction(func) or inspect.isasyncgenfunction(func):
            pytest.fixture = _original_pytest_fixture
            try:
                return pytest_asyncio.fixture(*args, **kwargs)(func)
            finally:
                pytest.fixture = _async_aware_fixture
        return _original_pytest_fixture(*args, **kwargs)(func)

    # Support bare decorator usage
    if args and callable(args[0]) and not kwargs:
        func = args[0]
        return _async_aware_fixture()(func)

    return decorator


pytest.fixture = _async_aware_fixture


@pytest.fixture(autouse=True, scope="session")
def mock_logger():
    with patch("ccproxy.logging._logger", MagicMock()) as mock_logger:
        mock_logger.log.return_value = None
        yield mock_logger
