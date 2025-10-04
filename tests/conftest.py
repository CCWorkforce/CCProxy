from unittest.mock import MagicMock, patch

import pytest


# Configure anyio to only use asyncio backend
@pytest.fixture
def anyio_backend():
    """Force tests to use asyncio backend only."""
    return "asyncio"


@pytest.fixture(autouse=True, scope="session")
def mock_logger():
    with patch("ccproxy.logging._logger", MagicMock()) as mock_logger:
        mock_logger.log.return_value = None
        yield mock_logger
