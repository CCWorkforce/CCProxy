import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture(autouse=True, scope="session")
def mock_logger():
    with patch("ccproxy.logging._logger", MagicMock()) as mock_logger:
        mock_logger.log.return_value = None
        yield mock_logger
