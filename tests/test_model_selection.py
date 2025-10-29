"""Tests for model selection logic."""

from ccproxy.application.model_selection import select_target_model


class TestModelSelection:
    """Test cases for model selection functionality."""

    def test_opus_model_selects_big_model(self) -> None:
        """Test that Opus models map to big model."""
        result = select_target_model(
            client_model_name="claude-3-opus-20240229",
            request_id="test-123",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_sonnet_model_selects_big_model(self) -> None:
        """Test that Sonnet models map to big model."""
        result = select_target_model(
            client_model_name="claude-3-sonnet-20240229",
            request_id="test-456",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_haiku_model_selects_small_model(self) -> None:
        """Test that Haiku models map to small model."""
        result = select_target_model(
            client_model_name="claude-3-haiku-20240307",
            request_id="test-789",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-3.5-turbo"

    def test_unknown_model_defaults_to_small_model(self) -> None:
        """Test that unknown models default to small model."""
        result = select_target_model(
            client_model_name="unknown-model",
            request_id="test-999",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-3.5-turbo"

    def test_case_insensitive_opus(self) -> None:
        """Test that model name matching is case-insensitive for Opus."""
        result = select_target_model(
            client_model_name="CLAUDE-3-OPUS-20240229",
            request_id="test-uppercase",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_case_insensitive_sonnet(self) -> None:
        """Test that model name matching is case-insensitive for Sonnet."""
        result = select_target_model(
            client_model_name="Claude-3-Sonnet-Latest",
            request_id="test-mixed-case",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_case_insensitive_haiku(self) -> None:
        """Test that model name matching is case-insensitive for Haiku."""
        result = select_target_model(
            client_model_name="claude-3-HAIKU-20240307",
            request_id="test-haiku-upper",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-3.5-turbo"

    def test_partial_match_opus(self) -> None:
        """Test that partial match works for Opus in model name."""
        result = select_target_model(
            client_model_name="my-custom-opus-model",
            request_id="test-partial-opus",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_partial_match_sonnet(self) -> None:
        """Test that partial match works for Sonnet in model name."""
        result = select_target_model(
            client_model_name="my-custom-sonnet-model",
            request_id="test-partial-sonnet",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"

    def test_partial_match_haiku(self) -> None:
        """Test that partial match works for Haiku in model name."""
        result = select_target_model(
            client_model_name="my-custom-haiku-model",
            request_id="test-partial-haiku",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-3.5-turbo"

    def test_different_big_model_name(self) -> None:
        """Test with different big model names."""
        result = select_target_model(
            client_model_name="claude-3-opus-20240229",
            request_id="test-custom-big",
            big_model_name="gpt-4-turbo",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4-turbo"

    def test_different_small_model_name(self) -> None:
        """Test with different small model names."""
        result = select_target_model(
            client_model_name="claude-3-haiku-20240307",
            request_id="test-custom-small",
            big_model_name="gpt-4",
            small_model_name="gpt-4o-mini",
        )
        assert result == "gpt-4o-mini"

    def test_empty_model_name(self) -> None:
        """Test with empty model name defaults to small."""
        result = select_target_model(
            client_model_name="",
            request_id="test-empty",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-3.5-turbo"

    def test_opus_and_haiku_in_name(self) -> None:
        """Test when both opus and haiku appear - opus takes precedence."""
        result = select_target_model(
            client_model_name="my-opus-haiku-model",
            request_id="test-both",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"  # opus has priority

    def test_sonnet_and_haiku_in_name(self) -> None:
        """Test when both sonnet and haiku appear - sonnet takes precedence."""
        result = select_target_model(
            client_model_name="my-sonnet-haiku-model",
            request_id="test-both-2",
            big_model_name="gpt-4",
            small_model_name="gpt-3.5-turbo",
        )
        assert result == "gpt-4"  # sonnet has priority
