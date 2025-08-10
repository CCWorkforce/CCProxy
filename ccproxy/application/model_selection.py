from ..logging import warning, debug, LogRecord, LogEvent


def select_target_model(client_model_name: str, request_id: str, big_model_name: str, small_model_name: str) -> str:
    """Map Anthropic model names to appropriate OpenAI target models."""

    client_model_lower = client_model_name.lower()

    if "opus" in client_model_lower or "sonnet" in client_model_lower:
        target_model = big_model_name
    elif "haiku" in client_model_lower:
        target_model = small_model_name
    else:
        target_model = small_model_name
        warning(
            LogRecord(
                event=LogEvent.MODEL_SELECTION.value,
                message=(
                    f"Unknown client model '{client_model_name}', defaulting to SMALL model '{target_model}'."
                ),
                request_id=request_id,
                data={
                    "client_model": client_model_name,
                    "default_target_model": target_model,
                },
            )
        )

    debug(
        LogRecord(
            event=LogEvent.MODEL_SELECTION.value,
            message=f"Client model '{client_model_name}' mapped to target model '{target_model}'.",
            request_id=request_id,
            data={"client_model": client_model_name, "target_model": target_model},
        )
    )
    return target_model
