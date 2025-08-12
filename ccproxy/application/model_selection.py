from typing import TypeAlias

from ..logging import warning, debug, LogRecord, LogEvent


ModelName: TypeAlias = str


def select_target_model(
    client_model_name: ModelName, request_id: str, big_model_name: ModelName, small_model_name: ModelName
) -> ModelName:
    """Map Anthropic model names to appropriate OpenAI target models.

    Args:
        client_model_name (str): The name of the client model to map.
        request_id (str): The unique identifier for the request.
        big_model_name (str): The name of the big model to use.
        small_model_name (str): The name of the small model to use.

    Returns:
        str: The name of the target model to use.
    """

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
