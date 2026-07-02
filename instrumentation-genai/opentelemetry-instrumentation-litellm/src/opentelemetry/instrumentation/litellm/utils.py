# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Request/response extractors shared between sync + async + stream paths.

Nothing in this module talks to OpenTelemetry directly. Everything either
reads litellm-shaped call arguments and returns plain values, or drives
``TelemetryHandler.start_*`` and mutates the resulting ``InferenceInvocation``
/ ``EmbeddingInvocation``. All span, metric, and event emission is owned by
``opentelemetry-util-genai`` — see ``instrumentation-genai/CLAUDE.md``.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Tuple
from urllib.parse import urlparse

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)
from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import (
    EmbeddingInvocation,
    InferenceInvocation,
)
from opentelemetry.util.genai.types import (
    InputMessage,
    OutputMessage,
    Text,
    ToolCallRequest,
    ToolCallResponse,
)

# LiteLLM exposes a large, evolving set of provider strings via
# ``custom_llm_provider``. The semantic-conventions
# ``GenAiProviderNameValues`` enum defines a curated subset — when there is a
# direct match we prefer the canonical enum value; otherwise we pass litellm's
# provider string through verbatim so the instrumentation remains useful for
# the long tail of providers litellm supports.
_SEMCONV_PROVIDER = {
    "openai": GenAIAttributes.GenAiProviderNameValues.OPENAI.value,
    "azure": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "azure_openai": GenAIAttributes.GenAiProviderNameValues.AZURE_AI_OPENAI.value,
    "anthropic": GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value,
    "bedrock": GenAIAttributes.GenAiProviderNameValues.AWS_BEDROCK.value,
    "cohere": GenAIAttributes.GenAiProviderNameValues.COHERE.value,
    "vertex_ai": GenAIAttributes.GenAiProviderNameValues.GCP_VERTEX_AI.value,
    "gemini": GenAIAttributes.GenAiProviderNameValues.GCP_GEMINI.value,
    "groq": GenAIAttributes.GenAiProviderNameValues.GROQ.value,
    "mistral": GenAIAttributes.GenAiProviderNameValues.MISTRAL_AI.value,
    "deepseek": GenAIAttributes.GenAiProviderNameValues.DEEPSEEK.value,
    "xai": GenAIAttributes.GenAiProviderNameValues.X_AI.value,
    "perplexity": GenAIAttributes.GenAiProviderNameValues.PERPLEXITY.value,
}


def get_property_value(obj: Any, property_name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(property_name, None)
    return getattr(obj, property_name, None)


def resolve_provider(kwargs: dict[str, Any], result: Any = None) -> Optional[str]:
    """Return litellm's ``custom_llm_provider`` for this call.

    Preference order:

    1. Explicit ``custom_llm_provider`` kwarg the caller passed.
    2. ``_hidden_params.custom_llm_provider`` on the response (populated by
       litellm after the call resolves).
    3. ``litellm.get_llm_provider(model)`` if the model string carries a
       provider prefix (e.g. ``anthropic/claude-3-5-sonnet``).
    """

    provider = kwargs.get("custom_llm_provider")
    if provider:
        return provider

    if result is not None:
        hidden = get_property_value(result, "_hidden_params")
        if hidden:
            provider = get_property_value(hidden, "custom_llm_provider")
            if provider:
                return provider

    model = kwargs.get("model")
    if not model:
        return None

    try:
        import litellm  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        _, provider, _, _ = litellm.get_llm_provider(model=model)
        return provider
    except Exception:  # pylint: disable=broad-except
        return None


def system_value(provider: Optional[str]) -> Optional[str]:
    """Map litellm's ``custom_llm_provider`` onto a ``gen_ai.provider.name``.

    Semconv calls this ``gen_ai.provider.name``; util-genai emits it based on
    the ``provider`` field on ``InferenceInvocation`` / ``EmbeddingInvocation``.
    """
    if not provider:
        return None
    return _SEMCONV_PROVIDER.get(provider, provider)


def get_server_address_and_port(
    kwargs: dict[str, Any],
) -> Tuple[Optional[str], Optional[int]]:
    """Extract ``server.address`` / ``server.port`` from ``api_base``.

    LiteLLM's ``completion()`` is a module-level function, so unlike the OpenAI
    SDK there is no persistent client to introspect. Callers may pass
    ``api_base`` (or ``base_url``) to point at a proxy or self-hosted endpoint;
    when they do, surface it. Otherwise return ``(None, None)`` — the default
    endpoint is provider-specific and would be misleading to hard-code.
    """
    base_url = kwargs.get("api_base") or kwargs.get("base_url")
    if not base_url or not isinstance(base_url, str):
        return None, None

    parsed = urlparse(base_url)
    port = parsed.port
    if port and (port == 443 or port <= 0):
        port = None
    return parsed.hostname, port


def _is_text_part(content: Any) -> bool:
    return isinstance(content, str) or (
        isinstance(content, Iterable)
        and all(isinstance(part, str) for part in content)
    )


def _extract_tool_calls(tool_calls: Any) -> List[ToolCallRequest]:
    calls: List[ToolCallRequest] = []
    for tc in tool_calls or []:
        func = get_property_value(tc, "function") or {}
        name = get_property_value(func, "name")
        arguments = get_property_value(func, "arguments")
        # Arguments come across the wire as a JSON string on OpenAI-shaped
        # responses; leave the parsing to util-genai's consumers by passing
        # the raw string through verbatim — it round-trips into the
        # ``ToolCallRequest.arguments`` slot.
        calls.append(
            ToolCallRequest(
                name=name,
                id=get_property_value(tc, "id"),
                arguments=arguments,
            )
        )
    return calls


def prepare_input_messages(messages: Iterable[Any]) -> List[InputMessage]:
    prepared: List[InputMessage] = []
    for message in messages or []:
        role = str(get_property_value(message, "role") or "")
        parts: List[Any] = []
        content = get_property_value(message, "content")

        if role == "assistant":
            tool_calls = get_property_value(message, "tool_calls")
            if tool_calls:
                parts.extend(_extract_tool_calls(tool_calls))
            if _is_text_part(content):
                parts.append(Text(content=str(content)))
        elif role == "tool":
            tool_call_id = get_property_value(message, "tool_call_id")
            parts.append(ToolCallResponse(id=tool_call_id, response=content))
        else:
            if _is_text_part(content):
                parts.append(Text(content=str(content)))

        prepared.append(InputMessage(role=role, parts=parts))
    return prepared


def prepare_output_messages(choices: Iterable[Any]) -> List[OutputMessage]:
    prepared: List[OutputMessage] = []
    for choice in choices or []:
        message = get_property_value(choice, "message")
        role = str(get_property_value(message, "role") or "assistant")
        parts: List[Any] = []
        content = get_property_value(message, "content")
        if _is_text_part(content):
            parts.append(Text(content=str(content)))
        tool_calls = get_property_value(message, "tool_calls")
        if tool_calls:
            parts.extend(_extract_tool_calls(tool_calls))
        prepared.append(
            OutputMessage(
                role=role,
                parts=parts,
                finish_reason=str(
                    get_property_value(choice, "finish_reason") or "error"
                ),
            )
        )
    return prepared


def create_chat_invocation(
    handler: TelemetryHandler,
    kwargs: dict[str, Any],
    capture_content: bool,
    provider: str,
) -> InferenceInvocation:
    """Start an inference invocation and populate its request-side fields.

    The invocation returned here has its span already open (per util-genai's
    factory-method contract). The caller must eventually call ``.stop()`` on
    success or ``.fail(exc)`` on failure — this module never does that itself.
    """
    server_address, server_port = get_server_address_and_port(kwargs)
    invocation = handler.start_inference(
        provider,
        request_model=kwargs.get("model") or "",
        server_address=server_address,
        server_port=server_port,
    )

    invocation.temperature = kwargs.get("temperature")
    invocation.top_p = kwargs.get("top_p")
    invocation.max_tokens = kwargs.get("max_tokens") or kwargs.get(
        "max_completion_tokens"
    )
    invocation.seed = kwargs.get("seed")
    stop_sequences = kwargs.get("stop")
    if stop_sequences is not None:
        if isinstance(stop_sequences, str):
            stop_sequences = [stop_sequences]
        invocation.stop_sequences = stop_sequences

    if capture_content:  # skip parsing when content isn't going anywhere
        invocation.input_messages = prepare_input_messages(
            kwargs.get("messages", [])
        )

    return invocation


def create_embedding_invocation(
    handler: TelemetryHandler,
    kwargs: dict[str, Any],
    provider: str,
) -> EmbeddingInvocation:
    server_address, server_port = get_server_address_and_port(kwargs)
    invocation = handler.start_embedding(
        provider,
        request_model=kwargs.get("model") or "",
        server_address=server_address,
        server_port=server_port,
    )
    encoding_format = kwargs.get("encoding_format")
    if encoding_format:
        invocation.encoding_formats = (
            [encoding_format]
            if isinstance(encoding_format, str)
            else list(encoding_format)
        )
    dimensions = kwargs.get("dimensions")
    if dimensions is not None:
        invocation.dimension_count = dimensions
    return invocation


def set_chat_response_properties(
    invocation: InferenceInvocation,
    result: Any,
    capture_content: bool,
) -> None:
    """Fill in response-side fields on the invocation from a completion result."""
    if getattr(result, "model", None):
        invocation.response_model_name = result.model
    if getattr(result, "id", None):
        invocation.response_id = result.id

    choices = getattr(result, "choices", None)
    if choices:
        invocation.finish_reasons = [
            str(get_property_value(c, "finish_reason") or "error")
            for c in choices
        ]
        if capture_content:
            invocation.output_messages = prepare_output_messages(choices)

    usage = getattr(result, "usage", None)
    if usage is not None:
        prompt_tokens = get_property_value(usage, "prompt_tokens")
        if prompt_tokens is not None:
            invocation.input_tokens = prompt_tokens
        completion_tokens = get_property_value(usage, "completion_tokens")
        if completion_tokens is not None:
            invocation.output_tokens = completion_tokens


def set_embedding_response_properties(
    invocation: EmbeddingInvocation, result: Any
) -> None:
    if getattr(result, "model", None):
        invocation.response_model_name = result.model
    # Embedding dimensions come from the first vector in the response when the
    # caller didn't hint them in the request.
    if invocation.dimension_count is None:
        data = getattr(result, "data", None)
        if data:
            embedding = get_property_value(data[0], "embedding")
            if embedding:
                invocation.dimension_count = len(embedding)
    usage = getattr(result, "usage", None)
    if usage is not None:
        prompt_tokens = get_property_value(usage, "prompt_tokens")
        if prompt_tokens is not None:
            invocation.input_tokens = prompt_tokens
        # Embeddings do not report output tokens.


def is_streaming(kwargs: Mapping[str, Any]) -> bool:
    return bool(kwargs.get("stream"))
