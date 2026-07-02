# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for LiteLLM instrumentation using a fake litellm module.

Rather than install and mock the real litellm package (which pulls a large
transitive dep tree and needs network keys), we install a stub ``litellm``
module into ``sys.modules`` that exposes the four entry points the
instrumentor wraps and returns fakes shaped like the real response types.
This isolates the tests to the wrapping and attribute-extraction logic —
which is what this instrumentation actually owns; the shared
``opentelemetry-util-genai`` layer covers span, metric, and event emission
in its own test suite.
"""

# Every test imports ``litellm`` AFTER the fake_litellm fixture registers the
# stub module — a top-level import here would resolve to the real (or missing)
# package instead.
# pylint: disable=import-outside-toplevel

from __future__ import annotations

import asyncio
import json
import sys

import pytest
from wrapt import FunctionWrapper

from opentelemetry.semconv._incubating.attributes import (
    gen_ai_attributes as GenAIAttributes,
)

from .fakes import (
    StreamIter,
    make_chat_response,
    make_embedding_response,
    make_stream_chunks,
)


def _configure_fake_litellm(
    *,
    completion_return=None,
    acompletion_return=None,
    embedding_return=None,
    aembedding_return=None,
    completion_raises: Exception | None = None,
    get_llm_provider_return=("model", "openai", None, None),
):
    """Rewrite the four wrapped functions on the stub ``litellm`` module."""
    module = sys.modules["litellm"]

    def completion(*args, **kwargs):
        if completion_raises:
            raise completion_raises
        return completion_return

    async def acompletion(*args, **kwargs):
        return acompletion_return

    def embedding(*args, **kwargs):
        return embedding_return

    async def aembedding(*args, **kwargs):
        return aembedding_return

    def get_llm_provider(model=None, **_):
        return get_llm_provider_return

    # wrapt's FunctionWrapper is transparent — reassigning __wrapped__ swaps
    # the underlying callable without disturbing the wrapper itself.
    for name, func in (
        ("completion", completion),
        ("acompletion", acompletion),
        ("embedding", embedding),
        ("aembedding", aembedding),
    ):
        current = getattr(module, name)
        if isinstance(current, FunctionWrapper):
            current.__wrapped__ = func
        else:
            setattr(module, name, func)
    module.get_llm_provider = get_llm_provider
    return module


def _get_finished_spans(span_exporter):
    return span_exporter.get_finished_spans()


def _messages_from_span(span, key: str):
    """Parse the JSON-encoded ``gen_ai.input.messages`` / ``output.messages``."""
    raw = span.attributes.get(key)
    if raw is None:
        return None
    return json.loads(raw)


def test_completion_records_span_and_metrics(
    span_exporter, metric_reader, instrument_no_content
):
    _configure_fake_litellm(
        completion_return=make_chat_response(
            content="hi there",
            provider="openai",
            prompt_tokens=3,
            completion_tokens=4,
        )
    )
    import litellm  # noqa: PLC0415

    result = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hey"}],
        temperature=0.5,
        max_tokens=42,
    )
    assert result.choices[0].message.content == "hi there"

    spans = _get_finished_spans(span_exporter)
    assert len(spans) == 1
    span = spans[0]
    assert span.name == "chat gpt-4o-mini"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.CHAT.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_REQUEST_MODEL] == "gpt-4o-mini"
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_TEMPERATURE] == 0.5
    assert span.attributes[GenAIAttributes.GEN_AI_REQUEST_MAX_TOKENS] == 42
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 3
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 4
    assert (
        span.attributes[GenAIAttributes.GEN_AI_RESPONSE_ID] == "chatcmpl-fake"
    )
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )

    # Metrics — util-genai records duration + token histograms per invocation.
    metric_data = metric_reader.get_metrics_data()
    all_metric_names = {
        m.name
        for rm in metric_data.resource_metrics
        for sm in rm.scope_metrics
        for m in sm.metrics
    }
    assert "gen_ai.client.operation.duration" in all_metric_names
    assert "gen_ai.client.token.usage" in all_metric_names


def test_completion_provider_from_response_hidden_params(
    span_exporter, instrument_no_content
):
    """Provider isn't hinted up-front — we pick it up from ``_hidden_params``."""
    _configure_fake_litellm(
        completion_return=make_chat_response(
            content="hello", provider="anthropic"
        ),
        # get_llm_provider returns unknown so we exercise the response fallback
        get_llm_provider_return=("model", None, None, None),
    )
    import litellm  # noqa: PLC0415

    litellm.completion(
        model="claude-3-5-sonnet",
        messages=[{"role": "user", "content": "hey"}],
    )

    span = _get_finished_spans(span_exporter)[0]
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.ANTHROPIC.value
    )


def test_completion_unknown_provider_passes_through(
    span_exporter, instrument_no_content
):
    """Providers not in the semconv enum pass through as their litellm name."""
    _configure_fake_litellm(
        completion_return=make_chat_response(
            content="hi", provider="together_ai"
        )
    )
    import litellm  # noqa: PLC0415

    litellm.completion(
        model="together_ai/mistralai/Mistral-7B",
        messages=[{"role": "user", "content": "hey"}],
        custom_llm_provider="together_ai",
    )

    span = _get_finished_spans(span_exporter)[0]
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME] == "together_ai"
    )


def test_completion_message_content_capture_gated(
    span_exporter, instrument_with_content
):
    _configure_fake_litellm(
        completion_return=make_chat_response(
            content="the answer", provider="openai"
        )
    )
    import litellm  # noqa: PLC0415

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "the question"}],
    )

    span = _get_finished_spans(span_exporter)[0]
    inputs = _messages_from_span(span, "gen_ai.input.messages")
    outputs = _messages_from_span(span, "gen_ai.output.messages")
    assert inputs is not None
    assert any(
        part.get("content") == "the question"
        for msg in inputs
        for part in msg.get("parts", [])
    )
    assert outputs is not None
    assert any(
        part.get("content") == "the answer"
        for msg in outputs
        for part in msg.get("parts", [])
    )


def test_completion_message_content_omitted_by_default(
    span_exporter, instrument_no_content
):
    _configure_fake_litellm(
        completion_return=make_chat_response(
            content="the answer", provider="openai"
        )
    )
    import litellm  # noqa: PLC0415

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "the question"}],
    )

    span = _get_finished_spans(span_exporter)[0]
    assert "gen_ai.input.messages" not in span.attributes
    assert "gen_ai.output.messages" not in span.attributes


def test_completion_error_sets_status_and_records_error_type(
    span_exporter, instrument_no_content
):
    _configure_fake_litellm(completion_raises=RuntimeError("boom"))
    import litellm  # noqa: PLC0415

    with pytest.raises(RuntimeError):
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hey"}],
        )

    span = _get_finished_spans(span_exporter)[0]
    assert span.status.status_code.name == "ERROR"
    assert span.attributes["error.type"] == "RuntimeError"


def test_acompletion_records_span(span_exporter, instrument_no_content):
    _configure_fake_litellm(
        acompletion_return=make_chat_response(content="ok", provider="openai")
    )
    import litellm  # noqa: PLC0415

    asyncio.run(
        litellm.acompletion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "hey"}],
        )
    )

    span = _get_finished_spans(span_exporter)[0]
    assert span.name == "chat gpt-4o-mini"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_PROVIDER_NAME]
        == GenAIAttributes.GenAiProviderNameValues.OPENAI.value
    )


def test_embedding_records_span_and_dimensions(
    span_exporter, instrument_no_content
):
    _configure_fake_litellm(
        embedding_return=make_embedding_response(
            model="text-embedding-3-small", provider="openai", dimensions=4
        )
    )
    import litellm  # noqa: PLC0415

    litellm.embedding(model="text-embedding-3-small", input="hello")

    span = _get_finished_spans(span_exporter)[0]
    assert span.name == "embeddings text-embedding-3-small"
    assert (
        span.attributes[GenAIAttributes.GEN_AI_OPERATION_NAME]
        == GenAIAttributes.GenAiOperationNameValues.EMBEDDINGS.value
    )
    assert (
        span.attributes[GenAIAttributes.GEN_AI_EMBEDDINGS_DIMENSION_COUNT] == 4
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    # Embeddings should NOT set output-token attribute.
    assert GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS not in span.attributes


def test_aembedding_records_span(span_exporter, instrument_no_content):
    _configure_fake_litellm(
        aembedding_return=make_embedding_response(provider="openai")
    )
    import litellm  # noqa: PLC0415

    asyncio.run(
        litellm.aembedding(model="text-embedding-3-small", input="hello")
    )

    span = _get_finished_spans(span_exporter)[0]
    assert span.name == "embeddings text-embedding-3-small"


def test_completion_streaming_reassembles_response(
    span_exporter, instrument_with_content
):
    chunks = make_stream_chunks(text="hello world friend")
    _configure_fake_litellm(completion_return=StreamIter(chunks))
    import litellm  # noqa: PLC0415

    stream = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
    )
    received = list(stream)
    assert len(received) == len(chunks)

    span = _get_finished_spans(span_exporter)[0]
    assert span.attributes[GenAIAttributes.GEN_AI_RESPONSE_FINISH_REASONS] == (
        "stop",
    )
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_INPUT_TOKENS] == 5
    assert span.attributes[GenAIAttributes.GEN_AI_USAGE_OUTPUT_TOKENS] == 3

    outputs = _messages_from_span(span, "gen_ai.output.messages")
    assert outputs is not None
    # Reassembled content should span all chunks concatenated.
    joined = "".join(
        part.get("content", "")
        for msg in outputs
        for part in msg.get("parts", [])
        if part.get("type") == "text"
    )
    assert joined == "hello world friend"


def test_server_address_from_api_base(span_exporter, instrument_no_content):
    _configure_fake_litellm(
        completion_return=make_chat_response(provider="openai")
    )
    import litellm  # noqa: PLC0415

    litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hey"}],
        api_base="http://localhost:4000/v1",
    )

    span = _get_finished_spans(span_exporter)[0]
    assert span.attributes["server.address"] == "localhost"
    assert span.attributes["server.port"] == 4000


def test_uninstrument_restores_original(instrument_no_content):
    _configure_fake_litellm(
        completion_return=make_chat_response(provider="openai")
    )
    import litellm  # noqa: PLC0415

    assert isinstance(litellm.completion, FunctionWrapper)
    instrument_no_content.uninstrument()
    assert not isinstance(litellm.completion, FunctionWrapper)
