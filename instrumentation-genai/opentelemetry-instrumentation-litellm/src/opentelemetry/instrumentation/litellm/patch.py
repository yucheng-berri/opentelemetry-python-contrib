# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Patch functions for LiteLLM's four module-level entry points.

Each of these returns a ``wrapt`` traced-method closure over the shared
``TelemetryHandler``. All actual telemetry emission (spans, metrics, events)
lives in ``opentelemetry-util-genai``; this layer only patches, extracts
request/response fields, and drives ``invocation.stop()`` / ``.fail()``.
"""

from __future__ import annotations

from typing import Any, Callable

from opentelemetry.util.genai.handler import TelemetryHandler
from opentelemetry.util.genai.invocation import Error

from .stream_wrappers import AsyncChatStreamWrapper, ChatStreamWrapper
from .utils import (
    create_chat_invocation,
    create_embedding_invocation,
    is_streaming,
    resolve_provider,
    set_chat_response_properties,
    set_embedding_response_properties,
    system_value,
)


def _make_error(exc: BaseException) -> Error:
    return Error(type=type(exc), message=str(exc))


def completion_wrapper(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``litellm.completion``."""
    capture_content = handler.should_capture_content()

    def traced_method(wrapped, instance, args, kwargs):
        # Try to resolve the provider now; if we can't, fall through with an
        # empty string and refine from the response's ``_hidden_params`` below.
        provider = system_value(resolve_provider(kwargs)) or ""
        invocation = create_chat_invocation(
            handler, kwargs, capture_content, provider
        )
        try:
            result = wrapped(*args, **kwargs)

            # If we couldn't tell the provider up-front (e.g. bare model
            # string), refine from the response now. The invocation's span
            # was created with an empty provider, but ``gen_ai.provider.name``
            # is only emitted from the base attributes when non-null — refresh
            # the field so downstream users see the resolved value.
            if not provider:
                refined = system_value(resolve_provider(kwargs, result))
                if refined:
                    invocation.provider = refined

            if is_streaming(kwargs):
                return ChatStreamWrapper(result, invocation, capture_content)

            set_chat_response_properties(invocation, result, capture_content)
            invocation.stop()
            return result
        except Exception as exc:  # noqa: BLE001 — re-raise below
            invocation.fail(_make_error(exc))
            raise

    return traced_method


def acompletion_wrapper(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``litellm.acompletion``."""
    capture_content = handler.should_capture_content()

    async def traced_method(wrapped, instance, args, kwargs):
        # Try to resolve the provider now; if we can't, fall through with an
        # empty string and refine from the response's ``_hidden_params`` below.
        provider = system_value(resolve_provider(kwargs)) or ""
        invocation = create_chat_invocation(
            handler, kwargs, capture_content, provider
        )
        try:
            result = await wrapped(*args, **kwargs)

            if not provider:
                refined = system_value(resolve_provider(kwargs, result))
                if refined:
                    invocation.provider = refined

            if is_streaming(kwargs):
                return AsyncChatStreamWrapper(
                    result, invocation, capture_content
                )

            set_chat_response_properties(invocation, result, capture_content)
            invocation.stop()
            return result
        except Exception as exc:
            invocation.fail(_make_error(exc))
            raise

    return traced_method


def embedding_wrapper(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``litellm.embedding``."""

    def traced_method(wrapped, instance, args, kwargs):
        provider = system_value(resolve_provider(kwargs)) or ""
        invocation = create_embedding_invocation(handler, kwargs, provider)
        try:
            result = wrapped(*args, **kwargs)
            if not provider:
                refined = system_value(resolve_provider(kwargs, result))
                if refined:
                    invocation.provider = refined
            set_embedding_response_properties(invocation, result)
            invocation.stop()
            return result
        except Exception as exc:
            invocation.fail(_make_error(exc))
            raise

    return traced_method


def aembedding_wrapper(handler: TelemetryHandler) -> Callable[..., Any]:
    """Wrap ``litellm.aembedding``."""

    async def traced_method(wrapped, instance, args, kwargs):
        provider = system_value(resolve_provider(kwargs)) or ""
        invocation = create_embedding_invocation(handler, kwargs, provider)
        try:
            result = await wrapped(*args, **kwargs)
            if not provider:
                refined = system_value(resolve_provider(kwargs, result))
                if refined:
                    invocation.provider = refined
            set_embedding_response_properties(invocation, result)
            invocation.stop()
            return result
        except Exception as exc:
            invocation.fail(_make_error(exc))
            raise

    return traced_method
