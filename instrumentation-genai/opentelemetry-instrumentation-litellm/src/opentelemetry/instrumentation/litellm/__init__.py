# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""
LiteLLM instrumentation supporting `litellm`_, it can be enabled by using
``LiteLLMInstrumentor``.

.. _litellm: https://pypi.org/project/litellm/

LiteLLM is a unified interface for many LLM providers (OpenAI, Anthropic,
Bedrock, Vertex AI, Gemini, Cohere, Groq, and many more) that exposes an
OpenAI-compatible response shape. This instrumentation records a
``gen_ai.*`` span, prompt/output messages, and duration + token metrics for
each call to ``litellm.completion``, ``litellm.acompletion``,
``litellm.embedding``, and ``litellm.aembedding`` — going through the shared
``opentelemetry-util-genai`` ``TelemetryHandler`` so the emitted telemetry
matches every other GenAI instrumentation in this repo.

``gen_ai.provider.name`` is resolved per call from litellm's
``custom_llm_provider`` (kwarg → response ``_hidden_params`` →
``litellm.get_llm_provider(model)`` fallback), so the same instrumentation
surfaces the correct provider regardless of which backend routed the request.

Usage
-----

.. code:: python

    import litellm
    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

    LiteLLMInstrumentor().instrument()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": "Write a short poem on open telemetry."},
        ],
    )

Configuration
-------------

Message content capture is controlled by the standard GenAI environment
variables read by ``opentelemetry-util-genai``:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` — enable capture of
  prompts, completions, tool arguments, and return values.
- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK`` — override the completion
  hook (e.g. ``upload`` to send content to blob storage).

A custom ``CompletionHook`` implementation can also be passed
programmatically::

    LiteLLMInstrumentor().instrument(completion_hook=my_hook)

When provided, this takes precedence over the hook resolved from
``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK``.

API
---
"""

from typing import Any, Collection

from wrapt import wrap_function_wrapper

from opentelemetry.instrumentation.instrumentor import BaseInstrumentor
from opentelemetry.instrumentation.litellm.package import _instruments
from opentelemetry.instrumentation.utils import unwrap
from opentelemetry.util.genai.completion_hook import load_completion_hook
from opentelemetry.util.genai.handler import TelemetryHandler

from .patch import (
    acompletion_wrapper,
    aembedding_wrapper,
    completion_wrapper,
    embedding_wrapper,
)


class LiteLLMInstrumentor(BaseInstrumentor):
    """An instrumentor for LiteLLM."""

    # pylint: disable=no-self-use
    def instrumentation_dependencies(self) -> Collection[str]:
        return _instruments

    def _instrument(self, **kwargs: Any) -> None:
        """Enable LiteLLM instrumentation."""
        tracer_provider = kwargs.get("tracer_provider")
        meter_provider = kwargs.get("meter_provider")
        logger_provider = kwargs.get("logger_provider")

        handler = TelemetryHandler(
            tracer_provider=tracer_provider,
            meter_provider=meter_provider,
            logger_provider=logger_provider,
            completion_hook=kwargs.get("completion_hook")
            or load_completion_hook(),
        )

        # LiteLLM's public entry points live at the module root. We patch them
        # there so ``litellm.completion(...)`` — and ``from litellm import
        # completion`` when done after ``instrument()`` — dispatch through
        # our wrappers.
        wrap_function_wrapper(
            "litellm", "completion", completion_wrapper(handler)
        )
        wrap_function_wrapper(
            "litellm", "acompletion", acompletion_wrapper(handler)
        )
        wrap_function_wrapper(
            "litellm", "embedding", embedding_wrapper(handler)
        )
        wrap_function_wrapper(
            "litellm", "aembedding", aembedding_wrapper(handler)
        )

    def _uninstrument(self, **kwargs: Any) -> None:
        import litellm  # pylint: disable=import-outside-toplevel  # noqa: PLC0415

        unwrap(litellm, "completion")
        unwrap(litellm, "acompletion")
        unwrap(litellm, "embedding")
        unwrap(litellm, "aembedding")
