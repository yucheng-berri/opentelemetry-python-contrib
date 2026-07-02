OpenTelemetry LiteLLM Instrumentation
======================================

|pypi|

.. |pypi| image:: https://badge.fury.io/py/opentelemetry-instrumentation-litellm.svg
   :target: https://pypi.org/project/opentelemetry-instrumentation-litellm/

This library allows tracing LLM requests and logging of messages made through
`LiteLLM <https://pypi.org/project/litellm/>`_. It captures the duration of
each operation and the number of tokens used as metrics.

LiteLLM is a unified interface for 100+ LLM providers. Because a single call
can route to any backend (OpenAI, Anthropic, Bedrock, Vertex AI, Gemini,
Cohere, Groq, Mistral, DeepSeek, xAI, Perplexity, and many more), this
instrumentation resolves ``gen_ai.system`` per-call from litellm's
``custom_llm_provider`` so the correct provider shows up on every span
regardless of the model you called.

Installation
------------

If your application is already instrumented with OpenTelemetry, add this
package to your requirements.
::

    pip install opentelemetry-instrumentation-litellm

If you don't have a LiteLLM application yet, try our `examples <examples>`_
which only need a valid API key for whichever provider you point at.

Usage
-----

This section describes how to set up LiteLLM instrumentation if you're
setting OpenTelemetry up manually. Check out the `manual example <examples/manual>`_
for a runnable version.

Instrumenting all calls
***********************

When using ``LiteLLMInstrumentor``, all calls made via ``litellm.completion``,
``litellm.acompletion``, ``litellm.embedding``, and ``litellm.aembedding`` will
be instrumented automatically.

.. code:: python

    import litellm
    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

    LiteLLMInstrumentor().instrument()

    response = litellm.completion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Write a short poem on open telemetry."}],
    )

Enabling message content
************************

Message content — such as the contents of the prompt and completion — is not
captured by default. This instrumentation goes through the shared
``opentelemetry-util-genai`` layer, so content capture uses the standard
GenAI variables:

- ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT`` — set to
  ``SPAN_ONLY``, ``EVENT_ONLY``, or ``SPAN_AND_EVENT`` to enable capture.
- ``OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`` — required to
  access the newer content-capture modes.
- ``OTEL_INSTRUMENTATION_GENAI_COMPLETION_HOOK=upload`` together with
  ``OTEL_INSTRUMENTATION_GENAI_UPLOAD_BASE_PATH=<fsspec-uri>`` — upload
  prompts and completions to an ``fsspec``-compatible destination and
  record reference URIs as ``gen_ai.input.messages.ref`` /
  ``gen_ai.output.messages.ref`` attributes.

See the `opentelemetry-util-genai README
<https://github.com/open-telemetry/opentelemetry-python-contrib/blob/main/util/opentelemetry-util-genai/README.rst>`_
for the full list of GenAI configuration variables.

Uninstrument
************

To uninstrument LiteLLM, call the ``uninstrument`` method:

.. code:: python

    from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor

    LiteLLMInstrumentor().uninstrument()

References
----------

* `OpenTelemetry LiteLLM Instrumentation <https://opentelemetry-python-contrib.readthedocs.io/en/latest/instrumentation-genai/litellm/litellm.html>`_
* `OpenTelemetry Project <https://opentelemetry.io/>`_
* `OpenTelemetry Python Examples <https://github.com/open-telemetry/opentelemetry-python/tree/main/docs/examples>`_
