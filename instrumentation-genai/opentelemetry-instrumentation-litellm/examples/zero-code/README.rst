OpenTelemetry LiteLLM Zero-Code Instrumentation Example
=======================================================

This example shows how to instrument LiteLLM calls with no code changes,
using ``opentelemetry-instrument`` from
`opentelemetry-distro <https://pypi.org/project/opentelemetry-distro/>`_.

Setup
-----

Set an API key for whichever provider your model targets (for the default
``gpt-4o-mini``, ``OPENAI_API_KEY``). An OTLP-compatible endpoint should be
listening on ``http://localhost:4317``; if not, override
``OTEL_EXPORTER_OTLP_ENDPOINT``.

::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Run
---

::

    opentelemetry-instrument python main.py

Setting ``OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT=true`` in the
environment enables capture of prompt and completion contents in log events.
