"""Unit tests configuration module."""

# pylint: disable=redefined-outer-name

from __future__ import annotations

import os
import sys
import types

import pytest

from opentelemetry.instrumentation._semconv import (  # pylint: disable=no-name-in-module
    OTEL_SEMCONV_STABILITY_OPT_IN,
    _OpenTelemetrySemanticConventionStability,
)
from opentelemetry.instrumentation.litellm import LiteLLMInstrumentor
from opentelemetry.sdk._logs import LoggerProvider

try:
    from opentelemetry.sdk._logs.export import (  # pylint: disable=no-name-in-module
        InMemoryLogRecordExporter,
        SimpleLogRecordProcessor,
    )
except ImportError:
    from opentelemetry.sdk._logs.export import (
        InMemoryLogExporter as InMemoryLogRecordExporter,
    )
    from opentelemetry.sdk._logs.export import SimpleLogRecordProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.util.genai.environment_variables import (
    OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT,
)


@pytest.fixture(scope="function", name="span_exporter")
def fixture_span_exporter():
    yield InMemorySpanExporter()


@pytest.fixture(scope="function", name="log_exporter")
def fixture_log_exporter():
    yield InMemoryLogRecordExporter()


@pytest.fixture(scope="function", name="metric_reader")
def fixture_metric_reader():
    yield InMemoryMetricReader()


@pytest.fixture(scope="function", name="tracer_provider")
def fixture_tracer_provider(span_exporter):
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    return provider


@pytest.fixture(scope="function", name="logger_provider")
def fixture_logger_provider(log_exporter):
    provider = LoggerProvider()
    provider.add_log_record_processor(SimpleLogRecordProcessor(log_exporter))
    return provider


@pytest.fixture(scope="function", name="meter_provider")
def fixture_meter_provider(metric_reader):
    return MeterProvider(metric_readers=[metric_reader])


@pytest.fixture
def fake_litellm():
    """Register a bare stub ``litellm`` module.

    The instrumentor wraps four functions on this module during
    ``instrument()``, so it must exist BEFORE the instrumentor fixture runs.
    Individual tests then reassign the wrapped functions (via
    ``fakes._configure_fake_litellm``) to control per-call behavior.
    """
    module = types.ModuleType("litellm")

    def _noop(*args, **kwargs):
        raise RuntimeError("fake litellm function not configured by test")

    async def _anoop(*args, **kwargs):
        raise RuntimeError("fake litellm function not configured by test")

    def _get_llm_provider(model=None, **_):
        return ("model", "openai", None, None)

    module.completion = _noop
    module.acompletion = _anoop
    module.embedding = _noop
    module.aembedding = _anoop
    module.get_llm_provider = _get_llm_provider
    sys.modules["litellm"] = module
    yield module
    sys.modules.pop("litellm", None)


def _instrument(
    tracer_provider,
    logger_provider,
    meter_provider,
    *,
    capture_content: bool,
):
    # Force re-initialization so the OTEL_SEMCONV_STABILITY_OPT_IN we just set
    # is picked up. Without this the module-level singleton would remember the
    # first value it ever saw.
    _OpenTelemetrySemanticConventionStability._initialized = False

    os.environ[OTEL_SEMCONV_STABILITY_OPT_IN] = "gen_ai_latest_experimental"
    if capture_content:
        # Under experimental semconv the value must be one of the enum names,
        # not "true" — util-genai warns and falls back to NO_CONTENT otherwise.
        os.environ[OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT] = (
            "SPAN_ONLY"
        )
    else:
        os.environ.pop(
            OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None
        )

    instrumentor = LiteLLMInstrumentor()
    instrumentor.instrument(
        tracer_provider=tracer_provider,
        logger_provider=logger_provider,
        meter_provider=meter_provider,
        skip_dep_check=True,
    )
    return instrumentor


@pytest.fixture
def instrument_no_content(
    fake_litellm, tracer_provider, logger_provider, meter_provider
):
    instrumentor = _instrument(
        tracer_provider,
        logger_provider,
        meter_provider,
        capture_content=False,
    )
    yield instrumentor
    os.environ.pop(OTEL_SEMCONV_STABILITY_OPT_IN, None)
    instrumentor.uninstrument()


@pytest.fixture
def instrument_with_content(
    fake_litellm, tracer_provider, logger_provider, meter_provider
):
    instrumentor = _instrument(
        tracer_provider,
        logger_provider,
        meter_provider,
        capture_content=True,
    )
    yield instrumentor
    os.environ.pop(OTEL_INSTRUMENTATION_GENAI_CAPTURE_MESSAGE_CONTENT, None)
    os.environ.pop(OTEL_SEMCONV_STABILITY_OPT_IN, None)
    instrumentor.uninstrument()
