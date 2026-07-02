# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

- Initial release. Instruments `litellm.completion`, `litellm.acompletion`,
  `litellm.embedding`, and `litellm.aembedding` on top of
  `opentelemetry-util-genai`'s `TelemetryHandler`, emitting `gen_ai.*` spans,
  input/output message content on the span, and duration + token-usage
  metrics. `gen_ai.provider.name` is resolved per call from litellm's
  `custom_llm_provider` so the same instrumentation works across every
  provider LiteLLM routes to. Streaming uses the shared util-genai
  `SyncStreamWrapper` / `AsyncStreamWrapper` base classes.
