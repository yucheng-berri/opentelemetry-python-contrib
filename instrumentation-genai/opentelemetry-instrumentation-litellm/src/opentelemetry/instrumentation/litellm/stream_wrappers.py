# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming wrappers built on top of ``opentelemetry-util-genai``'s shared
``SyncStreamWrapper`` / ``AsyncStreamWrapper`` bases.

These provide only litellm-specific chunk parsing and telemetry finalization;
the base classes own iteration, close, context-manager, and lifecycle
sequencing. See ``instrumentation-genai/CLAUDE.md`` for the boundary rules.
"""

from __future__ import annotations

from typing import Any, List, Optional

from opentelemetry.util.genai.invocation import InferenceInvocation
from opentelemetry.util.genai.stream import (
    AsyncStreamWrapper,
    SyncStreamWrapper,
)
from opentelemetry.util.genai.types import (
    OutputMessage,
    Text,
    ToolCallRequest,
)

from .utils import get_property_value


class _ChoiceAccumulator:
    """Reassembles a streamed choice from litellm's OpenAI-shaped deltas."""

    def __init__(self, index: int) -> None:
        self.index = index
        self.finish_reason: Optional[str] = None
        self.role: Optional[str] = None
        self.content_parts: List[str] = []
        # index -> {"id", "type", "name", "arguments" (accumulated str)}
        self.tool_calls: dict[int, dict[str, Any]] = {}

    def add_chunk(self, choice: Any) -> None:
        finish_reason = get_property_value(choice, "finish_reason")
        if finish_reason:
            self.finish_reason = finish_reason

        delta = get_property_value(choice, "delta")
        if delta is None:
            return

        role = get_property_value(delta, "role")
        if role and not self.role:
            self.role = role

        content = get_property_value(delta, "content")
        if content:
            self.content_parts.append(content)

        for tc in get_property_value(delta, "tool_calls") or []:
            idx = get_property_value(tc, "index") or 0
            slot = self.tool_calls.setdefault(
                idx,
                {"id": None, "type": None, "name": None, "arguments": ""},
            )
            call_id = get_property_value(tc, "id")
            if call_id:
                slot["id"] = call_id
            call_type = get_property_value(tc, "type")
            if call_type:
                slot["type"] = call_type
            func = get_property_value(tc, "function")
            if func:
                name = get_property_value(func, "name")
                if name:
                    slot["name"] = name
                arguments = get_property_value(func, "arguments")
                if arguments:
                    slot["arguments"] += arguments


class _ChatStreamMixin:
    """Chat-specific hooks shared by the sync + async stream wrappers."""

    # Slots — using ``_self_*`` because util-genai's stream bases derive from
    # wrapt.ObjectProxy, which reserves the ``_self_*`` namespace for state
    # that must NOT be forwarded to the wrapped stream.
    _self_invocation: InferenceInvocation
    _self_capture_content: bool
    _self_choices: dict[int, _ChoiceAccumulator]
    _self_response_id: Optional[str]
    _self_response_model: Optional[str]
    _self_prompt_tokens: Optional[int]
    _self_completion_tokens: Optional[int]

    def _process_chunk(self, chunk: Any) -> None:
        chunk_id = get_property_value(chunk, "id")
        if chunk_id and not self._self_response_id:
            self._self_response_id = chunk_id
        chunk_model = get_property_value(chunk, "model")
        if chunk_model and not self._self_response_model:
            self._self_response_model = chunk_model
        usage = get_property_value(chunk, "usage")
        # ``stream_options={"include_usage": True}`` produces a terminal chunk
        # carrying the final token counts.
        if usage:
            prompt_tokens = get_property_value(usage, "prompt_tokens")
            if prompt_tokens is not None:
                self._self_prompt_tokens = prompt_tokens
            completion_tokens = get_property_value(usage, "completion_tokens")
            if completion_tokens is not None:
                self._self_completion_tokens = completion_tokens
        for choice in get_property_value(chunk, "choices") or []:
            idx = get_property_value(choice, "index") or 0
            slot = self._self_choices.setdefault(idx, _ChoiceAccumulator(idx))
            slot.add_chunk(choice)

    def _on_stream_end(self) -> None:
        self._finalize_invocation()
        self._self_invocation.stop()

    def _on_stream_error(self, error: BaseException) -> None:
        # Populate what we managed to gather before failing so downstream
        # consumers still see partial finish_reasons / usage.
        self._finalize_invocation()
        self._self_invocation.fail(error)

    def _finalize_invocation(self) -> None:
        invocation = self._self_invocation
        if self._self_response_model:
            invocation.response_model_name = self._self_response_model
        if self._self_response_id:
            invocation.response_id = self._self_response_id
        if self._self_prompt_tokens is not None:
            invocation.input_tokens = self._self_prompt_tokens
        if self._self_completion_tokens is not None:
            invocation.output_tokens = self._self_completion_tokens

        if self._self_choices:
            ordered = [
                self._self_choices[i] for i in sorted(self._self_choices)
            ]
            finish_reasons = [
                c.finish_reason for c in ordered if c.finish_reason
            ]
            if finish_reasons:
                invocation.finish_reasons = finish_reasons
            if self._self_capture_content:
                invocation.output_messages = _build_output_messages(ordered)


def _build_output_messages(
    choices: List[_ChoiceAccumulator],
) -> List[OutputMessage]:
    messages: List[OutputMessage] = []
    for choice in choices:
        parts: list[Any] = []
        if choice.content_parts:
            parts.append(Text(content="".join(choice.content_parts)))
        for idx in sorted(choice.tool_calls):
            slot = choice.tool_calls[idx]
            parts.append(
                ToolCallRequest(
                    name=slot["name"],
                    id=slot["id"],
                    arguments=slot["arguments"] or None,
                )
            )
        messages.append(
            OutputMessage(
                role=choice.role or "assistant",
                parts=parts,
                finish_reason=choice.finish_reason or "error",
            )
        )
    return messages


class ChatStreamWrapper(_ChatStreamMixin, SyncStreamWrapper):
    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_choices = {}
        self._self_response_id = None
        self._self_response_model = None
        self._self_prompt_tokens = None
        self._self_completion_tokens = None


class AsyncChatStreamWrapper(_ChatStreamMixin, AsyncStreamWrapper):
    def __init__(
        self,
        stream: Any,
        invocation: InferenceInvocation,
        capture_content: bool,
    ) -> None:
        super().__init__(stream)
        self._self_invocation = invocation
        self._self_capture_content = capture_content
        self._self_choices = {}
        self._self_response_id = None
        self._self_response_model = None
        self._self_prompt_tokens = None
        self._self_completion_tokens = None


__all__ = ["AsyncChatStreamWrapper", "ChatStreamWrapper"]
