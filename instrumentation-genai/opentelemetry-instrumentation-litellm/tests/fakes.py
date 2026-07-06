# Copyright The OpenTelemetry Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Minimal fakes standing in for the shape of ``litellm``'s public surface.

The instrumentation wraps the four module-level functions ``completion``,
``acompletion``, ``embedding``, ``aembedding`` and reads attributes on their
return values (``choices[].message.{role,content,tool_calls}``, ``usage``,
``model``, ``id``, ``_hidden_params.custom_llm_provider``). Building a fake
that matches this shape keeps the unit tests hermetic — no HTTP, no provider
credentials, no version-dependent litellm behavior in the test surface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None


@dataclass
class Choice:
    index: int = 0
    finish_reason: Optional[str] = "stop"
    message: Optional[Message] = None


@dataclass
class HiddenParams:
    custom_llm_provider: Optional[str] = None


@dataclass
class ModelResponse:
    id: str = "resp-123"
    model: str = "gpt-4o-mini"
    choices: List[Choice] = field(default_factory=list)
    usage: Optional[Usage] = None
    _hidden_params: HiddenParams = field(default_factory=HiddenParams)


@dataclass
class Embedding:
    embedding: List[float] = field(default_factory=list)
    index: int = 0


@dataclass
class EmbeddingResponse:
    model: str = "text-embedding-3-small"
    data: List[Embedding] = field(default_factory=list)
    usage: Optional[Usage] = None
    _hidden_params: HiddenParams = field(default_factory=HiddenParams)


def make_chat_response(
    *,
    content: str = "hello",
    model: str = "gpt-4o-mini",
    provider: str = "openai",
    prompt_tokens: int = 5,
    completion_tokens: int = 7,
    finish_reason: str = "stop",
) -> ModelResponse:
    return ModelResponse(
        id="chatcmpl-fake",
        model=model,
        choices=[
            Choice(
                index=0,
                finish_reason=finish_reason,
                message=Message(role="assistant", content=content),
            )
        ],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
        _hidden_params=HiddenParams(custom_llm_provider=provider),
    )


def make_embedding_response(
    *,
    model: str = "text-embedding-3-small",
    provider: str = "openai",
    dimensions: int = 3,
    prompt_tokens: int = 5,
) -> EmbeddingResponse:
    return EmbeddingResponse(
        model=model,
        data=[Embedding(embedding=[0.1] * dimensions, index=0)],
        usage=Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens),
        _hidden_params=HiddenParams(custom_llm_provider=provider),
    )


class StreamIter:
    """Fake sync CustomStreamWrapper — yields the chunks it was given, then StopIteration."""

    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)
        self._pos = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self._pos >= len(self._chunks):
            raise StopIteration
        chunk = self._chunks[self._pos]
        self._pos += 1
        return chunk

    def close(self):
        pass


class AsyncStreamIter:
    def __init__(self, chunks: Iterable[Any]) -> None:
        self._chunks = list(chunks)
        self._pos = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._pos >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._pos]
        self._pos += 1
        return chunk

    async def aclose(self):
        pass


@dataclass
class Delta:
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: Optional[List[Any]] = None


@dataclass
class StreamChoice:
    index: int = 0
    finish_reason: Optional[str] = None
    delta: Optional[Delta] = None


@dataclass
class StreamChunk:
    id: str = "chatcmpl-fake"
    model: str = "gpt-4o-mini"
    choices: List[StreamChoice] = field(default_factory=list)
    usage: Optional[Usage] = None


def make_stream_chunks(
    *, text: str = "hello world", model: str = "gpt-4o-mini"
) -> List[StreamChunk]:
    words = text.split(" ")
    chunks: List[StreamChunk] = []
    # role-only first chunk
    chunks.append(
        StreamChunk(
            model=model,
            choices=[
                StreamChoice(
                    index=0, delta=Delta(role="assistant", content="")
                )
            ],
        )
    )
    for idx, word in enumerate(words):
        piece = word if idx == 0 else " " + word
        chunks.append(
            StreamChunk(
                model=model,
                choices=[StreamChoice(index=0, delta=Delta(content=piece))],
            )
        )
    # terminal chunk with finish_reason + usage (mirrors stream_options include_usage)
    chunks.append(
        StreamChunk(
            model=model,
            choices=[
                StreamChoice(index=0, finish_reason="stop", delta=Delta())
            ],
            usage=Usage(
                prompt_tokens=5,
                completion_tokens=len(words),
                total_tokens=5 + len(words),
            ),
        )
    )
    return chunks
