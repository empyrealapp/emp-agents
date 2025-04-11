import json
from enum import StrEnum

from pydantic import BaseModel

from emp_agents.models.provider import ResponseT
from emp_agents.models.shared import AssistantMessage, Message, ToolCall
from emp_agents.types import Role

from .types import AnthropicModelType as ModelType


class ResponseType(StrEnum):
    text = "text"
    tool_use = "tool_use"


class Content(BaseModel):
    id: str | None = None
    type: ResponseType
    text: str | None = None
    input: dict[str, str] | None = None
    name: str | None = None

    @property
    def function(self) -> ToolCall.Function:
        assert self.name is not None
        assert self.input is not None
        return ToolCall.Function(name=self.name, arguments=json.dumps(self.input))

    def to_message(self) -> AssistantMessage:
        return AssistantMessage(content=self.text)


class StopReason(StrEnum):
    tool_use = "tool_use"
    end_turn = "end_turn"
    max_tokens = "max_tokens"
    stop_sequence = "stop_sequence"


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class Response(ResponseT):
    id: str
    content: list[Content]
    model: ModelType | str
    role: Role
    stop_reason: StopReason
    stop_sequence: str | None
    type: str | None
    usage: Usage

    @property
    def text(self):
        return self.content[0].text

    @property
    def tool_calls(self):
        return [content for content in self.content if content.type == "tool_use"]

    @property
    def messages(self) -> list[Message]:
        return [content.to_message() for content in self.content]
