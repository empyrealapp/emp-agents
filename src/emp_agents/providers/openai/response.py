from datetime import datetime
from typing import Optional

from pydantic import BaseModel

from emp_agents.models import AssistantMessage, Message, ResponseT, ToolCall
from .types import OpenAIModelType

from .types import FinishReason


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    index: int
    message: AssistantMessage
    logprobs: Optional[str]
    finish_reason: FinishReason

    @property
    def content(self):
        return self.message.content


class Response(ResponseT):
    id: str
    object: str
    created: datetime
    model: OpenAIModelType | str
    choices: list[Choice]
    usage: Usage
    system_fingerprint: Optional[str]

    @property
    def text(self) -> str:
        return self.choices[0].content

    @property
    def messages(self) -> list[Message]:
        return [self.choices[0].message]

    @property
    def tool_calls(self) -> list[ToolCall] | None:
        return self.choices[0].message.tool_calls

    def __repr__(self):
        return f'<Response id="{self.id}">'

    def print(self):
        for choice in self.choices:
            print(choice.content)
            print("-" * 15)

    __str__ = __repr__
