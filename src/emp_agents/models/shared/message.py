import json
from abc import ABC
from datetime import datetime
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, Field, field_serializer
from pydantic.types import Json

from emp_agents.types.enums import Role


class ToolCall(BaseModel):
    class Function(BaseModel):
        name: str
        arguments: Json[Any]

        @field_serializer("arguments")
        def serialize_dt(self, dt: datetime, _info):
            return json.dumps(self.arguments)

    id: str
    type: str
    function: Function

    def __repr__(self):
        return f"<ToolCall function={self.function}>"


class Message(BaseModel, ABC):
    role: Role
    content: str | None

    @classmethod
    def build(
        self,
        content: str,
        role: Role = Role.user,
        tool_call_id: str | None = None,
        tool_calls: list[Any] | None = None,
    ):
        match role:
            case Role.user:
                return UserMessage(content=content)
            case Role.assistant:
                return AssistantMessage(
                    content=content,
                    tool_calls=tool_calls,
                )
            case Role.tool:
                return ToolMessage(
                    content=content,
                    tool_call_id=tool_call_id,
                )
            case Role.system:
                return SystemMessage(content=content)

    def serialize_anthropic(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role}
        data["content"] = [{"type": "text", "text": self.content}]
        return data

    def __repr__(self):
        if self.content:
            return f"{self.role.value}: {self.content}"
        return f"{self.role.value}: {self.tool_calls}"

    __str__ = __repr__


class SystemMessage(Message):
    role: Literal[Role.system] = Role.system


class UserMessage(Message):
    role: Literal[Role.user] = Role.user


class AssistantMessage(Message):
    role: Literal[Role.assistant] = Role.assistant
    refusal: str | None = Field(default=None)
    tool_calls: list[ToolCall] | None = Field(default=None)


class ToolMessage(Message):
    role: Literal[Role.tool] = Role.tool
    tool_call_id: str | None = Field(
        default=None, validation_alias=AliasChoices("tool_call_id")
    )
