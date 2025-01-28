from typing import Annotated

from anthropic.types.message_create_params import ToolChoice, ToolChoiceToolChoiceAuto
from pydantic import BaseModel, ConfigDict, Field, PlainSerializer

from emp_agents.models.anthropic.tool import Tool
from emp_agents.models.shared import Message

from .types import AnthropicModelType


class Request(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: AnthropicModelType = Field(default=AnthropicModelType.claude_3_5_sonnet)
    max_tokens: int | None = Field(default=None)
    temperature: float | None = Field(default=None)
    tool_choice: ToolChoice | None = Field(
        default_factory=lambda: ToolChoiceToolChoiceAuto({"type": "auto"})
    )
    tools: list[Tool] = Field(default_factory=list)

    system: str
    messages: Annotated[
        list[Message],
        PlainSerializer(lambda x: [row.serialize_anthropic() for row in x]),
    ]
