from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from emp_agents.models.shared import Message
from emp_agents.models.shared.tools import GenericTool

from .types import GrokModelType


class Request(BaseModel):
    """
    Request model for Grok API, which follows the OpenAI API format.
    """

    model_config = ConfigDict(populate_by_name=True)

    model: GrokModelType
    max_tokens: Optional[int] = Field(default=None)
    temperature: Optional[float] = Field(default=None, ge=0, le=2.0)
    tool_choice: Literal["none", "required", "auto", None] = Field(default=None)
    tools: Optional[list[GenericTool]] = Field(default=None)

    system: str | None = None
    messages: list[Message] | None = None

    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)
    num_responses: Optional[int] = Field(default=None, serialization_alias="n")
    top_p: Optional[int] = Field(default=None)

    def model_dump(self, *, exclude_none=True, by_alias=True, **kwargs):
        return super().model_dump(
            exclude_none=exclude_none, by_alias=by_alias, **kwargs
        )
