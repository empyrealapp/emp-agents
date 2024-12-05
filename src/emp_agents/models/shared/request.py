from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from emp_agents.models.shared.message import Message, SystemMessage
from emp_agents.models.shared.tools import GenericTool
from emp_agents.types import ModelType, Role


class Request(BaseModel):
    """
    https://platform.openai.com/docs/api-reference/chat/create
    """

    model_config = ConfigDict(populate_by_name=True)

    model: ModelType
    max_tokens: Optional[int] = Field(default=1_000, lt=16_384, gt=0)
    temperature: Optional[float] = Field(default=None, ge=0, le=2.0)
    tool_choice: Literal["none", "required", "auto", None] = Field(default=None)
    tools: Optional[list[GenericTool]] = None
    response_format: type[BaseModel] | None = None

    system: str | None = None  # anthropic field
    messages: list[Message]

    frequency_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)  # openai
    presence_penalty: Optional[float] = Field(default=None, ge=-2.0, le=2.0)  # openai
    num_responses: Optional[int] = Field(
        default=None, serialization_alias="n"
    )  # openai
    top_p: Optional[int] = Field(default=None)  # openai

    def model_dump(self, *, exclude_none=True, by_alias=True, **kwargs):
        return super().model_dump(
            exclude_none=exclude_none, by_alias=by_alias, **kwargs
        )

    def to_anthropic(self):
        exclude = ["frequency_penalty", "presence_penalty", "num_responses", "n"]
        result = self.model_dump(exclude_none=True)
        result["tools"] = [t.to_anthropic() for t in self.tools] if self.tools else []
        if "tool_choice" in result:
            result["tool_choice"] = {"type": result["tool_choice"]}
        for field in exclude:
            if field in result:
                del result[field]
        messages = [message for message in self.messages if message.role != Role.system]
        system_messages = [
            message for message in self.messages if message.role == Role.system
        ]
        result["system"] = result.get("system", "") + str(
            "\n".join([m.content for m in system_messages])
        )
        result["messages"] = [m.model_dump(exclude_none=True) for m in messages]
        return result

    def to_openai(self):
        exclude = ["system"]
        result = self.model_dump(exclude_none=True)
        if self.system:
            messages = [SystemMessage(content=self.system)] + self.messages
        else:
            messages = self.messages

        # Function to recursively set 'additionalProperties': False
        def set_additional_properties_false(schema):
            if isinstance(schema, dict):
                if schema.get("type") == "object":
                    schema["additionalProperties"] = False
                for key, value in schema.items():
                    set_additional_properties_false(value)
            elif isinstance(schema, list):
                for item in schema:
                    set_additional_properties_false(item)

        if "response_format" in result:
            model_schema = self.response_format.model_json_schema()
            set_additional_properties_false(model_schema)
            del result["response_format"]
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self.response_format.__name__,
                    "description": "response format",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "additionalProperties": False,
                        **model_schema,
                    },
                },
            }
        result["messages"] = [m.model_dump() for m in messages]
        result["tools"] = (
            [t.to_openai().model_dump(exclude_none=True) for t in self.tools]
            if self.tools
            else None
        )

        for field in exclude:
            if field in result:
                del result[field]
        return result
