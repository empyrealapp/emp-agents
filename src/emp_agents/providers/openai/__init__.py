import os
from typing import ClassVar

import httpx
from pydantic import Field

from emp_agents.models import GenericTool, Provider, Request, SystemMessage
from .request import Message, Tool
from .response import Response
from .tool import Function, Parameters, Property
from .types import Classification


class OpenAIProvider(Provider[Response]):
    URL: ClassVar[str] = "https://api.openai.com/v1/chat/completions"
    api_key: str = Field(default_factory=lambda: os.environ["OPENAI_API_KEY"])
    default_model: str = Field(default="gpt-4o-mini")

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _from_request(self, request: Request):
        exclude = ["system"]
        result = request.model_dump(exclude_none=True)
        if request.system:
            messages = [SystemMessage(content=request.system)] + request.messages
        else:
            messages = request.messages

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
            model_schema = request.response_format.model_json_schema()
            set_additional_properties_false(model_schema)
            del result["response_format"]
            result["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.response_format.__name__,
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
            [self.to_tool_call(t).model_dump(exclude_none=True) for t in request.tools]
            if request.tools
            else None
        )

        for field in exclude:
            if field in result:
                del result[field]
        return result

    async def completion(self, request: Request) -> Response:
        openai_request = self._from_request(request)
        async with httpx.AsyncClient(headers=self.headers) as client:
            response = await client.post(self.URL, json=openai_request, timeout=None)
        if response.status_code >= 400:
            raise ValueError(response.json())
        return Response(**response.json())

    def to_tool_call(self, tool: GenericTool):
        from .tool import Function, Parameters, Property
        from .tool import Tool as Tool

        return Tool(
            type="function",
            function=Function(
                description=tool.description,
                name=tool.name,
                parameters=Parameters(
                    properties={
                        key: Property(**param.model_dump(exclude_none=True))
                        for key, param in tool.parameters.items()
                    },
                    required=tool.required,
                ),
            ),
            strict=True,
        )


__all__ = [
    "Classification",
    "Message",
    "OpenAIBase",
    "Request",
    "Response",
    "Tool",
    "Function",
    "Property",
    "Parameters",
]
