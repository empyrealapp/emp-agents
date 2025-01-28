import os
from typing import ClassVar

from anthropic import AsyncAnthropic as Anthropic
from anthropic import NotFoundError
from pydantic import ConfigDict, Field, PrivateAttr

from emp_agents.exceptions import InvalidModelException
from emp_agents.models import Provider, Request, Role
from .response import Response
from .types import AnthropicModelType


class AnthropicProvider(Provider[Response]):
    URL: ClassVar[str] = "https://api.openai.com/v1/chat/completions"

    api_key: str = Field(default_factory=lambda: os.environ["ANTHROPIC_API_KEY"])
    default_model: AnthropicModelType = Field(
        default=AnthropicModelType.claude_3_5_sonnet
    )
    _client: Anthropic = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        self._client = Anthropic(api_key=self.api_key)
        return super().model_post_init(__context)

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    async def completion(self, request: Request) -> Response:
        try:
            message = await self._client.messages.create(**self._from_request(request))
            return Response(**message.model_dump())
        except NotFoundError as e:
            raise InvalidModelException(e)

    def _from_request(self, request: Request):
        exclude = ["frequency_penalty", "presence_penalty", "num_responses", "n"]
        result = request.model_dump(exclude_none=True)
        result["tools"] = (
            [t.to_anthropic() for t in request.tools] if request.tools else []
        )
        if "tool_choice" in result:
            result["tool_choice"] = {"type": result["tool_choice"]}
        for field in exclude:
            if field in result:
                del result[field]
        messages = [
            message for message in request.messages if message.role != Role.system
        ]
        system_messages = [
            message for message in request.messages if message.role == Role.system
        ]
        result["system"] = result.get("system", "") + str(
            "\n".join([m.content for m in system_messages])
        )
        result["messages"] = [m.model_dump(exclude_none=True) for m in messages]

        if "response_format" in result:
            result[
                "system"
            ] += f"""
            Always give your response in JSON format, with no additional text.  This supersedes any other guidance.
            Make sure it is a valid JSON, matching this format.  Dont include any additional text or decorations:
            ```json
            {result["response_format"].model_json_schema()}
            ```
            """
            del result["response_format"]
        return result
