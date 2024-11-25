import os
from typing import ClassVar

from anthropic import AsyncAnthropic as Anthropic
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from emp_agents.models.anthropic.response import Response
from emp_agents.models.shared import Request
from emp_agents.types import TCompletionAgent


class AnthropicBase(BaseModel, TCompletionAgent[Response]):
    URL: ClassVar[str] = "https://api.openai.com/v1/chat/completions"

    anthropic_api_key: str = Field(
        default_factory=lambda: os.environ["ANTHROPIC_API_KEY"]
    )
    _client: Anthropic = PrivateAttr()

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context) -> None:
        self._client = Anthropic(api_key=self.anthropic_api_key)
        return super().model_post_init(__context)

    @property
    def headers(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.anthropic_api_key}",
        }

    async def completion(self, request: Request) -> Response:
        message = await self._client.messages.create(**request.to_anthropic())
        try:
            return Response(**message.model_dump())
        except Exception as e:
            raise e
