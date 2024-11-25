from typing import Any, Callable

from fast_depends import Provider
from pydantic import BaseModel, ConfigDict, Field

from emp_agents.models.protocol import SkillSet
from emp_agents.types import AnthropicModelType, OpenAIModelType

from .base import AgentBase


class SkillsAgent(AgentBase):
    skills: list[type[SkillSet]] = Field(default_factory=list)
    scopes: list[tuple[Provider, Callable, Callable]] = Field(default_factory=list)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any):
        super().model_post_init(__context)

        for skill in self.skills:
            for tool in skill._tools:
                self._add_tool(tool)

    async def complete(
        self,
        model: OpenAIModelType | AnthropicModelType | None = None,
    ) -> str:
        for scope, old, new in self.scopes:
            scope.dependency_overrides[old] = new

        response = await super().complete(model)

        for scope, old, new in self.scopes:
            scope.dependency_overrides.pop(old, None)

        return response

    async def respond(
        self,
        question: str,
        model: OpenAIModelType | AnthropicModelType | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        for scope, old, new in self.scopes:
            scope.dependency_overrides[old] = new

        response = await super().respond(question, model, max_tokens, response_format)

        for scope, old, new in self.scopes:
            scope.dependency_overrides.pop(old, None)

        return response
