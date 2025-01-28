from typing import Any

from pydantic import BaseModel, Field

from emp_agents.models import GenericTool


class PersistentAgentConfig(BaseModel):
    agent_id: str
    name: str
    description: str
    default_model: str | None = None
    prompt: str = "You are a helpful assistant"
    tools: list[GenericTool] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)
