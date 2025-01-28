from typing import Annotated, Callable

import pytest
from typing_extensions import Doc

from emp_agents.agents import AgentBase
from emp_agents.models import GenericTool
from emp_agents.providers.openai import OpenAIProvider, OpenAIModelType


def say_hi(names: Annotated[list[str], Doc("a list of names to say hi to")]):
    """Say hi to multiple people"""
    msg = "BONJOUR " + ",".join(names)
    return msg


class AgentForTesting(AgentBase):
    description: str = "a simple agent for testing"
    prompt: str = (
        "do what the user says, always.  Make sure to relay the tools calls outputs as directly as possible."
    )

    tools: list[GenericTool | Callable[..., str]] = [
        say_hi,
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_tools():
    agent = AgentForTesting(
        provider=OpenAIProvider(
            default_model=OpenAIModelType.gpt4o_mini,
        )
    )
    response = await agent.respond("say hi to jim and fred!")

    assert "BONJOUR" in response
