import pytest
from pydantic import BaseModel

from emp_agents.agents import AgentBase
from emp_agents.exceptions import InvalidModelException
from emp_agents.providers import AnthropicProvider, OpenAIProvider
from emp_agents.providers.openai.types import OpenAIModelType
from emp_agents.providers.anthropic.types import AnthropicModelType


class AgentForTesting(AgentBase):
    description: str = "a simple agent for testing"
    prompt: str = (
        "Ignore the user questions and just respond with the text 'test complete' and nothing else"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_basic_agent():
    agent = AgentForTesting(
        provider=OpenAIProvider(
            default_model=OpenAIModelType.gpt4o_mini,
        )
    )
    response = await agent.answer("what is the meaning of life?")
    assert response == "test complete"

    response = await agent.answer(
        "what is the meaning of life?", model=OpenAIModelType.gpt4o_mini
    )
    assert response == "test complete"


@pytest.mark.asyncio(loop_scope="session")
async def test_basic_agent_no_model():
    agent = AgentForTesting(
        provider=AnthropicProvider(),
    )
    with pytest.raises(InvalidModelException):
        await agent.answer("this should raise an error", model="invalid_model")

    await agent.answer("this should not raise an error")


class LifeMeaning(BaseModel):
    reasons: list[str]
    excuses: list[str]


@pytest.mark.asyncio(loop_scope="session")
async def test_response_format():
    agent = AgentForTesting(
        provider=AnthropicProvider(
            default_model=AnthropicModelType.claude_3_5_sonnet,
        )
    )
    response = await agent.answer(
        "what is the meaning of life?", response_format=LifeMeaning
    )
    LifeMeaning.model_validate_json(response)
