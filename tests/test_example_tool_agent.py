import pytest
import os

from emp_agents import AgentBase
from emp_agents.providers import OpenAIProvider


async def get_cat_fact() -> str:
    """
    Get a random cat fact
    """
    return "cats sleep 70% of their lives"


@pytest.mark.asyncio(scope="session")
async def test_tool_agent():
    agent = AgentBase(
        prompt="you provide cat facts.  respond with the exact text from the tool.",
        tools=[get_cat_fact],
        provider=OpenAIProvider(
            model="gpt-4o-mini",
            openai_api_key=os.environ["OPENAI_API_KEY"],
        ),
    )
    response = await agent.answer("tell me a cat fact")
    assert "cats sleep 70% of their lives" in response.strip().lower()
