import pytest

from emp_agents.agents.planner import Planner, Task, TaskList
from emp_agents.providers.openai import OpenAIProvider


@pytest.mark.asyncio(scope="session")
async def test_generate():
    agent = Planner(
        provider=OpenAIProvider(),
    )
    subject = "write a blog post"
    task_list = await agent.generate(subject)
    assert isinstance(task_list, TaskList)

    assert len(task_list.tasks) > 0

    for task in task_list.tasks:
        assert isinstance(task, Task)
        assert isinstance(task.title, str)
        assert isinstance(task.description, str)
        assert task.title != ""
        assert task.description != ""
