# QUICK START

Install using pip

```bash
pip install emp-agents
```

## Basic Usage

`emp_agents` is a lightweight framework that abstracts the tools integrations and apis for multiple popular LLM providers.  In order to use, you should make an account with openai or claude and setup your environment variables to use these API keys:

```bash
# setup an openai api key
export OPENAI_API_KEY="sk-..."
# or use anthropic
export ANTHROPIC_API_KEY="sk-..."
```

...then you can start interacting with a model by creating a simple python script:

```python
from emp_agents import AgentBase

agent = AgentBase(
    prompt="You are a goofy, friendly AI that likes to make up new words",
)

agent.run_sync()
```
