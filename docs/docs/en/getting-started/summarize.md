# Summarizing Conversations
Conversations that become too long end up taking up too much memory in the context window for the agent.  To summarize a conversation, you can use the `summarize_conversation` function, or the agents `BaseAgent.summarize` method.
To summarize a conversation, you can use the `summarize_conversation` function.

```python
from emp_agents import AgentBase
from emp_agents.models import Role, UserMessage, AssistantMessage
from emp_agents.utils import summarize_conversation

messages = [
    UserMessage(content="Hello how are you?"),
    AssistantMessage(content="I'm doing great, thanks for asking!"),
    UserMessage(content="Tell me about baseball."),
    AssistantMessage(content="Baseball is a sport played with a bat and a ball.  It's a very popular sport in the United States.  The goal is to score runs by hitting the ball and running around the bases."),
    UserMessage(content="What's the best baseball team?"),
    AssistantMessage(content="The best baseball team is the Boston Red Sox. They are a very successful team that has won many championships and have a really interesting history."),
]
agent = AgentBase()
agent.add_messages(messages)
summary = await agent.summarize(
    prompt="Provide a summary in a single sentence.",
    max_tokens=200,
)
print(summary)
# Output: The user engages in a friendly conversation with the assistant about baseball, discussing its basics and identifying the Boston Red Sox as the best team due to their success and history.
```
