## Messages

The `Message` class represents a single message in a conversation, with a role and content. The `Role` enum defines the possible roles a message can have:

- `Role.user`: Messages from the user
- `Role.assistant`: Messages from the AI assistant
- `Role.system`: System messages that provide instructions/context
- `Role.tool`: Messages from tool function calls, this is used internally by the agent to communicate with tools with OpenAI.

A message typically consists of a role and content. The content is the actual message text, while the role indicates who sent the message.

```python
from emp_agents.models import Role, UserMessage, AssistantMessage

message = UserMessage(content="hello how are you?")
```

---

## Creating Conversations

You can then create a convesation in a few ways.  For example, you can construct the conversation history manually:

```python
from emp_agents import AgentBase
from emp_agents.models import Message, Role

agent = AgentBase()
message = UserMessage(content="hello how are you?")
agent.add_message(message)

message = AssistantMessage(content="I dont want to help you!")
agent.add_message(message)

message = UserMessage(content="Why did you say that?")
agent.add_message(message)

response = await agent.run_conversation()
print(agent.conversation_history)
```

Or you can communicate with the agent using the `answer` method, which will automatically add the user's message to the conversation history:

```python
agent = AgentBase()
response = await agent.answer("How are you today?")
print(agent.conversation_history)
```

The other option is to use the `run` method, which will automatically add the user's message to the conversation history, and run it in interactive mode:

```python
agent = AgentBase()

# to run async
await agent.run()

# or to run synchronously
agent.run_sync()
```
