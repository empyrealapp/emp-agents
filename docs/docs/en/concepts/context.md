# Context Window

The context window is a fundamental concept in Large Language Models (LLMs) that refers to the maximum amount of text that the model can process at once. This includes both the input (prompt) and the expected output.

## Key Points

### Tokens
The context window is measured in tokens (roughly 4 characters per token in English)

### Model Context Windows

Different models have different context window sizes.  For example:

- GPT-4 Turbo: 128K tokens
- Claude 3 Opus: 200K tokens
- GPT-3.5 Turbo: 16K tokens

Longer conversations need to be summarized to fit within the context window. This is because as conversations grow, they accumulate more tokens and can eventually exceed the model's context limit. When this happens, older messages need to be condensed or removed to make room for new ones while preserving the important context. `emp-agents` handles this by tracking token usage and summarizing conversations when needed.

---

### Context Window Impact

The context window affects:

- Memory of previous conversation
- Ability to process long documents
- Cost (as most models charge per token)

!!! tip "Context Window Efficiency"
    The model will perform slower, cost more, and be less accurate as the context window grows, so it is very important to be careful about how you use your context tokens.  The more efficient you are with your tokens, the more accurate and cost-effective your model will be.

---
## Managing Context

`emp-agents` provides tools to help manage the context window:

- Token counting via `count_tokens()` function
- Conversation summarization via `summarize_conversation()` or `agent.summarize()`
- Dynamic Tool Allocation
- Automatic tracking of conversation history

See the [Summarizing Conversations](../getting-started/summarize.md) section for practical examples of context management.

```python
from emp_agents import AgentBase
from emp_agents.models import Role, UserMessage, AssistantMessage

messages = [
    UserMessage(content="Hello, how are you?"),
    AssistantMessage(content="I'm doing great, thank you!"),
]

agent = AgentBase()
agent.add_messages(messages)
token_count = agent.get_token_count()
print(token_count)

# Output: 35
```
