from typing import TYPE_CHECKING

import tiktoken

from emp_agents.types import AnthropicModelType, OpenAIModelType, Role

if TYPE_CHECKING:
    from emp_agents.models import AnthropicBase, Message, OpenAIBase

DEFAULT_SUMMARY_PROMPT = """
You are an assistant that summarizes conversations concisely.
Dont worry about human readability, just focus on conciseness.
"""


def format_conversation(conversation: list["Message"]) -> str:
    """
    Formats the conversation list into a readable string.
    """
    formatted = ""
    for message in conversation:
        role = message.role
        content = message.content
        formatted += f"{role.capitalize()}: {content}\n"
    return formatted


def count_tokens(
    messages: list["Message"] | str,
    model: OpenAIModelType | AnthropicModelType = OpenAIModelType.gpt4o_mini,
) -> int:
    encoding = tiktoken.encoding_for_model(model)
    tokens = 0
    if isinstance(messages, list):
        for message in messages:
            tokens += 4  # Message formatting tokens
            for key, value in message.model_dump().items():
                if isinstance(value, str):
                    tokens += len(encoding.encode(value))
    else:
        tokens += len(encoding.encode(messages))
    tokens += 2  # Priming tokens
    return tokens


async def summarize_conversation(
    client: "OpenAIBase | AnthropicBase",
    messages: list["Message"],
    model: OpenAIModelType | AnthropicModelType,
    prompt: str | None = None,
    max_tokens: int = 500,
) -> "Message":
    from emp_agents.models import AssistantMessage, Request, SystemMessage, UserMessage

    summary_prompt = prompt or DEFAULT_SUMMARY_PROMPT
    assert summary_prompt is not None, "Summary prompt is required"

    try:
        request = Request(
            messages=[
                SystemMessage(
                    content=summary_prompt,
                ),
                UserMessage(
                    content=f"Summarize the following conversation:\n\n{format_conversation(messages)}",
                ),
            ],
            model=model,
            max_tokens=max_tokens,
            temperature=0.5,
        )
        response = await client.completion(
            request,
        )
        return AssistantMessage(content=response.text)
    except Exception as e:
        print(f"Error during summarization: {e}")
        return AssistantMessage(content="")
