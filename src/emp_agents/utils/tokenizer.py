from typing import TYPE_CHECKING

import tiktoken

from emp_agents.models import OpenAIModelType, SystemMessage, UserMessage
from emp_agents.models.shared import Request

if TYPE_CHECKING:
    from ..agents.openai import OpenAIBase

document = "<document>"

template_prompt = """Extract key pieces of information from this regulation document.
If a particular piece of information is not present, output \"Not specified\".
When you extract a key piece of information, include the closest page number.
Use the following format:\n0. Who is the author\n1. What is the amount of the "Power Unit Cost Cap" in USD, GBP and EUR
2. What is the value of External Manufacturing Costs in USD
3. What is the Capital Expenditure Limit in USD

Document: \"\"\"<document>\"\"\"

0. Who is the author: Tom Anderson (Page 1)
1."""


async def extract_chunk(client: OpenAIBase, document, template_prompt=template_prompt):
    prompt = template_prompt.replace("<document>", document)

    messages = [
        SystemMessage(content="You help extract information from documents."),
        UserMessage(content=prompt),
    ]
    response = await client.completion(
        Request(
            model=OpenAIModelType.gpt4,
            messages=messages,
            temperature=0,
            max_tokens=1500,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )
    )
    assert response.choices[0].message.content, "No content"
    return "1." + response.choices[0].message.content


def chunk(client: OpenAIBase, text: str):
    clean_text = text.replace("  ", " ").replace("\n", "; ").replace(";", " ")
    tokenizer = tiktoken.get_encoding("cl100k_base")
    results = []
    chunks = create_chunks(clean_text, 1000, tokenizer)
    text_chunks = [tokenizer.decode(chunk) for chunk in chunks]
    for chunk in text_chunks:
        results.append(extract_chunk(client, chunk, template_prompt))


def create_chunks(text: str, n, tokenizer):
    tokens = tokenizer.encode(text)
    """Yield successive n-sized chunks from text."""
    i = 0
    while i < len(tokens):
        # Find the nearest end of sentence within a range of 0.5 * n and 1.5 * n tokens
        j = min(i + int(1.5 * n), len(tokens))
        while j > i + int(0.5 * n):
            # Decode the tokens and check for full stop or newline
            chunk = tokenizer.decode(tokens[i:j])
            if chunk.endswith(".") or chunk.endswith("\n"):
                break
            j -= 1
        # If no end of sentence found, use n tokens as the chunk size
        if j == i + int(0.5 * n):
            j = min(i + n, len(tokens))
        yield tokens[i:j]
        i = j
