from enum import StrEnum

from pydantic import BaseModel


class Classification(BaseModel):
    name: str
    description: str


class FinishReason(StrEnum):
    stop = "stop"
    length = "length"
    function_call = "function_call"
    content_filter = "content_filter"
    null = "null"
    tool_calls = "tool_calls"


class OpenAIModelType(StrEnum):
    gpt3_5 = "gpt-3.5-turbo-0125"
    gpt3_5_turbo = "gpt-3.5-turbo"
    gpt4 = "gpt-4"
    gpt4_turbo = "gpt-4-turbo"
    gpt4o_mini = "gpt-4o-mini"  # 128_000 tokens
    gpt4o = "gpt-4o"
    gpt_o1_mini = "o1-mini"
    gpt_o1_preview = "o1-preview"
