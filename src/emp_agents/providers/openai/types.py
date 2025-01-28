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
