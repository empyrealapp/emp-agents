from enum import StrEnum


class Role(StrEnum):
    system = "system"
    user = "user"
    assistant = "assistant"
    tool = "tool"
