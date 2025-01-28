from enum import StrEnum


class AnthropicModelType(StrEnum):
    claude_3_5_sonnet = "claude-3-5-sonnet-20240620"
    claude_3_opus = "claude-3-opus-20240229"
    claude_3_sonnet = "claude-3-sonnet-20240229"
    claude_3_5_haiku = "claude-3-5-haiku-20241022"
    claude_3_haiku = "claude-3-haiku-20240307"
    claude_2_1 = "claude-2.1"
    claude_2_0 = "claude-2.0"
    claude_instant_1_2 = "claude-instant-1.2"
