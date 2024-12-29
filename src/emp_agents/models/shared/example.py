from emp_agents.models.shared.message import AssistantMessage, ToolMessage, UserMessage

QuestionAnswer = tuple[UserMessage, AssistantMessage]
TooledMessageSequence = tuple[UserMessage, AssistantMessage, list[ToolMessage]]
AgenticMessageSequence = tuple[
    UserMessage,
    list[tuple[AssistantMessage, list[ToolMessage]]],
    AssistantMessage,
]

Example = QuestionAnswer | TooledMessageSequence | AgenticMessageSequence
