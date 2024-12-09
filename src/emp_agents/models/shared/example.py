from emp_agents.models.shared.message import AssistantMessage, ToolMessage, UserMessage


QuestionAnswer = tuple[UserMessage, AssistantMessage]
TooledMessageSequence = tuple[UserMessage, AssistantMessage, list[ToolMessage]]

Example = QuestionAnswer | TooledMessageSequence
