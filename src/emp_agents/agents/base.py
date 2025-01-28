import asyncio
from textwrap import dedent
from typing import Any, Callable, Awaitable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_validator,
)

from emp_agents.agents.history import AbstractConversationProvider, ConversationProvider
from emp_agents.logger import logger
from emp_agents.models import (
    AssistantMessage,
    GenericTool,
    Message,
    Middleware,
    Provider,
    Request,
    ResponseT,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from emp_agents.types import Role
from emp_agents.utils import count_tokens, execute_tool, summarize_conversation


class AgentBase(BaseModel):
    agent_id: str = Field(default="")
    description: str = Field(default="")
    default_model: str | None = None
    prompt: str = Field(default="You are a helpful assistant")
    tools: list[GenericTool | Callable[..., str]] = Field(default_factory=list)
    requires: list[str] = []
    provider: Provider
    conversation: AbstractConversationProvider = Field(
        default_factory=ConversationProvider
    )

    # This can be used to modify the conversation before completion, such as RAG
    middleware: list[Middleware] = Field(default_factory=list)

    _tools: list[GenericTool] = PrivateAttr(default_factory=list)
    _tools_map: dict[str, Callable[..., Any]] = PrivateAttr(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def conversation_history(self) -> list[Message]:
        return self.conversation.get_history()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def _default_model(self) -> str:
        if self.default_model:
            return self.default_model
        _model = self.provider.default_model
        assert _model is not None, "Default model is required"
        return _model

    def _load_model(self, model: str | None) -> str:
        if model is None:
            model = self._default_model
        assert model is not None, "Model is required"
        return model

    @field_validator("prompt", mode="before")
    @classmethod
    def to_prompt(cls, v: str) -> str:
        return dedent(v).strip()

    @field_validator("tools", mode="before")
    @classmethod
    def to_generic_tools(
        cls, v: list[Callable[..., Any] | GenericTool]
    ) -> list[GenericTool]:
        return [
            GenericTool.from_func(tool) if not isinstance(tool, GenericTool) else tool
            for tool in v
        ]

    def _load_implicits(self):
        """Override this method to load implicits to the agent directly"""

    def model_post_init(self, _context: Any):
        if not (self.provider.api_key):
            raise ValueError("Must provide an api key")

        for tool in self.tools:
            if isinstance(tool, GenericTool):
                self._tools.append(tool)
            else:
                self._tools.append(GenericTool.from_func(tool))

        self._tools_map = {tool.name: tool.func for tool in self._tools}
        self.conversation.add_message(SystemMessage(content=self.system_prompt))

        self._load_implicits()

    def get_token_count(
        self,
        model: str = "gpt4o_mini",
    ) -> int:
        """A utility to get the token count for openai models, fairly accurate across all providers"""
        return count_tokens(self.conversation.get_history(), model)

    async def summarize(
        self,
        model: str | None = None,
        update: bool = True,
        prompt: str | None = None,
        max_tokens: int = 500,
    ) -> str:
        model = self._load_model(model)

        summary = await summarize_conversation(
            self.provider,
            self.conversation.get_history(),
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        if update:
            self.conversation.set_history([summary])
        assert summary.content is not None, "Summary content should always be present"
        return summary.content

    async def respond(
        self,
        question: str,
        model: str | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """Send a one-off question and get a response"""
        if model is None:
            model = self._default_model

        conversation = [
            SystemMessage(content=self.system_prompt),
            UserMessage(content=question),
        ]
        return await self._run_conversation(
            conversation,
            model=model,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def complete(
        self,
        model: str | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """Complete the current conversation until no more tool calls"""
        _model = self._load_model(model)
        return await self._run_conversation(
            self.conversation.get_history(),
            model=_model,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    async def _run_conversation(
        self,
        messages: list[Message],
        model: str,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        """Core conversation loop handling tool calls"""
        conversation = messages.copy()
        for middleware in self.middleware:
            _conversation = middleware.function(conversation)
            if isinstance(_conversation, Awaitable):
                conversation = await _conversation
            else:
                conversation = _conversation
        while True:
            request = Request(
                messages=conversation,
                model=model,
                tools=self._tools,
                max_tokens=max_tokens or 1_000,
                response_format=response_format,
            )
            response: ResponseT = await self.provider.completion(request)
            conversation += response.messages

            if not response.tool_calls:
                self.conversation.set_history(conversation)
                return response.text

            tool_invocation_coros = [
                execute_tool(
                    self._tools_map,
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                for tool_call in response.tool_calls
            ]
            tool_results = await asyncio.gather(*tool_invocation_coros)
            for result, tool_call in zip(tool_results, response.tool_calls):
                message = ToolMessage(
                    content=result,
                    tool_call_id=(
                        tool_call.id if tool_call and hasattr(tool_call, "id") else None
                    ),
                )
                if hasattr(self, "conversation_history"):
                    logger.info(message)
                conversation += [message]
                self.conversation.set_history(conversation)

    async def answer(
        self,
        question: str,
        model: str | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        self.conversation.add_message(Message(role=Role.user, content=question))

        return await self.complete(
            model=model,
            response_format=response_format,
        )

    def add_message(
        self,
        message: Message,
    ) -> None:
        self.conversation.add_message(message)

    def add_messages(
        self,
        messages: list[Message],
    ) -> None:
        self.conversation.add_messages(messages)

    async def __call__(
        self,
        question: str,
        model: str | None = None,
    ) -> str:
        return await self.answer(question, model=model)

    async def reset(self):
        self.conversation.reset()

    @property
    def system_prompt(self) -> str:
        prompt = self.prompt
        return prompt.strip()

    def print_conversation(self) -> None:
        for message in self.conversation.get_history():
            print(f"{message.role}: {message.content}")

    def _make_message(self, content: str, role: Role = Role.user) -> Message:
        return Message.build(content, role)

    async def run(self):
        conversation = [SystemMessage(content=self.system_prompt)]
        while True:
            question = input("You: ")
            if question == "":
                break
            conversation += [UserMessage(content=question)]
            response = await self.answer(question)
            print(response)
            conversation += [AssistantMessage(content=response)]

    def _add_tool(self, tool: GenericTool) -> None:
        self._tools.append(tool)
        self._tools_map[tool.name] = tool.func

    def run_sync(self):
        asyncio.run(self.run())

    def __repr__(self):
        prompt = self.prompt[:100].strip().replace("\n", " ")
        if len(prompt) >= 50:
            prompt = prompt[:50] + "..."
        return dedent(
            """
            <{class_name}
                prompt="{prompt}..."
                tools=[
                    {tools}
                ]
            >
        """.format(
                class_name=self.__class__.__name__,
                prompt=prompt,
                tools="\n".join([repr(tool) for tool in self.tools]),
            )
        ).strip()

    __str__ = __repr__
