import asyncio
import os
from textwrap import dedent
from typing import Any, Callable

from pydantic import BaseModel, Field, PrivateAttr, computed_field, field_validator

from emp_agents.exceptions import InvalidModelException
from emp_agents.logger import logger
from emp_agents.models import AnthropicBase, GenericTool, Message, OpenAIBase, Request
from emp_agents.types import AnthropicModelType, OpenAIModelType, Role
from emp_agents.utils import count_tokens, execute_tool, summarize_conversation


class AgentBase(BaseModel):
    agent_id: str = Field(default="")
    description: str = Field(default="")
    default_model: OpenAIModelType | AnthropicModelType | None = None
    prompt: str = Field(default="You are a helpful assistant")
    personality: str | None = Field(default=None)
    tools: list[GenericTool] = Field(default_factory=list)
    conversation_history: list[Message] = Field(default_factory=list)
    requires: list[str] = []
    openai_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
    )
    anthropic_api_key: str | None = Field(
        default_factory=lambda: os.environ.get("ANTHROPIC_API_KEY")
    )

    _tools: list[GenericTool] = PrivateAttr(default_factory=list)
    _tools_map: dict[str, Callable[..., Any]] = PrivateAttr(default_factory=dict)

    @computed_field
    @property
    def _default_model(self) -> OpenAIModelType | AnthropicModelType:
        if self.default_model:
            return self.default_model
        if self.openai_api_key:
            return OpenAIModelType.gpt4o_mini
        elif self.anthropic_api_key:
            return AnthropicModelType.claude_3_opus
        raise ValueError("No API key found")

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

    def model_post_init(self, _context: Any):
        if not (self.openai_api_key or self.anthropic_api_key):
            raise ValueError("Must provide either openai or anthropic api key")

        for tool in self.tools:
            if isinstance(tool, GenericTool):
                self._tools.append(tool)
            else:
                self._tools.append(GenericTool.from_func(tool))

        self._tools_map = {tool.name: tool.func for tool in self._tools}
        self.conversation_history = [
            Message(role=Role.system, content=self.system_prompt)
        ] + self.conversation_history

    def get_token_count(
        self, model: OpenAIModelType | AnthropicModelType = OpenAIModelType.gpt4o_mini
    ) -> int:
        return count_tokens(self.conversation_history, model)

    async def summarize(
        self,
        model: OpenAIModelType | AnthropicModelType | None = None,
        update: bool = True,
        prompt: str | None = None,
        max_tokens: int = 500,
    ) -> str:
        if model is None:
            model = self._default_model
        assert model is not None, "Model is required"

        summary = await summarize_conversation(
            self._make_client(model),
            self.conversation_history,
            model=model,
            prompt=prompt,
            max_tokens=max_tokens,
        )
        if update:
            self.conversation_history = [summary]
        assert summary.content is not None, "Summary content should always be present"
        return summary.content

    async def respond(
        self,
        question: str,
        model: OpenAIModelType | AnthropicModelType | None = None,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
    ) -> str:
        if self.default_model is None and model is None:
            raise InvalidModelException("Model is required")
        if model is None:
            assert self.default_model is not None, "Model is required"
            model = self.default_model
        if not isinstance(model, (OpenAIModelType, AnthropicModelType)):
            raise InvalidModelException(f"Invalid model: {model}")
        client: OpenAIBase | AnthropicBase = self._make_client(model)
        conversation = [
            Message(role=Role.system, content=self.system_prompt),
            Message(role=Role.user, content=question),
        ]

        while True:
            request = Request(
                messages=conversation,
                model=model,
                tools=self._tools,
                max_tokens=max_tokens or 1_000,
                response_format=response_format,
            )
            response = await client.completion(
                request,
            )

            if isinstance(model, OpenAIModelType):
                conversation += response.messages
            else:
                conversation += [Message(role=Role.assistant, content=response.text)]

            tool_calls = response.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    result = await execute_tool(
                        self._tools_map,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    message = Message(
                        role=(
                            Role.user
                            if isinstance(model, AnthropicModelType)
                            else Role.tool
                        ),
                        content=result,
                        tool_call_id=(
                            tool_call.id if isinstance(model, OpenAIModelType) else None
                        ),
                    )
                    conversation += [message]
                continue
            else:
                return response.text

    async def answer(
        self,
        question: str,
        model: OpenAIModelType | AnthropicModelType | None = None,
    ) -> str:
        self.conversation_history += [Message(role=Role.user, content=question)]
        response = await self.complete(
            model=model,
        )
        return response

    async def step(
        self,
        model: OpenAIModelType | AnthropicModelType = OpenAIModelType.gpt4o_mini,
    ) -> None:
        client: OpenAIBase | AnthropicBase = self._make_client(model)
        request = Request(
            messages=self.conversation_history, model=model, tools=self._tools
        )
        response = await client.completion(
            request,
        )
        if isinstance(model, OpenAIModelType):
            self.conversation_history += response.messages
        else:
            self.conversation_history += [
                Message(role=Role.assistant, content=response.text)
            ]

        tool_calls = response.tool_calls
        if tool_calls:
            for tool_call in tool_calls:
                result = await execute_tool(
                    self._tools_map,
                    tool_call.function.name,
                    tool_call.function.arguments,
                )
                message = Message(
                    role=(
                        Role.user
                        if isinstance(model, AnthropicModelType)
                        else Role.tool
                    ),
                    content=result,
                    tool_call_id=(
                        tool_call.id if isinstance(model, OpenAIModelType) else None
                    ),
                )
                self.conversation_history += [message]
        return None

    def add_message(
        self,
        message: Message,
    ) -> None:
        self.conversation_history += [message]

    def add_messages(
        self,
        messages: list[Message],
    ) -> None:
        self.conversation_history += messages

    async def complete(
        self,
        model: OpenAIModelType | AnthropicModelType | None = None,
        verbose: bool = False,
    ) -> str:
        if not model:
            model = self._default_model
        assert model is not None, "Model is required"
        client: OpenAIBase | AnthropicBase = self._make_client(model)

        while True:
            request = Request(
                messages=self.conversation_history,
                model=model,
                tools=self._tools,
            )
            response = await client.completion(
                request,
            )

            if isinstance(model, OpenAIModelType):
                self.conversation_history += response.messages
            else:
                self.conversation_history += [
                    Message(role=Role.assistant, content=response.text)
                ]

            tool_calls = response.tool_calls

            if tool_calls:
                for tool_call in tool_calls:
                    result = await execute_tool(
                        self._tools_map,
                        tool_call.function.name,
                        tool_call.function.arguments,
                    )
                    message = Message(
                        role=(
                            Role.user
                            if isinstance(model, AnthropicModelType)
                            else Role.tool
                        ),
                        content=result,
                        tool_call_id=(
                            tool_call.id if isinstance(model, OpenAIModelType) else None
                        ),
                    )
                    logger.info(message)
                    self.conversation_history += [message]
                continue
            else:
                return response.text

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

    def _make_client(
        self, model: OpenAIModelType | AnthropicModelType
    ) -> OpenAIBase | AnthropicBase:
        if isinstance(model, OpenAIModelType):
            if not self.openai_api_key:
                raise ValueError("OpenAI API key is required")
            return OpenAIBase(openai_api_key=self.openai_api_key)
        else:
            if not self.anthropic_api_key:
                raise ValueError("Anthropic API key is required")
            return AnthropicBase(anthropic_api_key=self.anthropic_api_key)

    async def __call__(
        self,
        question: str,
        model: OpenAIModelType | AnthropicModelType = OpenAIModelType.gpt4o_mini,
    ) -> str:
        return await self.answer(question, model)

    async def reset(self):
        self.conversation_history = []

    @property
    def system_prompt(self) -> str:
        prompt = self.prompt
        if self.personality:
            prompt = f"{prompt}\n\nPERSONALITY: {self.personality}"
        return prompt.strip()

    def print_conversation(self) -> None:
        for message in self.conversation_history:
            if not message.tool_call_id:
                print(f"{message.role}: {message.content}")

    def _make_message(self, content: str, role: Role = Role.user):
        return Message(role=role, content=content)

    async def run(self):
        conversation = [Message(role=Role.system, content=self.system_prompt)]
        while True:
            question = input("You: ")
            if question == "":
                break
            conversation += [Message(role=Role.user, content=question)]
            response = await self.answer(question)
            conversation += [Message(role=Role.assistant, content=response)]

    def _add_tool(self, tool: GenericTool) -> None:
        self._tools.append(tool)
        self._tools_map[tool.name] = tool.func

    def run_sync(self):
        asyncio.run(self.run())
