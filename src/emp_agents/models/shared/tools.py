import inspect
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, get_args, get_origin

from pydantic import BaseModel, Field

from emp_agents.utils import FunctionSchema, get_function_schema

if TYPE_CHECKING:
    from emp_agents.agents.base import AgentBase


class Property(BaseModel):
    type: str | None = Field(default="string")
    description: str = ""
    enum: list[str] | None = None
    properties: dict[str, "Property"] | None = None
    required: list[str] | None = None
    items: "Property | None" = None


class GenericTool(BaseModel):
    """
    This converts a function to a generic tool.
    """

    strict: bool | None = None
    description: str
    name: str
    parameters: dict[str, Property]
    required: list[str]
    type: str = "object"
    additional_properties: bool = Field(default=False)

    func: Callable

    @classmethod
    def _convert_type(cls, type):
        if type is str:
            return "string"
        elif get_origin(type) is list:
            return "array"
        elif inspect.isclass(type) and issubclass(type, Enum):
            return "enum"
        return str(type)

    @classmethod
    def _get_enum(cls, type):
        if inspect.isclass(type) and issubclass(type, Enum):
            return [x.name for x in list(type)]
        return None

    @classmethod
    def _get_items(cls, items):
        if get_origin(type) == list:
            return cls._convert_type(get_args(type)[0])
        return None

    @classmethod
    def _extract_params(cls, lines: list[str]):
        output = {}
        cur_key = ""
        for line in lines:
            split_line = line.split(":")
            if len(split_line) == 2:
                cur_key = split_line[0]
                output[cur_key] = split_line[1]
            else:
                output[cur_key] += split_line[0]
        return output

    @classmethod
    def from_agent(cls, agent: "AgentBase"):
        func = agent.answer
        description: str = agent.description
        parameters_ = {
            "question": Property(
                type="string",
                description="the question you want answered by the agent",
                enum=None,
            ),
        }
        return GenericTool(
            description=description,
            name="answer",
            parameters=parameters_,
            required=[
                key
                for key, value in inspect.signature(func).parameters.items()
                if value.default == inspect._empty
            ],
            func=agent.answer,
        )

    @classmethod
    def from_func(cls, func: Callable[..., Any]):
        description: str = func.__doc__ or ""
        schema: FunctionSchema = get_function_schema(func)
        parameters_ = {
            name: Property(
                type=type_["type"],
                description=type_["description"],
                enum=type_.get("enum"),
                properties=type_.get("properties"),  # type: ignore
                required=type_.get("required"),
                items=type_.get("items"),  # type: ignore
            )
            for name, type_ in schema["parameters"]["properties"].items()
        }

        return GenericTool(
            description=description,
            name=schema["name"],
            parameters=parameters_,
            required=[
                key
                for key, value in inspect.signature(func).parameters.items()
                if value.default == inspect._empty
            ],
            func=func,
        )

    def to_openai(self):
        from ..openai.tool import Function, Parameters, Property
        from ..openai.tool import Tool as Tool

        return Tool(
            type="function",
            function=Function(
                description=self.description,
                name=self.name,
                parameters=Parameters(
                    properties={
                        key: Property(**param.model_dump(exclude_none=True))
                        for key, param in self.parameters.items()
                    },
                    required=self.required,
                ),
            ),
            strict=True,
        )

    def to_anthropic(self):
        from ..anthropic.tool import Property, Tool, ToolSchema

        return Tool(
            name=self.name,
            description=self.description,
            input_schema=ToolSchema(
                type="object",
                properties={
                    key: Property(**param.model_dump(exclude_none=True))
                    for key, param in self.parameters.items()
                },
                required=self.required,
            ),
        )

    def __repr__(self):
        description = self.description.strip().replace("\n", " ")[:100]
        if len(description) >= 50:
            description = description[:50] + "..."
        return f'<GenericTool name="{self.name}" description="{description}">'

    __str__ = __repr__
