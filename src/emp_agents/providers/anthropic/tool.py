from pydantic import BaseModel, Field


class Property(BaseModel):
    type: str | None = None
    description: str
    enum: list[str] | None = None
    items: dict[str, "str | Property"] | None = None
    properties: dict[str, "Property"] | None = None
    required: list[str] | None = None


class ToolSchema(BaseModel):
    type: str = "object"
    properties: dict[str, Property]
    required: list[str] = Field(default_factory=list)


class Tool(BaseModel):
    name: str
    description: str
    input_schema: ToolSchema
