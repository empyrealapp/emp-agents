from pydantic import BaseModel, Field


class Property(BaseModel):
    type: str | None = None
    description: str
    enum: list[str] | None = None
    items: dict[str, "str | Property"] | None = None
    properties: dict[str, "Property"] | None = None
    required: list[str] | None = None


class Parameters(BaseModel):
    type: str = Field(default="object")
    properties: dict[str, Property]
    required: list[str] | None = None

    @classmethod
    def build(cls, type: str = "object", **kwargs):
        return cls(type=type, **kwargs)


class Function(BaseModel):
    description: str
    name: str
    parameters: Parameters | None
    # required: list[str] | None = Field(default=None)
    additional_properties: bool = False


class Tool(BaseModel):
    type: str = Field(default="function")
    function: Function
    strict: bool = True

    @classmethod
    def build(cls, type="function", **kwargs):
        return cls(type=type, **kwargs)
