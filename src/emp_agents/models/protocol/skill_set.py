from typing import ClassVar

from pydantic import BaseModel, PrivateAttr

from emp_agents.models import GenericTool
from emp_agents.models.protocol.registry import ToolRegistry


class SkillSet(BaseModel):
    _tools_map: ClassVar[dict[str, GenericTool]] = PrivateAttr(default_factory=dict)
    _tools: ClassVar[list[GenericTool]] = PrivateAttr(default_factory=list)

    async def setup(self):
        """Any setup commands post init"""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        cls._tools_map = {}

        items = cls.__dict__.values()
        for method in items:
            if hasattr(method, "_is_tool_method"):
                cls._tools_map[method.__name__] = GenericTool.from_func(method)

        cls._tools = list(cls._tools_map.values())
        ToolRegistry.register_class(cls)

    def __iter__(self):
        return iter(self._tools)
