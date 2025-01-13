from typing import Callable, Awaitable

from pydantic import BaseModel

from emp_agents.models.shared import Message


class Middleware(BaseModel):
    name: str
    description: str
    function: Callable[[list[Message]], Awaitable[list[Message]] | list[Message]]
