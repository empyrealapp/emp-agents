from abc import abstractmethod
from typing import TypeVar

from pydantic import BaseModel

from .shared import Request, Message, ToolCall
from ..types import TCompletionAgent


class ResponseT(BaseModel):
    @property
    @abstractmethod
    def text(self) -> str:
        """The text from the API response"""

    @property
    @abstractmethod
    def messages(self) -> list[Message]:
        """The API response messages"""

    @property
    @abstractmethod
    def tool_calls(self) -> list[ToolCall]: ...


Response = TypeVar("Response", bound=ResponseT)


class Provider(BaseModel, TCompletionAgent[Response]):
    api_key: str | None = None
    default_model: str | None = None

    def _load_model(self, model: str | None) -> str:
        if model is None:
            model = self.default_model
        if not model:
            raise ValueError("No model provided")
        return model

    @abstractmethod
    async def completion(self, request: Request) -> Response: ...
