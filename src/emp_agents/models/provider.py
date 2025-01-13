from abc import abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel

from .shared.request import Request
from .shared.message import Message, ToolCall


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


class Provider(BaseModel, Generic[Response]):
    api_key: str | None = None

    def _load_model(self, model: str | None) -> str:
        if model is None:
            model = self.default_model()
        return model

    @abstractmethod
    def default_model(self) -> str: ...

    @abstractmethod
    async def _run_conversation(self, messages: list[Message]) -> list[Message]: ...

    @abstractmethod
    async def completion(self, request: Request) -> Response: ...
