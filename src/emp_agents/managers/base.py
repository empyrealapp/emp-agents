from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from pydantic import BaseModel

from emp_agents.models import GenericTool, UserMessage

if TYPE_CHECKING:
    from emp_agents.agents.base import AgentBase

class ToolManager(BaseModel, ABC):
    """
    Abstract base class for tool managers.
    
    Tool managers are responsible for dynamically updating the tools available
    to an agent based on the conversation context.
    """
    
    @abstractmethod
    def update_tools(self, 
                     agent: 'AgentBase', 
                     message: UserMessage, 
                     current_tools: List[GenericTool]) -> List[GenericTool]:
        pass
        
    async def update_tools_async(self, 
                     agent: 'AgentBase', 
                     message: UserMessage, 
                     current_tools: List[GenericTool]) -> List[GenericTool]:
        """
        Async version of update_tools for use in async contexts.
        
        By default, just calls the sync version. Child classes can override this
        with a true async implementation.
        """
        return self.update_tools(agent, message, current_tools)


class PromptManager(BaseModel, ABC):
    """
    Abstract base class for prompt managers.
    
    Prompt managers are responsible for dynamically updating the system prompt
    used by an agent based on the conversation context.
    """
    
    @abstractmethod
    def update_prompt(self, 
                      agent: 'AgentBase', 
                      message: UserMessage, 
                      current_prompt: str) -> str:
        pass
        
    async def update_prompt_async(self, 
                      agent: 'AgentBase', 
                      message: UserMessage, 
                      current_prompt: str) -> str:
        """
        Async version of update_prompt for use in async contexts.
        
        By default, just calls the sync version. Child classes can override this
        with a true async implementation.
        """
        return self.update_prompt(agent, message, current_prompt)