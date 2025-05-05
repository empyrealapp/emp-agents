import os
from typing import TYPE_CHECKING, Dict, Optional
from pydantic import Field, PrivateAttr

from emp_agents.managers.base import PromptManager
from emp_agents.models import UserMessage, SystemMessage, Request
from emp_agents.providers.groq import GroqProvider

if TYPE_CHECKING:
    from emp_agents.agents.base import AgentBase

if TYPE_CHECKING:
    from emp_agents.agents.base import AgentBase


class NoOpPromptManager(PromptManager):
    """
    Default prompt manager that does not modify the prompt.
    
    This is the default implementation that simply returns the current prompt
    without modification, effectively doing nothing.
    """
    
    def update_prompt(self, 
                      agent: 'AgentBase', 
                      message: UserMessage, 
                      current_prompt: str) -> str:
        """Return the current prompt unchanged."""
        return current_prompt
        
    async def update_prompt_async(self, 
                      agent: 'AgentBase', 
                      message: UserMessage, 
                      current_prompt: str) -> str:
        """Async version that returns the current prompt unchanged."""
        return current_prompt

    
class GroqPromptManager(PromptManager):
    """
    Prompt manager powered by Groq's fast inference API.
    
    This manager uses Groq's fast inference capabilities to dynamically
    adjust the agent's system prompt based on conversation context.
    """
    
    api_key: str = Field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))
    model: str = Field(default="llama3-8b-8192")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=300)
    base_prompt: str = Field(default="You are a helpful assistant")
    
    _provider: GroqProvider = PrivateAttr(default=None)
    
    def model_post_init(self, __context):
        """Initialize the Groq provider"""
        self._provider = GroqProvider(
            api_key=self.api_key,
            default_model=self.model
        )
    
    def update_prompt(self, 
                      agent: 'AgentBase',
                      message: UserMessage, 
                      current_prompt: str) -> str:
        """
        Dynamically update the system prompt based on the message using Groq.
        
        This sync version is a fallback that returns the current prompt in
        async contexts to avoid event loop conflicts. Use update_prompt_async
        in async contexts.
        """
        # In async contexts, we can't safely run the async code
        # Try to detect if we're in an async context
        try:
            # Try to get the current event loop - if this fails, we're not in an event loop
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're in a running event loop, but that will cause problems
                # with nested loops. Return current prompt as a fallback.
                return current_prompt
            else:
                # If we have an event loop but it's not running, use it
                result = loop.run_until_complete(self.update_prompt_async(agent, message, current_prompt))
                return result
        except RuntimeError:
            # No event loop exists, create one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.update_prompt_async(agent, message, current_prompt))
            return result
    
    async def update_prompt_async(self, 
                      agent: 'AgentBase',
                      message: UserMessage, 
                      current_prompt: str) -> str:
        """
        Async version of update_prompt that can be used in async contexts.
        
        Dynamically update the system prompt based on the message using Groq.
        """
        # Create a focused prompt for prompt adjustment
        from emp_agents.models import SystemMessage
        
        prompt_adjustment_prompt = f"""
        You are an AI system that adjusts the system prompt for another assistant.
        
        Current system prompt:
        "{current_prompt}"
        
        Latest user message:
        "{message.content}"
        
        Your task: Determine if the system prompt needs adjustment to better address the user's message.
        
        Rules:
        1. If no changes are needed, respond with the EXACT original prompt.
        2. If changes are needed, create an improved prompt that helps the assistant better address the user's needs.
        3. Changes should be minimal but effective.
        4. The prompt should be complete and ready to use.
        
        Respond with ONLY the new system prompt, no explanation:
        """
        
        # Create request for Groq
        request = Request(
            model=self.model,
            messages=[SystemMessage(content=prompt_adjustment_prompt)],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        # Get the response - directly await it since we're in an async context
        response = await self._provider.completion(request)
        new_prompt = response.text.strip()
        
        # If the response is too short, return the original prompt
        if len(new_prompt) < 10:
            return current_prompt
            
        return new_prompt