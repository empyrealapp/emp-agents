from typing import Dict, List, Set, TYPE_CHECKING, Optional
import os
import asyncio
import json
from pydantic import Field, PrivateAttr

from emp_agents.managers.base import ToolManager
from emp_agents.models import GenericTool, UserMessage, SystemMessage, Request
from emp_agents.types.mcp import MCPClient, SSEParams
from emp_agents.exceptions import DuplicateToolException
from emp_agents.providers.groq import GroqProvider

if TYPE_CHECKING:
    from emp_agents.agents.base import AgentBase


class NoOpToolManager(ToolManager):
    """
    Default tool manager that does not modify the tools.
    
    This is the default implementation that simply returns the current tools
    without modification, effectively doing nothing.
    """
    
    def update_tools(self, 
                     agent: 'AgentBase', 
                     message: UserMessage, 
                     current_tools: List[GenericTool]) -> List[GenericTool]:
        """Return the current tools unchanged."""
        return current_tools
        
    async def update_tools_async(self, 
                     agent: 'AgentBase', 
                     message: UserMessage, 
                     current_tools: List[GenericTool]) -> List[GenericTool]:
        """Async version that returns the current tools unchanged."""
        return current_tools




class GroqMCPToolManager(ToolManager):
    """
    Tool manager that uses Groq to decide which MCP servers to connect to.
    
    This manager uses Groq's fast inference API to analyze messages and
    dynamically connect to or disconnect from MCP servers based on the
    conversation context, allowing for efficient management of tool groups.
    """
    
    api_key: str = Field(default_factory=lambda: os.environ.get("GROQ_API_KEY"))
    model: str = Field(default="llama3-8b-8192")
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=100)
    
    available_mcp_servers: Dict[str, str] = Field(default_factory=dict)
    
    server_descriptions: Dict[str, str] = Field(default_factory=dict)
    
    default_connect_all: bool = Field(default=False)
    
    _provider: Optional[GroqProvider] = PrivateAttr(default=None)
    _connected_servers: Set[str] = PrivateAttr(default_factory=set)
    _server_tools: Dict[str, List[GenericTool]] = PrivateAttr(default_factory=dict)
    
    def model_post_init(self, __context):
        """Initialize the Groq provider"""
        self._provider = GroqProvider(
            api_key=self.api_key,
            default_model=self.model
        )
        
        if not self.server_descriptions:
            self.server_descriptions = {
                name: f"Tools from {name} server" 
                for name in self.available_mcp_servers.keys()
            }
            
    def update_tools(self, 
                    agent: 'AgentBase', 
                    message: UserMessage, 
                    current_tools: List[GenericTool]) -> List[GenericTool]:
        """
        Analyze the message and connect to appropriate MCP servers.
        
        This method creates a new event loop to run the async version when called
        from a synchronous context. In async contexts, use update_tools_async instead.
        
        Args:
            agent: The agent instance
            message: The latest user message
            current_tools: The current list of tools
            
        Returns:
            An updated list of tools
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return current_tools
            else:
                result = loop.run_until_complete(self.update_tools_async(agent, message, current_tools))
                return result
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(self.update_tools_async(agent, message, current_tools))
            return result
    
    async def update_tools_async(self, 
                    agent: 'AgentBase', 
                    message: UserMessage, 
                    current_tools: List[GenericTool]) -> List[GenericTool]:
        """
        Async version of update_tools that can be used in async contexts.
        
        Analyzes the message and connects to appropriate MCP servers.
        
        Args:
            agent: The agent instance
            message: The latest user message
            current_tools: The current list of tools
            
        Returns:
            An updated list of tools
        """
        if not self.available_mcp_servers:
            return current_tools
        
        server_descriptions = "\n".join([
            f"- {name}: {desc}" 
            for name, desc in self.server_descriptions.items()
        ])
        
        selection_prompt = f"""
        Based on this user message: "{message.content}"
        
        The following tool servers are available:
        {server_descriptions}
        
        Choose which servers would provide tools most useful for answering this message.
        Return ONLY a JSON list of server names, like: ["math", "weather"]
        If no servers are relevant, return an empty list: []
        """
        
        request = Request(
            model=self.model,
            messages=[SystemMessage(content=selection_prompt)],
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        
        response = await self._provider.completion(request)
        response_text = response.text.strip()
        
        try:
            server_names = json.loads(response_text)
            
            if not isinstance(server_names, list) or not all(isinstance(name, str) for name in server_names):
                if self.default_connect_all:
                    server_names = list(self.available_mcp_servers.keys())
                else:
                    return current_tools
            
            valid_server_names = [
                name for name in server_names 
                if name in self.available_mcp_servers
            ]
            
            if not valid_server_names and self.default_connect_all:
                valid_server_names = list(self.available_mcp_servers.keys())
            
            await self._update_server_connections(agent, valid_server_names)
            
        except (json.JSONDecodeError, ValueError):
            if self.default_connect_all:
                await self._update_server_connections(agent, list(self.available_mcp_servers.keys()))
        
        return agent._tools
    
    async def _update_server_connections(self, 
                                        agent: 'AgentBase', 
                                        server_names: List[str]):
        """
        Update MCP server connections based on the selected server names.
        
        Args:
            agent: The agent instance
            server_names: List of server names to connect to
        """
        servers_to_connect = {
            name: self.available_mcp_servers[name] 
            for name in server_names 
            if name in self.available_mcp_servers
        }
        
        for server_name in list(self._connected_servers):
            if server_name not in servers_to_connect:
                await self._disconnect_from_server(agent, server_name)
        
        for server_name, url in servers_to_connect.items():
            if server_name not in self._connected_servers:
                await self._connect_to_server(agent, server_name, url)
    
    async def _connect_to_server(self, agent: 'AgentBase', server_name: str, url: str):
        """
        Connect to an MCP server and add its tools.
        
        Args:
            agent: The agent instance
            server_name: Name of the server
            url: URL of the server
        """
        mcp_client = MCPClient(params=SSEParams(url=url))
        
        client_urls = [client.params.url for client in agent._mcp_clients]
        if url not in client_urls:
            await mcp_client._create_session()
            agent._mcp_clients.append(mcp_client)
        else:
            for client in agent._mcp_clients:
                if client.params.url == url:
                    mcp_client = client
                    break
        
        tools = await mcp_client.list_tools()
        
        self._server_tools[server_name] = tools.copy()
        
        for tool in tools:
            try:
                agent._add_tool(tool)
            except DuplicateToolException:
                pass
            
        self._connected_servers.add(server_name)
    
    async def _disconnect_from_server(self, agent: 'AgentBase', server_name: str):
        """
        Disconnect from an MCP server and remove its tools.
        
        Args:
            agent: The agent instance
            server_name: Name of the server
        """
        server_tools = self._server_tools.get(server_name, [])
        
        tool_names = [tool.name for tool in server_tools]
        agent._tools = [tool for tool in agent._tools if tool.name not in tool_names]
        
        for tool_name in tool_names:
            if tool_name in agent._tools_map:
                del agent._tools_map[tool_name]
        
        server_url = self.available_mcp_servers.get(server_name)
        if server_url:
            clients_to_remove = []
            for client in agent._mcp_clients:
                if client.params.url == server_url:
                    await client.close()
                    clients_to_remove.append(client)
            
            agent._mcp_clients = [
                client for client in agent._mcp_clients 
                if client not in clients_to_remove
            ]
        
        if server_name in self._server_tools:
            del self._server_tools[server_name]
            
        self._connected_servers.remove(server_name)