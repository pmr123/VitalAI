"""
MCP (Model Context Protocol) Server for VitalAI Health Platform.

This module implements an MCP-compatible server that:
1. Exposes available tools via a discovery endpoint
2. Executes tools via a call endpoint
3. Returns results in a standardized format

MCP Protocol Overview:
- Tools are functions that LLMs can call to perform actions
- Each tool has a schema describing its parameters
- The server validates inputs and returns structured outputs
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json
import logging
from datetime import datetime

from services.mcp_tools import (
    list_tools, 
    get_tool, 
    execute_tool,
    TOOL_REGISTRY
)

logger = logging.getLogger(__name__)


@dataclass
class MCPRequest:
    """Represents an MCP tool call request."""
    tool_name: str
    arguments: Dict[str, Any]
    request_id: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MCPRequest':
        return cls(
            tool_name=data.get('tool_name', data.get('name', '')),
            arguments=data.get('arguments', data.get('parameters', {})),
            request_id=data.get('request_id', data.get('id'))
        )


@dataclass 
class MCPResponse:
    """Represents an MCP tool call response."""
    success: bool
    result: Any
    error: Optional[str] = None
    request_id: Optional[str] = None
    execution_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        response = {
            "success": self.success,
            "result": self.result
        }
        if self.error:
            response["error"] = self.error
        if self.request_id:
            response["request_id"] = self.request_id
        if self.execution_time_ms is not None:
            response["execution_time_ms"] = self.execution_time_ms
        return response


class MCPServer:
    """
    MCP Server that handles tool discovery and execution.
    
    Usage:
        server = MCPServer()
        
        # Get available tools
        tools = server.list_tools()
        
        # Execute a tool
        result = server.execute("get_health_summary", {"user_id": 1})
    """
    
    def __init__(self):
        self.name = "VitalAI MCP Server"
        self.version = "1.0.0"
        self.description = "Health AI platform tools for managing health data, medications, and reminders"
        logger.info(f"MCP Server initialized: {self.name} v{self.version}")
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server metadata."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "protocol_version": "1.0",
            "capabilities": {
                "tools": True,
                "resources": False,  # Not implemented yet
                "prompts": False     # Not implemented yet
            },
            "tool_count": len(TOOL_REGISTRY)
        }
    
    def list_tools(self) -> List[Dict[str, Any]]:
        """
        List all available tools with their schemas.
        
        Returns a list of tool definitions that can be shown to an LLM
        so it knows what tools are available and how to call them.
        """
        return list_tools()
    
    def get_tool_schema(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """Get the schema for a specific tool."""
        tool = get_tool(tool_name)
        if tool:
            return tool.to_schema()
        return None
    
    def execute(self, tool_name: str, arguments: Dict[str, Any]) -> MCPResponse:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Dictionary of arguments to pass to the tool
            
        Returns:
            MCPResponse with the result or error
        """
        start_time = datetime.now()
        
        # Validate tool exists
        tool = get_tool(tool_name)
        if not tool:
            return MCPResponse(
                success=False,
                result=None,
                error=f"Tool not found: {tool_name}. Available tools: {list(TOOL_REGISTRY.keys())}"
            )
        
        # Execute the tool
        try:
            result = execute_tool(tool_name, **arguments)
            
            # Check if result contains an error
            if isinstance(result, dict) and "error" in result:
                return MCPResponse(
                    success=False,
                    result=result,
                    error=result["error"],
                    execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
                )
            
            return MCPResponse(
                success=True,
                result=result,
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
            
        except Exception as e:
            logger.error(f"Error executing tool {tool_name}: {e}", exc_info=True)
            return MCPResponse(
                success=False,
                result=None,
                error=str(e),
                execution_time_ms=(datetime.now() - start_time).total_seconds() * 1000
            )
    
    def handle_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle an incoming MCP request.
        
        This is the main entry point for MCP protocol requests.
        Supports both single requests and batch requests.
        """
        # Check if it's a batch request
        if isinstance(request_data, list):
            return [self._handle_single_request(req) for req in request_data]
        
        return self._handle_single_request(request_data)
    
    def _handle_single_request(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a single MCP request."""
        request = MCPRequest.from_dict(request_data)
        
        response = self.execute(request.tool_name, request.arguments)
        response.request_id = request.request_id
        
        return response.to_dict()
    
    def format_tools_for_llm(self) -> str:
        """
        Format tool descriptions for inclusion in LLM system prompt.
        
        This creates a human-readable description of available tools
        that helps the LLM understand what it can do.
        """
        tools = self.list_tools()
        
        lines = ["## Available Tools\n"]
        lines.append("You can use the following tools to help users:\n")
        
        for tool in tools:
            lines.append(f"### {tool['name']}")
            lines.append(f"{tool['description']}\n")
            
            if tool['parameters']['properties']:
                lines.append("**Parameters:**")
                for param_name, param_info in tool['parameters']['properties'].items():
                    required = param_name in tool['parameters'].get('required', [])
                    req_str = " (required)" if required else " (optional)"
                    lines.append(f"- `{param_name}`: {param_info['description']}{req_str}")
            lines.append("")
        
        return "\n".join(lines)
    
    def format_tools_for_ollama(self) -> List[Dict[str, Any]]:
        """
        Format tools in Ollama's expected tool format.
        
        Ollama expects tools in a specific format for function calling.
        """
        tools = self.list_tools()
        ollama_tools = []
        
        for tool in tools:
            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            ollama_tools.append(ollama_tool)
        
        return ollama_tools


# =============================================================================
# Global MCP Server Instance
# =============================================================================

_mcp_server: Optional[MCPServer] = None


def get_mcp_server() -> MCPServer:
    """Get or create the global MCP server instance."""
    global _mcp_server
    if _mcp_server is None:
        _mcp_server = MCPServer()
    return _mcp_server

