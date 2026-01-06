"""
Base Agent Class and Types for the Multi-Agent System.

All specialized agents inherit from BaseAgent and implement:
- process(): Handle a user message and return a response
- get_tools(): Return the tools this agent can use
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class AgentType(Enum):
    """Types of agents in the system."""
    SUPERVISOR = "supervisor"
    HEALTH_ANALYST = "health_analyst"
    MEDICATION_MANAGER = "medication_manager"
    KNOWLEDGE_EXPERT = "knowledge_expert"
    DIGITAL_CLONE = "digital_clone"


@dataclass
class AgentResponse:
    """Response from an agent."""
    content: str
    agent_type: AgentType
    tool_used: Optional[str] = None
    tool_result: Optional[Dict[str, Any]] = None
    sources: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "content": self.content,
            "agent": self.agent_type.value,
            "tool_used": self.tool_used,
            "sources": self.sources,
            "confidence": self.confidence,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Each agent:
    - Has a specific type and system prompt
    - Can use a subset of available tools
    - Processes messages and returns structured responses
    """
    
    def __init__(self, user_id: int = 1):
        self.user_id = user_id
        self._llm_service = None
        self._mcp_server = None
    
    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        raise NotImplementedError
    
    @property
    def name(self) -> str:
        """Human-readable name of the agent."""
        return self.agent_type.value.replace("_", " ").title()
    
    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """System prompt defining the agent's behavior and expertise."""
        pass
    
    @abstractmethod
    def get_tools(self) -> List[str]:
        """Return list of tool names this agent can use."""
        pass
    
    @property
    def llm_service(self):
        """Get the LLM service (lazy initialization)."""
        if self._llm_service is None:
            from services.llm_service import get_llm_service
            self._llm_service = get_llm_service()
        return self._llm_service
    
    @property
    def mcp_server(self):
        """Get the MCP server (lazy initialization)."""
        if self._mcp_server is None:
            from services.mcp_server import get_mcp_server
            self._mcp_server = get_mcp_server()
        return self._mcp_server
    
    def get_formatted_tools(self) -> List[Dict[str, Any]]:
        """Get this agent's tools in Ollama format."""
        all_tools = self.mcp_server.format_tools_for_ollama()
        my_tool_names = self.get_tools()
        return [t for t in all_tools if t['function']['name'] in my_tool_names]
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get user context for personalization."""
        from models import User
        from services.health_scoring import calculate_health_summary
        
        user = User.query.get(self.user_id)
        if not user:
            return {}
        
        summary = calculate_health_summary(self.user_id)
        
        return {
            'user_profile': {
                'name': user.name,
                'age': user.age,
                'conditions': user.conditions or [],
                'health_goals': user.health_goals or []
            },
            'scores': summary.get('scores', {})
        }
    
    @abstractmethod
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's input message
            chat_history: Previous messages in the conversation
            context: Additional context (e.g., routed from supervisor)
            
        Returns:
            AgentResponse with the agent's answer
        """
        pass
    
    def _execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a tool and return the result."""
        # Always ensure user_id is set correctly
        arguments['user_id'] = self.user_id
        
        result = self.mcp_server.execute(tool_name, arguments)
        
        if result.success:
            return result.result
        else:
            logger.error(f"Tool {tool_name} failed: {result.error}")
            return {"error": result.error}
    
    def _generate_response(
        self,
        message: str,
        tool_result: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None,
        additional_context: str = ""
    ) -> str:
        """Generate a natural language response using the LLM."""
        
        if tool_result:
            # Generate response with tool result
            return self.llm_service.generate_response_with_tool_result(
                original_message=message,
                tool_name=tool_name,
                tool_result=tool_result,
                chat_history=chat_history
            )
        else:
            # Generate response without tool (direct LLM)
            user_context = self.get_user_context()
            return self.llm_service.chat(
                message=message,
                context=additional_context,
                chat_history=chat_history,
                user_data=user_context
            )
    
    def _try_tool_call(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Try to determine if a tool should be called and execute it.
        
        Returns dict with tool_name and result if successful, None otherwise.
        """
        tools = self.get_formatted_tools()
        if not tools:
            return None
        
        user_context = self.get_user_context()
        
        # Ask LLM if it wants to use a tool
        llm_result = self.llm_service.chat_with_tools(
            message=message,
            tools=tools,
            chat_history=chat_history,
            user_data=user_context
        )
        
        if llm_result.get('type') == 'tool_call':
            tool_call = llm_result['tool_call']
            tool_name = tool_call['name']
            arguments = tool_call.get('arguments', {})
            
            logger.debug(f"[{self.name}] Executing tool: {tool_name}")
            
            result = self._execute_tool(tool_name, arguments)
            
            if 'error' not in result:
                return {
                    'tool_name': tool_name,
                    'result': result
                }
        
        return None
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(user_id={self.user_id})>"

