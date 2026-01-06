"""
Chat Service - Combines RAG, Tools, Agents, and LLM for intelligent health conversations.

This service orchestrates:
1. Multi-agent routing (new in Phase 4)
2. RAG retrieval for knowledge base queries
3. Tool calling for user-specific data (health metrics, medications)
4. LLM generation for natural responses
"""
from typing import Optional, Generator, Dict, Any
from datetime import datetime
import logging

from services.rag_service import get_rag_service
from services.llm_service import get_llm_service
from services.health_scoring import calculate_health_summary
from services.mcp_server import get_mcp_server
from models import User, HealthData, MetricType
from extensions import db

logger = logging.getLogger(__name__)


class ChatService:
    """
    Orchestrates RAG-powered chat conversations with tool calling and multi-agent support.
    
    Flow (with agents):
    1. User sends message
    2. Orchestrator routes to appropriate specialist agent
    3. Agent processes and uses tools if needed
    4. Response returned with agent metadata
    
    Flow (without agents - legacy mode):
    1. User sends message
    2. LLM decides if it needs a tool or can answer directly
    3. If tool needed: execute tool, then generate response with results
    4. If no tool: use RAG context + LLM to generate response
    """
    
    def __init__(self):
        self.rag = get_rag_service()
        self.llm = get_llm_service()
        self.mcp = get_mcp_server()
        self._orchestrator = None
    
    @property
    def orchestrator(self):
        """Get or create the agent orchestrator."""
        if self._orchestrator is None:
            from services.agents import get_orchestrator
            self._orchestrator = get_orchestrator()
        return self._orchestrator
    
    def chat(
        self, 
        user_id: int, 
        message: str,
        chat_history: Optional[list] = None,
        use_rag: bool = True,
        use_tools: bool = True,
        use_agents: bool = True
    ) -> dict:
        """
        Process a chat message and return response.
        
        Args:
            user_id: The user's ID
            message: User's message
            chat_history: Previous messages for context
            use_rag: Whether to search knowledge base
            use_tools: Whether to allow tool calling
            use_agents: Whether to use the multi-agent system (Phase 4)
        
        Returns:
            Dict with response, agent info, and metadata
        """
        result = {
            'response': '',
            'sources': [],
            'tool_used': None,
            'agent': None,
            'agent_type': None,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        }
        
        # Use multi-agent system if enabled
        if use_agents:
            try:
                agent_response = self._process_with_agents(user_id, message, chat_history)
                
                result['response'] = agent_response.content
                result['agent'] = agent_response.agent_type.value
                result['agent_type'] = agent_response.agent_type.value.replace('_', ' ').title()
                result['tool_used'] = agent_response.tool_used
                result['sources'] = agent_response.sources
                
                # Convert routing metadata to JSON-serializable format
                routing = agent_response.metadata.get('routing', {})
                if routing:
                    result['routing'] = self._serialize_routing(routing)
                
                return result
                
            except Exception as e:
                logger.error(f"Agent system error, falling back to legacy: {e}")
                import traceback
                traceback.print_exc()
                # Fall through to legacy system
        
        # Legacy system (Phase 2/3 approach)
        # First, try tool-calling approach if enabled
        if use_tools:
            tool_result = self._try_tool_call(user_id, message, chat_history)
            if tool_result:
                result['response'] = tool_result['response']
                result['tool_used'] = tool_result['tool_name']
                return result
        
        # Fall back to RAG-based response
        rag_context = ""
        rag_sources = []
        
        if use_rag:
            results = self.rag.search(message, top_k=2)
            if results:
                rag_context = self.rag.get_context(message, top_k=2)
                rag_sources = [
                    {
                        'title': r['metadata'].get('title', 'Unknown'),
                        'source': r['metadata'].get('source', 'Unknown'),
                        'relevance': round(r['relevance'], 2)
                    }
                    for r in results
                ]
        
        # Get user health data context
        user_data = self._get_user_context(user_id)
        
        # Generate response using RAG
        response_text = self.llm.chat(
            message=message,
            context=rag_context,
            chat_history=chat_history,
            user_data=user_data
        )
        
        result['response'] = response_text
        result['sources'] = rag_sources
        return result
    
    def _process_with_agents(
        self, 
        user_id: int, 
        message: str, 
        chat_history: Optional[list]
    ):
        """
        Process message using the multi-agent system.
        
        The orchestrator handles:
        1. Intent classification via Supervisor
        2. Routing to specialist agent
        3. Tool execution if needed
        4. Response generation
        """
        from app import app
        from services.agents import get_orchestrator, AgentResponse
        
        with app.app_context():
            # Get orchestrator for this user
            orchestrator = get_orchestrator(user_id)
            
            # Process through agent system
            response = orchestrator.process(
                message=message,
                chat_history=chat_history
            )
            
            logger.info(f"Agent response from {response.agent_type.value}")
            
            return response
    
    def _try_tool_call(
        self, 
        user_id: int, 
        message: str, 
        chat_history: Optional[list]
    ) -> Optional[Dict[str, Any]]:
        """
        Try to handle the message using tool calling (legacy approach).
        
        Returns None if no tool is appropriate for the message.
        """
        from app import app
        
        with app.app_context():
            # Get tools in Ollama format
            tools = self.mcp.format_tools_for_ollama()
            
            # Get basic user context
            user_data = self._get_user_context(user_id)
            
            # Ask LLM if it wants to use a tool
            llm_result = self.llm.chat_with_tools(
                message=message,
                tools=tools,
                chat_history=chat_history,
                user_data=user_data
            )
            
            logger.info(f"LLM tool response type: {llm_result.get('type')}")
            if llm_result.get('type') == 'text':
                logger.info(f"LLM responded with text (no tool call): {llm_result.get('response', '')[:100]}...")
            
            if llm_result.get('type') == 'tool_call':
                tool_call = llm_result['tool_call']
                tool_name = tool_call['name']
                arguments = tool_call.get('arguments', {})
                
                logger.info(f"Tool call detected: {tool_name} with raw args: {arguments}")
                
                # ALWAYS set user_id to the correct integer value
                # LLM might pass wrong values like user names instead of IDs
                arguments['user_id'] = user_id
                
                logger.info(f"Executing tool: {tool_name} with fixed args: {arguments}")
                
                # Execute the tool
                tool_response = self.mcp.execute(tool_name, arguments)
                
                if tool_response.success:
                    # Generate natural language response from tool result
                    response_text = self.llm.generate_response_with_tool_result(
                        original_message=message,
                        tool_name=tool_name,
                        tool_result=tool_response.result,
                        chat_history=chat_history
                    )
                    
                    return {
                        'response': response_text,
                        'tool_name': tool_name,
                        'tool_result': tool_response.result
                    }
                else:
                    logger.warning(f"Tool execution failed: {tool_response.error}")
            
            return None
    
    def chat_stream(
        self, 
        user_id: int, 
        message: str,
        chat_history: Optional[list] = None,
        use_rag: bool = True
    ) -> Generator[str, None, None]:
        """
        Stream a chat response.
        Yields response chunks as they're generated.
        """
        # Get RAG context
        rag_context = ""
        if use_rag:
            rag_context = self.rag.get_context(message, top_k=3)
        
        # Get user health data context
        user_data = self._get_user_context(user_id)
        
        # Stream response
        for chunk in self.llm.chat_stream(
            message=message,
            context=rag_context,
            chat_history=chat_history,
            user_data=user_data
        ):
            yield chunk
    
    def chat_with_agent(
        self,
        user_id: int,
        message: str,
        agent_type: str,
        chat_history: Optional[list] = None
    ) -> dict:
        """
        Process a message with a specific agent (bypassing supervisor routing).
        
        Args:
            user_id: User's ID
            message: User's message
            agent_type: Type of agent to use (e.g., "health_analyst")
            chat_history: Previous messages
            
        Returns:
            Dict with response and metadata
        """
        from app import app
        from services.agents import get_orchestrator, AgentType
        
        # Map string to AgentType
        agent_type_map = {
            'health_analyst': AgentType.HEALTH_ANALYST,
            'medication_manager': AgentType.MEDICATION_MANAGER,
            'knowledge_expert': AgentType.KNOWLEDGE_EXPERT,
            'digital_clone': AgentType.DIGITAL_CLONE
        }
        
        target_agent = agent_type_map.get(agent_type.lower())
        if not target_agent:
            return {
                'error': f"Unknown agent type: {agent_type}",
                'valid_agents': list(agent_type_map.keys())
            }
        
        with app.app_context():
            orchestrator = get_orchestrator(user_id)
            response = orchestrator.process(
                message=message,
                chat_history=chat_history,
                force_agent=target_agent
            )
            
            return {
                'response': response.content,
                'agent': response.agent_type.value,
                'agent_type': response.agent_type.value.replace('_', ' ').title(),
                'tool_used': response.tool_used,
                'sources': response.sources,
                'timestamp': response.timestamp.isoformat()
            }
    
    def _get_user_context(self, user_id: int) -> dict:
        """Get user's health context for personalization."""
        from app import app
        
        with app.app_context():
            user = User.query.get(user_id)
            if not user:
                return {}
            
            # Get health summary
            health_summary = calculate_health_summary(user_id)
            
            # Get latest metrics
            latest_metrics = self._get_latest_metrics(user_id)
            
            return {
                'user_profile': {
                    'name': user.name,
                    'age': user.age,
                    'conditions': user.conditions or [],
                    'health_goals': user.health_goals or []
                },
                'scores': health_summary.get('scores', {}),
                'latest_metrics': latest_metrics
            }
    
    def _get_latest_metrics(self, user_id: int) -> dict:
        """Get user's latest health metrics."""
        metrics = {}
        
        metric_types = [
            MetricType.RESTING_HR,
            MetricType.STEPS,
            MetricType.SLEEP_DURATION,
            MetricType.SLEEP_SCORE,
            MetricType.HRV
        ]
        
        for metric_type in metric_types:
            latest = HealthData.query.filter_by(
                user_id=user_id,
                metric_type=metric_type
            ).order_by(HealthData.timestamp.desc()).first()
            
            if latest:
                metrics[metric_type] = f"{latest.value} {latest.unit or ''}"
        
        return metrics
    
    def _serialize_routing(self, routing: dict) -> dict:
        """Convert routing dict to JSON-serializable format."""
        from enum import Enum
        
        serialized = {}
        for key, value in routing.items():
            if isinstance(value, Enum):
                serialized[key] = value.value
            elif isinstance(value, dict):
                serialized[key] = self._serialize_routing(value)
            else:
                serialized[key] = value
        return serialized


# Global instance
_chat_service: Optional[ChatService] = None


def get_chat_service() -> ChatService:
    """Get or create the global chat service instance."""
    global _chat_service
    if _chat_service is None:
        _chat_service = ChatService()
    return _chat_service
