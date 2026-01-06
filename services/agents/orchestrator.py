"""
Agent Orchestrator - Coordinates the multi-agent system.

The orchestrator:
1. Receives user messages
2. Uses SupervisorAgent to classify intent and route
3. Dispatches to appropriate specialist agent
4. Handles multi-step queries requiring multiple agents
5. Aggregates responses if needed
"""

from typing import Dict, Any, List, Optional
import logging

from .base_agent import AgentType, AgentResponse
from .supervisor_agent import SupervisorAgent
from .health_analyst_agent import HealthAnalystAgent
from .medication_agent import MedicationManagerAgent
from .knowledge_agent import KnowledgeExpertAgent
from .digital_clone_agent import DigitalCloneAgent

logger = logging.getLogger(__name__)


class AgentOrchestrator:
    """
    Central coordinator for the multi-agent system.
    
    Flow:
    1. User message → Supervisor (intent classification)
    2. Supervisor routes → Specialist Agent
    3. Specialist processes → Response
    4. Orchestrator returns → Final response with metadata
    """
    
    def __init__(self, user_id: int = 1):
        self.user_id = user_id
        self._agents: Dict[AgentType, Any] = {}
        self._conversation_history: List[Dict] = []
        self._max_history = 10  # Keep last N exchanges
    
    @property
    def supervisor(self) -> SupervisorAgent:
        """Get or create the supervisor agent."""
        if AgentType.SUPERVISOR not in self._agents:
            self._agents[AgentType.SUPERVISOR] = SupervisorAgent(self.user_id)
        return self._agents[AgentType.SUPERVISOR]
    
    def get_agent(self, agent_type: AgentType):
        """Get or create an agent of the specified type."""
        if agent_type not in self._agents:
            agent_classes = {
                AgentType.SUPERVISOR: SupervisorAgent,
                AgentType.HEALTH_ANALYST: HealthAnalystAgent,
                AgentType.MEDICATION_MANAGER: MedicationManagerAgent,
                AgentType.KNOWLEDGE_EXPERT: KnowledgeExpertAgent,
                AgentType.DIGITAL_CLONE: DigitalCloneAgent
            }
            
            agent_class = agent_classes.get(agent_type)
            if agent_class:
                self._agents[agent_type] = agent_class(self.user_id)
            else:
                raise ValueError(f"Unknown agent type: {agent_type}")
        
        return self._agents[agent_type]
    
    def process(
        self,
        message: str,
        chat_history: Optional[List[Dict]] = None,
        force_agent: Optional[AgentType] = None
    ) -> AgentResponse:
        """
        Process a user message through the multi-agent system.
        
        Args:
            message: User's input message
            chat_history: External chat history (if any)
            force_agent: Force routing to a specific agent (bypass supervisor)
            
        Returns:
            AgentResponse from the handling agent
        """
        # Combine external and internal history
        history = self._get_combined_history(chat_history)
        
        logger.debug(f"[Orchestrator] Processing message: {message[:50]}...")
        
        # Step 1: Determine which agent should handle this
        if force_agent:
            target_agent_type = force_agent
            routing_info = {"forced": True, "agent": force_agent.value}
        else:
            # Use supervisor to classify and route
            routing = self.supervisor.classify_intent(message, history)
            target_agent_type = routing["primary_agent"]
            routing_info = routing
            
            logger.debug(f"[Orchestrator] Routing to: {target_agent_type.value} "
                       f"(confidence: {routing['confidence']:.2f})")
        
        # Step 2: Get the appropriate specialist agent
        agent = self.get_agent(target_agent_type)
        
        # Step 3: Process the message with the specialist
        response = agent.process(
            message=message,
            chat_history=history,
            context={"routing": routing_info}
        )
        
        # Step 4: Store in conversation history
        self._update_history(message, response)
        
        # Step 5: Add routing metadata to response
        response.metadata["routing"] = routing_info
        
        logger.debug(f"[Orchestrator] Response from {response.agent_type.value}, "
                   f"tool_used={response.tool_used}")
        
        return response
    
    def process_multi_agent(
        self,
        message: str,
        agents: List[AgentType],
        chat_history: Optional[List[Dict]] = None
    ) -> List[AgentResponse]:
        """
        Process a message with multiple agents and aggregate responses.
        
        Useful for complex queries like:
        "What's my CMI score (Health Analyst) and how can I improve it (Knowledge Expert)?"
        
        Args:
            message: User's input message
            agents: List of agent types to query
            chat_history: External chat history
            
        Returns:
            List of responses from each agent
        """
        history = self._get_combined_history(chat_history)
        responses = []
        
        for agent_type in agents:
            agent = self.get_agent(agent_type)
            response = agent.process(
                message=message,
                chat_history=history,
                context={"multi_agent": True, "agents": [a.value for a in agents]}
            )
            responses.append(response)
        
        return responses
    
    def aggregate_responses(self, responses: List[AgentResponse]) -> AgentResponse:
        """
        Aggregate multiple agent responses into a single coherent response.
        
        Args:
            responses: List of responses from different agents
            
        Returns:
            Combined AgentResponse
        """
        if not responses:
            return AgentResponse(
                content="I couldn't process your request.",
                agent_type=AgentType.SUPERVISOR
            )
        
        if len(responses) == 1:
            return responses[0]
        
        # Combine contents
        combined_parts = []
        all_sources = []
        tools_used = []
        
        for resp in responses:
            combined_parts.append(f"**{resp.agent_type.value.replace('_', ' ').title()}:**\n{resp.content}")
            all_sources.extend(resp.sources)
            if resp.tool_used:
                tools_used.append(resp.tool_used)
        
        return AgentResponse(
            content="\n\n".join(combined_parts),
            agent_type=AgentType.SUPERVISOR,  # Aggregated by supervisor
            sources=all_sources,
            metadata={
                "aggregated": True,
                "agents_used": [r.agent_type.value for r in responses],
                "tools_used": tools_used
            }
        )
    
    def _get_combined_history(
        self, 
        external_history: Optional[List[Dict]] = None
    ) -> List[Dict]:
        """Combine external history with internal conversation memory."""
        combined = []
        
        if external_history:
            combined.extend(external_history)
        
        combined.extend(self._conversation_history)
        
        # Limit total history
        return combined[-self._max_history * 2:]  # *2 for user+assistant pairs
    
    def _update_history(self, user_message: str, response: AgentResponse) -> None:
        """Update internal conversation history."""
        self._conversation_history.append({
            "role": "user",
            "content": user_message
        })
        self._conversation_history.append({
            "role": "assistant",
            "content": response.content,
            "agent": response.agent_type.value
        })
        
        # Trim if too long
        if len(self._conversation_history) > self._max_history * 2:
            self._conversation_history = self._conversation_history[-self._max_history * 2:]
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self._conversation_history = []
    
    def get_agent_info(self) -> List[Dict[str, Any]]:
        """Get information about all available agents."""
        return [
            {
                "type": AgentType.SUPERVISOR.value,
                "name": "Supervisor",
                "description": "Routes queries to specialist agents"
            },
            {
                "type": AgentType.HEALTH_ANALYST.value,
                "name": "Health Analyst",
                "description": "Analyzes health data, scores, and metrics"
            },
            {
                "type": AgentType.MEDICATION_MANAGER.value,
                "name": "Medication Manager",
                "description": "Manages medications and reminders"
            },
            {
                "type": AgentType.KNOWLEDGE_EXPERT.value,
                "name": "Knowledge Expert",
                "description": "Answers medical and health questions"
            },
            {
                "type": AgentType.DIGITAL_CLONE.value,
                "name": "Digital Clone",
                "description": "Personalized AI assistant"
            }
        ]


# Global orchestrator instance
_orchestrator: Optional[AgentOrchestrator] = None


def get_orchestrator(user_id: int = 1) -> AgentOrchestrator:
    """Get or create the global orchestrator instance."""
    global _orchestrator
    if _orchestrator is None or _orchestrator.user_id != user_id:
        _orchestrator = AgentOrchestrator(user_id)
    return _orchestrator


def reset_orchestrator() -> None:
    """Reset the global orchestrator instance."""
    global _orchestrator
    _orchestrator = None

