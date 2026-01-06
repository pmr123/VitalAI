"""
VitalAI Multi-Agent System

This package contains specialized AI agents for different health-related tasks:
- SupervisorAgent: Routes queries to appropriate specialist agents
- HealthAnalystAgent: Analyzes health data and provides insights
- MedicationManagerAgent: Manages medications and reminders
- KnowledgeExpertAgent: Answers medical knowledge questions via RAG
- DigitalCloneAgent: Personalized AI assistant that learns user preferences
"""

from .base_agent import BaseAgent, AgentType, AgentResponse
from .supervisor_agent import SupervisorAgent
from .health_analyst_agent import HealthAnalystAgent
from .medication_agent import MedicationManagerAgent
from .knowledge_agent import KnowledgeExpertAgent
from .digital_clone_agent import DigitalCloneAgent
from .onboarding_agent import OnboardingAgent
from .orchestrator import AgentOrchestrator, get_orchestrator

__all__ = [
    "BaseAgent",
    "AgentType", 
    "AgentResponse",
    "SupervisorAgent",
    "HealthAnalystAgent",
    "MedicationManagerAgent",
    "KnowledgeExpertAgent",
    "DigitalCloneAgent",
    "OnboardingAgent",
    "AgentOrchestrator",
    "get_orchestrator"
]

