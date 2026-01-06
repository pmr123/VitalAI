"""
Onboarding Service - Orchestrates the user onboarding flow.

Manages the conversation state, collects user information, and creates
the user profile when onboarding is complete.
"""

from typing import Dict, Any, Optional
import logging
from models import User
from extensions import db
from services.agents.onboarding_agent import OnboardingAgent

logger = logging.getLogger(__name__)


class OnboardingService:
    """Service to manage user onboarding flow"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.agent = OnboardingAgent(user_id)
        self._conversation_history = []
    
    def process_message(self, message: str) -> Dict[str, Any]:
        """
        Process a user message during onboarding
        
        Args:
            message: User's message/answer
        
        Returns:
            Dict with agent response and onboarding state
        """
        # Get user info for context
        user = User.query.get(self.user_id)
        user_name = user.name if user else "there"
        
        # Build context with current onboarding state and user info
        context = {
            "onboarding_state": self.agent.get_collected_data(),
            "user_name": user_name
        }
        
        # Process with agent
        response = self.agent.process(message, context=context)
        
        # Get updated state
        onboarding_state = response.metadata.get("onboarding_state", {})
        step = response.metadata.get("step", "unknown")
        
        # Check if completed - MUST have all required fields
        is_complete = self._is_complete(onboarding_state)
        
        # Only use fallback completion detection if we're very close (have 3 out of 4 required fields)
        # This prevents premature completion
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg']
        collected_count = sum(1 for field in required_fields 
                             if field in onboarding_state 
                             and onboarding_state[field] is not None)
        
        response_lower = response.content.lower()
        response_indicates_completion = any(phrase in response_lower for phrase in [
            'all the information',
            'complete your profile',
            'ready to be created',
            'profile setup',
            "you're all set",
            'account is now active',
            'i have all the information i need',
            'let me complete your profile'
        ])
        
        # Only allow fallback completion if we have at least 3 of 4 required fields
        # This prevents premature completion when only 1-2 fields are collected
        allow_fallback = collected_count >= 3
        
        # IMPORTANT: Only complete if:
        # 1. We have all required fields AND
        # 2. The agent explicitly says it's ready to complete (not just that it has required fields)
        # This gives the agent a chance to ask about optional fields first
        final_completed = is_complete and response_indicates_completion
        
        # If we have all fields but agent hasn't said it's ready, don't complete yet
        # This allows the agent to ask about optional fields
        if is_complete and not response_indicates_completion:
            logger.debug(f"All required fields collected but agent hasn't indicated completion yet. Allowing optional field collection.")
            final_completed = False
        
        if final_completed and step != "confirmation":
            step = "confirmation"
        
        logger.info(f"Onboarding completion check: is_complete={is_complete}, response_indicates={response_indicates_completion}, collected={collected_count}/4, final={final_completed}")
        
        return {
            "response": response.content,  # AgentResponse uses 'content' field
            "onboarding_state": onboarding_state,
            "step": step,
            "completed": final_completed  # Only complete if we have all fields AND agent says it's ready
        }
    
    def _is_complete(self, state: Dict[str, Any]) -> bool:
        """Check if onboarding is complete (all required fields collected)"""
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg']
        is_complete = all(
            field in state 
            and state[field] is not None 
            and state[field] != '' 
            for field in required_fields
        )
        logger.debug(f"Onboarding completion check: {is_complete}, state: {state}")
        return is_complete
    
    def complete_onboarding(self, final_data: Optional[Dict[str, Any]] = None) -> User:
        """
        Complete onboarding and create/update user profile
        
        Args:
            final_data: Optional final data to merge with collected data
        
        Returns:
            Updated User object
        """
        user = User.query.get(self.user_id)
        if not user:
            raise ValueError(f"User {self.user_id} not found")
        
        # Get all collected data
        collected_data = self.agent.get_collected_data()
        if final_data:
            collected_data.update(final_data)
        
        # Update user profile - ensure all fields are properly set with type conversion
        if 'age' in collected_data and collected_data['age'] is not None:
            user.age = int(collected_data['age'])
        if 'gender' in collected_data and collected_data['gender'] is not None:
            user.gender = str(collected_data['gender'])
        if 'height_cm' in collected_data and collected_data['height_cm'] is not None:
            user.height_cm = float(collected_data['height_cm'])
        if 'weight_kg' in collected_data and collected_data['weight_kg'] is not None:
            user.weight_kg = float(collected_data['weight_kg'])
        if 'preferred_language' in collected_data and collected_data['preferred_language'] is not None:
            user.preferred_language = str(collected_data['preferred_language'])
        if 'health_goals' in collected_data and collected_data['health_goals'] is not None:
            user.health_goals = collected_data['health_goals'] if isinstance(collected_data['health_goals'], list) else [collected_data['health_goals']]
        if 'conditions' in collected_data and collected_data['conditions'] is not None:
            user.conditions = collected_data['conditions'] if isinstance(collected_data['conditions'], list) else [collected_data['conditions']]
        if 'allergies' in collected_data and collected_data['allergies'] is not None:
            user.allergies = collected_data['allergies'] if isinstance(collected_data['allergies'], list) else [collected_data['allergies']]
        if 'daily_step_goal' in collected_data and collected_data['daily_step_goal'] is not None:
            user.daily_step_goal = int(collected_data['daily_step_goal'])
        
        # Mark onboarding as complete
        user.onboarding_completed = True
        
        db.session.add(user)
        db.session.commit()
        logger.info(f"Completed onboarding for user {self.user_id}: age={user.age}, gender={user.gender}, height={user.height_cm}cm, weight={user.weight_kg}kg, BMI={user.bmi}")
        
        return user
    
    def get_state(self) -> Dict[str, Any]:
        """Get current onboarding state"""
        return {
            "onboarding_state": self.agent.get_collected_data(),
            "step": self.agent._determine_step(),
            "completed": self._is_complete(self.agent.get_collected_data())
        }
    
    def reset(self):
        """Reset onboarding state"""
        self.agent.reset_state()
        self._conversation_history = []


# Global instances (per user)
_onboarding_services = {}


def get_onboarding_service(user_id: int) -> OnboardingService:
    """Get or create onboarding service for a user"""
    if user_id not in _onboarding_services:
        _onboarding_services[user_id] = OnboardingService(user_id)
    return _onboarding_services[user_id]

