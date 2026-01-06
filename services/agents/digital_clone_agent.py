"""
Digital Clone Agent - Personalized AI assistant that learns user preferences.

Specializes in:
- Remembering user preferences and patterns
- Providing personalized health advice
- Learning from past interactions
- Adapting communication style to user
"""

from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import logging
import json

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


class DigitalCloneAgent(BaseAgent):
    """
    Personalized AI assistant that maintains a "digital clone" of user preferences.
    
    Features:
    - Stores user preferences and patterns
    - Provides personalized recommendations
    - Remembers past conversations context
    - Adapts advice based on user history
    """
    
    def __init__(self, user_id: int = 1):
        super().__init__(user_id)
        self._user_profile_cache = None
        self._preferences_cache = None
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.DIGITAL_CLONE
    
    @property
    def system_prompt(self) -> str:
        return """You are the user's personal VitalAI Clone - a personalized health companion that knows them well.

Your personality:
- You speak as if you truly know the user personally
- You remember their preferences, patterns, and history
- You give advice tailored specifically to them
- You're supportive, encouraging, and understanding

Personalization approach:
- Reference their health goals and progress
- Consider their health conditions when giving advice
- Adapt suggestions to their lifestyle patterns
- Use their name naturally in conversation
- Remember what has worked (or hasn't) for them before

You're not just an AI - you're THEIR AI, customized to help them specifically."""
    
    def get_tools(self) -> List[str]:
        # Digital clone can use all data tools for personalization
        return [
            "get_health_summary",
            "get_health_metrics",
            "get_user_profile",
            "list_medications",
            "list_reminders",
            "search_knowledge_base"
        ]
    
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process message with full personalization context.
        
        Strategy:
        1. Load comprehensive user profile
        2. Analyze patterns from health data
        3. Generate highly personalized response
        """
        logger.debug(f"[DigitalClone] Processing: {message[:50]}...")
        
        # Build comprehensive personalization context
        personalization = self._build_personalization_context()
        
        # Check if message needs tool data
        tool_result = self._try_tool_call(message, chat_history)
        
        if tool_result:
            response_text = self._generate_personalized_response(
                message=message,
                tool_result=tool_result['result'],
                tool_name=tool_result['tool_name'],
                personalization=personalization,
                chat_history=chat_history
            )
            
            return AgentResponse(
                content=response_text,
                agent_type=self.agent_type,
                tool_used=tool_result['tool_name'],
                tool_result=tool_result['result'],
                metadata={"personalization": "full"}
            )
        
        # Generate personalized response without specific tool
        response_text = self._generate_personalized_response(
            message=message,
            personalization=personalization,
            chat_history=chat_history
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type,
            metadata={"personalization": "full"}
        )
    
    def _build_personalization_context(self) -> Dict[str, Any]:
        """
        Build comprehensive context about the user for personalization.
        
        Includes:
        - Profile info (name, age, goals, conditions)
        - Current health scores
        - Recent health trends
        - Medication history
        - Behavioral patterns
        """
        from models import User, HealthData, Medication, Reminder
        from services.health_scoring import calculate_health_summary
        
        context = {
            "profile": {},
            "health_summary": {},
            "patterns": {},
            "medications": [],
            "preferences": {}
        }
        
        # Get user profile
        user = User.query.get(self.user_id)
        if user:
            context["profile"] = {
                "name": user.name,
                "age": user.age,
                "gender": user.gender,
                "health_goals": user.health_goals or [],
                "conditions": user.conditions or [],
                "preferred_language": user.preferred_language
            }
        
        # Get health summary
        summary = calculate_health_summary(self.user_id)
        context["health_summary"] = summary.get("scores", {})
        
        # Analyze patterns
        context["patterns"] = self._analyze_user_patterns()
        
        # Get medications
        meds = Medication.query.filter_by(user_id=self.user_id, active=True).all()
        context["medications"] = [m.name for m in meds]
        
        # Infer preferences (could be stored in DB later)
        context["preferences"] = self._infer_preferences()
        
        return context
    
    def _analyze_user_patterns(self) -> Dict[str, Any]:
        """
        Analyze user's health data patterns.
        
        Identifies:
        - Sleep patterns (good/poor sleeper)
        - Activity levels (active/sedentary)
        - Consistency (regular/irregular patterns)
        """
        from models import HealthData, MetricType
        
        patterns = {
            "sleep_quality": "unknown",
            "activity_level": "unknown",
            "consistency": "unknown"
        }
        
        week_ago = datetime.utcnow() - timedelta(days=7)
        
        # Analyze sleep
        sleep_data = HealthData.query.filter(
            HealthData.user_id == self.user_id,
            HealthData.metric_type == MetricType.SLEEP_SCORE,
            HealthData.timestamp >= week_ago
        ).all()
        
        if sleep_data:
            avg_sleep = sum(d.value for d in sleep_data) / len(sleep_data)
            patterns["sleep_quality"] = "good" if avg_sleep >= 70 else "needs_improvement"
        
        # Analyze activity
        step_data = HealthData.query.filter(
            HealthData.user_id == self.user_id,
            HealthData.metric_type == MetricType.STEPS,
            HealthData.timestamp >= week_ago
        ).all()
        
        if step_data:
            avg_steps = sum(d.value for d in step_data) / len(step_data)
            if avg_steps >= 10000:
                patterns["activity_level"] = "very_active"
            elif avg_steps >= 7000:
                patterns["activity_level"] = "active"
            elif avg_steps >= 4000:
                patterns["activity_level"] = "moderate"
            else:
                patterns["activity_level"] = "sedentary"
        
        return patterns
    
    def _infer_preferences(self) -> Dict[str, Any]:
        """
        Infer user preferences from their profile and behavior.
        
        In production, this would be learned and stored.
        """
        from models import User
        
        user = User.query.get(self.user_id)
        if not user:
            return {}
        
        preferences = {
            "communication_style": "friendly",  # Could be formal/casual/friendly
            "detail_level": "moderate",  # brief/moderate/detailed
            "motivation_style": "encouraging"  # encouraging/direct/analytical
        }
        
        # Infer from health goals
        if user.health_goals:
            goals = [g.lower() for g in user.health_goals]
            if any("weight" in g for g in goals):
                preferences["focus_areas"] = ["weight", "activity"]
            if any("sleep" in g for g in goals):
                preferences["focus_areas"] = preferences.get("focus_areas", []) + ["sleep"]
        
        return preferences
    
    def _generate_personalized_response(
        self,
        message: str,
        personalization: Dict[str, Any],
        tool_result: Optional[Dict[str, Any]] = None,
        tool_name: Optional[str] = None,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate a highly personalized response."""
        from services.llm_service import get_llm_service
        
        llm = get_llm_service()
        
        # Build personalized system prompt
        system_prompt = self._build_personalized_system_prompt(personalization)
        
        # Build message with context
        user_content = message
        if tool_result:
            user_content = f"""User asked: {message}

Relevant data:
{json.dumps(tool_result, indent=2)}

Provide a personalized response based on this data and what you know about the user."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            messages.extend(chat_history[-4:])
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = llm.client.chat(
                model=llm.model,
                messages=messages,
                options={"temperature": 0.8, "top_p": 0.9}  # Slightly higher temp for personality
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating personalized response: {e}")
            return "I'm having trouble connecting right now. Let me try again."
    
    def _build_personalized_system_prompt(self, personalization: Dict[str, Any]) -> str:
        """Build system prompt with personalization context."""
        profile = personalization.get("profile", {})
        patterns = personalization.get("patterns", {})
        health = personalization.get("health_summary", {})
        
        prompt_parts = [self.system_prompt]
        
        # Add user-specific context
        if profile.get("name"):
            prompt_parts.append(f"\nUser's name: {profile['name']}")
        
        if profile.get("age"):
            prompt_parts.append(f"Age: {profile['age']}")
        
        if profile.get("health_goals"):
            prompt_parts.append(f"Health Goals: {', '.join(profile['health_goals'])}")
        
        if profile.get("conditions"):
            prompt_parts.append(f"Health Conditions: {', '.join(profile['conditions'])}")
        
        if health:
            prompt_parts.append(f"\nCurrent Scores: CMI={health.get('cmi', 'N/A')}, "
                              f"Cardiac={health.get('cardiac', 'N/A')}, "
                              f"Activity={health.get('activity', 'N/A')}")
        
        if patterns.get("sleep_quality") != "unknown":
            prompt_parts.append(f"Sleep Pattern: {patterns['sleep_quality']}")
        
        if patterns.get("activity_level") != "unknown":
            prompt_parts.append(f"Activity Level: {patterns['activity_level']}")
        
        medications = personalization.get("medications", [])
        if medications:
            prompt_parts.append(f"Current Medications: {', '.join(medications)}")
        
        prompt_parts.append("\nUse this information to make your responses highly personalized and relevant.")
        
        return "\n".join(prompt_parts)
    
    def remember_preference(self, key: str, value: Any) -> None:
        """
        Store a user preference for future use.
        
        In production, this would persist to the database.
        """
        # TODO: Implement persistent preference storage
        logger.debug(f"[DigitalClone] Remembering preference: {key}={value}")
    
    def get_personalized_greeting(self) -> str:
        """Generate a personalized greeting based on time and user data."""
        from datetime import datetime
        
        hour = datetime.now().hour
        personalization = self._build_personalization_context()
        name = personalization.get("profile", {}).get("name", "there")
        
        if hour < 12:
            time_greeting = "Good morning"
        elif hour < 17:
            time_greeting = "Good afternoon"
        else:
            time_greeting = "Good evening"
        
        return f"{time_greeting}, {name}! How can I help you today?"

