"""
Health Analyst Agent - Analyzes health data and provides insights.

Specializes in:
- Health scores (CMI, cardiac, activity, recovery, metabolic)
- Health metrics (heart rate, HRV, steps, sleep, etc.)
- Trend analysis and comparisons
- Health data interpretation
"""

from typing import Dict, Any, List, Optional
import logging

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


class HealthAnalystAgent(BaseAgent):
    """
    Specialist agent for health data analysis.
    
    Tools:
    - get_health_summary: Overall health scores
    - get_health_metrics: Specific metric data
    - get_user_profile: User info for context
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.HEALTH_ANALYST
    
    @property
    def system_prompt(self) -> str:
        return """You are the VitalAI Health Analyst, an expert in interpreting health data.

Your expertise:
- Understanding health scores (CMI, cardiac, activity, recovery, metabolic)
- Analyzing health metrics (heart rate, HRV, sleep, steps, calories)
- Identifying trends and patterns in health data
- Explaining what health numbers mean in simple terms

Guidelines:
- Always use tools to get actual data - never make up numbers
- Explain scores in context (what's good, what needs attention)
- Compare to healthy ranges when relevant
- Be encouraging while being honest about areas for improvement
- Use data to tell the user's health story

Remember: You have access to the user's real health data. Use it to provide personalized, accurate insights."""
    
    def get_tools(self) -> List[str]:
        return [
            "get_health_summary",
            "get_health_metrics", 
            "get_user_profile"
        ]
    
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a health data query.
        
        Strategy:
        1. Use tools to get relevant health data
        2. Generate insightful response based on the data
        """
        logger.debug(f"[HealthAnalyst] Processing: {message[:50]}...")
        
        # Try to use appropriate tool
        tool_result = self._try_tool_call(message, chat_history)
        
        if tool_result:
            tool_name = tool_result['tool_name']
            logger.debug(f"[HealthAnalyst] Tool {tool_name} was called successfully")
            
            # Generate response with tool data
            response_text = self._generate_response(
                message=message,
                tool_result=tool_result['result'],
                tool_name=tool_name,
                chat_history=chat_history
            )
            
            return AgentResponse(
                content=response_text,
                agent_type=self.agent_type,
                tool_used=tool_name,  # Explicitly set tool_used
                tool_result=tool_result['result']
            )
        else:
            logger.debug(f"[HealthAnalyst] No tool was called for this message")
        
        # No tool needed - provide general health advice
        user_context = self.get_user_context()
        context_str = self._format_health_context(user_context)
        
        response_text = self._generate_response(
            message=message,
            chat_history=chat_history,
            additional_context=context_str
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type
        )
    
    def _format_health_context(self, user_context: Dict[str, Any]) -> str:
        """Format user health context for the LLM prompt."""
        if not user_context:
            return ""
        
        parts = []
        
        scores = user_context.get('scores', {})
        if scores:
            parts.append("Current Health Scores:")
            for name, value in scores.items():
                if value is not None:
                    parts.append(f"  - {name.upper()}: {value}/100")
        
        profile = user_context.get('user_profile', {})
        if profile:
            if profile.get('health_goals'):
                parts.append(f"Health Goals: {', '.join(profile['health_goals'])}")
            if profile.get('conditions'):
                parts.append(f"Conditions: {', '.join(profile['conditions'])}")
        
        return "\n".join(parts)
    
    def analyze_trends(
        self, 
        metric_type: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Analyze trends in a specific metric.
        
        Returns trend direction, averages, and notable patterns.
        """
        from models import HealthData
        from datetime import datetime, timedelta
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        data = HealthData.query.filter(
            HealthData.user_id == self.user_id,
            HealthData.metric_type == metric_type,
            HealthData.timestamp >= start_date
        ).order_by(HealthData.timestamp).all()
        
        if len(data) < 2:
            return {"trend": "insufficient_data", "message": "Not enough data for trend analysis"}
        
        values = [d.value for d in data]
        
        # Calculate basic trend
        first_half = sum(values[:len(values)//2]) / (len(values)//2)
        second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
        
        if second_half > first_half * 1.05:
            trend = "increasing"
        elif second_half < first_half * 0.95:
            trend = "decreasing"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "average": round(sum(values) / len(values), 2),
            "min": min(values),
            "max": max(values),
            "data_points": len(values),
            "period_days": days
        }

