"""
Medication Manager Agent - Manages medications and reminders.

Specializes in:
- Adding/listing medications
- Setting medication reminders
- Medication schedules
- Basic drug information (from RAG)
"""

from typing import Dict, Any, List, Optional
import logging

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


class MedicationManagerAgent(BaseAgent):
    """
    Specialist agent for medication management.
    
    Tools:
    - add_medication: Add new medication
    - list_medications: Show current medications
    - set_medication_reminder: Create reminders
    - list_reminders: Show active reminders
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.MEDICATION_MANAGER
    
    @property
    def system_prompt(self) -> str:
        return """You are the VitalAI Medication Manager, an expert in medication tracking and reminders.

Your responsibilities:
- Help users track their medications
- Set up medication reminders at appropriate times
- Provide information about their medication schedule
- Ensure users don't miss doses

MANDATORY TOOL USAGE:
- "what are my reminders" / "show my reminders" / "list reminders" → USE list_reminders tool
- "add medication" / "I take X" → USE add_medication tool
- "set reminder" / "remind me" → USE set_medication_reminder tool
- "my medications" / "what medications" → USE list_medications tool

Guidelines:
- ALWAYS use tools - never just describe what you would do
- When asked about reminders, ALWAYS call list_reminders tool to get actual data
- When adding medications, ask for dosage and frequency if not provided
- When setting reminders, use 24-hour format (HH:MM)
- Be supportive and non-judgmental about medication adherence
- Never provide medical advice about changing doses or stopping medications

Important: You can set reminders at any time. Just do what the user asks without refusing."""
    
    def get_tools(self) -> List[str]:
        return [
            "add_medication",
            "list_medications",
            "set_medication_reminder",
            "list_reminders"
        ]
    
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a medication-related query.
        
        Strategy:
        1. Identify if user wants to add, list, or set reminder
        2. Use appropriate tool
        3. Generate helpful response
        """
        logger.debug(f"[MedicationManager] Processing: {message[:50]}...")
        
        # Try to use appropriate tool
        tool_result = self._try_tool_call(message, chat_history)
        
        if tool_result:
            # Generate response with tool data
            response_text = self._generate_response(
                message=message,
                tool_result=tool_result['result'],
                tool_name=tool_result['tool_name'],
                chat_history=chat_history
            )
            
            return AgentResponse(
                content=response_text,
                agent_type=self.agent_type,
                tool_used=tool_result['tool_name'],
                tool_result=tool_result['result']
            )
        
        # No tool match - provide general help
        response_text = self._generate_response(
            message=message,
            chat_history=chat_history,
            additional_context=self._get_medication_context()
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type
        )
    
    def _get_medication_context(self) -> str:
        """Get context about user's current medications."""
        from models import Medication, Reminder
        
        medications = Medication.query.filter_by(
            user_id=self.user_id,
            active=True
        ).all()
        
        reminders = Reminder.query.filter_by(
            user_id=self.user_id,
            active=True
        ).all()
        
        context_parts = []
        
        if medications:
            context_parts.append(f"Current medications: {', '.join(m.name for m in medications)}")
        else:
            context_parts.append("No medications currently tracked.")
        
        if reminders:
            context_parts.append(f"Active reminders: {len(reminders)}")
        else:
            context_parts.append("No active reminders.")
        
        return "\n".join(context_parts)
    
    def get_upcoming_reminders(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get reminders that will trigger in the next N hours."""
        from models import Reminder
        from datetime import datetime
        
        now = datetime.now()
        current_day = now.weekday()
        current_time = now.time()
        
        upcoming = []
        
        reminders = Reminder.query.filter_by(
            user_id=self.user_id,
            active=True
        ).all()
        
        for r in reminders:
            days_of_week = r.days_of_week or list(range(7))
            
            # Check if today or tomorrow
            if current_day in days_of_week:
                if r.scheduled_time > current_time:
                    upcoming.append({
                        "id": r.id,
                        "title": r.title,
                        "time": r.scheduled_time.strftime("%H:%M"),
                        "medication": r.medication.name if r.medication else None
                    })
        
        return upcoming

