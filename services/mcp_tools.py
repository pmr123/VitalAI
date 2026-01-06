"""
MCP Tool Definitions and Implementations for VitalAI Health Platform.

This module defines tools that can be called by the LLM to perform actions
and retrieve data. Each tool has:
- name: Unique identifier
- description: What the tool does (shown to LLM)
- parameters: JSON schema of expected inputs
- handler: Function that executes the tool
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import json
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Type Conversion Utilities (LLM might pass strings instead of proper types)
# =============================================================================

def safe_int(value: Any, default: int = 0) -> int:
    """Safely convert a value to int."""
    if value is None:
        return default
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def safe_bool(value: Any, default: bool = True) -> bool:
    """Safely convert a value to bool."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes', 'on')
    try:
        return bool(value)
    except:
        return default


def safe_list(value: Any, default: List = None) -> List:
    """Safely convert a value to list (handles JSON strings)."""
    if default is None:
        default = []
    if value is None:
        return default
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
    return default


def safe_int_list(value: Any, default: List[int] = None) -> List[int]:
    """Safely convert a value to list of integers."""
    if default is None:
        default = []
    lst = safe_list(value, default)
    try:
        return [int(x) for x in lst]
    except (ValueError, TypeError):
        return default


@dataclass
class ToolParameter:
    """Definition of a tool parameter."""
    name: str
    type: str  # "string", "integer", "number", "boolean", "array", "object"
    description: str
    required: bool = True
    enum: Optional[List[str]] = None
    default: Any = None


@dataclass
class Tool:
    """Definition of an MCP tool."""
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    handler: Optional[Callable] = None
    
    def to_schema(self) -> Dict[str, Any]:
        """Convert to JSON schema format for LLM consumption."""
        properties = {}
        required = []
        
        for param in self.parameters:
            prop = {
                "type": param.type,
                "description": param.description
            }
            if param.enum:
                prop["enum"] = param.enum
            if param.default is not None:
                prop["default"] = param.default
            
            properties[param.name] = prop
            
            if param.required:
                required.append(param.name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class HealthTools:
    """Health-related tools for the MCP server."""
    
    @staticmethod
    def get_health_summary(user_id: int) -> Dict[str, Any]:
        """Get a summary of user's health scores."""
        from models import User
        from services.health_scoring import calculate_health_summary
        
        user_id = safe_int(user_id, 1)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Calculate scores dynamically (same as dashboard)
        summary = calculate_health_summary(user_id)
        
        if not summary.get('scores'):
            return {
                "user_name": user.name,
                "message": "No health data available yet. Generate some test data first.",
                "scores": None
            }
        
        scores = summary.get('scores', {})
        return {
            "user_name": user.name,
            "date": summary.get('date', datetime.now().strftime('%Y-%m-%d')),
            "scores": {
                "cmi": scores.get('cmi'),
                "cardiac": scores.get('cardiac'),
                "activity": scores.get('activity'),
                "recovery": scores.get('recovery'),
                "metabolic": scores.get('metabolic')
            },
            "score_interpretation": {
                "cmi": HealthTools._interpret_score(scores.get('cmi'), "CMI"),
                "cardiac": HealthTools._interpret_score(scores.get('cardiac'), "Cardiac"),
                "activity": HealthTools._interpret_score(scores.get('activity'), "Activity"),
                "recovery": HealthTools._interpret_score(scores.get('recovery'), "Recovery"),
                "metabolic": HealthTools._interpret_score(scores.get('metabolic'), "Metabolic")
            }
        }
    
    @staticmethod
    def _interpret_score(score: Optional[int], name: str) -> str:
        """Interpret a health score value."""
        if score is None:
            return "Not available"
        if score >= 80:
            return f"Excellent {name.lower()} health"
        elif score >= 60:
            return f"Good {name.lower()} health"
        elif score >= 40:
            return f"Fair {name.lower()} health - room for improvement"
        else:
            return f"Needs attention - consider lifestyle changes"
    
    @staticmethod
    def get_health_metrics(
        user_id: int, 
        metric_type: str, 
        days: int = 7
    ) -> Dict[str, Any]:
        """Get specific health metrics for a user."""
        from models import User, HealthData, MetricType
        from extensions import db
        
        user_id = safe_int(user_id, 1)
        days = safe_int(days, 7)
        
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Valid metric types (from MetricType class constants)
        valid_metrics = [
            MetricType.HEART_RATE, MetricType.RESTING_HR, MetricType.HRV,
            MetricType.STEPS, MetricType.CALORIES, MetricType.ACTIVE_MINUTES,
            MetricType.SLEEP_DURATION, MetricType.SLEEP_DEEP, MetricType.SLEEP_SCORE,
            MetricType.SPO2, MetricType.STRESS_LEVEL
        ]
        if metric_type not in valid_metrics:
            return {
                "error": f"Invalid metric type: {metric_type}",
                "valid_types": valid_metrics
            }
        
        # Get data for the specified period
        start_date = datetime.utcnow() - timedelta(days=days)
        
        data = HealthData.query.filter(
            HealthData.user_id == user_id,
            HealthData.metric_type == metric_type,
            HealthData.timestamp >= start_date
        ).order_by(HealthData.timestamp.desc()).all()
        
        if not data:
            return {
                "metric_type": metric_type,
                "days": days,
                "message": "No data found for this metric and period",
                "records": []
            }
        
        # Calculate stats
        values = [d.value for d in data]
        
        return {
            "metric_type": metric_type,
            "days": days,
            "record_count": len(data),
            "latest": {
                "value": data[0].value,
                "unit": data[0].unit,
                "timestamp": data[0].timestamp.isoformat()
            },
            "statistics": {
                "average": round(sum(values) / len(values), 2),
                "min": min(values),
                "max": max(values)
            },
            "recent_values": [
                {"value": d.value, "timestamp": d.timestamp.isoformat()}
                for d in data[:10]  # Last 10 records
            ]
        }
    
    @staticmethod
    def get_user_profile(user_id: int) -> Dict[str, Any]:
        """Get user profile information."""
        from models import User
        
        user_id = safe_int(user_id, 1)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Calculate BMI if height and weight available
        bmi = None
        bmi_category = None
        if user.height_cm and user.weight_kg:
            height_m = user.height_cm / 100
            bmi = round(user.weight_kg / (height_m ** 2), 1)
            if bmi < 18.5:
                bmi_category = "underweight"
            elif bmi < 25:
                bmi_category = "normal"
            elif bmi < 30:
                bmi_category = "overweight"
            else:
                bmi_category = "obese"
        
        return {
            "name": user.name,
            "age": user.age,
            "gender": user.gender,
            "height_cm": user.height_cm,
            "weight_kg": user.weight_kg,
            "bmi": bmi,
            "bmi_category": bmi_category,
            "conditions": user.conditions,
            "allergies": user.allergies,
            "health_goals": user.health_goals,
            "daily_step_goal": user.daily_step_goal,
            "preferred_language": user.preferred_language
        }


class MedicationTools:
    """Medication-related tools for the MCP server."""
    
    @staticmethod
    def add_medication(
        user_id: int,
        medication_name: str,
        dosage: Optional[str] = None,
        frequency: Optional[str] = None,
        instructions: Optional[str] = None
    ) -> Dict[str, Any]:
        """Add a new medication for the user."""
        from models import User, Medication
        from extensions import db
        
        user_id = safe_int(user_id, 1)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Check if medication already exists
        existing = Medication.query.filter_by(
            user_id=user_id,
            name=medication_name
        ).first()
        
        if existing:
            if existing.active:
                return {
                    "success": False,
                    "message": f"{medication_name} is already in your medication list.",
                    "medication": {
                        "id": existing.id,
                        "name": existing.name,
                        "dosage": existing.dosage,
                        "frequency": existing.frequency
                    }
                }
            else:
                # Reactivate existing medication
                existing.active = True
                existing.dosage = dosage or existing.dosage
                existing.frequency = frequency or existing.frequency
                existing.instructions = instructions or existing.instructions
                db.session.commit()
                return {
                    "success": True,
                    "message": f"Reactivated {medication_name} in your medication list.",
                    "medication": {
                        "id": existing.id,
                        "name": existing.name,
                        "dosage": existing.dosage,
                        "frequency": existing.frequency
                    }
                }
        
        # Create new medication
        medication = Medication(
            user_id=user_id,
            name=medication_name,
            dosage=dosage,
            frequency=frequency,
            instructions=instructions,
            active=True
        )
        db.session.add(medication)
        db.session.commit()
        
        return {
            "success": True,
            "message": f"Added {medication_name} to your medication list.",
            "medication": {
                "id": medication.id,
                "name": medication.name,
                "dosage": medication.dosage,
                "frequency": medication.frequency,
                "instructions": medication.instructions
            }
        }
    
    @staticmethod
    def list_medications(user_id: int, active_only: bool = True) -> Dict[str, Any]:
        """List user's medications."""
        from models import User, Medication
        
        user_id = safe_int(user_id, 1)
        active_only = safe_bool(active_only, True)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        query = Medication.query.filter_by(user_id=user_id)
        if active_only:
            query = query.filter_by(active=True)
        
        medications = query.all()
        
        return {
            "user_name": user.name,
            "medication_count": len(medications),
            "medications": [
                {
                    "id": med.id,
                    "name": med.name,
                    "dosage": med.dosage,
                    "frequency": med.frequency,
                    "instructions": med.instructions,
                    "active": med.active
                }
                for med in medications
            ]
        }
    
    @staticmethod
    def set_medication_reminder(
        user_id: int,
        medication_name: str,
        time: str,  # HH:MM format
        days_of_week: Optional[List[int]] = None,  # 0=Mon, 6=Sun
        message: Optional[str] = None
    ) -> Dict[str, Any]:
        """Set a medication reminder."""
        from models import User, Medication, Reminder, ReminderType
        from extensions import db
        from datetime import time as time_type
        
        user_id = safe_int(user_id, 1)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Parse time
        try:
            hour, minute = map(int, time.split(':'))
            reminder_time = time_type(hour, minute)
        except ValueError:
            return {"error": f"Invalid time format: {time}. Use HH:MM format."}
        
        # Find or create medication
        medication = Medication.query.filter_by(
            user_id=user_id, 
            name=medication_name
        ).first()
        
        if not medication:
            # Create new medication entry
            medication = Medication(
                user_id=user_id,
                name=medication_name,
                active=True
            )
            db.session.add(medication)
            db.session.flush()  # Get the ID
        
        # Default to every day if not specified, handle string/list conversion
        days_of_week = safe_int_list(days_of_week, [0, 1, 2, 3, 4, 5, 6])
        
        # Create reminder title and message
        title = f"Take {medication_name}"
        if message is None:
            message = f"Time to take your {medication_name}!"
        
        # Create reminder
        reminder = Reminder(
            user_id=user_id,
            medication_id=medication.id,
            reminder_type=ReminderType.MEDICATION,
            title=title,
            message=message,
            scheduled_time=reminder_time,
            days_of_week=days_of_week,
            channel="push",  # Use string, not enum
            active=True
        )
        db.session.add(reminder)
        db.session.commit()
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        scheduled_days = [day_names[d] for d in days_of_week]
        
        return {
            "success": True,
            "reminder_id": reminder.id,
            "medication": medication_name,
            "time": time,
            "days": scheduled_days,
            "message": message,
            "confirmation": f"Reminder set for {medication_name} at {time} on {', '.join(scheduled_days)}"
        }
    
    @staticmethod
    def list_reminders(user_id: int, active_only: bool = True) -> Dict[str, Any]:
        """List user's medication reminders."""
        from models import User, Reminder, Medication
        import logging
        
        logger = logging.getLogger(__name__)
        
        user_id = safe_int(user_id, 1)
        active_only = safe_bool(active_only, True)
        user = User.query.get(user_id)
        if not user:
            return {"error": f"User {user_id} not found"}
        
        # Query all reminders for this user
        query = Reminder.query.filter_by(user_id=user_id)
        if active_only:
            query = query.filter_by(active=True)
        
        # Order by creation time (newest first)
        reminders = query.order_by(Reminder.created_at.desc()).all()
        
        logger.info(f"[list_reminders] Found {len(reminders)} reminders for user {user_id} (active_only={active_only})")
        
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        
        def safe_day_name(d):
            """Safely convert day index to name."""
            try:
                return day_names[int(d)]
            except (ValueError, IndexError):
                return str(d)
        
        reminder_list = []
        for r in reminders:
            reminder_list.append({
                "id": r.id,
                "title": r.title,
                "medication": r.medication.name if r.medication else None,
                "message": r.message,
                "time": r.scheduled_time.strftime("%H:%M"),
                "days": [safe_day_name(d) for d in (r.days_of_week or [])],
                "channel": r.channel,
                "active": r.active,
                "created_at": r.created_at.isoformat() if r.created_at else None
            })
        
        result = {
            "user_name": user.name,
            "reminder_count": len(reminder_list),
            "reminders": reminder_list
        }
        
        logger.debug(f"[list_reminders] Returning {len(reminder_list)} reminders for user {user_id}")
        
        return result


class RAGTools:
    """RAG/Knowledge base tools for the MCP server."""
    
    @staticmethod
    def search_knowledge_base(query: str, top_k: int = 3) -> Dict[str, Any]:
        """Search the medical knowledge base."""
        from services.rag_service import get_rag_service
        
        top_k = safe_int(top_k, 3)
        rag_service = get_rag_service()
        results = rag_service.search(query, top_k=top_k)
        
        return {
            "query": query,
            "result_count": len(results),
            "results": results
        }


# =============================================================================
# Tool Registry - All available tools
# =============================================================================

TOOL_REGISTRY: Dict[str, Tool] = {}


def register_tool(tool: Tool) -> None:
    """Register a tool in the global registry."""
    TOOL_REGISTRY[tool.name] = tool
    logger.info(f"Registered tool: {tool.name}")


def get_tool(name: str) -> Optional[Tool]:
    """Get a tool by name."""
    return TOOL_REGISTRY.get(name)


def list_tools() -> List[Dict[str, Any]]:
    """List all available tools as schemas."""
    return [tool.to_schema() for tool in TOOL_REGISTRY.values()]


def execute_tool(name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name with given arguments."""
    tool = get_tool(name)
    if not tool:
        return {"error": f"Tool not found: {name}"}
    
    if not tool.handler:
        return {"error": f"Tool has no handler: {name}"}
    
    try:
        result = tool.handler(**kwargs)
        return result
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return {"error": str(e)}


# =============================================================================
# Initialize Tool Registry
# =============================================================================

def initialize_tools():
    """Register all available tools."""
    
    # Health Tools
    register_tool(Tool(
        name="get_health_summary",
        description="Get a summary of the user's current health scores including CMI (Cardio-Metabolic Index), cardiac score, activity score, recovery score, and metabolic score. Use this when the user asks about their overall health status or scores.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1)
        ],
        handler=HealthTools.get_health_summary
    ))
    
    register_tool(Tool(
        name="get_health_metrics",
        description="Get specific health metrics like heart rate, steps, sleep duration, HRV, etc. for a specified time period. Use this when the user asks about specific measurements.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1),
            ToolParameter(
                "metric_type", "string", 
                "Type of metric to retrieve",
                enum=["heart_rate", "steps", "sleep_duration", "sleep_deep", 
                      "sleep_score", "hrv", "resting_hr", "calories"]
            ),
            ToolParameter("days", "integer", "Number of days to look back", required=False, default=7)
        ],
        handler=HealthTools.get_health_metrics
    ))
    
    register_tool(Tool(
        name="get_user_profile",
        description="Get the user's profile information including age, weight, height, health conditions, goals, and preferences. Use this when the user asks about their profile or personal information.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1)
        ],
        handler=HealthTools.get_user_profile
    ))
    
    # Medication Tools
    register_tool(Tool(
        name="add_medication",
        description="Add a new medication to the user's medication list. Use this when the user wants to add or track a medication they are taking.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1),
            ToolParameter("medication_name", "string", "Name of the medication to add"),
            ToolParameter("dosage", "string", "Dosage (e.g., '100mg', '1 tablet')", required=False),
            ToolParameter("frequency", "string", "How often to take (e.g., 'daily', 'twice daily')", required=False),
            ToolParameter("instructions", "string", "Special instructions (e.g., 'take with food')", required=False)
        ],
        handler=MedicationTools.add_medication
    ))
    
    register_tool(Tool(
        name="list_medications",
        description="List all medications the user is currently taking. Use this when the user asks about their medications.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1),
            ToolParameter("active_only", "boolean", "Only show active medications", required=False, default=True)
        ],
        handler=MedicationTools.list_medications
    ))
    
    register_tool(Tool(
        name="set_medication_reminder",
        description="Create a new medication reminder for the user. Use this when the user wants to be reminded to take medication at a specific time.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1),
            ToolParameter("medication_name", "string", "Name of the medication"),
            ToolParameter("time", "string", "Time for the reminder in HH:MM format (24-hour)"),
            ToolParameter(
                "days_of_week", "array", 
                "Days to remind (0=Monday, 6=Sunday). Default: every day",
                required=False
            ),
            ToolParameter("message", "string", "Custom reminder message", required=False)
        ],
        handler=MedicationTools.set_medication_reminder
    ))
    
    register_tool(Tool(
        name="list_reminders",
        description="List ALL medication reminders set for the user. Use this when the user asks 'what are my reminders', 'show my reminders', 'list my reminders', or any variation asking about their reminder list. ALWAYS use this tool when asked about reminders - never guess or make up reminder information.",
        parameters=[
            ToolParameter("user_id", "integer", "The user's ID", default=1),
            ToolParameter("active_only", "boolean", "Only show active reminders", required=False, default=True)
        ],
        handler=MedicationTools.list_reminders
    ))
    
    # RAG Tools
    register_tool(Tool(
        name="search_knowledge_base",
        description="Search the medical knowledge base for information about health topics, medications, conditions, etc. Use this when the user asks general health questions that require factual medical information.",
        parameters=[
            ToolParameter("query", "string", "Search query"),
            ToolParameter("top_k", "integer", "Number of results to return", required=False, default=3)
        ],
        handler=RAGTools.search_knowledge_base
    ))
    
    logger.info(f"Initialized {len(TOOL_REGISTRY)} tools")


# Initialize tools when module is imported
initialize_tools()

