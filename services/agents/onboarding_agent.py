"""
Onboarding Agent - Guides new users through account setup.

This agent conducts a conversational onboarding flow to collect:
- Basic information (name, age, gender)
- Health goals and preferences
- Initial health metrics
- Language preferences
"""

from typing import Dict, Any, Optional, List
import logging

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


class OnboardingAgent(BaseAgent):
    """Agent specialized for user onboarding conversations"""
    
    def __init__(self, user_id: int):
        super().__init__(user_id)
        # Don't set llm_service directly - it's a property from BaseAgent
        self._onboarding_state = {}  # Store collected information
        self._conversation_history = []  # Store conversation for context
    
    @property
    def agent_type(self) -> AgentType:
        """Return the type of this agent."""
        return AgentType.SUPERVISOR  # Using supervisor type for now
    
    @property
    def system_prompt(self) -> str:
        """System prompt for onboarding agent."""
        return self._build_system_prompt()
    
    def get_tools(self) -> List[str]:
        """Onboarding agent doesn't use tools - it's conversational only."""
        return []
    
    def _build_system_prompt(self) -> str:
        """Build system prompt for onboarding agent"""
        # Get current state to include in prompt
        collected = self._onboarding_state
        state_info = []
        
        if collected.get('age'):
            state_info.append(f"Age: {collected['age']}")
        if collected.get('gender'):
            state_info.append(f"Gender: {collected['gender']}")
        if collected.get('height_cm'):
            state_info.append(f"Height: {collected['height_cm']} cm")
        if collected.get('weight_kg'):
            state_info.append(f"Weight: {collected['weight_kg']} kg")
        if collected.get('preferred_language'):
            state_info.append(f"Language: {collected['preferred_language']}")
        if collected.get('health_goals'):
            state_info.append(f"Health Goals: {', '.join(collected['health_goals']) if isinstance(collected['health_goals'], list) else collected['health_goals']}")
        
        state_str = "\n".join(state_info) if state_info else "No information collected yet."
        
        return f"""You are a friendly and helpful health assistant guiding a new user through onboarding.

IMPORTANT: Check what information you already have before asking questions!

Information already collected:
{state_str}

Your goal is to collect the following information in a natural, conversational way:
1. Basic information: age, gender (if not already collected)
2. Health goals: what they want to achieve (weight loss, fitness, monitoring, etc.) - OPTIONAL
3. Preferences: preferred language, notification preferences - OPTIONAL
4. Initial health metrics: height, weight (if not already collected)
5. Optional: medical conditions, medications, allergies

Guidelines:
- CRITICAL: DO NOT ask for information you already have! Check the "Information already collected" section above.
- Be warm, friendly, and encouraging
- Ask one question at a time
- Don't be pushy - allow users to skip optional questions
- ONLY say "I have all the information I need" when ALL FOUR required fields are present: age, gender, height, weight
- Check the "Information already collected" section - if ANY of age, gender, height, or weight is missing, you MUST ask for it
- After collecting all 4 required fields, ask if they want to share optional information (health goals, conditions, medications, allergies)
- Give them a chance to provide optional information before completing
- Only say "I have all the information I need" and offer to complete after asking about optional fields

Required fields: age, gender, height, weight (ALL FOUR must be present before asking about optional fields)
Optional fields: health goals, conditions, medications, allergies, preferred_language

IMPORTANT: 
- Count the required fields in "Information already collected"
- If you see fewer than 4 required fields (age, gender, height, weight), you MUST continue asking for them
- When you see all 4 required fields, ask: "Would you like to share any health goals, medical conditions, medications, or allergies? This is optional but helps us personalize your experience."
- After they respond (or say no/skip), then say "Great! I have all the information I need. Let me complete your profile setup."
- Do NOT complete prematurely - missing even one required field means you must ask for it"""
    
    def process(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process onboarding conversation
        
        Args:
            message: User's message
            context: Optional context including onboarding_state and user_name
        
        Returns:
            AgentResponse with onboarding guidance
        """
        if context:
            if 'onboarding_state' in context:
                self._onboarding_state = context['onboarding_state']
            # Use user name in conversation if available
            user_name = context.get('user_name', 'there')
            if user_name and 'Hello' in message or 'Hi' in message or 'name' in message.lower():
                # Update state with name if mentioned
                if 'name' not in self._onboarding_state:
                    self._onboarding_state['name'] = user_name
        
        # Build conversation history from stored history
        # For first message, use empty history
        chat_history = []
        if hasattr(self, '_conversation_history') and self._conversation_history:
            # Convert to format expected by LLM service
            chat_history = self._conversation_history[-6:]  # Last 6 messages
        
        # Get LLM response
        # We need to call Ollama directly with custom system prompt
        # since LLMService.chat() doesn't accept a custom system_prompt parameter
        try:
            import ollama
            import config
            
            client = ollama.Client(host=config.OLLAMA_HOST)
            # Rebuild system prompt with current state (includes what we already know)
            current_system_prompt = self._build_system_prompt()
            messages = [{"role": "system", "content": current_system_prompt}]
            
            # Add chat history if provided
            if chat_history:
                messages.extend(chat_history)
            
            # Add current message
            messages.append({"role": "user", "content": message})
            
            response = client.chat(
                model=config.OLLAMA_MODEL,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            response_text = response['message']['content']
            
            # Extract structured data from conversation if possible
            # Try extracting from current message
            extracted_data = self._extract_data_from_message(message)
            
            # Also try extracting from recent conversation history (last 3 user messages)
            if not extracted_data or len(extracted_data) < 2:
                for hist_msg in reversed(self._conversation_history[-6:]):  # Check last 6 messages
                    if hist_msg.get('role') == 'user':
                        hist_extracted = self._extract_data_from_message(hist_msg.get('content', ''))
                        if hist_extracted:
                            # Merge with current extraction
                            for k, v in hist_extracted.items():
                                if k not in extracted_data:
                                    extracted_data[k] = v
            
            if extracted_data:
                logger.info(f"Extracted data from message: {extracted_data}")
                # Only update fields that aren't already set (don't overwrite)
                for key, value in extracted_data.items():
                    if key not in self._onboarding_state or self._onboarding_state[key] is None:
                        self._onboarding_state[key] = value
                        logger.info(f"Updated onboarding state: {key} = {value}")
                    elif isinstance(value, list) and key in self._onboarding_state:
                        # For lists like health_goals, merge them
                        existing = self._onboarding_state.get(key, [])
                        if not isinstance(existing, list):
                            existing = [existing] if existing else []
                        self._onboarding_state[key] = list(set(existing + value))
            
            logger.info(f"Current onboarding state: {self._onboarding_state}")
            
            # Update conversation history
            self._conversation_history.append({"role": "user", "content": message})
            self._conversation_history.append({"role": "assistant", "content": response_text})
            
            return AgentResponse(
                content=response_text,
                agent_type=AgentType.SUPERVISOR,
                metadata={
                    "onboarding_state": self._onboarding_state.copy(),
                    "step": self._determine_step()
                }
            )
            
        except Exception as e:
            logger.error(f"Onboarding agent error: {e}")
            return AgentResponse(
                content="I apologize, but I encountered an error. Could you please try again?",
                agent_type=AgentType.SUPERVISOR,
                metadata={"error": str(e)}
            )
    
    def _extract_data_from_message(self, message: str) -> Dict[str, Any]:
        """Extract structured data from user message"""
        extracted = {}
        message_lower = message.lower()
        
        # Extract age - more flexible patterns
        import re
        age_patterns = [
            r'\b(\d{1,3})\s*(?:years?\s*old|age|aged)\b',
            r'\b(?:i\s+am|i\'m|age\s+is|i\'m\s+)\s*(\d{1,3})\b',
            r'\b(\d{1,3})\b',  # Simple number - but only if context suggests age
        ]
        for pattern in age_patterns:
            age_match = re.search(pattern, message_lower)
            if age_match:
                try:
                    age = int(age_match.group(1))
                    # Reasonable age range check
                    if 1 <= age <= 150:
                        extracted['age'] = age
                        logger.debug(f"Extracted age: {age} from '{message}'")
                        break
                except:
                    pass
        
        # Extract gender
        if any(word in message_lower for word in ['male', 'man', 'boy']):
            extracted['gender'] = 'male'
        elif any(word in message_lower for word in ['female', 'woman', 'girl']):
            extracted['gender'] = 'female'
        elif any(word in message_lower for word in ['other', 'non-binary', 'nonbinary']):
            extracted['gender'] = 'other'
        
        # Extract height (in cm or feet/inches) - more flexible patterns
        # Handle feet'inches" format first (e.g., 5'7", 5' 7", 5'7)
        feet_inches_pattern = r"(\d+)['']\s*(\d+)[\"]?"
        feet_inches_match = re.search(feet_inches_pattern, message)
        if feet_inches_match:
            try:
                feet = float(feet_inches_match.group(1))
                inches = float(feet_inches_match.group(2))
                if 1 <= feet <= 8 and 0 <= inches <= 11:  # Reasonable range
                    extracted['height_cm'] = (feet * 30.48) + (inches * 2.54)
                    logger.debug(f"Extracted height: {extracted['height_cm']} cm from '{message}' (feet'inches format)")
            except Exception as e:
                logger.debug(f"Error extracting height from feet'inches format: {e}")
        
        # If not already extracted, try other patterns
        if 'height_cm' not in extracted:
            height_patterns = [
                r'\b(\d+(?:\.\d+)?)\s*(?:cm|centimeters?)\b',
                r'\b(\d+(?:\.\d+)?)\s*(?:feet|ft)\s*(?:and)?\s*(\d+(?:\.\d+)?)\s*(?:inches?|in)?\b',  # e.g., "5 feet 10 inches"
                r'\b(\d+(?:\.\d+)?)\s*(?:meters?|m)\b',
                r'height.*?(\d+(?:\.\d+)?)\s*(?:cm|centimeters?|meters?|m)',  # "height 175 cm"
                r'\b(\d{3})\b',  # 3-digit number (likely height in cm: 150-250)
            ]
            for pattern in height_patterns:
                height_match = re.search(pattern, message_lower)
                if height_match:
                    try:
                        value = float(height_match.group(1))
                        # Check if it's a reasonable height (50-300 cm)
                        if 50 <= value <= 300:
                            if 'cm' in message_lower or 'centimeter' in message_lower:
                                extracted['height_cm'] = value
                            elif 'meter' in message_lower or ('m\b' in message_lower and 'cm' not in message_lower):
                                extracted['height_cm'] = value * 100
                            elif 'feet' in message_lower or 'ft' in message_lower:
                                # Convert feet to cm (1 ft = 30.48 cm)
                                inches_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:inches?|in)', message_lower)
                                inches = float(inches_match.group(1)) if inches_match else 0
                                extracted['height_cm'] = (value * 30.48) + (inches * 2.54)
                            else:
                                # Default to cm if no unit specified but number is reasonable (likely cm)
                                if 100 <= value <= 250:  # Reasonable height range in cm
                                    extracted['height_cm'] = value
                            if 'height_cm' in extracted:
                                logger.debug(f"Extracted height: {extracted['height_cm']} cm from '{message}'")
                                break
                    except Exception as e:
                        logger.debug(f"Error extracting height: {e}")
                        pass
        
        # Extract weight (in kg or lbs) - more flexible patterns
        weight_patterns = [
            r'\b(\d+(?:\.\d+)?)\s*(?:kg|kilograms?)\b',
            r'\b(\d+(?:\.\d+)?)\s*(?:lbs?|pounds?)\b',
            r'^(\d+(?:\.\d+)?)\s*kg\b',  # "76 kg" at start
            r'\b(\d+(?:\.\d+)?)\s*kg\b',  # "76 kg" anywhere
        ]
        for pattern in weight_patterns:
            weight_match = re.search(pattern, message_lower)
            if weight_match:
                try:
                    value = float(weight_match.group(1))
                    # Check if it's a reasonable weight (10-500 kg)
                    if 10 <= value <= 500:
                        if 'lb' in message_lower or 'pound' in message_lower:
                            extracted['weight_kg'] = value * 0.453592  # Convert lbs to kg
                        else:
                            extracted['weight_kg'] = value  # Assume kg
                        logger.debug(f"Extracted weight: {extracted['weight_kg']} kg from '{message}'")
                        break
                except Exception as e:
                    logger.debug(f"Error extracting weight: {e}")
                    pass
        
        return extracted
    
    def _determine_step(self) -> str:
        """Determine current onboarding step based on collected data"""
        if not self._onboarding_state:
            return "welcome"
        
        # Check required fields
        required_fields = ['age', 'gender', 'height_cm', 'weight_kg']
        has_all_required = all(
            field in self._onboarding_state 
            and self._onboarding_state[field] is not None 
            for field in required_fields
        )
        
        # If all required fields are present, we're at confirmation
        if has_all_required:
            return "confirmation"
        
        # Otherwise, determine which step we're on
        if 'age' not in self._onboarding_state or self._onboarding_state.get('age') is None:
            return "basic_info"
        if 'gender' not in self._onboarding_state or self._onboarding_state.get('gender') is None:
            return "basic_info"
        if 'height_cm' not in self._onboarding_state or self._onboarding_state.get('height_cm') is None:
            return "health_metrics"
        if 'weight_kg' not in self._onboarding_state or self._onboarding_state.get('weight_kg') is None:
            return "health_metrics"
        
        # All required fields collected, optional goals step
        if 'health_goals' not in self._onboarding_state:
            return "goals"
        
        return "confirmation"
    
    def get_collected_data(self) -> Dict[str, Any]:
        """Get all collected onboarding data"""
        return self._onboarding_state.copy()
    
    def reset_state(self):
        """Reset onboarding state"""
        self._onboarding_state = {}
        self._conversation_history = []
        self._conversation_history = []

