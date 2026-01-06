"""
LLM Service - Interface to Ollama for local LLM inference.

Supports both regular chat and tool-calling modes.
"""
from typing import Optional, Generator, List, Dict, Any
import ollama
import json
import re
import logging

import config

logger = logging.getLogger(__name__)


class LLMService:
    """
    Service for interacting with Ollama local LLM.
    Provides both regular and streaming responses, plus tool calling.
    """
    
    def __init__(self, model: Optional[str] = None):
        self.model = model or config.OLLAMA_MODEL
        self.client = ollama.Client(host=config.OLLAMA_HOST)
        
        # System prompt for health assistant
        self.system_prompt = """You are VitalAI, a helpful health assistant. You help users understand their health data, answer health-related questions, and provide guidance on wellness topics.

Guidelines:
- Be helpful, accurate, and caring in your responses
- Use the provided context to answer questions when available
- If you don't know something, say so honestly
- Always remind users to consult healthcare professionals for medical advice
- Keep responses concise but informative
- Reference specific data when the user asks about their health metrics

Remember: You are not a doctor. Encourage users to seek professional medical advice for health concerns."""
        
        # System prompt for tool-calling mode
        self.tool_system_prompt = """You are VitalAI, a health assistant. You MUST use the provided tools to help users.

MANDATORY TOOL USAGE:
- "add medication" / "I take X" → USE add_medication tool
- "set reminder" / "remind me" → USE set_medication_reminder tool  
- "health scores" / "how am I doing" → USE get_health_summary tool
- "heart rate" / "steps" / "sleep" → USE get_health_metrics tool
- "my medications" / "what am I taking" → USE list_medications tool
- "my reminders" → USE list_reminders tool

NEVER refuse to set a reminder or add a medication. Just do it.
NEVER make up health data. Use tools to get real data.

You are a helpful assistant that executes user requests."""
    
    def chat(
        self, 
        message: str, 
        context: Optional[str] = None,
        chat_history: Optional[list] = None,
        user_data: Optional[dict] = None
    ) -> str:
        """
        Send a message and get a response.
        
        Args:
            message: User's message
            context: Retrieved RAG context (from knowledge base)
            chat_history: Previous messages in conversation
            user_data: User's health data for personalization
        
        Returns:
            Assistant's response text
        """
        messages = [{"role": "system", "content": self._build_system_prompt(context, user_data)}]
        
        # Add chat history if provided
        if chat_history:
            messages.extend(chat_history)
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            return response['message']['content']
        
        except Exception as e:
            print(f"LLM Error: {e}")
            return f"I'm sorry, I encountered an error connecting to the AI model. Please make sure Ollama is running. Error: {str(e)}"
    
    def chat_stream(
        self, 
        message: str, 
        context: Optional[str] = None,
        chat_history: Optional[list] = None,
        user_data: Optional[dict] = None
    ) -> Generator[str, None, None]:
        """
        Send a message and stream the response.
        Yields response chunks as they arrive.
        """
        messages = [{"role": "system", "content": self._build_system_prompt(context, user_data)}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": message})
        
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        
        except Exception as e:
            yield f"Error: {str(e)}"
    
    def _build_system_prompt(
        self, 
        context: Optional[str] = None, 
        user_data: Optional[dict] = None
    ) -> str:
        """Build the system prompt with context and user data."""
        prompt_parts = [self.system_prompt]
        
        # Add user health data if available
        if user_data:
            user_context = self._format_user_data(user_data)
            if user_context:
                prompt_parts.append(f"\n\nUser's Current Health Data:\n{user_context}")
        
        # Add RAG context if available
        if context:
            prompt_parts.append(f"\n\nRelevant Information from Knowledge Base:\n{context}")
            prompt_parts.append("\nUse the above information to help answer the user's question.")
        
        return "\n".join(prompt_parts)
    
    def _format_user_data(self, user_data: dict) -> str:
        """Format user health data for the prompt."""
        if not user_data:
            return ""
        
        parts = []
        
        if 'scores' in user_data:
            scores = user_data['scores']
            parts.append(f"- CMI Score: {scores.get('cmi', 'N/A')}/100")
            parts.append(f"- Cardiac Score: {scores.get('cardiac', 'N/A')}/100")
            parts.append(f"- Activity Score: {scores.get('activity', 'N/A')}/100")
            parts.append(f"- Recovery Score: {scores.get('recovery', 'N/A')}/100")
            parts.append(f"- Metabolic Score: {scores.get('metabolic', 'N/A')}/100")
        
        if 'latest_metrics' in user_data:
            metrics = user_data['latest_metrics']
            for metric, value in metrics.items():
                parts.append(f"- {metric.replace('_', ' ').title()}: {value}")
        
        if 'user_profile' in user_data:
            profile = user_data['user_profile']
            if profile.get('name'):
                parts.insert(0, f"User: {profile['name']}")
            if profile.get('age'):
                parts.append(f"- Age: {profile['age']}")
            if profile.get('conditions'):
                parts.append(f"- Health Conditions: {', '.join(profile['conditions'])}")
        
        return "\n".join(parts)
    
    def chat_with_tools(
        self,
        message: str,
        tools: List[Dict[str, Any]],
        chat_history: Optional[list] = None,
        user_data: Optional[dict] = None
    ) -> Dict[str, Any]:
        """
        Chat with tool-calling capability using Ollama's native tool support.
        
        Args:
            message: User's message
            tools: List of available tools in Ollama format
            chat_history: Previous messages
            user_data: User's health data
            
        Returns:
            Dict with 'response' (text) and optionally 'tool_call' (if LLM wants to call a tool)
        """
        # Build system prompt
        system_content = self.tool_system_prompt
        
        if user_data:
            user_context = self._format_user_data(user_data)
            if user_context:
                system_content += f"\n\nUser's Current Profile:\n{user_context}"
        
        messages = [{"role": "system", "content": system_content}]
        
        if chat_history:
            messages.extend(chat_history)
        
        messages.append({"role": "user", "content": message})
        
        try:
            # Use Ollama's native tool calling
            response = self.client.chat(
                model=self.model,
                messages=messages,
                tools=tools,  # Pass tools for native function calling
                options={
                    "temperature": 0.3,
                    "top_p": 0.9,
                }
            )
            
            message_response = response.get('message', {})
            
            # Check if model made a tool call (native Ollama format)
            tool_calls = message_response.get('tool_calls', [])
            
            if tool_calls:
                # Model decided to call a tool
                tool_call = tool_calls[0]  # Take the first tool call
                function_info = tool_call.get('function', {})
                
                return {
                    'type': 'tool_call',
                    'tool_call': {
                        'name': function_info.get('name'),
                        'arguments': function_info.get('arguments', {})
                    },
                    'raw_response': message_response.get('content', '')
                }
            
            # No tool call - check for JSON in response text as fallback
            response_text = message_response.get('content', '')
            tool_call = self._extract_tool_call(response_text)
            
            if tool_call:
                return {
                    'type': 'tool_call',
                    'tool_call': tool_call,
                    'raw_response': response_text
                }
            else:
                return {
                    'type': 'text',
                    'response': response_text
                }
        
        except Exception as e:
            logger.error(f"Error in chat_with_tools: {e}")
            return {
                'type': 'error',
                'error': str(e)
            }
    
    def _extract_tool_call(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Extract tool call from LLM response.
        
        Looks for JSON with tool_call structure.
        """
        try:
            # Try to find JSON in response
            # Look for {"tool_call": ...} pattern
            json_match = re.search(r'\{[\s\S]*"tool_call"[\s\S]*\}', response)
            
            if json_match:
                json_str = json_match.group()
                parsed = json.loads(json_str)
                
                if 'tool_call' in parsed:
                    tool_call = parsed['tool_call']
                    # Validate structure
                    if 'name' in tool_call:
                        return {
                            'name': tool_call['name'],
                            'arguments': tool_call.get('arguments', {})
                        }
            
            return None
            
        except json.JSONDecodeError:
            return None
        except Exception as e:
            logger.debug(f"Error extracting tool call: {e}")
            return None
    
    def generate_response_with_tool_result(
        self,
        original_message: str,
        tool_name: str,
        tool_result: Dict[str, Any],
        chat_history: Optional[list] = None
    ) -> str:
        """
        Generate a natural language response incorporating tool results.
        
        Args:
            original_message: The user's original question
            tool_name: Name of the tool that was called
            tool_result: Result from the tool execution
            chat_history: Previous messages
            
        Returns:
            Natural language response
        """
        system_prompt = """You are VitalAI, a helpful health assistant. 
You just retrieved data using a tool. Now provide a helpful, natural response to the user based on this data.
Be conversational, caring, and explain what the data means in simple terms.
Do NOT mention that you used a tool - just present the information naturally."""
        
        # Format tool result for the prompt
        result_str = json.dumps(tool_result, indent=2)
        
        user_content = f"""User asked: {original_message}

Data retrieved ({tool_name}):
{result_str}

Provide a helpful response based on this data."""
        
        messages = [
            {"role": "system", "content": system_prompt},
        ]
        
        if chat_history:
            # Add recent history for context
            messages.extend(chat_history[-4:])
        
        messages.append({"role": "user", "content": user_content})
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={
                    "temperature": 0.7,
                    "top_p": 0.9,
                }
            )
            return response['message']['content']
        
        except Exception as e:
            logger.error(f"Error generating response with tool result: {e}")
            # Fallback: return formatted tool result
            return f"Here's what I found: {json.dumps(tool_result, indent=2)}"
    
    def check_connection(self) -> dict:
        """Check if Ollama is accessible and model is available."""
        try:
            # List available models
            response = self.client.list()
            
            # Handle different response formats from ollama library
            model_names = []
            models_data = response.get('models', []) if isinstance(response, dict) else getattr(response, 'models', [])
            
            for m in models_data:
                # Try attribute access first (newer ollama library), then dict access
                if hasattr(m, 'model'):
                    model_names.append(m.model)
                elif hasattr(m, 'name'):
                    model_names.append(m.name)
                elif isinstance(m, dict):
                    model_names.append(m.get('model') or m.get('name', 'unknown'))
            
            model_available = any(self.model in name for name in model_names)
            
            return {
                'status': 'connected',
                'available_models': model_names,
                'configured_model': self.model,
                'model_available': model_available
            }
        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'configured_model': self.model
            }


# Global instance
_llm_service: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """Get or create the global LLM service instance."""
    global _llm_service
    if _llm_service is None:
        _llm_service = LLMService()
    return _llm_service

