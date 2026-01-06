"""
Supervisor Agent - Routes queries to appropriate specialist agents.

The Supervisor analyzes user messages and determines:
1. What type of query it is (health data, medication, knowledge, personal)
2. Which specialist agent should handle it
3. If multiple agents are needed (multi-step tasks)
"""

from typing import Dict, Any, List, Optional
import logging
import re
import json

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


# Intent keywords mapping to agent types
INTENT_PATTERNS = {
    AgentType.HEALTH_ANALYST: [
        # Health scores (singular and plural)
        r'\b(health\s*score|cmi|cardiac|metabolic|recovery|activity)\b',
        r'\b(health\s*scores)\b',
        r'\b(current\s+health\s+score|my\s+health\s+score|my\s+health\s+scores)\b',
        # Metrics
        r'\b(heart\s*rate|hrv|sleep|steps|calories|blood\s*pressure|spo2)\b',
        # User's data queries (but NOT reminders - those are handled by priority check)
        r'\b(my\s+(health|data|metrics|stats|score|scores))\b',
        r'\b(show\s+me\s+my|what\s+are\s+my|tell\s+me\s+my)\s+(health|data|metrics|scores|score)\b',
        r'\b(how\s+am\s+i\s+doing|health\s+summary|check\s+my)\b',
        # Analysis keywords
        r'\b(trend|progress|average|history|past\s+week|last\s+week)\b',
        r'\b(analyze|analysis|insight)\b',
    ],
    AgentType.MEDICATION_MANAGER: [
        r'\b(medication|medicine|drug|pill|prescription)\b',
        r'\b(remind|reminder|reminders|alert|notify|notification)\b',
        r'\b(take|taking|dose|dosage)\b',
        r'\b(add\s+.+\s+to\s+my|add\s+medication)\b',
        r'\b(set\s+.+\s+reminder|create\s+reminder)\b',
        r'\b(medication\s+list|my\s+meds)\b',
        r'\b(what\s+are\s+my\s+reminders|show\s+my\s+reminders|list\s+my\s+reminders|current\s+reminders)\b',
    ],
    AgentType.KNOWLEDGE_EXPERT: [
        # General knowledge questions (priority checks handle "my" queries first)
        r'\b(what\s+is|what\s+are|explain|tell\s+me\s+about)\b',
        r'\b(how\s+does|how\s+do|why\s+does|why\s+do)\b',
        r'\b(symptoms|treatment|cause|effect|side\s+effect)\b',
        r'\b(interaction|contraindication)\b',
        r'\b(normal\s+range|healthy\s+level)\b',
        r'\b(best\s+practice|recommendation|guideline|tip)\b',
    ],
    AgentType.DIGITAL_CLONE: [
        r'\b(remember|recall|you\s+know|last\s+time)\b',
        r'\b(my\s+preference|i\s+like|i\s+prefer|i\s+usually)\b',
        # "Based on my" patterns (plural and singular)
        r'\b(based\s+on\s+my|for\s+me\s+personally|personalize)\b',
        r'\b(based\s+on\s+my\s+(pattern|patterns|history|data|habits))\b',
        r'\b(advice|suggest|recommend)\s+(for\s+me|based\s+on)\b',
        r'\b(my\s+history|my\s+pattern|my\s+patterns|my\s+habit|my\s+habits)\b',
        r'\b(what\s+should\s+i\s+focus\s+on|what\s+should\s+i\s+do)\b',
    ],
}


class SupervisorAgent(BaseAgent):
    """
    Supervisor agent that routes queries to specialist agents.
    
    Responsibilities:
    1. Classify user intent
    2. Select appropriate specialist agent
    3. Handle multi-step queries requiring multiple agents
    4. Provide fallback for ambiguous queries
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.SUPERVISOR
    
    @property
    def system_prompt(self) -> str:
        return """You are the VitalAI Supervisor. Your job is to analyze user queries and determine the best way to help them.

You coordinate between specialist agents:
- HEALTH_ANALYST: For health scores, metrics, data analysis, trends
- MEDICATION_MANAGER: For medications, reminders, doses
- KNOWLEDGE_EXPERT: For medical information, explanations, guidelines
- DIGITAL_CLONE: For personalized advice based on user history

Analyze the user's message and respond with a JSON object:
{
    "primary_agent": "agent_type",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation",
    "needs_follow_up": false,
    "secondary_agents": []
}

Only respond with the JSON object, nothing else."""
    
    def get_tools(self) -> List[str]:
        """Supervisor doesn't use tools directly - it routes to other agents."""
        return []
    
    def classify_intent(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Classify the user's intent and determine which agent should handle it.
        
        Uses a hybrid approach:
        1. Rule-based pattern matching (fast, deterministic)
        2. Priority checks for specific patterns (user data queries)
        3. LLM-based classification (for ambiguous cases)
        
        Returns:
            Dict with primary_agent, confidence, and optional secondary_agents
        """
        message_lower = message.lower()
        
        # Priority checks: Specific patterns that should override generic ones
        # Check for reminder queries first (these should go to Medication Manager)
        reminder_patterns = [
            r'\b(what\s+are\s+my\s+(current\s+)?reminders|show\s+my\s+reminders|list\s+my\s+reminders|current\s+reminders|my\s+reminders)\b',
            r'\b(what\s+reminders|show\s+reminders|list\s+reminders)\b',
        ]
        for pattern in reminder_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.debug(f"[Supervisor] Priority match: Reminder query -> Medication Manager")
                return {
                    "primary_agent": AgentType.MEDICATION_MANAGER,
                    "confidence": 0.95,
                    "method": "priority_pattern",
                    "reasoning": "User asking about their reminders"
                }
        
        # Check for user data queries (these should go to Health Analyst)
        user_data_patterns = [
            r'\b(what\s+are\s+my|show\s+me\s+my|tell\s+me\s+my|my\s+current)\s+(health\s+)?(score|scores|data|metrics)\b',
            r'\b(what\s+are\s+my\s+health\s+scores)\b',
            r'\b(my\s+health\s+score|my\s+health\s+scores)\b',
            # User asking about improving THEIR specific score
            r'\b(how\s+can\s+i\s+improve\s+my\s+(sleep|health|cardiac|activity|recovery|metabolic|cmi)\s+score)\b',
        ]
        for pattern in user_data_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.debug(f"[Supervisor] Priority match: User data query -> Health Analyst")
                return {
                    "primary_agent": AgentType.HEALTH_ANALYST,
                    "confidence": 0.95,
                    "method": "priority_pattern",
                    "reasoning": "User asking about their own health data/scores"
                }
        
        # Check for "based on my" patterns (should go to Digital Clone)
        personal_patterns = [
            r'\bbased\s+on\s+my\s+(pattern|patterns|history|data|habits)',
            r'\bwhat\s+should\s+i\s+(focus\s+on|do)\s+(based\s+on|given)\s+my\b',
            r'\b(based\s+on|given)\s+my\s+(pattern|patterns|history|data|habits)',
        ]
        for pattern in personal_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.debug(f"[Supervisor] Priority match: Personalization query -> Digital Clone")
                return {
                    "primary_agent": AgentType.DIGITAL_CLONE,
                    "confidence": 0.95,
                    "method": "priority_pattern",
                    "reasoning": "User asking for personalized advice based on their patterns"
                }
        
        # Step 1: Rule-based pattern matching
        scores = {agent_type: 0.0 for agent_type in INTENT_PATTERNS.keys()}
        
        for agent_type, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    scores[agent_type] += 1.0
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {k: v / total_score for k, v in scores.items()}
        
        # Find best match
        best_agent = max(scores, key=scores.get)
        best_score = scores[best_agent]
        
        logger.debug(f"[Supervisor] Pattern scores: {scores}")
        
        # If pattern matching is confident enough, use it
        if best_score >= 0.6:
            return {
                "primary_agent": best_agent,
                "confidence": min(best_score, 0.95),
                "method": "pattern_matching",
                "reasoning": f"Detected {best_agent.value} keywords in message"
            }
        
        # Step 2: LLM-based classification for ambiguous cases
        if best_score >= 0.3:
            # Pattern gave some signal but not enough
            return {
                "primary_agent": best_agent,
                "confidence": best_score,
                "method": "pattern_matching_low_confidence",
                "reasoning": "Some keyword matches but uncertain"
            }
        
        # Step 3: Fall back to knowledge expert for general questions
        # (Most health questions are knowledge-seeking)
        return {
            "primary_agent": AgentType.KNOWLEDGE_EXPERT,
            "confidence": 0.5,
            "method": "default",
            "reasoning": "No strong pattern match, defaulting to knowledge expert"
        }
    
    def classify_intent_with_llm(
        self, 
        message: str,
        chat_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Use LLM to classify intent when pattern matching is uncertain.
        
        This is slower but more accurate for complex queries.
        """
        classification_prompt = f"""Analyze this health-related query and classify it.

User message: "{message}"

Which specialist should handle this?
- HEALTH_ANALYST: User wants their own health data, scores, metrics, trends
- MEDICATION_MANAGER: User wants to manage medications or reminders
- KNOWLEDGE_EXPERT: User wants general medical/health information
- DIGITAL_CLONE: User wants personalized advice based on their history

Respond with JSON only:
{{"primary_agent": "AGENT_TYPE", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        try:
            response = self.llm_service.chat(
                message=classification_prompt,
                chat_history=chat_history
            )
            
            # Parse JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                agent_name = result.get('primary_agent', 'KNOWLEDGE_EXPERT').upper()
                
                # Map string to AgentType
                agent_map = {
                    'HEALTH_ANALYST': AgentType.HEALTH_ANALYST,
                    'MEDICATION_MANAGER': AgentType.MEDICATION_MANAGER,
                    'KNOWLEDGE_EXPERT': AgentType.KNOWLEDGE_EXPERT,
                    'DIGITAL_CLONE': AgentType.DIGITAL_CLONE
                }
                
                return {
                    "primary_agent": agent_map.get(agent_name, AgentType.KNOWLEDGE_EXPERT),
                    "confidence": float(result.get('confidence', 0.7)),
                    "method": "llm_classification",
                    "reasoning": result.get('reasoning', 'LLM classification')
                }
        
        except Exception as e:
            logger.error(f"LLM classification failed: {e}")
        
        # Fallback
        return {
            "primary_agent": AgentType.KNOWLEDGE_EXPERT,
            "confidence": 0.5,
            "method": "fallback",
            "reasoning": "Classification failed, using default"
        }
    
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a message by routing to the appropriate specialist.
        
        Note: The supervisor doesn't generate content itself - it returns
        routing information that the orchestrator uses.
        """
        classification = self.classify_intent(message, chat_history)
        
        return AgentResponse(
            content="",  # Supervisor doesn't generate content
            agent_type=self.agent_type,
            metadata={
                "routing": classification,
                "target_agent": classification["primary_agent"].value
            },
            confidence=classification["confidence"]
        )
    
    def should_use_multi_agent(self, message: str) -> bool:
        """
        Determine if a query needs multiple agents.
        
        Examples:
        - "What's my CMI score and how can I improve it?"
          → Health Analyst (data) + Knowledge Expert (improvement tips)
        - "Based on my sleep data, what should I do?"
          → Health Analyst (data) + Digital Clone (personalized advice)
        """
        # Look for connector words that suggest multi-part questions
        multi_patterns = [
            r'\band\s+(how|what|why|should)',
            r'\bthen\b',
            r'\balso\b',
            r'\bbased\s+on\s+(my|the)\b',
        ]
        
        for pattern in multi_patterns:
            if re.search(pattern, message.lower()):
                return True
        
        return False

