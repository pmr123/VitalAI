"""
Knowledge Expert Agent - Answers medical/health knowledge questions using RAG.

Specializes in:
- Explaining health concepts (HRV, CMI, sleep stages)
- Providing general health guidelines
- Answering "what is X" and "how to improve Y" questions
- Drug information and interactions
"""

from typing import Dict, Any, List, Optional
import logging

from .base_agent import BaseAgent, AgentType, AgentResponse

logger = logging.getLogger(__name__)


class KnowledgeExpertAgent(BaseAgent):
    """
    Specialist agent for medical/health knowledge questions.
    
    Uses RAG to provide accurate, sourced answers from the knowledge base.
    
    Tools:
    - search_knowledge_base: Search medical knowledge
    """
    
    @property
    def agent_type(self) -> AgentType:
        return AgentType.KNOWLEDGE_EXPERT
    
    @property
    def system_prompt(self) -> str:
        return """You are the VitalAI Knowledge Expert, specializing in health and medical information.

Your expertise:
- Explaining health metrics and scores (HRV, CMI, blood pressure, etc.)
- Providing evidence-based health guidelines
- Explaining medical concepts in simple terms
- Information about medications, their uses, and interactions

Guidelines:
- Use the knowledge base to provide accurate information
- Cite sources when available
- Explain complex concepts in plain language
- Always recommend consulting a healthcare provider for personal medical decisions
- If information isn't in the knowledge base, say so honestly

Remember: You provide information, not medical advice. Always encourage users to consult healthcare professionals."""
    
    def get_tools(self) -> List[str]:
        return ["search_knowledge_base"]
    
    def process(
        self, 
        message: str, 
        chat_history: Optional[List[Dict]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> AgentResponse:
        """
        Process a knowledge question using RAG.
        
        Strategy:
        1. Search knowledge base for relevant information
        2. Generate informative response with citations
        """
        logger.debug(f"[KnowledgeExpert] Processing: {message[:50]}...")
        
        # Always use RAG for knowledge questions
        rag_context, sources = self._search_knowledge_base(message)
        
        if rag_context:
            # Generate response with RAG context
            response_text = self._generate_response_with_rag(
                message=message,
                rag_context=rag_context,
                chat_history=chat_history
            )
            
            return AgentResponse(
                content=response_text,
                agent_type=self.agent_type,
                sources=sources,
                tool_used="search_knowledge_base"
            )
        
        # No relevant knowledge found - use general LLM knowledge
        logger.debug("[KnowledgeExpert] No RAG results, using general knowledge")
        
        response_text = self._generate_response(
            message=message,
            chat_history=chat_history,
            additional_context="Note: This information is from general knowledge, not the curated knowledge base."
        )
        
        return AgentResponse(
            content=response_text,
            agent_type=self.agent_type,
            metadata={"source": "general_knowledge"}
        )
    
    def _search_knowledge_base(self, query: str) -> tuple:
        """
        Search the knowledge base and return context + sources.
        
        Returns:
            Tuple of (context_string, list_of_sources)
        """
        from services.rag_service import get_rag_service
        
        rag = get_rag_service()
        results = rag.search(query, top_k=3)
        
        if not results:
            return "", []
        
        # Build context string
        context_parts = []
        sources = []
        
        for i, result in enumerate(results):
            context_parts.append(f"[{i+1}] {result['content']}")
            
            sources.append({
                'title': result['metadata'].get('title', 'Unknown'),
                'source': result['metadata'].get('source', 'Unknown'),
                'relevance': round(result['relevance'], 2)
            })
        
        return "\n\n".join(context_parts), sources
    
    def _generate_response_with_rag(
        self,
        message: str,
        rag_context: str,
        chat_history: Optional[List[Dict]] = None
    ) -> str:
        """Generate response using RAG context."""
        from services.llm_service import get_llm_service
        
        llm = get_llm_service()
        
        system_prompt = self.system_prompt + f"""

Knowledge Base Information:
{rag_context}

Use the above information to answer the user's question. If the information doesn't fully answer the question, say so and provide what you can."""
        
        # Create custom chat call with RAG context in system prompt
        messages = [{"role": "system", "content": system_prompt}]
        
        if chat_history:
            messages.extend(chat_history[-4:])  # Last 4 messages for context
        
        messages.append({"role": "user", "content": message})
        
        try:
            response = llm.client.chat(
                model=llm.model,
                messages=messages,
                options={"temperature": 0.7, "top_p": 0.9}
            )
            return response['message']['content']
        except Exception as e:
            logger.error(f"Error generating RAG response: {e}")
            return f"I found relevant information but had trouble generating a response. Error: {str(e)}"
    
    def explain_concept(self, concept: str) -> str:
        """
        Get an explanation of a health concept.
        
        Helper method for common explanations.
        """
        response = self.process(f"Explain what {concept} is and why it matters")
        return response.content

