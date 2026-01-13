"""
LLM Utilities for RAG System
Handles Groq API integration and answer generation
"""

import os
import time
from typing import List, Dict, Optional
from groq import Groq
from dotenv import load_dotenv
from logger import logger, log_llm_call, log_error
from error_handling import retry_with_backoff, APIError, validate_input
from rate_limiting import api_rate_limiter

# Load environment variables
load_dotenv()

# Groq Configuration
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
DEFAULT_MODEL = "llama-3.1-8b-instant"  # Fast and reliable model
# Alternative models: "llama-3.1-70b-versatile", "llama3-8b-8192", "llama3-70b-8192"

# Improved prompt template for better LLM responses
SYSTEM_PROMPT = """You are an expert assistant for the FAST-NUCES Final Year Project (FYP) Handbook. Your role is to help students understand FYP guidelines, requirements, and procedures.

INSTRUCTIONS:
1. Answer questions using ONLY the provided context from the handbook
2. Be conversational and friendly, but precise and accurate
3. ALWAYS cite page numbers using format "(p. X)" after each key point or fact
4. If information spans multiple pages, cite all relevant pages: "(p. 5-7)"
5. If the answer is not in the context, politely say "I don't have that information in the handbook"
6. Break down complex information into clear, organized points
7. Use formatting (bullet points, numbering) to improve readability
8. Be concise but comprehensive - provide complete answers

CITATION RULES:
- Every factual claim MUST have a page citation
- Place citations immediately after the relevant information
- Example: "The project proposal should be 15-20 pages (p. 12)."
- For multi-page references: "Chapters include Introduction, Literature Review, and Methodology (p. 8-10)."

RESPONSE STYLE:
- Start with a direct answer to the question
- Organize information logically
- Use natural, conversational language
- End with a helpful summary or next steps if appropriate"""

USER_PROMPT_TEMPLATE = """Based on the FYP Handbook context below, please answer this question:

**Question:** {question}

**Context from Handbook:**
{context}

**Your Answer (with page citations):**"""


class GroqLLM:
    """Wrapper for Groq API integration"""
    
    def __init__(self, api_key: Optional[str] = None, model: str = DEFAULT_MODEL):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key (if None, loads from environment)
            model: Model name to use
        """
        self.api_key = api_key or GROQ_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "Groq API key not found. Please set GROQ_API_KEY environment variable "
                "or pass it to the constructor."
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
    
    @retry_with_backoff(max_retries=3, initial_delay=1.0, exceptions=(Exception,))
    def generate_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 1500,
        temperature: float = 0.3
    ) -> Dict:
        """
        Generate answer using Groq LLM with retry logic.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature (0.0-1.0, lower = more focused)
            
        Returns:
            Dictionary with answer and metadata
        """
        # Check rate limit
        allowed, retry_after = api_rate_limiter.is_allowed("groq_api")
        if not allowed:
            logger.warning(f"Groq API rate limit exceeded. Retry after {retry_after:.1f}s")
            raise APIError(f"Rate limit exceeded. Please wait {int(retry_after) + 1} seconds.")
        
        start_time = time.time()
        
        try:
            # Format user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=question,
                context=context
            )
            
            logger.debug(f"Calling Groq API with model: {self.model}")
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=False
            )
            
            elapsed = time.time() - start_time
            
            # Extract answer
            answer = response.choices[0].message.content.strip()
            
            # Get usage stats
            usage = response.usage
            
            # Log successful call
            log_llm_call(
                model=self.model,
                tokens=usage.total_tokens,
                latency=elapsed,
                success=True
            )
            
            return {
                'success': True,
                'answer': answer,
                'model': self.model,
                'tokens_used': {
                    'prompt': usage.prompt_tokens,
                    'completion': usage.completion_tokens,
                    'total': usage.total_tokens
                },
                'finish_reason': response.choices[0].finish_reason,
                'latency': elapsed
            }
            
        except Exception as e:
            elapsed = time.time() - start_time
            
            # Log failed call
            log_llm_call(
                model=self.model,
                tokens=0,
                latency=elapsed,
                success=False
            )
            log_error(e, {"function": "generate_answer", "model": self.model})
            
            # Return user-friendly error
            error_message = self._format_error_message(e)
            
            return {
                'success': False,
                'answer': error_message,
                'error': str(e),
                'latency': elapsed
            }
    
    def _format_error_message(self, error: Exception) -> str:
        """Format user-friendly error message"""
        error_str = str(error).lower()
        
        if "rate" in error_str or "limit" in error_str:
            return "⚠️ The AI service is temporarily busy. Please wait a moment and try again."
        elif "key" in error_str or "auth" in error_str:
            return "⚠️ API authentication error. Please contact support."
        elif "timeout" in error_str:
            return "⚠️ The request timed out. Please try again with a shorter question."
        elif "model" in error_str:
            return "⚠️ The AI model is currently unavailable. Please try again later."
        else:
            return f"⚠️ An error occurred while generating the answer. Please try again.\n\nError: {str(error)}"
    
    def generate_answer_stream(
        self,
        question: str,
        context: str,
        max_tokens: int = 1500,
        temperature: float = 0.3
    ):
        """
        Generate answer with streaming for real-time display.
        
        Args:
            question: User's question
            context: Retrieved context from RAG
            max_tokens: Maximum tokens in response
            temperature: Sampling temperature
            
        Yields:
            Chunks of the generated answer
        """
        try:
            # Format user prompt
            user_prompt = USER_PROMPT_TEMPLATE.format(
                question=question,
                context=context
            )
            
            # Call Groq API with streaming
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                stream=True
            )
            
            # Stream response chunks
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            yield f"\n\n❌ Error: {str(e)}"


def format_context_for_llm(chunks: List[Dict], max_chunks: int = 5) -> str:
    """
    Format retrieved chunks into a context string for the LLM.
    
    Args:
        chunks: List of retrieved chunk dictionaries
        max_chunks: Maximum number of chunks to include
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, chunk in enumerate(chunks[:max_chunks], 1):
        page = chunk['page_number']
        text = chunk['text'].strip()
        section = chunk.get('section_hint', 'General Section')
        
        # Format each chunk with clear structure
        context_parts.append(
            f"--- Excerpt {i} (Page {page} - {section}) ---\n{text}\n"
        )
    
    return "\n".join(context_parts)


def validate_api_key(api_key: Optional[str] = None) -> bool:
    """
    Validate that Groq API key is available and working.
    
    Args:
        api_key: API key to validate (if None, checks environment)
        
    Returns:
        True if valid, False otherwise
    """
    test_key = api_key or GROQ_API_KEY
    
    if not test_key:
        return False
    
    try:
        client = Groq(api_key=test_key)
        # Try a minimal request to verify the key works
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=5
        )
        return True
    except Exception:
        return False


# Convenience function for quick usage
def generate_rag_answer(
    question: str,
    chunks: List[Dict],
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    stream: bool = False
) -> Dict:
    """
    Convenience function to generate RAG answer in one call.
    
    Args:
        question: User's question
        chunks: Retrieved chunks from RAG system
        api_key: Groq API key (optional)
        model: Model to use
        stream: Whether to use streaming
        
    Returns:
        Dictionary with answer and metadata
    """
    # Format context
    context = format_context_for_llm(chunks)
    
    # Initialize LLM
    llm = GroqLLM(api_key=api_key, model=model)
    
    # Generate answer
    if stream:
        return llm.generate_answer_stream(question, context)
    else:
        return llm.generate_answer(question, context)
