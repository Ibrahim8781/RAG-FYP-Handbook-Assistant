"""
FYP Handbook RAG - Streamlit Web Interface
Minimal UI with input box, Ask button, Answer panel, and collapsible Sources
"""

import streamlit as st
import os
import pickle
import json
import re
import time
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from llm_utils import GroqLLM, format_context_for_llm, validate_api_key
from config_env import config, print_config_info
from logger import logger, log_query, log_retrieval, log_error
from error_handling import validate_input, ErrorContext
from rate_limiting import check_rate_limit, query_rate_limiter, user_rate_limiter
from caching import embedding_cache

# Use environment-based configuration
FAISS_INDEX_PATH = config.FAISS_INDEX_PATH
METADATA_PATH = config.METADATA_PATH
CONFIG_PATH = config.CONFIG_PATH
TOP_K = config.TOP_K
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD
GROQ_API_KEY = config.GROQ_API_KEY
GROQ_MODEL = config.GROQ_MODEL


def clean_ocr_errors(text: str) -> str:
    """Fix common OCR errors in text output."""
    text = re.sub(r'Y\s+ear', 'Year', text, flags=re.IGNORECASE)
    text = re.sub(r'Pr\s+oject', 'Project', text, flags=re.IGNORECASE)
    text = re.sub(r'F\s+inal', 'Final', text, flags=re.IGNORECASE)
    text = re.sub(r'R\s+eport', 'Report', text, flags=re.IGNORECASE)
    text = re.sub(r'D\s+evelopment', 'Development', text, flags=re.IGNORECASE)
    text = re.sub(r'H\s+andbook', 'Handbook', text, flags=re.IGNORECASE)
    text = re.sub(r'\b([A-Z])\s+([a-z]{2,})\b', r'\1\2', text)
    text = re.sub(r' +', ' ', text)
    return text


@st.cache_resource
def load_rag_system():
    """Load RAG system components (cached for performance)"""
    # Print config info (only in development)
    if config.DEBUG:
        print_config_info()
    
    # Check if required files exist
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, METADATA_PATH, CONFIG_PATH]):
        return None, None, None, None, None
    
    # Load configuration
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    
    # Load FAISS index
    index = faiss.read_index(FAISS_INDEX_PATH)
    
    # Load metadata
    with open(METADATA_PATH, 'rb') as f:
        chunks = pickle.load(f)
    
    # Load embedding model
    model = SentenceTransformer(config['embedding_model'])
    
    # Initialize Groq LLM
    llm = None
    if GROQ_API_KEY:
        try:
            llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        except Exception as e:
            st.warning(f"Failed to initialize Groq LLM: {e}")
    
    return index, chunks, model, config, llm


def retrieve_chunks(query: str, model, index, chunks, top_k: int = TOP_K) -> Tuple[List[Dict], List[float]]:
    """Retrieve top-k most relevant chunks with caching"""
    start_time = time.time()
    
    # Check cache first
    cached_embedding = embedding_cache.get(query)
    
    if cached_embedding is not None:
        query_embedding = cached_embedding
        logger.debug("Using cached embedding")
    else:
        # Embed query
        query_embedding = model.encode([query])
        # Cache the embedding
        embedding_cache.set(query, query_embedding)
        logger.debug("Generated and cached new embedding")
    
    # Normalize for cosine similarity
    faiss.normalize_L2(query_embedding)
    
    # Search in FAISS index
    scores, indices = index.search(query_embedding, top_k)
    
    # Get corresponding chunks
    retrieved_chunks = []
    retrieved_scores = []
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            retrieved_scores.append(float(score))
    
    elapsed = time.time() - start_time
    
    # Log retrieval metrics
    top_score = retrieved_scores[0] if retrieved_scores else 0.0
    log_retrieval(len(retrieved_chunks), top_score, elapsed)
    
    return retrieved_chunks, retrieved_scores


def format_context(chunks: List[Dict]) -> str:
    """Format retrieved chunks into context string"""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        page = chunk['page_number']
        text = chunk['text'].strip()
        section = chunk.get('section_hint', 'General')
        
        context_parts.append(
            f"[Chunk {i} - Page {page} - {section}]\n{text}\n"
        )
    
    return "\n".join(context_parts)


def generate_llm_answer(query: str, chunks: List[Dict], llm: GroqLLM) -> Dict:
    """
    Generate answer using Groq LLM with retrieved chunks.
    Returns conversational, well-formatted answer with citations.
    """
    try:
        # Start timing (if enabled)
        start_time = time.time() if config.SHOW_TIMING else None
        
        # Format context for LLM
        context = format_context_for_llm(chunks, max_chunks=5)
        
        # Generate answer using LLM
        result = llm.generate_answer(
            question=query,
            context=context,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        
        # Add timing info (if enabled)
        if config.SHOW_TIMING and start_time:
            elapsed = time.time() - start_time
            result['timing'] = f"{elapsed:.2f}s"
            if config.DEBUG:
                print(f"⏱️ LLM generation took {elapsed:.2f}s")
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'answer': f"Error generating answer: {str(e)}. Please check your API key and internet connection.",
            'error': str(e)
        }


def extract_answer_from_chunks(query: str, chunks: List[Dict]) -> str:
    """
    Extract answer from chunks with page citations.
    Simple implementation - in production, use an LLM like GPT-4.
    """
    # Organize content by topic/section
    answer_content = []
    references = []
    seen_pages = set()
    
    for chunk in chunks[:3]:  # Use top 3 chunks
        page = chunk['page_number']
        text = clean_ocr_errors(chunk['text'].strip())
        
        if page not in seen_pages:
            # Extract key points (look for bullet-style content)
            lines = text.split('\n')
            key_points = []
            
            for line in lines:
                line = line.strip()
                if len(line) > 20 and not line.isupper():  # Skip very short lines and headers
                    # Check if it's a complete sentence or key point
                    if line.endswith('.') or line.endswith(':') or line.endswith('!') or len(line) > 50:
                        # Remove handbook prefix if present in the line
                        if 'Handbook 2023' in line:
                            # Find the year and take everything after it
                            match = re.search(r'Handbook 2023\s*(.+)', line)
                            if match:
                                cleaned_line = match.group(1).strip()
                                # Remove leading asterisk or bullet if present
                                cleaned_line = re.sub(r'^[*•]\s*', '', cleaned_line)
                                if cleaned_line:
                                    key_points.append(cleaned_line)
                            else:
                                key_points.append(line)
                        else:
                            key_points.append(line)
            
            # If we found key points, format them nicely
            if key_points:
                # Group related content
                for point in key_points[:4]:  # Limit to 4 points per chunk
                    if point:
                        answer_content.append(f"• {point}")
                
                # Add reference for this page
                references.append(f"Page {page}")
                seen_pages.add(page)
    
    # Build final answer with clear sections
    if answer_content:
        answer = "\n\n".join(answer_content)
        ref_section = "\n\n" + "─" * 50 + "\n\n📚 **Quick References:** " + ", ".join(references)
        return answer + ref_section
    else:
        return "I found relevant information but couldn't extract a clear answer. Please refer to the detailed sources below."


def generate_answer(query: str, chunks: List[Dict], scores: List[float], llm: GroqLLM) -> Dict:
    """Generate answer from retrieved chunks using Groq LLM"""
    # Check if best match is above threshold
    if not scores or scores[0] < SIMILARITY_THRESHOLD:
        return {
            'answer': "I don't have that information in the handbook. Please make sure your question is about the FYP handbook content.",
            'sources': [],
            'scores': scores,
            'confidence': 'low',
            'tokens_used': None,
            'llm_used': False
        }
    
    # Generate answer using LLM
    llm_result = generate_llm_answer(query, chunks, llm)
    
    if not llm_result.get('success', False):
        # Fallback if LLM fails
        return {
            'answer': llm_result.get('answer', 'Error generating answer.'),
            'sources': [],
            'scores': scores,
            'confidence': 'low',
            'tokens_used': None,
            'llm_used': False,
            'error': llm_result.get('error', 'Unknown error')
        }
    
    # Format sources
    sources = []
    seen_pages = set()
    for chunk, score in zip(chunks, scores):
        page = chunk['page_number']
        if page not in seen_pages:
            sources.append({
                'page': page,
                'section': chunk.get('section_hint', 'General'),
                'score': score,
                'snippet': chunk['text'][:300] + "..." if len(chunk['text']) > 300 else chunk['text']
            })
            seen_pages.add(page)
    
    return {
        'answer': llm_result['answer'],
        'sources': sources,
        'scores': scores,
        'confidence': 'high' if scores[0] > 0.5 else 'medium',
        'tokens_used': llm_result.get('tokens_used'),
        'model': llm_result.get('model'),
        'llm_used': True
    }


def main():
    """Main Streamlit app"""
    st.set_page_config(
        page_title="FYP Handbook Assistant",
        page_icon="📚",
        layout="wide"
    )
    
    # Header
    st.title("📚 FYP Handbook Assistant")
    st.markdown("*Ask questions about the FAST-NUCES FYP process*")
    st.divider()
    
    # Load RAG system
    index, chunks, model, config, llm = load_rag_system()
    
    # Check if system is loaded
    if index is None:
        st.error("⚠️ RAG system not initialized!")
        st.info("Please run `python ingest.py` first to create the FAISS index.")
        st.stop()
    
    # Check API key
    if not GROQ_API_KEY:
        st.error("⚠️ Groq API key not found!")
        st.info("Please set GROQ_API_KEY in your .env file or environment variables.")
        st.code('GROQ_API_KEY=your_api_key_here', language='bash')
        st.stop()
    
    if llm is None:
        st.error("⚠️ Failed to initialize Groq LLM!")
        st.info("Please check your API key and internet connection.")
        st.stop()
    
    # Display system info in sidebar
    with st.sidebar:
        st.header("System Information")
        
        # Show environment mode
        env_mode = "🏭 PRODUCTION" if not config.DEBUG else "🔧 DEVELOPMENT"
        st.info(env_mode)
        
        # Rate limit status
        rate_status = query_rate_limiter.get_status()
        st.metric(
            "Request Capacity", 
            f"{rate_status['remaining']}/{rate_status['max_requests']}",
            help="Available requests in current time window"
        )
        
        # Cache statistics (in debug mode)
        if config.DEBUG:
            cache_stats = embedding_cache.get_stats()
            st.metric(
                "Cache Hit Rate",
                f"{cache_stats['valid_entries']}/{cache_stats['total_entries']}",
                help="Cached embeddings reduce API costs"
            )
        
        st.metric("Total Chunks", config['num_chunks'])
        st.metric("Chunk Size", f"{config['chunk_size']} words")
        st.metric("Overlap", f"{config['overlap']} words")
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.metric("LLM Model", GROQ_MODEL)
        st.metric("Top-K Retrieval", TOP_K)
        st.metric("Similarity Threshold", SIMILARITY_THRESHOLD)
        
        # Show debug info in development mode
        if config.DEBUG:
            st.metric("Max Tokens", config.LLM_MAX_TOKENS)
            st.metric("Temperature", config.LLM_TEMPERATURE)
        
        # API Status
        if validate_api_key(GROQ_API_KEY):
            st.success("✅ Groq API Connected")
        else:
            st.error("❌ Groq API Not Connected")
        
        st.divider()
        
        st.header("Sample Questions")
        st.markdown("""
        1. What headings, fonts, and sizes are required?
        2. What margins and spacing do we use?
        3. Required chapters of Development FYP?
        4. Required chapters of R&D FYP?
        5. How to use 'Ibid.' and 'op. cit.'?
        6. What goes into Executive Summary?
        """)
    
    # Main input area
    st.subheader("Ask a Question")
    
    # Query input
    query = st.text_input(
        "Enter your question about the FYP handbook:",
        placeholder="e.g., What are the required chapters of a Development FYP report?",
        key="query_input"
    )
    
    # Ask button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        ask_button = st.button("🔍 Ask", type="primary", use_container_width=True)
    
    # Process query
    if ask_button and query.strip():
        # Log the query
        log_query(query)
        
        # Validate input
        is_valid, error_msg = validate_input(query, min_length=3, max_length=500)
        if not is_valid:
            st.error(f"❌ {error_msg}")
            logger.warning(f"Invalid query: {error_msg}")
        else:
            # Check rate limits
            user_id = st.session_state.get('user_id', 'anonymous')
            allowed, rate_limit_msg = check_rate_limit(user_id)
            
            if not allowed:
                st.error(f"❌ {rate_limit_msg}")
                logger.warning(f"Rate limit exceeded for user: {user_id}")
            else:
                with st.spinner("Searching handbook..."):
                    try:
                        # Retrieve chunks
                        retrieved_chunks, scores = retrieve_chunks(query, model, index, chunks, TOP_K)
                        
                        # Generate answer using LLM
                        result = generate_answer(query, retrieved_chunks, scores, llm)
                        
                        # Display answer
                        st.divider()
                        st.subheader("📝 Answer")
                        
                        # Confidence badge
                        confidence_color = {
                            'high': '🟢',
                            'medium': '🟡',
                            'low': '🔴'
                        }
                        st.markdown(f"{confidence_color[result['confidence']]} Confidence: **{result['confidence'].title()}**")
                        
                        # Answer text - use a container for better styling
                        answer_container = st.container()
                        with answer_container:
                            st.markdown(result['answer'])
                        
                        # Show timing in debug mode
                        if config.SHOW_TIMING and 'timing' in result:
                            st.caption(f"⏱️ Response time: {result['timing']}")
                        
                    except Exception as e:
                        log_error(e, {"query": query, "user_id": user_id})
                        st.error(f"❌ An error occurred while processing your query. Please try again.")
                        if config.DEBUG:
                            st.exception(e)
        
        # Show token usage if available
        if result.get('tokens_used'):
            tokens = result['tokens_used']
            col_t1, col_t2, col_t3 = st.columns(3)
            with col_t1:
                st.caption(f"📊 Prompt: {tokens['prompt']} tokens")
            with col_t2:
                st.caption(f"📊 Response: {tokens['completion']} tokens")
            with col_t3:
                st.caption(f"📊 Total: {tokens['total']} tokens")
        
        # Sources section (collapsible)
        st.divider()
        with st.expander("📚 **Sources (Page References)**", expanded=True):
            if result['sources']:
                for i, source in enumerate(result['sources'], 1):
                    st.markdown(f"**{i}. Page {source['page']}** - *{source['section']}*")
                    st.caption(f"Relevance Score: {source['score']:.3f}")
                    st.text(source['snippet'])
                    st.markdown("---")
            else:
                st.info("No sources found above the similarity threshold.")
        
        # Debug info (collapsible)
        with st.expander("🔧 Debug Information"):
            st.markdown("**Similarity Scores:**")
            st.write(result['scores'])
            
            if result.get('model'):
                st.markdown(f"**LLM Model Used:** {result['model']}")
            
            if result.get('tokens_used'):
                st.markdown("**Token Usage:**")
                st.json(result['tokens_used'])
    
    elif ask_button and not query.strip():
        st.warning("Please enter a question!")
    
    # Footer
    st.divider()
    st.caption("FYP Handbook RAG Assistant | Built with Sentence-BERT & FAISS")


if __name__ == "__main__":
    main()
