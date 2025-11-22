"""
FYP Handbook RAG - Streamlit Web Interface
Minimal UI with input box, Ask button, Answer panel, and collapsible Sources
"""

import streamlit as st
import os
import pickle
import json
import re
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configuration
FAISS_INDEX_PATH = "faiss_index.bin"
METADATA_PATH = "chunks_metadata.pkl"
CONFIG_PATH = "config.json"
TOP_K = 5
SIMILARITY_THRESHOLD = 0.25

# Prompt template
PROMPT_TEMPLATE = """You are a handbook assistant. Answer ONLY from the context.
Cite page numbers like "(p. X)". If unsure, say you don't know.

Question: {user_question}

Context:
{top_chunks_text}

Answer:"""


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
    # Check if required files exist
    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, METADATA_PATH, CONFIG_PATH]):
        return None, None, None, None
    
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
    
    return index, chunks, model, config


def retrieve_chunks(query: str, model, index, chunks, top_k: int = TOP_K) -> Tuple[List[Dict], List[float]]:
    """Retrieve top-k most relevant chunks"""
    # Embed query
    query_embedding = model.encode([query])
    
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


def generate_answer(query: str, chunks: List[Dict], scores: List[float]) -> Dict:
    """Generate answer from retrieved chunks"""
    # Check if best match is above threshold
    if not scores or scores[0] < SIMILARITY_THRESHOLD:
        return {
            'answer': "I don't have that information in the handbook. Please make sure your question is about the FYP handbook content.",
            'sources': [],
            'scores': scores,
            'confidence': 'low',
            'prompt': ''
        }
    
    # Format context
    context = format_context(chunks)
    
    # Create prompt
    prompt = PROMPT_TEMPLATE.format(
        user_question=query,
        top_chunks_text=context
    )
    
    # Extract answer
    answer = extract_answer_from_chunks(query, chunks)
    
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
        'answer': answer,
        'sources': sources,
        'scores': scores,
        'confidence': 'high' if scores[0] > 0.5 else 'medium',
        'prompt': prompt
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
    index, chunks, model, config = load_rag_system()
    
    # Check if system is loaded
    if index is None:
        st.error("⚠️ RAG system not initialized!")
        st.info("Please run `python ingest.py` first to create the FAISS index.")
        st.stop()
    
    # Display system info in sidebar
    with st.sidebar:
        st.header("System Information")
        st.metric("Total Chunks", config['num_chunks'])
        st.metric("Chunk Size", f"{config['chunk_size']} words")
        st.metric("Overlap", f"{config['overlap']} words")
        st.metric("Embedding Model", "all-MiniLM-L6-v2")
        st.metric("Top-K Retrieval", TOP_K)
        st.metric("Similarity Threshold", SIMILARITY_THRESHOLD)
        
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
        with st.spinner("Searching handbook..."):
            # Retrieve chunks
            retrieved_chunks, scores = retrieve_chunks(query, model, index, chunks, TOP_K)
            
            # Generate answer
            result = generate_answer(query, retrieved_chunks, scores)
        
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
            # Split answer and references if they exist
            if "─" in result['answer']:
                answer_part, ref_part = result['answer'].split("─" * 50, 1)
                st.markdown(answer_part.strip())
                st.markdown("---")
                st.markdown(ref_part.strip())
            else:
                st.markdown(result['answer'])
        
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
            
            st.markdown("**Generated Prompt:**")
            st.code(result['prompt'], language='text')
    
    elif ask_button and not query.strip():
        st.warning("Please enter a question!")
    
    # Footer
    st.divider()
    st.caption("FYP Handbook RAG Assistant | Built with Sentence-BERT & FAISS")


if __name__ == "__main__":
    main()
