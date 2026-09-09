"""
FYP Handbook RAG - Developer Console & Inference Engine
Streamlit web interface with developer console design system
"""

import streamlit as st
import os
import pickle
import json
import re
import time
import threading
from typing import List, Dict, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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

# Lucide SVG Icon Constants
SVG_SEARCH = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/></svg>'
SVG_CPU = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect width="16" height="16" x="4" y="4" rx="2"/><rect width="6" height="6" x="9" y="9" rx="1"/><path d="M15 2v2"/><path d="M15 20v2"/><path d="M2 15h2"/><path d="M2 9h2"/><path d="M20 15h2"/><path d="M20 9h2"/><path d="M9 2v2"/><path d="M9 20v2"/></svg>'
SVG_DATABASE = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/><path d="M3 12c0 1.66 4 3 9 3s9-1.34 9-3"/></svg>'
SVG_BOOK = '<svg xmlns="http://www.w3.org/2000/svg" width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M4 19.5v-15A2.5 2.5 0 0 1 6.5 2H20v20H6.5a2.5 2.5 0 0 1 0-5H20"/></svg>'
SVG_CHECK = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="20 6 9 17 4 12"/></svg>'
SVG_ALERT = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><line x1="12" x2="12" y1="8" y2="12"/><line x1="12" x2="12.01" y1="16" y2="16"/></svg>'
SVG_TERMINAL = '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="4 17 10 11 4 5"/><line x1="12" x2="20" y1="19" y2="19"/></svg>'
SVG_GAUGE = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="m12 14 4-4"/><path d="M3.34 19a10 10 0 1 1 17.32 0"/></svg>'
SVG_LAYERS = '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>'


@st.cache_data
def clean_ocr_errors(text: str) -> str:
    """Fix common OCR errors in text output (cached)"""
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
    """Load RAG system components (cached for performance).
    
    Heavy steps (in order of cost):
      1. SentenceTransformer — loads ~90 MB of PyTorch weights  (~4-7s cold)
      2. FAISS index read                                        (~0.2s)
      3. Pickle metadata load                                    (~0.1s)
      4. Groq client init                                        (~0.3s)
    
    @st.cache_resource ensures this runs ONCE per server process and the
    result is shared across all user sessions.
    """
    if config.DEBUG:
        print_config_info()

    if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, METADATA_PATH, CONFIG_PATH]):
        return None, None, None, None, None

    with open(CONFIG_PATH, 'r') as f:
        rag_config = json.load(f)

    index = faiss.read_index(FAISS_INDEX_PATH)

    with open(METADATA_PATH, 'rb') as f:
        chunks = pickle.load(f)

    # Primary bottleneck: loading PyTorch weights for the embedding model
    model = SentenceTransformer(rag_config['embedding_model'])

    llm = None
    if GROQ_API_KEY:
        try:
            llm = GroqLLM(api_key=GROQ_API_KEY, model=GROQ_MODEL)
        except Exception as e:
            logger.warning(f"Failed to initialize Groq LLM: {e}")

    return index, chunks, model, rag_config, llm


def _prewarm_rag_system():
    """Background pre-warm: trigger load_rag_system() immediately when the
    server process starts so the cache is already populated by the time the
    first HTTP request arrives.  Runs in a daemon thread — any exception is
    logged but never propagates to the Streamlit main thread."""
    try:
        logger.info("[prewarm] Starting background RAG system pre-warm...")
        t0 = time.time()
        load_rag_system()          # fills @st.cache_resource
        elapsed = time.time() - t0
        logger.info(f"[prewarm] RAG system ready in {elapsed:.1f}s")
    except Exception as exc:
        logger.error(f"[prewarm] Pre-warm failed: {exc}")


# Kick off pre-warm exactly once when this module is first imported.
# Streamlit re-imports the module on every server restart but not on every
# page refresh, so this runs at most once per server session.
_prewarm_thread = threading.Thread(target=_prewarm_rag_system, daemon=True, name="rag-prewarm")
_prewarm_thread.start()


def retrieve_chunks(query: str, model, index, chunks, top_k: int = TOP_K) -> Tuple[List[Dict], List[float]]:
    """Retrieve top-k most relevant chunks with caching"""
    start_time = time.time()
    
    cached_embedding = embedding_cache.get(query)
    
    if cached_embedding is not None:
        query_embedding = cached_embedding
        logger.debug("Using cached embedding")
    else:
        query_embedding = model.encode([query])
        embedding_cache.set(query, query_embedding)
        logger.debug("Generated and cached new embedding")
    
    faiss.normalize_L2(query_embedding)
    scores, indices = index.search(query_embedding, top_k)
    
    retrieved_chunks = []
    retrieved_scores = []
    
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks):
            retrieved_chunks.append(chunks[idx])
            retrieved_scores.append(float(score))
    
    elapsed = time.time() - start_time
    top_score = retrieved_scores[0] if retrieved_scores else 0.0
    log_retrieval(len(retrieved_chunks), top_score, elapsed)
    
    return retrieved_chunks, retrieved_scores


def generate_llm_answer(query: str, chunks: List[Dict], llm: GroqLLM) -> Dict:
    """Generate answer using Groq LLM with retrieved chunks."""
    try:
        start_time = time.time()
        context = format_context_for_llm(chunks, max_chunks=5)
        
        result = llm.generate_answer(
            question=query,
            context=context,
            max_tokens=config.LLM_MAX_TOKENS,
            temperature=config.LLM_TEMPERATURE
        )
        
        elapsed = time.time() - start_time
        result['timing'] = f"{elapsed:.2f}s"
        return result
        
    except Exception as e:
        return {
            'success': False,
            'answer': f"Error generating answer: {str(e)}",
            'error': str(e)
        }


def generate_answer(query: str, chunks: List[Dict], scores: List[float], llm: GroqLLM) -> Dict:
    """Generate answer from retrieved chunks using Groq LLM"""
    if not scores or scores[0] < SIMILARITY_THRESHOLD:
        return {
            'answer': "I don't have that information in the handbook. Please make sure your question is about the FYP handbook content.",
            'sources': [],
            'scores': scores,
            'confidence': 'low',
            'tokens_used': None,
            'llm_used': False
        }
    
    llm_result = generate_llm_answer(query, chunks, llm)
    
    if not llm_result.get('success', False):
        return {
            'answer': llm_result.get('answer', 'Error generating answer.'),
            'sources': [],
            'scores': scores,
            'confidence': 'low',
            'tokens_used': None,
            'llm_used': False,
            'error': llm_result.get('error', 'Unknown error')
        }
    
    sources = []
    seen_pages = set()
    for chunk, score in zip(chunks, scores):
        page = chunk['page_number']
        if page not in seen_pages:
            sources.append({
                'page': page,
                'section': chunk.get('section_hint', 'General Section'),
                'score': score,
                'snippet': chunk['text'][:350] + "..." if len(chunk['text']) > 350 else chunk['text']
            })
            seen_pages.add(page)
    
    return {
        'answer': llm_result['answer'],
        'sources': sources,
        'scores': scores,
        'confidence': 'high' if scores[0] > 0.5 else 'medium',
        'tokens_used': llm_result.get('tokens_used'),
        'model': llm_result.get('model', GROQ_MODEL),
        'timing': llm_result.get('timing', 'N/A'),
        'llm_used': True
    }


def inject_developer_css():
    """Inject custom developer console CSS theme — enhanced with full accent color system"""
    st.markdown("""
        <style>
        /* ====================================================
           FYP Handbook Inference Engine — Developer Console
           Enhanced Color System & Visual Polish
        ==================================================== */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

        /* ---------- Design Tokens ---------- */
        :root {
            /* Backgrounds */
            --bg-base:        #080c14;
            --bg-surface:     #0e1420;
            --bg-card:        #131c2e;
            --bg-card-hover:  #192438;
            --bg-elevated:    #1a2540;

            /* Borders */
            --border-subtle:  #1c2a42;
            --border-default: #243350;
            --border-strong:  #2e4268;

            /* Text */
            --text-primary:   #f0f6ff;
            --text-secondary: #a0b4cc;
            --text-muted:     #5a7095;
            --text-dim:       #3a5070;

            /* Accent — Cyan (primary actions, links, tags) */
            --cyan-500:   #0ea5e9;
            --cyan-400:   #38bdf8;
            --cyan-glow:  rgba(14, 165, 233, 0.25);
            --cyan-faint: rgba(14, 165, 233, 0.08);

            /* Accent — Green (status, online, high-confidence) */
            --green-500:  #10b981;
            --green-400:  #34d399;
            --green-glow: rgba(16, 185, 129, 0.22);
            --green-faint:rgba(16, 185, 129, 0.08);

            /* Accent — Amber (section headers, highlights, stats) */
            --amber-500:  #f59e0b;
            --amber-400:  #fbbf24;
            --amber-glow: rgba(245, 158, 11, 0.22);
            --amber-faint:rgba(245, 158, 11, 0.08);

            /* Accent — Purple (secondary decorative) */
            --purple-500: #8b5cf6;
            --purple-400: #a78bfa;
            --purple-faint:rgba(139, 92, 246, 0.08);

            /* Danger / Low */
            --red-400:    #f87171;
            --red-faint:  rgba(248, 113, 113, 0.08);

            /* Fonts & Shape */
            --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
            --radius-xs: 3px;
            --radius-sm: 5px;
            --radius-md: 8px;
            --radius-lg: 12px;
        }

        /* ---------- Streamlit Chrome Cleanup ---------- */
        #MainMenu          { visibility: hidden; }
        footer             { visibility: hidden; }
        .stDeployButton    { display: none; }
        .stDivider         { display: none !important; }

        /* ---------- Page Foundation ---------- */
        .stApp {
            background-color: var(--bg-base);
            color: var(--text-primary);
            font-family: var(--font-sans);
        }

        /* Subtle grid texture overlay on main content */
        [data-testid="stAppViewContainer"] > .main {
            background-image:
                linear-gradient(rgba(14,165,233,0.015) 1px, transparent 1px),
                linear-gradient(90deg, rgba(14,165,233,0.015) 1px, transparent 1px);
            background-size: 40px 40px;
        }

        /* ---------- Sidebar — Pinned Left ---------- */
        [data-testid="stSidebar"] {
            background-color: var(--bg-surface) !important;
            border-right: 1px solid var(--border-default) !important;
            display: flex !important;
            visibility: visible !important;
            transform: none !important;
            min-width: 270px !important;
            max-width: 310px !important;
            padding-top: 0.75rem !important;
            margin-left: 0 !important;
        }

        [data-testid="stHeader"] {
            background-color: transparent !important;
        }

        [data-testid="stSidebarCollapseButton"],
        button[aria-label="Close sidebar"],
        button[aria-label="Open sidebar"] {
            display: none !important;
        }

        /* ---------- Sidebar Inner Styling ---------- */

        /* Section title — amber accent, higher contrast */
        .sidebar-section-title {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--amber-500);
            margin: 1.4rem 0 0.6rem 0;
            padding-bottom: 0.3rem;
            border-bottom: 1px solid var(--border-default);
        }

        /* Stat rows — boosted contrast on labels */
        .console-stat-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 0.38rem 0;
            font-size: 0.82rem;
            border-bottom: 1px solid transparent;
        }

        .console-stat-row:hover {
            border-bottom-color: var(--border-subtle);
        }

        /* Label: was --text-muted (#64748b), now much more readable */
        .console-stat-label {
            color: var(--text-secondary);
            font-weight: 400;
        }

        /* Values: bright white mono */
        .console-stat-val {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--text-primary);
        }

        /* Highlighted / accent stat values */
        .console-stat-val-cyan {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--cyan-400);
        }

        .console-stat-val-amber {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--amber-400);
        }

        .console-stat-val-green {
            font-family: var(--font-mono);
            font-weight: 600;
            font-size: 0.8rem;
            color: var(--green-400);
        }

        /* Metric card (optional usage in sidebar) */
        .sidebar-metric-card {
            background: var(--bg-card);
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 0.55rem 0.75rem;
            margin-bottom: 0.5rem;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        /* ---------- Badges ---------- */
        .console-badge {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: var(--radius-xs);
            background: var(--cyan-faint);
            color: var(--cyan-400);
            border: 1px solid rgba(14, 165, 233, 0.28);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-green {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: var(--radius-xs);
            background: var(--green-faint);
            color: var(--green-400);
            border: 1px solid rgba(16, 185, 129, 0.28);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-amber {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: var(--radius-xs);
            background: var(--amber-faint);
            color: var(--amber-400);
            border: 1px solid rgba(245, 158, 11, 0.28);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .badge-red {
            font-family: var(--font-mono);
            font-size: 0.65rem;
            font-weight: 700;
            padding: 2px 8px;
            border-radius: var(--radius-xs);
            background: var(--red-faint);
            color: var(--red-400);
            border: 1px solid rgba(248, 113, 113, 0.28);
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        /* Animated pulse dot for ONLINE status */
        .status-dot {
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--green-500);
            box-shadow: 0 0 0 2px var(--green-faint);
            animation: pulse-dot 2s ease-in-out infinite;
            vertical-align: middle;
            margin-right: 5px;
        }

        @keyframes pulse-dot {
            0%, 100% { box-shadow: 0 0 0 2px var(--green-faint); }
            50%       { box-shadow: 0 0 0 5px transparent; }
        }

        /* ---------- Console Header ---------- */
        .console-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.6rem 0 1.1rem 0;
            border-bottom: 1px solid var(--border-default);
            margin-bottom: 1.4rem;
        }

        .console-title {
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: -0.025em;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .console-subtitle {
            font-size: 0.78rem;
            color: var(--text-secondary);
            margin-top: 3px;
            letter-spacing: 0.01em;
        }

        /* ---------- Input ---------- */
        div[data-baseweb="input"] {
            background-color: var(--bg-surface) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        div[data-baseweb="input"]:focus-within {
            border-color: var(--cyan-500) !important;
            box-shadow: 0 0 0 2px var(--cyan-glow) !important;
        }

        input {
            color: var(--text-primary) !important;
            font-size: 0.91rem !important;
        }

        /* ---------- Buttons ---------- */
        button[kind="primary"],
        button[data-testid="stBaseButton-primary"] {
            background: linear-gradient(135deg, var(--cyan-500), #0284c7) !important;
            color: #ffffff !important;
            font-weight: 600 !important;
            font-size: 0.88rem !important;
            border: none !important;
            border-radius: var(--radius-md) !important;
            padding: 0.55rem 1.25rem !important;
            transition: all 0.18s ease !important;
            width: 100%;
            letter-spacing: 0.02em;
        }

        button[kind="primary"]:hover,
        button[data-testid="stBaseButton-primary"]:hover {
            background: linear-gradient(135deg, var(--cyan-400), var(--cyan-500)) !important;
            box-shadow: 0 0 18px var(--cyan-glow) !important;
            transform: translateY(-1px) !important;
        }

        button[kind="secondary"],
        button[data-testid="stBaseButton-secondary"] {
            background-color: var(--bg-card) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            border-radius: var(--radius-md) !important;
            padding: 0.55rem 0.85rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 400 !important;
            font-size: 0.82rem !important;
            width: 100% !important;
            height: auto !important;
            min-height: 44px !important;
            white-space: normal !important;
            word-wrap: break-word !important;
            transition: all 0.15s ease !important;
        }

        button[kind="secondary"]:hover,
        button[data-testid="stBaseButton-secondary"]:hover {
            border-color: var(--cyan-500) !important;
            color: var(--text-primary) !important;
            background-color: var(--bg-card-hover) !important;
            box-shadow: 0 0 8px rgba(14, 165, 233, 0.12) !important;
        }

        /* ---------- Answer Panel ---------- */
        .answer-panel {
            background-color: var(--bg-surface);
            border: 1px solid var(--border-default);
            border-top: 2px solid var(--cyan-500);
            border-radius: var(--radius-md);
            padding: 1.5rem;
            margin-top: 1.25rem;
            box-shadow: 0 4px 24px rgba(0, 0, 0, 0.35), 0 0 40px -15px var(--cyan-glow);
        }

        .answer-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
            padding-bottom: 0.75rem;
            border-bottom: 1px solid var(--border-subtle);
        }

        .answer-header-title {
            font-size: 0.9rem;
            font-weight: 600;
            color: var(--text-primary);
            display: flex;
            align-items: center;
            gap: 0.5rem;
            letter-spacing: 0.02em;
        }

        .answer-body {
            font-size: 0.96rem;
            line-height: 1.7;
            color: #daeaf8;
        }

        .answer-body ul, .answer-body ol {
            padding-left: 1.25rem;
            margin-top: 0.5rem;
            margin-bottom: 0.5rem;
        }

        .answer-body li {
            margin-bottom: 0.4rem;
        }

        /* ---------- Metadata Strip ---------- */
        .meta-strip {
            display: flex;
            flex-wrap: wrap;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            background: linear-gradient(90deg, var(--bg-card), var(--bg-elevated));
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--cyan-500);
            border-radius: var(--radius-sm);
            padding: 0.65rem 1rem;
            margin-top: 0.9rem;
            font-size: 0.78rem;
        }

        .meta-group {
            display: flex;
            align-items: center;
            gap: 1.25rem;
        }

        .meta-item {
            display: flex;
            align-items: center;
            gap: 0.35rem;
            color: var(--text-secondary);
        }

        .meta-value {
            font-family: var(--font-mono);
            color: var(--cyan-400);
            font-weight: 600;
        }

        /* ---------- Confidence Pills ---------- */
        .pill-high {
            font-family: var(--font-mono);
            font-size: 0.67rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 20px;
            background: var(--green-faint);
            color: var(--green-400);
            border: 1px solid rgba(16, 185, 129, 0.35);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            box-shadow: 0 0 8px var(--green-glow);
        }

        .pill-medium {
            font-family: var(--font-mono);
            font-size: 0.67rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 20px;
            background: var(--amber-faint);
            color: var(--amber-400);
            border: 1px solid rgba(245, 158, 11, 0.35);
            text-transform: uppercase;
            letter-spacing: 0.06em;
            box-shadow: 0 0 8px var(--amber-glow);
        }

        .pill-low {
            font-family: var(--font-mono);
            font-size: 0.67rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: 20px;
            background: var(--red-faint);
            color: var(--red-400);
            border: 1px solid rgba(248, 113, 113, 0.3);
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        /* ---------- Source Cards ---------- */
        .source-card {
            background: linear-gradient(135deg, var(--bg-surface), var(--bg-card));
            border: 1px solid var(--border-subtle);
            border-left: 3px solid var(--cyan-500);
            border-radius: var(--radius-md);
            padding: 0.9rem 1.1rem;
            margin-bottom: 0.75rem;
            transition: border-color 0.15s ease, box-shadow 0.15s ease;
        }

        .source-card:hover {
            border-color: var(--border-strong);
            border-left-color: var(--cyan-400);
            box-shadow: 0 2px 12px rgba(0, 0, 0, 0.3);
        }

        .source-card-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 0.6rem;
        }

        .page-badge {
            font-family: var(--font-mono);
            font-size: 0.68rem;
            font-weight: 700;
            padding: 3px 9px;
            border-radius: var(--radius-xs);
            background: var(--cyan-faint);
            color: var(--cyan-400);
            border: 1px solid rgba(14, 165, 233, 0.3);
            letter-spacing: 0.04em;
        }

        .score-bar-container {
            display: flex;
            align-items: center;
            gap: 0.5rem;
            font-family: var(--font-mono);
            font-size: 0.72rem;
            color: var(--text-secondary);
        }

        .score-progress {
            width: 65px;
            height: 4px;
            background: var(--bg-card);
            border-radius: 3px;
            overflow: hidden;
        }

        .score-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--cyan-500), var(--cyan-400));
            border-radius: 3px;
        }

        .source-snippet {
            font-family: var(--font-mono);
            font-size: 0.79rem;
            color: #a8c0d8;
            background-color: #060b12;
            border: 1px solid var(--border-subtle);
            border-radius: var(--radius-sm);
            padding: 0.65rem 0.85rem;
            white-space: pre-wrap;
            word-break: break-word;
            margin: 0;
            line-height: 1.55;
        }

        /* ---------- Expander ---------- */
        .streamlit-expanderHeader {
            background-color: var(--bg-surface) !important;
            border: 1px solid var(--border-default) !important;
            border-radius: var(--radius-md) !important;
            color: var(--text-primary) !important;
            font-size: 0.86rem !important;
            font-weight: 600 !important;
            font-family: var(--font-mono) !important;
            letter-spacing: 0.02em !important;
        }

        .streamlit-expanderContent {
            background-color: var(--bg-base) !important;
            border: 1px solid var(--border-default) !important;
            border-top: none !important;
            border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
            padding: 1rem !important;
        }

        /* ---------- Suggested Question Buttons ---------- */
        .sample-query-btn button {
            background-color: var(--bg-card) !important;
            color: var(--text-secondary) !important;
            border: 1px solid var(--border-subtle) !important;
            font-size: 0.78rem !important;
            padding: 0.4rem 0.75rem !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-weight: 400 !important;
        }

        .sample-query-btn button:hover {
            border-color: var(--cyan-500) !important;
            color: var(--cyan-400) !important;
            background-color: var(--bg-card-hover) !important;
        }

        /* ---------- Scrollbar Styling ---------- */
        ::-webkit-scrollbar       { width: 5px; height: 5px; }
        ::-webkit-scrollbar-track { background: var(--bg-base); }
        ::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 3px; }
        ::-webkit-scrollbar-thumb:hover { background: var(--cyan-500); }
        /* ---------- Spinner override ---------- */
        .stSpinner > div { border-top-color: var(--cyan-500) !important; }

        </style>
    """, unsafe_allow_html=True)


def render_sidebar(rag_config):
    """Render developer console status sidebar — with full accent color system"""
    with st.sidebar:
        # Sidebar header with animated pulse dot for ONLINE status
        st.markdown(f"""
            <div style="display: flex; align-items: center; justify-content: space-between; padding-bottom: 0.75rem; border-bottom: 1px solid var(--border-default);">
                <div style="font-weight: 700; font-size: 0.9rem; letter-spacing: 0.02em; display: flex; align-items: center; gap: 0.5rem; color: var(--text-primary);">
                    {SVG_TERMINAL} DEVELOPER CONSOLE
                </div>
                <span class="badge-green"><span class="status-dot"></span>ONLINE</span>
            </div>
        """, unsafe_allow_html=True)

        # ── Section 1: System & Performance ──────────────────────────
        env_mode = "PRODUCTION" if not config.DEBUG else "DEVELOPMENT"
        api_connected = validate_api_key(GROQ_API_KEY)
        rate_status = query_rate_limiter.get_status()
        env_badge_class = "badge-amber" if not config.DEBUG else "console-badge"
        api_badge_class = "badge-green" if api_connected else "badge-red"
        api_label = "CONNECTED" if api_connected else "DISCONNECTED"

        st.markdown('<div class="sidebar-section-title">System &amp; Performance</div>', unsafe_allow_html=True)
        st.markdown(f'''
            <div class="console-stat-row">
                <span class="console-stat-label">Environment</span>
                <span class="{env_badge_class}">{env_mode}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">API Status</span>
                <span class="{api_badge_class}">{api_label}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Rate Remaining</span>
                <span class="console-stat-val-cyan">{rate_status['remaining']} / {rate_status['max_requests']}</span>
            </div>
        ''', unsafe_allow_html=True)

        if config.DEBUG:
            cache_stats = embedding_cache.get_stats()
            st.markdown(f'''
                <div class="console-stat-row">
                    <span class="console-stat-label">Embedding Cache</span>
                    <span class="console-stat-val-cyan">{cache_stats['valid_entries']} cached</span>
                </div>
            ''', unsafe_allow_html=True)

        # ── Section 2: Vector Indexing ────────────────────────────────
        st.markdown('<div class="sidebar-section-title">Vector Indexing</div>', unsafe_allow_html=True)
        st.markdown(f'''
            <div class="console-stat-row">
                <span class="console-stat-label">Total Index Chunks</span>
                <span class="console-stat-val-cyan">{rag_config['num_chunks']}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Chunk Size</span>
                <span class="console-stat-val">{rag_config['chunk_size']} words</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Overlap</span>
                <span class="console-stat-val">{rag_config['overlap']} words</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Embedding Model</span>
                <span class="console-stat-val">MiniLM-L6-v2</span>
            </div>
        ''', unsafe_allow_html=True)

        # ── Section 3: Inference Engine ───────────────────────────────
        st.markdown('<div class="sidebar-section-title">Inference Engine</div>', unsafe_allow_html=True)
        st.markdown(f'''
            <div class="console-stat-row">
                <span class="console-stat-label">LLM Provider</span>
                <span class="console-stat-val-green">Groq</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Active Model</span>
                <span class="console-stat-val">{GROQ_MODEL}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Top-K Retrieval</span>
                <span class="console-stat-val-amber">{TOP_K}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Sim. Threshold</span>
                <span class="console-stat-val-amber">{SIMILARITY_THRESHOLD}</span>
            </div>
            <div class="console-stat-row">
                <span class="console-stat-label">Max Tokens</span>
                <span class="console-stat-val-cyan">{config.LLM_MAX_TOKENS}</span>
            </div>
        ''', unsafe_allow_html=True)


def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="FYP Handbook Console",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Inject styling
    inject_developer_css()
    
    # Top Console Header Bar
    st.markdown(f"""
        <div class="console-header">
            <div>
                <div class="console-title">
                    {SVG_CPU} FYP HANDBOOK INFERENCE ENGINE
                </div>
                <div style="font-size: 0.8rem; color: var(--text-secondary); margin-top: 2px;">
                    FAST-NUCES Final Year Project RAG Assistant & Knowledge Base
                </div>
            </div>
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span class="console-badge">FAISS v1.13</span>
                <span class="console-badge">GROQ API</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Load system — show a styled progress screen on cold start
    # load_rag_system() is @st.cache_resource so it runs once per server session.
    # If the pre-warm thread already finished, this returns instantly.
    # If it's still loading (first user hit the page during warm-up), we show
    # a step-by-step progress indicator so the user knows something is happening.
    if not _prewarm_thread.is_alive():
        # Already cached — silent, instant return
        index, chunks, model, rag_config, llm = load_rag_system()
    else:
        # Still loading — show a progress screen
        with st.status("⚡ Initializing inference engine...", expanded=True) as status:
            st.write("📦 Loading vector index (FAISS)...")
            st.write("🧠 Loading embedding model weights (~90 MB)...")
            st.write("🔗 Connecting to Groq API...")
            st.write("")
            st.caption("This only happens once per server session. Subsequent visits will be instant.")
            # Block until the prewarm thread finishes (or 30s timeout)
            _prewarm_thread.join(timeout=30)
            index, chunks, model, rag_config, llm = load_rag_system()
            status.update(label="✅ Inference engine ready", state="complete", expanded=False)

    if index is None:
        st.error("RAG Index not found. Please run 'python ingest.py' to build the vector store.")
        st.stop()

    if not GROQ_API_KEY or llm is None:
        st.error("Groq API configuration error. Please verify GROQ_API_KEY in environment.")
        st.stop()

    # Render Developer Sidebar
    render_sidebar(rag_config)

    # Input Section
    col_input, col_btn = st.columns([5, 1], vertical_alignment="bottom")
    
    with col_input:
        query = st.text_input(
            "Query Input",
            placeholder="Ask questions about FYP guidelines, formatting, chapters, or grading...",
            key="query_input",
            label_visibility="collapsed"
        )
    
    with col_btn:
        ask_button = st.button("Ask Query", type="primary")

    # Sample Quick Queries
    st.markdown('<div style="font-size: 0.72rem; color: var(--text-muted); margin: 0.8rem 0 0.4rem 0; font-family: var(--font-mono); letter-spacing: 0.05em; text-transform: uppercase;">SUGGESTED QUESTIONS:</div>', unsafe_allow_html=True)
    
    sq_col1, sq_col2 = st.columns(2)
    preset_query = None
    
    q1 = "What are the required chapters of a Development FYP report?"
    q2 = "What headings, fonts, and sizes are required for formatting?"
    q3 = "What margins and line spacing standards do we use?"
    q4 = "How to use Ibid. and op. cit. in citations?"
    
    with sq_col1:
        if st.button(f"› {q1}", key="sq1", type="secondary", use_container_width=True):
            preset_query = q1
        if st.button(f"› {q3}", key="sq3", type="secondary", use_container_width=True):
            preset_query = q3
            
    with sq_col2:
        if st.button(f"› {q2}", key="sq2", type="secondary", use_container_width=True):
            preset_query = q2
        if st.button(f"› {q4}", key="sq4", type="secondary", use_container_width=True):
            preset_query = q4

    active_query = preset_query or (query.strip() if ask_button else None)

    # Process and Render Query Response
    if active_query:
        log_query(active_query)
        is_valid, error_msg = validate_input(active_query, min_length=3, max_length=500)
        
        if not is_valid:
            st.error(f"Input Validation Error: {error_msg}")
        else:
            user_id = st.session_state.get('user_id', 'anonymous')
            allowed, rate_limit_msg = check_rate_limit(user_id)
            
            if not allowed:
                st.error(f"Rate Limit Exceeded: {rate_limit_msg}")
            else:
                with st.spinner("Executing retrieval & inference..."):
                    try:
                        retrieved_chunks, scores = retrieve_chunks(active_query, model, index, chunks, TOP_K)
                        result = generate_answer(active_query, retrieved_chunks, scores, llm)
                        
                        # 1. PRIMARY CONTENT: Prominent Answer Panel
                        confidence_class = f"pill-{result['confidence']}"
                        confidence_label = result['confidence'].upper()
                        
                        st.markdown(f"""
                            <div class="answer-panel">
                                <div class="answer-header">
                                    <div class="answer-header-title">
                                        {SVG_TERMINAL} GENERATED ANSWER
                                    </div>
                                    <div>
                                        <span class="{confidence_class}">{confidence_label} CONFIDENCE</span>
                                    </div>
                                </div>
                                <div class="answer-body">
                        """, unsafe_allow_html=True)
                        
                        st.markdown(result['answer'])
                        
                        st.markdown("</div></div>", unsafe_allow_html=True)
                        
                        # 2. SECONDARY CONTENT: Compact Metadata & Token Stat Strip
                        tokens = result.get('tokens_used') or {'prompt': 0, 'completion': 0, 'total': 0}
                        latency_val = result.get('timing', 'N/A')
                        model_name = result.get('model', GROQ_MODEL)
                        
                        st.markdown(f"""
                            <div class="meta-strip">
                                <div class="meta-group">
                                    <div class="meta-item">
                                        <span>Model:</span>
                                        <span class="meta-value">{model_name}</span>
                                    </div>
                                    <div class="meta-item">
                                        <span>Latency:</span>
                                        <span class="meta-value">{latency_val}</span>
                                    </div>
                                </div>
                                <div class="meta-group">
                                    <div class="meta-item">
                                        <span>Prompt Tokens:</span>
                                        <span class="meta-value">{tokens['prompt']:,}</span>
                                    </div>
                                    <div class="meta-item">•</div>
                                    <div class="meta-item">
                                        <span>Response Tokens:</span>
                                        <span class="meta-value">{tokens['completion']:,}</span>
                                    </div>
                                    <div class="meta-item">•</div>
                                    <div class="meta-item">
                                        <span>Total:</span>
                                        <span class="meta-value">{tokens['total']:,} tokens</span>
                                    </div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        # 3. TERTIARY CONTENT: Distinct Card Treatment for Sources
                        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)
                        
                        with st.expander(f"RETRIVED SOURCES ({len(result['sources'])} References)", expanded=True):
                            if result['sources']:
                                for i, source in enumerate(result['sources'], 1):
                                    score_pct = min(100, max(0, int(source['score'] * 100)))
                                    
                                    st.markdown(f"""
                                        <div class="source-card">
                                            <div class="source-card-header">
                                                <div style="display: flex; align-items: center; gap: 0.6rem;">
                                                    <span class="page-badge">PAGE {source['page']}</span>
                                                    <span style="font-size: 0.85rem; font-weight: 600; color: var(--text-primary);">
                                                        {source['section']}
                                                    </span>
                                                </div>
                                                <div class="score-bar-container">
                                                    <span>Relevance: {score_pct}%</span>
                                                    <div class="score-progress">
                                                        <div class="score-fill" style="width: {score_pct}%;"></div>
                                                    </div>
                                                </div>
                                            </div>
                                            <pre class="source-snippet">{source['snippet']}</pre>
                                        </div>
                                    """, unsafe_allow_html=True)
                            else:
                                st.info("No sources retrieved above the similarity threshold.")
                                
                    except Exception as e:
                        log_error(e, {"query": active_query, "user_id": user_id})
                        st.error(f"Inference Failure: {str(e)}")


if __name__ == "__main__":
    main()
