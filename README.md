# 📚 FYP Handbook RAG Assistant

> An intelligent Retrieval-Augmented Generation (RAG) system for the FAST-NUCES Final Year Project Handbook, powered by Groq LLM and production-ready features.

## 🎯 Overview

This RAG system allows students to query the FYP Handbook using natural language and receive accurate, conversational answers with page citations. The system features:

- ✅ **Natural Language Understanding** - Ask questions in plain English
- ✅ **Accurate Answers** - Powered by Llama 3.1 (via Groq API)
- ✅ **Source Citations** - Every answer includes page numbers
- ✅ **Production-Ready** - Logging, rate limiting, caching, error handling
- ✅ **Fast Retrieval** - FAISS vector search with sentence transformers
- ✅ **Web Interface** - Clean Streamlit UI with minimal design

---

## 🏗️ Architecture

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE                            │
│              Streamlit Web App (app.py)                          │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ├─► Input Validation (error_handling.py)
                        ├─► Rate Limiting (rate_limiting.py)
                        └─► Logging (logger.py)
                        │
        ┌───────────────┴──────────────────────────┐
        │                                           │
┌───────▼──────────┐                    ┌──────────▼─────────┐
│  RETRIEVAL       │                    │  GENERATION        │
│  (Vector Search) │                    │  (LLM)             │
├──────────────────┤                    ├────────────────────┤
│ • Embedding      │────────────────────│ • Groq API         │
│   Cache          │   Context Chunks   │ • Llama 3.1 8B     │
│ • FAISS Index    │                    │ • Retry Logic      │
│ • Sentence-BERT  │                    │ • Rate Limiting    │
└──────────────────┘                    └────────────────────┘
        │                                           │
        └───────────────┬───────────────────────────┘
                        │
                        ▼
              Final Answer with Citations
```

### Data Flow

```
1. USER QUERY
   ↓
2. INPUT VALIDATION
   • Min 3 chars, Max 500 chars
   • Not empty
   ↓
3. RATE LIMITING
   • Global: 10 queries/min
   • Per-User: 20 queries/hour
   • API: 30 calls/min
   ↓
4. EMBEDDING (with caching)
   ┌─────────────┐
   │ Cache Check │ ──► HIT → Use cached (1ms)
   └─────────────┘
         │ MISS
         ↓
   Generate Embedding (100-200ms)
         │
         ↓
   Cache for Future Use
   ↓
5. VECTOR SEARCH (FAISS)
   • Cosine similarity
   • Top-K retrieval (K=5)
   • Threshold: 0.25
   ↓
6. LLM GENERATION (with retry)
   ┌──────────────┐
   │ Groq API Call│ ──► Retry on failure (3 attempts)
   └──────────────┘     Backoff: 1s, 2s, 4s
   ↓
7. RESPONSE
   • Answer with citations
   • Confidence score
   • Source chunks
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11 or higher
- Groq API key 

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Ibrahim8781/RAG-FYP-Handbook-Assistant.git
cd RAG-FYP-Handbook-Assistant

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment
cp .env.example .env
# Edit .env and add your Groq API key:
# GROQ_API_KEY=your_api_key_here

# 5. Ingest the handbook (first time only)
python ingest.py

# 6. Run the application
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 📖 Usage

### Web Interface

1. Open http://localhost:8501
2. Type your question in the text box
3. Click "🔍 Ask"
4. Get answer with page citations
5. Expand "View Sources" to see retrieved chunks

**Sample Questions:**
- "What are the required chapters of a Development FYP report?"
- "What headings, fonts, and sizes are required?"
- "How do I use 'Ibid.' and 'op. cit.' in citations?"
- "What goes into the Executive Summary?"


## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# Required: Groq API Key
GROQ_API_KEY=your_api_key_here

# Optional: Environment (auto-detected)
ENV=development  # or production
```

## 📊 Production Features

### 1. Structured Logging ✅

**JSON logs in production** for monitoring tools (CloudWatch, Datadog)

```json
{
  "timestamp": "2026-01-13T18:30:45Z",
  "level": "INFO",
  "message": "LLM call successful",
  "model": "llama-3.1-8b-instant",
  "tokens": 512,
  "latency": 1.45
}
```

### 2. Error Handling & Retry Logic ✅

**Automatic retries** with exponential backoff

- Retry pattern: 0s → 1s → 2s → 4s
- User-friendly error messages
- Input validation (min/max length)

### 3. Rate Limiting ✅

**Prevents abuse** and protects API costs

- Global: 10 queries/minute
- Per-User: 20 queries/hour
- API: 30 Groq calls/minute

### 4. Embedding Caching ✅

**90% faster** on cache hits, **50% cost reduction**

- TTL: 24 hours
- LRU eviction (max 1000)
- Persistent across restarts

---

## 🚂 Railway Deployment

### Quick Deploy

1. **Connect GitHub to Railway**
   - Go to [railway.app/dashboard]
   - New Project → Deploy from GitHub
   - Select: RAG-FYP-Handbook-Assistant

2. **Set Environment Variable**
   ```
   GROQ_API_KEY=your_api_key_here
   ```

3. **Deploy**
   - Railway auto-deploys on git push
   - Build time: ~2 minutes
   - Your app: https://your-project.up.railway.app


## 📁 Project Structure

```
RAG-FYP-Handbook-Assistant/
│
├── 📄 Core Application
│   ├── app.py                      # Streamlit web interface
│   ├── ask.py                      # CLI interface
│   ├── ingest.py                   # Document ingestion
│   ├── llm_utils.py                # Groq LLM wrapper
│   └── config_env.py               # Environment config
│
├── 🏗️ Production Infrastructure
│   ├── logger.py                   # Structured logging
│   ├── error_handling.py           # Retry logic
│   ├── rate_limiting.py            # Request throttling
│   └── caching.py                  # Embedding cache
│
├── ⚙️ Configuration
│   ├── .env.example                # Environment template
│   ├── Procfile                    # Railway start command
│   ├── railway.json                # Railway config
│   └── requirements.txt            # Dependencies
│
├── 📊 Data & Artifacts
│   ├── FYP-Handbook-2023.pdf       # Source document
│   ├── faiss_index.bin             # Vector index (generated)
│   └── chunks_metadata.pkl         # Text chunks (generated)
│
└── 📚 Documentation
    ├── README.md                   # This file
    ├── PRODUCTION_FEATURES.md      # Feature details
    └── RAILWAY_DEPLOY.md           # Deployment guide
```

---

### GroqLLM Class

