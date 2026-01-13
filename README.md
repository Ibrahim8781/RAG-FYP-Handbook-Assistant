# 📚 FYP Handbook RAG Assistant

> An intelligent Retrieval-Augmented Generation (RAG) system for the FAST-NUCES Final Year Project Handbook, powered by Groq LLM and production-ready features.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.52.2-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Demo:** [https://web-production-5db8b.up.railway.app](https://web-production-5db8b.up.railway.app)

---

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
- Groq API key ([Get one free](https://console.groq.com/keys))
- 2GB+ RAM
- Git

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

### CLI Interface

```bash
python ask.py "What are the FYP requirements?"
```

---

## 🔧 Configuration

### Environment Variables

Create `.env` file:

```bash
# Required: Groq API Key
GROQ_API_KEY=your_api_key_here

# Optional: Environment (auto-detected)
ENV=development  # or production
```

### Customization

Edit `config_env.py`:

```python
class Config:
    # RAG parameters
    CHUNK_SIZE = 300           # Words per chunk
    OVERLAP = 90               # Word overlap
    TOP_K = 5                  # Chunks to retrieve
    SIMILARITY_THRESHOLD = 0.25
    
    # LLM parameters
    GROQ_MODEL = "llama-3.1-8b-instant"
    LLM_TEMPERATURE = 0.3
    LLM_MAX_TOKENS = 1024
    
    # Caching
    CACHE_TTL = 86400          # 24 hours
    CACHE_MAX_SIZE = 1000
```

---

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
   - Go to [railway.app/dashboard](https://railway.app/dashboard)
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

### Environment Detection

- **Local**: Shows "🔧 DEVELOPMENT" badge
- **Railway**: Shows "🏭 PRODUCTION" badge
- **Auto-configured**: Based on `RAILWAY_ENVIRONMENT`

---

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

## 🛠️ API Documentation

### GroqLLM Class

```python
from llm_utils import GroqLLM

# Initialize
llm = GroqLLM(api_key="your_key", model="llama-3.1-8b-instant")

# Generate answer
result = llm.generate_answer(
    question="What are FYP requirements?",
    context="<retrieved chunks>",
    max_tokens=1024,
    temperature=0.3
)

# Response
{
    'success': True,
    'answer': "The FYP requirements include...",
    'tokens_used': {'total': 570},
    'latency': 1.45
}
```

### Caching

```python
from caching import embedding_cache

# Get cached embedding
cached = embedding_cache.get(query)

# Set new embedding
embedding_cache.set(query, embedding)

# Get stats
stats = embedding_cache.get_stats()
```

### Rate Limiting

```python
from rate_limiting import check_rate_limit

allowed, msg = check_rate_limit(user_id)
if not allowed:
    print(f"Rate limited: {msg}")
```

### Logging

```python
from logger import log_query, log_llm_call

log_query("What are FYP requirements?", user_id="user_123")
log_llm_call(model="llama-3.1-8b-instant", tokens=512, latency=1.45, success=True)
```

---

## 🧪 Testing

```bash
# Test individual modules
python logger.py              # Structured logging
python error_handling.py      # Retry logic
python rate_limiting.py       # Rate limits
python caching.py             # Cache operations

# Comprehensive test suite
python test_rag.py 1          # Test local CLI
python test_rag.py 2          # Test Railway
```

---

## 📈 Performance

### Latency

```
Typical Query (cached):     ~1.2 seconds
First Query (uncached):     ~1.4 seconds
Cache Hit:                  ~0.001 seconds (1ms)
FAISS Search:               ~0.010 seconds (10ms)
LLM Generation:             ~1.2 seconds
```

### Cost Analysis

**Per 1000 queries (50% cache hit):**
- Embeddings: $0.05
- LLM (Groq): $0.50
- **Total: $0.55** (vs $0.60 without cache)

### Resource Usage

**Local:**
- RAM: ~300-400 MB
- CPU: 10-20% (idle), 80-100% (query)

**Railway:**
- RAM: ~200-300 MB
- Build time: ~2 minutes
- Cold start: ~10 seconds

---

## 🐛 Troubleshooting

### "Groq API key not found"

```bash
# Check .env file
cat .env

# Should contain:
GROQ_API_KEY=gsk_your_key_here
```

### "RAG system not initialized"

```bash
# Run ingestion
python ingest.py

# Verify files
ls faiss_index.bin chunks_metadata.pkl
```

### "Rate limit exceeded"

Wait the specified time (shown in error message).

### Slow first query

Normal! Models load on first use (~10-15s). Subsequent queries are fast (~2-3s).

---

## 🤝 Contributing

```bash
# 1. Fork and clone
git clone https://github.com/your-username/RAG-FYP-Handbook-Assistant.git

# 2. Create branch
git checkout -b feature/your-feature

# 3. Make changes and test
python test_rag.py 1

# 4. Commit and push
git commit -m "Add feature"
git push origin feature/your-feature

# 5. Create Pull Request
```

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🙏 Acknowledgments

- **FAST-NUCES** - FYP Handbook source
- **Groq** - Fast LLM inference
- **Streamlit** - Web framework
- **Sentence-Transformers** - Embeddings
- **FAISS** - Vector search
- **Railway** - Deployment platform

---

## 📞 Support

- **Documentation**: [PRODUCTION_FEATURES.md](PRODUCTION_FEATURES.md)
- **Issues**: [GitHub Issues](https://github.com/Ibrahim8781/RAG-FYP-Handbook-Assistant/issues)
- **Repository**: [GitHub](https://github.com/Ibrahim8781/RAG-FYP-Handbook-Assistant)

---

<div align="center">

**Built with ❤️ for FAST-NUCES Students**

[Live Demo](https://web-production-5db8b.up.railway.app) • [Documentation](PRODUCTION_FEATURES.md) • [Report Issue](https://github.com/Ibrahim8781/RAG-FYP-Handbook-Assistant/issues)

⭐ Star this repo if you found it helpful!

</div>
