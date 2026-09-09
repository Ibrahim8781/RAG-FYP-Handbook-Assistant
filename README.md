# 📚 FYP Handbook RAG Assistant

> An intelligent Retrieval-Augmented Generation (RAG) system for the FAST-NUCES Final Year Project Handbook, powered by Groq LLM.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This RAG system allows students to query the FYP Handbook using natural language and receive accurate, conversational answers with page citations. The system features:

- ✅ **Natural Language Understanding** - Ask questions in plain English
- ✅ **Accurate Answers** - Powered by Groq API (`groq/compound-mini`)
- ✅ **Source Citations** - Every answer includes page numbers
- ✅ **Production-Ready** - Logging, rate limiting, caching, error handling
- ✅ **Fast Retrieval** - FAISS vector search with sentence transformers
- ✅ **Developer Console UI** - Sleek Streamlit dashboard

---

## 🚀 Quick Start

### Installation & Local Run

```bash
# 1. Clone repository
git clone https://github.com/Ibrahim8781/RAG-FYP-Handbook-Assistant.git
cd RAG-FYP-Handbook-Assistant

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set environment variable
GROQ_API_KEY=your_groq_api_key

# 4. Run application
streamlit run app.py
```

The app will open at **http://localhost:8501**

---

## 🚂 Railway Deployment Instructions

1. **Connect GitHub to Railway**
   - Go to [Railway Dashboard](https://railway.app/dashboard)
   - Click **+ New Project** → **Deploy from GitHub repo**
   - Select repository: `Ibrahim8781/RAG-FYP-Handbook-Assistant`

2. **Configure Environment Variables**
   In Railway project settings under **Variables**:
   - `GROQ_API_KEY`: `your_groq_api_key_here`

3. **Deploy**
   - Railway uses `NIXPACKS` (configured in `railway.json` and `railway.toml`) to build and start `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true`.

---

## 📄 License

MIT License - see LICENSE file for details.
