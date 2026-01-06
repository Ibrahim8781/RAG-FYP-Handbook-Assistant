# FYP Handbook RAG Assistant

A Retrieval-Augmented Generation (RAG) pipeline that answers students' questions about the FAST-NUCES FYP process using the BS Final Year Project Handbook 2023.

**Assignment:** Generative AI - Assignment #3  
**Submission Date:** November 22, 2025  
**Built with:** Sentence-BERT • FAISS • Streamlit • PyPDF2

---

## 🎯 Overview

This RAG system provides accurate, grounded answers about the FYP handbook with page citations. It uses semantic search to find relevant information and presents it with source references.

**Key Features:**
- 📄 Single PDF corpus (BS FYP Handbook 2023)
- 🧠 Semantic embeddings (all-MiniLM-L6-v2)
- 🔍 Fast vector search (FAISS)
- 💻 Web UI (Streamlit) + CLI
- 📌 Page-referenced answers (p. X)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Add Handbook PDF
Place your handbook in the project directory as:
```
FYP_Handbook_2023.pdf
```

### 3. Create Index
```bash
python ingest.py
```
This creates the FAISS index (~1-2 minutes, one-time only).

### 4. Launch App
```bash
# Web Interface (Recommended)
streamlit run app.py

# Or CLI
python ask.py

# Or run validation tests
python test_validation.py
```

---

## 📋 System Configuration

### Chunking Settings
- **Chunk Size:** 300 words (within 250-400 range)
- **Overlap:** 90 words (30%)
- **Metadata:** page_number, section_hint, chunk_id

### Model & Retrieval
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- **Vector Store:** FAISS IndexFlatIP (cosine similarity)
- **Top-K:** 5 chunks
- **Threshold:** 0.25 (filters irrelevant queries)

---

## 💡 Usage Examples

### Sample Questions (6 Validation Tests)

1. **Fonts:** "What headings, fonts, and sizes are required in the FYP report?"
2. **Margins:** "What margins and spacing do we use?"
3. **Dev Chapters:** "What are the required chapters/sections of a Development FYP report?"
4. **R&D Chapters:** "What are the required chapters of an R&D-based FYP report?"
5. **Endnotes:** "How should endnotes like 'Ibid.' and 'op. cit.' be used?"
6. **Executive Summary:** "What goes into the Executive Summary and Abstract?"

### Expected Behavior
- Answers cite pages: "(p. 15)"
- Sources shown with relevance scores
- Out-of-scope queries rejected (threshold < 0.25)

---

## 📁 Project Structure

```
Assignment-3/
├── ingest.py              # PDF → FAISS index
├── ask.py                 # CLI interface
├── app.py                 # Streamlit UI
├── test_validation.py     # Validation tests
├── requirements.txt       # Dependencies
├── prompt_log.txt         # Prompt templates
├── README.md              # This file
├── ARCHITECTURE.md        # System design
├── quickstart.py          # Setup helper
└── FYP_Handbook_2023.pdf  # Your handbook (add this)
```

**Generated files (not in repo):**
- `faiss_index.bin` - Vector index
- `chunks_metadata.pkl` - Chunk metadata
- `config.json` - Configuration


## Implementation Details

### Ingestion Pipeline (ingest.py)
1. **Load PDF:** Extract text per page (PyPDF2)
2. **Chunk:** Sliding window (300 words, 90 overlap)
3. **Embed:** Sentence-BERT (384-dim vectors)
4. **Index:** FAISS IndexFlatIP (cosine similarity)
5. **Save:** faiss_index.bin, chunks_metadata.pkl, config.json

### Query Pipeline (ask.py / app.py)
1. **Embed Query:** Same model as corpus
2. **Search:** FAISS returns top-5 chunks + scores
3. **Filter:** Reject if score < 0.25
4. **Format:** Extract answer with page citations
5. **Display:** Answer + sources + confidence

### Grounding Mechanism
- Every answer includes page references: "(p. X)"
- Retrieved chunks shown with relevance scores
- Source snippets for verification
- Confidence levels: high (>0.5), medium (0.25-0.5), low (<0.25)

---

##  Technical Specs

- **Python:** 3.8+
- **RAM:** 2 GB minimum
- **Disk:** 500 MB
- **Indexing Time:** 1-2 minutes
- **Query Time:** <1 second
- **Memory Usage:** ~200-300 MB

---

## 🎓 Notes

### Limitations
- **Answer Generation:** Uses rule-based extraction (for production, integrate LLM like GPT-4)
- **Single Document:** One handbook at a time
- **No Conversation:** Each query is independent

### Future Enhancements
- LLM integration (OpenAI/Anthropic) for better answers
- Hybrid search (BM25 + semantic)
- Multi-turn conversation
- Semantic chunking

### Contributors
- Muhammad Umar Farooq github@: yuri8822
- Ibrahim Ahmad github@: ibrahim8781
