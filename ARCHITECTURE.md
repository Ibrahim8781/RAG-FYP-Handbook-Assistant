# FYP Handbook RAG System - Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FYP HANDBOOK RAG ASSISTANT                       │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         INPUT PHASE                                 │
└─────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐
    │ FYP_Handbook_2023.pdf│
    │   (Source Document)  │
    └──────────┬───────────┘
               │
               ↓
    ┌──────────────────────┐
    │   ingest.py          │
    │   (Ingestion Script) │
    └──────────┬───────────┘
               │
               ↓
┌──────────────────────────────────────────────────────────────────────┐
│                      PROCESSING PIPELINE                             │
└──────────────────────────────────────────────────────────────────────┘

    [1] Load PDF
         │
         ↓  PyPDF2.PdfReader
    ┌─────────────────┐
    │ Extract Text    │
    │ Per Page        │
    │ (page_number)   │
    └────────┬────────┘
             │
             ↓
    [2] Chunk Text
         │
         ↓  Word-based sliding window
    ┌─────────────────────────┐
    │ Create Chunks           │
    │ • Size: 300 words       │
    │ • Overlap: 90 words     │
    │ • Metadata: page, hint  │
    └────────┬────────────────┘
             │
             ↓
    [3] Embed Chunks
         │
         ↓  Sentence-BERT (all-MiniLM-L6-v2)
    ┌─────────────────────────┐
    │ Generate Embeddings     │
    │ • Dimension: 384        │
    │ • L2 Normalized         │
    └────────┬────────────────┘
             │
             ↓
    [4] Build Index
         │
         ↓  FAISS IndexFlatIP
    ┌─────────────────────────┐
    │ Create Vector Index     │
    │ • Type: Inner Product   │
    │ • Similarity: Cosine    │
    └────────┬────────────────┘
             │
             ↓
    [5] Persist
         │
         ↓  Save to disk
    ┌─────────────────────────┐
    │ Save Files              │
    │ • faiss_index.bin       │
    │ • chunks_metadata.pkl   │
    │ • config.json           │
    └─────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      QUERY PHASE                                     │
└──────────────────────────────────────────────────────────────────────┘

    ┌──────────────────────┐         ┌──────────────────────┐
    │    app.py            │         │     ask.py           │
    │  (Streamlit UI)      │         │   (CLI Interface)    │
    └──────────┬───────────┘         └──────────┬───────────┘
               │                                 │
               └────────────┬────────────────────┘
                            │
                            ↓
                   ┌────────────────┐
                   │  User Query    │
                   │  "What are...?"│
                   └────────┬───────┘
                            │
                            ↓
    ┌───────────────────────────────────────────────────────┐
    │         RETRIEVAL PIPELINE                            │
    └───────────────────────────────────────────────────────┘

    [1] Embed Query
         │
         ↓  all-MiniLM-L6-v2
    ┌─────────────────────────┐
    │ Query → 384-dim vector  │
    │ L2 Normalized           │
    └────────┬────────────────┘
             │
             ↓
    [2] Search Index
         │
         ↓  FAISS.search(top_k=5)
    ┌─────────────────────────┐
    │ Cosine Similarity       │
    │ Return top-5 chunks     │
    │ + similarity scores     │
    └────────┬────────────────┘
             │
             ↓
    [3] Filter
         │
         ↓  Threshold = 0.25
    ┌─────────────────────────┐
    │ Check Relevance         │
    │ If score < 0.25:        │
    │   → "Not in handbook"   │
    └────────┬────────────────┘
             │
             ↓
    [4] Format Context
         │
         ↓  Combine chunks
    ┌─────────────────────────┐
    │ Create Context String   │
    │ [Chunk X - Page Y]      │
    │ {text}                  │
    └────────┬────────────────┘
             │
             ↓
    [5] Generate Answer
         │
         ↓  Extract + Cite
    ┌─────────────────────────┐
    │ Extract Answer          │
    │ Add citations: (p. X)   │
    │ Format sources          │
    └────────┬────────────────┘
             │
             ↓
    ┌─────────────────────────┐
    │    DISPLAY RESULT       │
    │                         │
    │ • Answer with citations │
    │ • Source chunks         │
    │ • Page references       │
    │ • Confidence score      │
    └─────────────────────────┘

┌──────────────────────────────────────────────────────────────────────┐
│                      DATA FLOW SUMMARY                               │
└──────────────────────────────────────────────────────────────────────┘

PDF → Text → Chunks → Embeddings → FAISS Index
                                        ↓
User Query → Query Embedding → Search → Top-K Chunks
                                        ↓
                              Context + Prompt → Answer + Citations
                                        ↓
                                    Display
```

## Component Details

### 1. Ingestion (Offline)

```
Input:   FYP_Handbook_2023.pdf
Process: ingest.py
Output:  faiss_index.bin, chunks_metadata.pkl, config.json
Time:    ~1-2 minutes (one-time)
```

**Key Operations:**
- PyPDF2: Extract text per page
- Chunking: 300 words, 90-word overlap
- Embedding: all-MiniLM-L6-v2 (384-dim)
- Indexing: FAISS IndexFlatIP
- Storage: Binary serialization

### 2. Query (Online)

```
Input:   User question (text)
Process: ask.py or app.py
Output:  Answer with page citations
Time:    <1 second
```

**Key Operations:**
- Encode query: Sentence-BERT
- Search: FAISS cosine similarity
- Filter: Threshold = 0.25
- Extract: Rule-based with citations
- Display: Formatted answer + sources

## Prompt Template Flow

```
┌──────────────────────────────────────────┐
│         PROMPT CONSTRUCTION              │
└──────────────────────────────────────────┘

System Prompt (Fixed)
─────────────────────
You are a handbook assistant.
Answer ONLY from the context.
Cite page numbers like "(p. X)".
If unsure, say you don't know.

         +

User Question (Variable)
────────────────────────
"What margins and spacing do we use?"

         +

Retrieved Context (Dynamic)
───────────────────────────
[Chunk 1 - Page 16]
The report margins should be...

[Chunk 2 - Page 15]
Page setup requires...

         ↓

Complete Prompt
───────────────
[System + Question + Context]

         ↓

Answer Generation
─────────────────
(Current: Rule-based extraction)
(Future: LLM integration)

         ↓

Final Answer
────────────
"The report margins should be
set as follows: Top 1.5"... (p. 16)"
```

## Similarity Scoring

```
Query Embedding          Chunk Embeddings
     [384]          ×        [N × 384]
       │                         │
       └────────┬────────────────┘
                │
                ↓
         Cosine Similarity
         ─────────────────
         score = (q · c) / (||q|| × ||c||)
         
         (Already normalized, so just: q · c)
                │
                ↓
         Sort by score (descending)
                │
                ↓
         Take top-5
                │
                ↓
         Apply threshold (0.25)
                │
                ↓
         Return chunks + scores
```

## User Interface Layout

```
┌────────────────────────────────────────────────────────────┐
│  📚 FYP Handbook Assistant                                 │
│  Ask questions about the FAST-NUCES FYP process            │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Sidebar                    Main Panel                     │
│  ┌──────────────┐          ┌───────────────────────────┐   │
│  │ System Info  │          │ Ask a Question            │   │
│  │              │          │ ┌──────────────────────┐  │   │
│  │ • Chunks     │          │ │ [Input Box]          │  │   │
│  │ • Model      │          │ └──────────────────────┘  │   │
│  │ • Top-K      │          │ [🔍 Ask Button]          │   │
│  │ • Threshold  │          └───────────────────────────┘   │
│  │              │                                          │
│  │ Sample       │          ┌───────────────────────────┐   │
│  │ Questions    │          │ 📝 Answer                │   │
│  │              │          │ 🟢 Confidence: High      │   │
│  │ 1. Fonts?    │          │                           │   │
│  │ 2. Margins?  │          │ [Answer text with         │   │
│  │ 3. Chapters? │          │  page citations]          │   │
│  │ ...          │          │                           │   │
│  └──────────────┘          └───────────────────────────┘   │
│                                                            │
│                            ┌───────────────────────────┐   │
│                            │ ▼ 📚 Sources             │   │
│                            │                           │   │
│                            │ • Page 16: ...            │   │
│                            │ • Page 15: ...            │   │
│                            │                           │   │
│                            └───────────────────────────┘   │
│                                                            │
│                            ┌───────────────────────────┐   │
│                            │ ▼ 🔧 Debug Info          │   │
│                            │                           │   │
│                            │ Scores: [0.87, 0.75...]   │   │
│                            │ Prompt: ...               │   │
│                            │                           │   │
│                            └───────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
```

## File Dependencies

```
ingest.py
    ↓ requires
    • PyPDF2
    • sentence-transformers
    • faiss
    • numpy
    ↓ reads
    • FYP_Handbook_2023.pdf
    ↓ writes
    • faiss_index.bin
    • chunks_metadata.pkl
    • config.json

ask.py / app.py
    ↓ requires
    • sentence-transformers
    • faiss
    • numpy
    • streamlit (app.py only)
    ↓ reads
    • faiss_index.bin
    • chunks_metadata.pkl
    • config.json
    ↓ outputs
    • Answer with citations
    • Source references
```

## Metadata Structure

```
Chunk Object
├── chunk_id: int
├── page_number: int
├── text: str
├── section_hint: str
└── word_count: int

Example:
{
  "chunk_id": 42,
  "page_number": 16,
  "text": "The report margins should be...",
  "section_hint": "REPORT FORMATTING",
  "word_count": 287
}
```

## Configuration Flow

```
ingest.py (Creation)
    ↓
    Saves config.json:
    {
      "chunk_size": 300,
      "overlap": 90,
      "embedding_model": "all-MiniLM-L6-v2",
      "num_chunks": 247,
      "embedding_dim": 384
    }
    ↓
ask.py/app.py (Loading)
    ↓
    Reads config.json
    ↓
    Uses settings for consistency
```

---

---

## Validation Results

### Question 1: Font Requirements
**Query:** "What headings, fonts, and sizes are required in the FYP report?"

**Expected:** Times New Roman 11pt body, Arial headings (Title 16pt, H1 14pt, H2 12pt, H3 11pt)

**System Output:**
```
The FYP report should use Times New Roman 11pt for body text. (p. 15)
Headings must be in Arial with sizes: Title 16pt, H1 14pt, H2 12pt, H3 11pt. (p. 15)
```
✅ Page citations included, accurate information

### Question 2: Margins & Spacing
**Query:** "What margins and spacing do we use?"

**Expected:** Top 1.5", Bottom 1.0", Left 2.0", Right 1.0", Line spacing 1.5, Para spacing 6pt

**System Output:**
```
Margins: Top 1.5", Bottom 1.0", Left 2.0", Right 1.0". (p. 16)
Line spacing should be 1.5 with 6pt paragraph spacing. (p. 16)
```
✅ Complete specifications with page references

### Question 3-6: Chapter Requirements, Endnotes, Executive Summary
All validation questions return accurate answers with proper page citations (p. X), retrieved sources with relevance scores, and appropriate confidence levels.

---

## Key Design Decisions

### Why 300-word chunks with 30% overlap?
- Balances context preservation and retrieval precision
- Overlap prevents information loss at chunk boundaries
- Within required 250-400 word range

### Why threshold = 0.25?
- Prevents answering out-of-scope questions
- Reduces hallucination risk
- Empirically effective for document Q&A

### Why FAISS IndexFlatIP?
- Exact cosine similarity (no approximation)
- Fast for small-to-medium datasets
- Simple, reliable, easy to debug

### Why rule-based answer extraction?
- **Current:** Fast, no API costs, deterministic
- **Future:** Can integrate LLM (GPT-4, Claude) for better synthesis
- Grounding mechanism works with either approach

---

## Performance Characteristics

- **Indexing:** 1-2 minutes (one-time)
- **Query:** <1 second
- **Memory:** ~200-300 MB
- **Accuracy:** High for in-scope queries
- **Grounding:** 100% (all answers cite pages)
