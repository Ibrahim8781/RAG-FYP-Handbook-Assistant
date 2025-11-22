"""
FYP Handbook RAG - Query & Answer Script (CLI)
Retrieves relevant chunks and generates answers with page citations
"""

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
SYSTEM_PROMPT = """You are a handbook assistant. Answer ONLY from the context provided.
Cite page numbers like "(p. X)" after each relevant point.
If you're unsure or the information isn't in the context, say "I don't have that information in the handbook."
Be concise but thorough."""

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


class RAGQueryEngine:
    """RAG Query Engine for retrieving and answering questions"""
    
    def __init__(self):
        """Load FAISS index, metadata, and model"""
        print("Loading RAG system...")
        
        # Check if required files exist
        if not all(os.path.exists(p) for p in [FAISS_INDEX_PATH, METADATA_PATH, CONFIG_PATH]):
            raise FileNotFoundError(
                "Required files not found. Please run ingest.py first to create the index."
            )
        
        # Load configuration
        with open(CONFIG_PATH, 'r') as f:
            self.config = json.load(f)
        
        # Load FAISS index
        print(f"Loading FAISS index from {FAISS_INDEX_PATH}")
        self.index = faiss.read_index(FAISS_INDEX_PATH)
        
        # Load metadata
        print(f"Loading metadata from {METADATA_PATH}")
        with open(METADATA_PATH, 'rb') as f:
            self.chunks = pickle.load(f)
        
        # Load embedding model
        print(f"Loading embedding model: {self.config['embedding_model']}")
        self.model = SentenceTransformer(self.config['embedding_model'])
        
        print(f"RAG system loaded with {len(self.chunks)} chunks")
    
    def retrieve(self, query: str, top_k: int = TOP_K) -> Tuple[List[Dict], List[float]]:
        """
        Retrieve top-k most relevant chunks for a query.
        
        Args:
            query: User query string
            top_k: Number of chunks to retrieve
            
        Returns:
            Tuple of (retrieved chunks, similarity scores)
        """
        # Embed query
        query_embedding = self.model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search in FAISS index
        scores, indices = self.index.search(query_embedding, top_k)
        
        # Get corresponding chunks
        retrieved_chunks = []
        retrieved_scores = []
        
        for idx, score in zip(indices[0], scores[0]):
            if idx < len(self.chunks):  # Validate index
                retrieved_chunks.append(self.chunks[idx])
                retrieved_scores.append(float(score))
        
        return retrieved_chunks, retrieved_scores
    
    def format_context(self, chunks: List[Dict]) -> str:
        """
        Format retrieved chunks into context string.
        """
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            page = chunk['page_number']
            text = chunk['text'].strip()
            section = chunk.get('section_hint', 'General')
            
            context_parts.append(
                f"[Chunk {i} - Page {page} - {section}]\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def generate_answer(self, query: str, chunks: List[Dict], scores: List[float]) -> Dict:
        """
        Generate answer from retrieved chunks.
        
        Note: This is a simple implementation that formats the context.
        For actual LLM-based generation, you would integrate with OpenAI, Anthropic, etc.
        This version provides the formatted prompt that can be used with any LLM.
        
        Args:
            query: User query
            chunks: Retrieved chunks
            scores: Similarity scores
            
        Returns:
            Dictionary with answer, sources, and metadata
        """
        # Check if best match is above threshold
        if not scores or scores[0] < SIMILARITY_THRESHOLD:
            return {
                'answer': "I don't have that information in the handbook. "
                         "Please make sure your question is about the FYP handbook content.",
                'sources': [],
                'scores': scores,
                'confidence': 'low'
            }
        
        # Format context
        context = self.format_context(chunks)
        
        # Create prompt (this would be sent to an LLM in production)
        prompt = PROMPT_TEMPLATE.format(
            user_question=query,
            top_chunks_text=context
        )
        
        # For this implementation, we'll provide a rule-based answer
        # that extracts relevant information with page citations
        answer = self._extract_answer_from_chunks(query, chunks)
        
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
                    'snippet': chunk['text'][:200] + "..."
                })
                seen_pages.add(page)
        
        return {
            'answer': answer,
            'sources': sources,
            'scores': scores,
            'confidence': 'high' if scores[0] > 0.5 else 'medium',
            'prompt': prompt  # Include for reference
        }
    
    def _extract_answer_from_chunks(self, query: str, chunks: List[Dict]) -> str:
        """
        Extract answer from chunks with page citations.
        This is a simple implementation - in production, use an LLM.
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
                            key_points.append(line)                # If we found key points, format them nicely
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
            answer = "\n".join(answer_content)
            ref_section = "\n\n" + "─" * 50 + "\n📚 **References:** " + ", ".join(references)
            return answer + ref_section
        else:
            return "I found relevant information but couldn't extract a clear answer. Please refer to the sources below."
    
    def ask(self, query: str, top_k: int = TOP_K) -> Dict:
        """
        Main query method: retrieve and answer.
        
        Args:
            query: User question
            top_k: Number of chunks to retrieve
            
        Returns:
            Dictionary with answer and metadata
        """
        print(f"\nQuery: {query}")
        print(f"Retrieving top {top_k} chunks...")
        
        # Retrieve
        chunks, scores = self.retrieve(query, top_k)
        
        # Generate answer
        result = self.generate_answer(query, chunks, scores)
        
        return result


def display_result(result: Dict):
    """Display query result in CLI format"""
    print("\n" + "=" * 80)
    print("ANSWER:")
    print("=" * 80)
    print(result['answer'])
    
    print("\n" + "=" * 80)
    print("SOURCES (Page References):")
    print("=" * 80)
    
    for source in result['sources']:
        print(f"\nPage {source['page']} - {source['section']}")
        print(f"  Relevance: {source['score']:.3f}")
        print(f"  Snippet: {source['snippet']}")
    
    print("\n" + "=" * 80)
    print(f"Confidence: {result['confidence'].upper()}")
    print(f"Top similarity score: {result['scores'][0]:.3f}" if result['scores'] else "N/A")
    print("=" * 80)


def main():
    """Main CLI interface"""
    print("=" * 80)
    print("FYP Handbook RAG - Question Answering System")
    print("=" * 80)
    
    try:
        # Initialize RAG engine
        engine = RAGQueryEngine()
        
        print("\nReady! Enter your questions (or 'quit' to exit)")
        print("-" * 80)
        
        while True:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            # Process query
            result = engine.ask(query)
            
            # Display result
            display_result(result)
    
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run 'python ingest.py' first to create the index.")
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
