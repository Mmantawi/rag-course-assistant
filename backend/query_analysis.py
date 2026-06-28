import os
import re
import json
import unicodedata
import numpy as np
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

try:
    from langchain_ollama import ChatOllama, OllamaEmbeddings
except ImportError:
    from langchain_community.chat_models import ChatOllama
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from backend.config import LLM_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH, SIMILARITY_THRESHOLD
except ModuleNotFoundError:
    from config import LLM_MODEL, EMBEDDING_MODEL, VECTOR_DB_PATH, SIMILARITY_THRESHOLD

class QueryAnalyzer:
    """
    Normalizes the user's query while preserving semantic meaning.
    Performs Unicode normalization, strips extra whitespace, and normalizes punctuation
    while preserving stopwords and acronym capitalization.
    """
    @staticmethod
    def normalize(query: str) -> str:
        if not query:
            return ""
        
        # 1. Unicode normalization (NFKC decomposes compatibility characters)
        normalized = unicodedata.normalize("NFKC", query)
        
        # 2. Normalize punctuation (convert smart quotes/dashes to regular ones)
        normalized = normalized.replace("“", "\"").replace("”", "\"")
        normalized = normalized.replace("‘", "'").replace("’", "'")
        normalized = normalized.replace("—", "-").replace("–", "-")
        
        # 3. Remove extra whitespaces
        normalized = re.sub(r"\s+", " ", normalized).strip()
        
        return normalized

class ConceptExtractor:
    """
    Uses a local LLM to extract semantic concepts from the user's question.
    """
    def __init__(self, model_name: str = LLM_MODEL):
        self.model_name = model_name
        self.llm = ChatOllama(model=model_name, temperature=0)
        
    def extract_concepts(self, question: str) -> list[str]:
        if not question:
            return []
            
        prompt = f"""You are a Query Analysis assistant. Your task is to extract the core semantic concepts from the user's question.
These concepts will be used individually to query a vector database.
Extract only the main nouns, technical terms, algorithms, or entities that are the subjects of the query.
Exclude task words (like "compare", "difference", "explain", "vs", "versus", "what is", "advantages of", "over"), articles, and grammatical helper words.

Examples:
Question: "What is the difference between regression and classification?"
Concepts: ["regression", "classification"]

Question: "Compare CNNs and Vision Transformers."
Concepts: ["CNN", "Vision Transformer"]

Question: "Explain backpropagation."
Concepts: ["backpropagation"]

Question: "What are the advantages of Adam over SGD?"
Concepts: ["Adam", "SGD"]

Respond ONLY with a JSON list of strings representing the extracted concepts. Do not include markdown formatting (like ```json), commentary, or extra text.

Question: "{question}"
JSON List:"""
        
        try:
            response = self.llm.invoke(prompt)
            content = response.content.strip()
            
            # Clean markdown JSON block formatting if returned
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\n", "", content)
                content = re.sub(r"\n```$", "", content)
            content = content.strip()
            
            concepts = json.loads(content)
            if isinstance(concepts, list):
                # Ensure all elements are strings and not empty
                parsed_concepts = [str(c).strip() for c in concepts if str(c).strip()]
                if parsed_concepts:
                    return parsed_concepts
        except Exception as e:
            print(f"[ConceptExtractor] Error parsing response: {e}")
            
        # Fallback heuristic: Split by common logical separators if LLM fails
        # Look for "and", "vs", "versus", "or", commas
        cleaned = re.sub(r"\b(what is|explain|compare|difference between|advantages of|over|vs|versus)\b", "", question, flags=re.IGNORECASE)
        splits = re.split(r",|\band\b|\bor\b", cleaned, flags=re.IGNORECASE)
        fallback_concepts = [s.strip() for s in splits if s.strip()]
        
        return fallback_concepts if fallback_concepts else [question]

class MultiRetriever:
    """
    Performs independent hybrid searches (dense Chroma vector search + lexical BM25 search)
    for each extracted concept, merging results using Reciprocal Rank Fusion (RRF).
    """
    def __init__(self, vector_store: Chroma):
        self.vector_store = vector_store
        
        # Load serialized BM25 index
        try:
            from backend.bm25 import BM25Index
        except ModuleNotFoundError:
            from bm25 import BM25Index
            
        self.bm25_index = BM25Index()
        bm25_path = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")
        if os.path.exists(bm25_path):
            try:
                self.bm25_index.load(bm25_path)
            except Exception as e:
                print(f"[MultiRetriever] Warning: Failed to load BM25 index: {e}")
        else:
            print(f"[MultiRetriever] Warning: BM25 index file not found at '{bm25_path}'. "
                  f"Exact keyword match will be skipped until ingestion runs.")
        
    def retrieve(self, concepts: list[str], top_k: int = 5) -> list[tuple[Document, float]]:
        all_results = []
        
        # Query for each concept
        for concept in concepts:
            # 1. Dense similarity vector search
            candidates_vector = self.vector_store.similarity_search_with_score(concept, k=top_k)
            
            # 2. Lexical keyword BM25 search
            candidates_bm25 = self.bm25_index.score(concept, top_k=top_k)
            
            # 3. Reciprocal Rank Fusion (RRF) merge
            merged = {}
            
            # Process vector candidates (higher rank index = worse)
            for rank_idx, (doc, score) in enumerate(candidates_vector):
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id not in merged:
                    # Clone document object to prevent modifying in-place Chroma objects
                    doc_clone = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                    merged[chunk_id] = {
                        "doc": doc_clone,
                        "vector_rank": rank_idx + 1,
                        "vector_score": float(score),
                        "bm25_rank": None,
                        "bm25_score": 0.0
                    }
                    
            # Process BM25 candidates
            for rank_idx, (doc, score) in enumerate(candidates_bm25):
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id not in merged:
                    doc_clone = Document(page_content=doc.page_content, metadata=doc.metadata.copy())
                    merged[chunk_id] = {
                        "doc": doc_clone,
                        "vector_rank": None,
                        "vector_score": 1.0,  # Default worst L2 score
                        "bm25_rank": rank_idx + 1,
                        "bm25_score": float(score)
                    }
                else:
                    merged[chunk_id]["bm25_rank"] = rank_idx + 1
                    merged[chunk_id]["bm25_score"] = float(score)
                    
            # Compute RRF score for each merged document
            for chunk_id, info in merged.items():
                doc = info["doc"]
                v_rank = info["vector_rank"]
                b_rank = info["bm25_rank"]
                
                rrf_score = 0.0
                if v_rank is not None:
                    rrf_score += 1.0 / (60.0 + v_rank)
                if b_rank is not None:
                    rrf_score += 1.0 / (60.0 + b_rank)
                    
                # Save debug statistics inside metadata
                doc.metadata["vector_score"] = info["vector_score"]
                doc.metadata["bm25_score"] = info["bm25_score"]
                doc.metadata["rrf_score"] = rrf_score
                
                all_results.append((doc, rrf_score))
                
        # Sort candidates descending by combined RRF score (higher RRF is better)
        all_results.sort(key=lambda x: x[1], reverse=True)
        return all_results

class ResultMerger:
    """
    Merges retrieved results, deduplicates them, filters by similarity threshold,
    and applies a page diversity filter.
    """
    def __init__(self, embeddings: OllamaEmbeddings, threshold: float = SIMILARITY_THRESHOLD):
        self.embeddings = embeddings
        self.threshold = threshold
        
    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))
        
    def merge_and_deduplicate(self, candidates: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        # 1. Deduplicate by chunk_id, keeping the HIGHEST combined RRF score
        dedup_map = {}
        for doc, score in candidates:
            chunk_id = doc.metadata.get("chunk_id")
            key = chunk_id if chunk_id else doc.page_content
            
            if key not in dedup_map or score > dedup_map[key][1]:
                dedup_map[key] = (doc, score)
                
        # Filter: keep if vector score passes threshold OR if there is a keyword BM25 match
        filtered_list = []
        for doc, rrf_score in dedup_map.values():
            vector_score = doc.metadata.get("vector_score", 1.0)
            bm25_score = doc.metadata.get("bm25_score", 0.0)
            
            if vector_score <= self.threshold or bm25_score > 0.0:
                filtered_list.append((doc, rrf_score))
        
        if not filtered_list:
            return []
            
        # 2. Group candidates by (source, page) to apply page diversity
        page_groups = {}
        for doc, score in filtered_list:
            key = (doc.metadata.get("source"), doc.metadata.get("page"))
            if key not in page_groups:
                page_groups[key] = []
            page_groups[key].append((doc, score))
            
        final_candidates = []
        for key, docs_with_scores in page_groups.items():
            if len(docs_with_scores) == 1:
                final_candidates.append(docs_with_scores[0])
            else:
                # Sort documents by score descending (highest RRF score is best)
                docs_with_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Embed each chunk's content in the group to compute cosine similarity
                texts = [x[0].page_content for x in docs_with_scores]
                embeddings_list = self.embeddings.embed_documents(texts)
                
                selected = []
                for idx, (doc, score) in enumerate(docs_with_scores):
                    emb = embeddings_list[idx]
                    is_redundant = False
                    
                    for sel_doc, sel_score, sel_emb in selected:
                        sim = self._cosine_similarity(emb, sel_emb)
                        # If two chunks on the same page are >85% similar, drop the lower-scoring one
                        if sim > 0.85:
                            is_redundant = True
                            break
                            
                    if not is_redundant:
                        selected.append((doc, score, emb))
                        
                for doc, score, _ in selected:
                    final_candidates.append((doc, score))
                    
        return final_candidates

class ResultReranker:
    """
    Reranks documents using cosine similarity against the original normalized query.
    Converts cosine similarity (higher is better) to a distance score (lower is better)
    to match the application's L2 distance metric interface.
    """
    def __init__(self, embeddings: OllamaEmbeddings):
        self.embeddings = embeddings
        
    @staticmethod
    def _cosine_similarity(v1, v2) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))
        
    def rerank(self, question: str, candidates: list[tuple[Document, float]]) -> list[tuple[Document, float]]:
        if not candidates:
            return []
            
        # 1. Embed query
        query_emb = self.embeddings.embed_query(question)
        
        # 2. Embed all candidate chunks
        texts = [doc.page_content for doc, _ in candidates]
        doc_embs = self.embeddings.embed_documents(texts)
        
        # 3. Calculate cosine similarity and convert to distance (distance = 1.0 - cosine_similarity)
        reranked = []
        for idx, (doc, rrf_score) in enumerate(candidates):
            sim = self._cosine_similarity(query_emb, doc_embs[idx])
            # Normalize distance to be in [0, 1] range for match percentage computation
            distance = max(0.0, min(1.0, 1.0 - sim))
            
            # Preserve the debug scores and RRF score in document metadata
            doc.metadata["vector_score"] = float(doc.metadata.get("vector_score", 1.0))
            doc.metadata["bm25_score"] = float(doc.metadata.get("bm25_score", 0.0))
            doc.metadata["rrf_score"] = float(rrf_score)
            doc.metadata["rerank_distance"] = distance
            
            reranked.append((doc, distance))
            
        # Sort in ascending order of distance (lower distance = closer match)
        reranked.sort(key=lambda x: x[1])
        return reranked
