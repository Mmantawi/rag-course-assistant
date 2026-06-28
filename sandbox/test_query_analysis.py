import os
import sys

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.query_analysis import (
    QueryAnalyzer,
    ConceptExtractor,
    MultiRetriever,
    ResultMerger,
    ResultReranker
)
from langchain_core.documents import Document
try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

def test_query_normalization():
    print("Testing QueryAnalyzer.normalize()...")
    test_cases = [
        ("  What  is   regression?  ", "What is regression?"),
        ("Compare CNNs and Vision Transformers.", "Compare CNNs and Vision Transformers."),
        ("What are “smart quotes”?", "What are \"smart quotes\"?"),
        ("unicode \u2122 trademark", "unicode TM trademark")
    ]
    for q, expected in test_cases:
        res = QueryAnalyzer.normalize(q)
        print(f"  Input:    {repr(q)}")
        print(f"  Expected: {repr(expected)}")
        print(f"  Result:   {repr(res)}")
        assert res == expected
    print("[SUCCESS] Query normalization passed!\n")

def test_concept_extraction():
    print("Testing ConceptExtractor.extract_concepts()...")
    extractor = ConceptExtractor()
    test_cases = [
        "What is the difference between regression and classification?",
        "Compare CNNs and Vision Transformers.",
        "Explain backpropagation."
    ]
    for q in test_cases:
        concepts = extractor.extract_concepts(q)
        print(f"  Question: '{q}'")
        print(f"  Concepts: {concepts}")
        assert isinstance(concepts, list)
        assert len(concepts) > 0
        for c in concepts:
            assert isinstance(c, str)
            assert len(c) > 0
    print("[SUCCESS] Concept extraction passed!\n")

def test_merger_and_reranker():
    print("Testing ResultMerger and ResultReranker...")
    # Setup dummy documents
    doc1 = Document(page_content="Regression is a supervised learning task to predict continuous values.", metadata={"source": "lec1.pdf", "page": 1, "chunk_id": "c1"})
    doc2 = Document(page_content="Classification predicts discrete classes or categories.", metadata={"source": "lec1.pdf", "page": 2, "chunk_id": "c2"})
    doc3 = Document(page_content="Neural networks use backpropagation to update weights.", metadata={"source": "lec2.pdf", "page": 5, "chunk_id": "c3"})
    
    # Duplicate document with worse score
    doc1_dup = Document(page_content="Regression is a supervised learning task to predict continuous values.", metadata={"source": "lec1.pdf", "page": 1, "chunk_id": "c1"})
    
    # List of candidates (Doc, Score) - Chroma L2 score: lower is better
    candidates = [
        (doc1, 0.4),
        (doc2, 0.5),
        (doc3, 0.9),      # Should be filtered out by SIMILARITY_THRESHOLD (0.75)
        (doc1_dup, 0.6)   # Should be deduplicated, keeping doc1's score (0.4)
    ]
    
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    embeddings = OllamaEmbeddings(model=embed_model)
    
    merger = ResultMerger(embeddings, threshold=0.75)
    merged = merger.merge_and_deduplicate(candidates)
    
    print(f"  Original count: {len(candidates)}")
    print(f"  Merged count: {len(merged)}")
    for doc, score in merged:
        print(f"    - Chunk {doc.metadata['chunk_id']}: Score {score}")
        
    # Check deduplication & threshold filter
    chunk_ids = [d.metadata["chunk_id"] for d, _ in merged]
    assert "c1" in chunk_ids
    assert "c2" in chunk_ids
    assert "c3" not in chunk_ids  # Filtered out (0.9 > 0.75)
    assert len(merged) == 2       # c1 and c2
    
    # Check that it kept the better score (0.4) for c1
    for doc, score in merged:
        if doc.metadata["chunk_id"] == "c1":
            assert score == 0.4
            
    # Test Reranking
    print("  Testing Reranker...")
    reranker = ResultReranker(embeddings)
    reranked = reranker.rerank("supervised learning regression classification", merged)
    
    print("  Reranked results:")
    for idx, (doc, dist) in enumerate(reranked):
        print(f"    [{idx+1}] Chunk {doc.metadata['chunk_id']}: Distance {dist:.4f}")
        
    assert len(reranked) == 2
    # Ensure distances are valid
    for _, dist in reranked:
        assert 0.0 <= dist <= 1.0
        
    print("[SUCCESS] Merger and Reranker passed!\n")

if __name__ == "__main__":
    print("=== STARTING PRE-RETRIEVAL QUERY ANALYSIS TESTS ===\n")
    test_query_normalization()
    test_concept_extraction()
    test_merger_and_reranker()
    print("=== ALL TESTS COMPLETED SUCCESSFULLY ===")
