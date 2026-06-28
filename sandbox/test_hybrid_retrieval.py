import os
import sys
import tempfile
import pickle

# Ensure backend imports work
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.bm25 import BM25Index
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

def test_bm25_index():
    print("Testing BM25Index operations...")
    doc1 = Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"chunk_id": "doc1"})
    doc2 = Document(page_content="Lazy dogs are cute and sleep all day.", metadata={"chunk_id": "doc2"})
    doc3 = Document(page_content="Quick brown foxes are fast and smart.", metadata={"chunk_id": "doc3"})
    
    index = BM25Index()
    index.fit([doc1, doc2, doc3])
    
    # Verify tokenization
    assert index._tokenize("Hello World!") == ["hello", "world"]
    
    # Test scoring: keyword "lazy" is present in doc1 and doc2
    lazy_results = index.score("lazy")
    print(f"  Scoring 'lazy': {[(d.metadata['chunk_id'], s) for d, s in lazy_results]}")
    assert len(lazy_results) == 2
    assert lazy_results[0][0].metadata["chunk_id"] in ["doc1", "doc2"]
    
    # Test scoring: keyword "foxes" is present in doc3
    fox_results = index.score("foxes")
    print(f"  Scoring 'foxes': {[(d.metadata['chunk_id'], s) for d, s in fox_results]}")
    assert len(fox_results) == 1  # doc3 contains "foxes", doc1 contains "fox" (no stemming)
    
    # Test Serialization
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "bm25.pkl")
        index.save(filepath)
        assert os.path.exists(filepath)
        
        loaded_index = BM25Index()
        loaded_index.load(filepath)
        assert loaded_index.num_docs == 3
        assert len(loaded_index.idf) == len(index.idf)
        
        # Verify scores match
        orig_score = index.score("lazy")[0][1]
        load_score = loaded_index.score("lazy")[0][1]
        assert orig_score == load_score
        
    print("[SUCCESS] BM25Index tests passed!\n")

class MockChroma:
    """Mock Chroma DB for testing hybrid retrieval in MultiRetriever"""
    def __init__(self, doc_score_list):
        self.doc_score_list = doc_score_list
        
    def similarity_search_with_score(self, query, k=5):
        return self.doc_score_list[:k]

def test_hybrid_rrf_merging():
    print("Testing Hybrid Search RRF Merging...")
    
    # Setup dummy document library
    doc1 = Document(page_content="Regression is a supervised learning task.", metadata={"source": "lec1.pdf", "page": 1, "chunk_id": "c1"})
    doc2 = Document(page_content="Classification predicts discrete categories.", metadata={"source": "lec1.pdf", "page": 2, "chunk_id": "c2"})
    doc3 = Document(page_content="Adam is an optimization algorithm.", metadata={"source": "lec2.pdf", "page": 5, "chunk_id": "c3"})
    
    # Mock Chroma vector database search results for query "Adam optimization"
    mock_chroma_results = [
        (doc3, 0.4), # doc3 is top match in vector search
        (doc1, 0.8)  # doc1 is weak match in vector search
    ]
    mock_store = MockChroma(mock_chroma_results)
    
    # Setup a mock BM25 Index containing these documents
    bm25_index = BM25Index()
    bm25_index.fit([doc1, doc2, doc3])
    
    # Save the index to vector_db temporarily (so MultiRetriever init can load it)
    bm25_path = os.path.join("vector_db", "bm25_index.pkl")
    os.makedirs("vector_db", exist_ok=True)
    bm25_index.save(bm25_path)
    
    # Instantiate MultiRetriever with our mock Chroma
    retriever = MultiRetriever(mock_store)
    
    # Retrieve using query "Adam optimization"
    results = retriever.retrieve(["Adam optimization"], top_k=5)
    
    print("  Hybrid RRF Merged Results:")
    for idx, (doc, rrf) in enumerate(results):
        print(f"    [{idx+1}] Chunk {doc.metadata['chunk_id']} | RRF Score: {rrf:.6f}")
        print(f"      Vector Dist: {doc.metadata.get('vector_score')} | BM25 Score: {doc.metadata.get('bm25_score')}")
        
    chunk_ids = [d.metadata["chunk_id"] for d, _ in results]
    assert "c3" in chunk_ids
    
    # Check specific doc scores
    for doc, rrf in results:
        cid = doc.metadata["chunk_id"]
        if cid == "c3":
            assert abs(rrf - (1.0/61.0 + 1.0/61.0)) < 1e-5
            assert doc.metadata["vector_score"] == 0.4
            assert doc.metadata["bm25_score"] > 0.0
        elif cid == "c1":
            assert abs(rrf - (1.0/62.0)) < 1e-5
            assert doc.metadata["vector_score"] == 0.8
            assert doc.metadata["bm25_score"] == 0.0
            
    # Clean up mock file
    if os.path.exists(bm25_path):
        os.remove(bm25_path)
        
    print("[SUCCESS] Hybrid search RRF merging tests passed!\n")

if __name__ == "__main__":
    print("=== STARTING HYBRID RETRIEVAL (BM25 + DENSE) TESTS ===\n")
    test_bm25_index()
    test_hybrid_rrf_merging()
    print("=== ALL TESTS COMPLETED SUCCESSFULLY ===")
