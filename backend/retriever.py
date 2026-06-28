import os
import sys
from dotenv import load_dotenv

# Load configuration
load_dotenv()

try:
    from langchain_ollama import OllamaEmbeddings
except ImportError:
    from langchain_community.embeddings import OllamaEmbeddings

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    print("[ERROR] Could not import Chroma. Make sure chromadb is installed.")
    sys.exit(1)

def retrieve_relevant_chunks(question, db_path, embedding_model, top_k=5):
    """
    Connects to ChromaDB and performs a pre-retrieval query analysis and
    post-retrieval reranking pipeline:
    1. Query Normalization (QueryAnalyzer)
    2. Concept Extraction via LLM (ConceptExtractor)
    3. Independent Searches per Concept (MultiRetriever)
    4. Merging, Deduplication & Page Diversity Filtering (ResultMerger)
    5. Cosine Similarity Reranking against original normalized query (ResultReranker)
    
    Returns a list of tuples: (Document, similarity_score)
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"ChromaDB directory not found at '{db_path}'. Please run ingestion first.")
        
    # Import modular components locally to avoid circular dependencies
    try:
        from backend.query_analysis import (
            QueryAnalyzer,
            ConceptExtractor,
            MultiRetriever,
            ResultMerger,
            ResultReranker
        )
    except ModuleNotFoundError:
        from query_analysis import (
            QueryAnalyzer,
            ConceptExtractor,
            MultiRetriever,
            ResultMerger,
            ResultReranker
        )

    # Initialize embeddings model
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    # Connect to the existing vector store
    vector_store = Chroma(
        collection_name="pdf_rag_collection",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    # 1. Query Normalization
    normalized_q = QueryAnalyzer.normalize(question)
    
    # 2. Concept Extraction
    extractor = ConceptExtractor()
    concepts = extractor.extract_concepts(normalized_q)
    print(f"[Retrieval] Extracted concepts for '{normalized_q}': {concepts}")
    
    # 3. Multi-Concept Retrieval
    retriever = MultiRetriever(vector_store)
    raw_candidates = retriever.retrieve(concepts, top_k=top_k)
    
    # 4. Result Merging & Deduplication & Page Diversity Filter
    merger = ResultMerger(embeddings)
    merged_candidates = merger.merge_and_deduplicate(raw_candidates)
    
    # 5. Cosine Similarity Reranking against original normalized query
    reranker = ResultReranker(embeddings)
    reranked_candidates = reranker.rerank(normalized_q, merged_candidates)
    
    # Return the top k reranked documents
    return reranked_candidates[:top_k]

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Read settings from environment variables
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db/")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    default_top_k = int(os.getenv("TOP_K", 5))
    
    # Use command-line arguments as the test question, or default to a slide topic
    if len(sys.argv) > 1:
        test_question = " ".join(sys.argv[1:])
    else:
        test_question = "What are the grades in year work and final?"
        
    print(f"Querying database with question: \"{test_question}\"")
    print(f"Top K configured: {default_top_k}\n")
    
    try:
        retrieved_results = retrieve_relevant_chunks(
            question=test_question,
            db_path=vector_db_path,
            embedding_model=embed_model,
            top_k=default_top_k
        )
        
        print(f"Found {len(retrieved_results)} relevant chunk(s):\n")
        
        for idx, (doc, score) in enumerate(retrieved_results):
            source = doc.metadata.get("source", "Unknown source")
            page = doc.metadata.get("page", "Unknown page")
            
            print(f"--- Result {idx + 1} (Score: {score:.4f}) ---")
            print(f"Source: {source} (Page {page})")
            print(f"Content:\n{doc.page_content.strip()}")
            print("-" * 50 + "\n")
            
    except Exception as e:
        print(f"[ERROR] Retrieval failed: {e}", file=sys.stderr)
        sys.exit(1)
