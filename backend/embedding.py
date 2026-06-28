import os
import sys
from dotenv import load_dotenv

# Load environment variables
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

def embed_and_store_documents(documents, db_path, embedding_model):
    """
    Generates embeddings for a list of Document objects using Ollama
    and stores them in ChromaDB. Prevents duplicates using chunk_id.
    """
    if not documents:
        print("[INFO] No documents provided for embedding.")
        return
        
    print(f"Initializing embeddings ('{embedding_model}')...")
    embeddings = OllamaEmbeddings(model=embedding_model)
    
    print(f"Connecting to ChromaDB at '{db_path}'...")
    vector_store = Chroma(
        collection_name="pdf_rag_collection",
        embedding_function=embeddings,
        persist_directory=db_path
    )
    
    # Extract unique IDs from metadata to prevent duplicate document insertion
    ids = [doc.metadata["chunk_id"] for doc in documents]
    
    print(f"Writing {len(documents)} chunk(s) to ChromaDB...")
    vector_store.add_documents(documents, ids=ids)
    print("      -> Embeddings successfully saved to ChromaDB!")
    
    # Build and serialize the BM25 Index
    print("Building BM25 Index over the exact same chunks...")
    try:
        from backend.bm25 import BM25Index
    except ModuleNotFoundError:
        from bm25 import BM25Index
        
    bm25_index = BM25Index()
    bm25_index.fit(documents)
    
    bm25_path = os.path.join(db_path, "bm25_index.pkl")
    print(f"Saving BM25 Index to '{bm25_path}'...")
    bm25_index.save(bm25_path)
    print("      -> BM25 Index successfully saved!")

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    # Standard script fallback for isolated testing
    from backend.chunker import chunk_text
    
    processed_folder = "data/processed"
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db/")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    
    if not os.path.exists(processed_folder):
        print(f"[ERROR] Directory '{processed_folder}' does not exist.")
        sys.exit(1)
        
    txt_files = [f for f in os.listdir(processed_folder) if f.endswith(".txt")]
    all_docs = []
    
    print(f"Testing embedding.py standalone on {len(txt_files)} file(s)...")
    for txt in txt_files:
        docs = chunk_text(os.path.join(processed_folder, txt))
        all_docs.extend(docs)
        
    embed_and_store_documents(all_docs, vector_db_path, embed_model)
