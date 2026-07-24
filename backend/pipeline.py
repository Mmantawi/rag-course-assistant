import os
import sys
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Import modular components
try:
    from backend.retriever import retrieve_relevant_chunks
    from backend.generator import generate_answer
except ModuleNotFoundError:
    from retriever import retrieve_relevant_chunks
    from generator import generate_answer

def run_rag_pipeline(question, db_path, embedding_model, llm_model, top_k=5, chat_history="", chat_id=None):
    """
    Orchestrates the complete RAG pipeline:
    1. Similarity search to retrieve top_k chunks.
    2. Builds the context text and compiles the metadata for citations.
    3. Runs the LLM generator to get the final grounded answer.
    
    Returns a dictionary:
    {
        "answer": str,
        "sources": list[dict]
    }
    """
    # 1. Retrieve the relevant chunks with scores
    retrieved_results = retrieve_relevant_chunks(
        question=question,
        db_path=db_path,
        embedding_model=embedding_model,
        top_k=top_k,
        chat_id=chat_id
    )
    
    if not retrieved_results:
        return {
            "answer": "I cannot find the answer in the provided documents.",
            "sources": []
        }
        
    # 2. Extract content for the context window and metadata for citation display
    context_snippets = []
    sources = []
    
    for doc, score in retrieved_results:
        source_name = doc.metadata.get("source", "Unknown source")
        page_num = doc.metadata.get("page", "Unknown page")
        
        # Collect text content for prompting the LLM
        context_snippets.append(doc.page_content)
        
        # Track metadata for frontend display/citations
        sources.append({
            "source": source_name,
            "page": page_num,
            "content": doc.page_content,
            "score": float(score),
            "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
            "vector_score": float(doc.metadata.get("vector_score", 1.0)),
            "bm25_score": float(doc.metadata.get("bm25_score", 0.0)),
            "rrf_score": float(doc.metadata.get("rrf_score", 0.0))
        })
        
    # Join the chunks together
    combined_context = "\n\n".join(context_snippets)
    
    # 3. Generate response using ChatOllama
    answer = generate_answer(
        question=question,
        context=combined_context,
        llm_model=llm_model,
        chat_history=chat_history
    )
    
    return {
        "answer": answer,
        "sources": sources
    }

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows
    sys.stdout.reconfigure(encoding='utf-8')
    
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db/")
    embed_model = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
    llm_model = os.getenv("LLM_MODEL", "llama3.2")
    top_k = int(os.getenv("TOP_K", 5))
    
    print("==================================================")
    print("Welcome to the Local RAG Chatbot CLI!")
    print(f"Database Path: {vector_db_path}")
    print(f"Embedding Model: {embed_model}")
    print(f"LLM: {llm_model}")
    print("Type 'exit' or 'quit' to end the session.")
    print("==================================================")
    
    while True:
        try:
            # Prompt the user for input
            question = input("\nAsk a question: ").strip()
            if not question:
                continue
            if question.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            print("\nRetrieving slides and generating answer...")
            result = run_rag_pipeline(
                question=question,
                db_path=vector_db_path,
                embedding_model=embed_model,
                llm_model=llm_model,
                top_k=top_k
            )
            
            print("\n=== ANSWER ===")
            print(result["answer"])
            print("==============")
            
            print("\n=== SOURCES ===")
            if not result["sources"]:
                print("No relevant slides found.")
            else:
                for idx, src in enumerate(result["sources"]):
                    print(f"[{idx+1}] PDF: '{src['source']}' (Page {src['page']}) (Distance: {src['score']:.4f})")
                    print(f"    Snippet: {src['content'][:150].replace(chr(10), ' ').strip()}...")
            print("==================================================")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\n[ERROR] Pipeline execution failed: {e}", file=sys.stderr)
