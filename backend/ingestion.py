import os
import sys
from dotenv import load_dotenv

# Load configuration
load_dotenv()

# Import modular functions
try:
    from backend.pdf_loader import extract_text_from_pdf
except ModuleNotFoundError:
    from pdf_loader import extract_text_from_pdf
try:
    from backend.chunker import chunk_text
    from backend.embedding import embed_and_store_documents
except ModuleNotFoundError:
    from chunker import chunk_text
    from embedding import embed_and_store_documents

def run_ingestion_pipeline(pdf_dir, processed_dir, db_path, embedding_model):
    """
    Orchestrates the modular RAG ingestion pipeline:
    1. Extracts text from PDFs in pdf_dir -> saves .txt files to processed_dir.
    2. Chunks the saved .txt files using page-aware splitting.
    3. Generates embeddings and stores chunks in ChromaDB.
    """
    print("=== Starting RAG Ingestion Pipeline ===")
    
    # 1. Validation and Directory setup
    if not os.path.exists(pdf_dir):
        print(f"[ERROR] PDF source directory '{pdf_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(processed_dir, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'. Ingestion aborted.")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) to process.")
    
    # 2. Extract text and save to processed directory
    processed_txt_paths = []
    for pdf_name in pdf_files:
        pdf_path = os.path.join(pdf_dir, pdf_name)
        txt_name = os.path.splitext(pdf_name)[0] + ".txt"
        txt_path = os.path.join(processed_dir, txt_name)
        
        print(f"\n[1/3] Extracting text: '{pdf_name}'...")
        try:
            text_content = extract_text_from_pdf(pdf_path)
            
            # Save extracted text to a .txt file (UTF-8)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text_content)
            print(f"      -> Saved plain text to '{txt_path}'")
            processed_txt_paths.append(txt_path)
        except Exception as e:
            print(f"      -> [ERROR] Failed to extract text: {e}", file=sys.stderr)
            
    # 3. Chunk the extracted text files
    all_chunks = []
    print("\n[2/3] Chunking text files with page-metadata preservation...")
    for txt_path in processed_txt_paths:
        try:
            chunks = chunk_text(txt_path)
            all_chunks.extend(chunks)
            print(f"      -> '{os.path.basename(txt_path)}': split into {len(chunks)} chunks.")
        except Exception as e:
            print(f"      -> [ERROR] Failed to chunk '{txt_path}': {e}", file=sys.stderr)
            
    if not all_chunks:
        print("\n[INFO] No text chunks generated. Stopping pipeline.")
        return
        
    print(f"\nTotal chunks to index: {len(all_chunks)}")
    
    # 4. Embed chunks and save in ChromaDB
    print(f"\n[3/3] Sending chunks to embedding and storage module...")
    try:
        embed_and_store_documents(all_chunks, db_path, embedding_model)
    except Exception as e:
        print(f"      -> [ERROR] Ingestion failed at database write stage: {e}", file=sys.stderr)
        
    print("\n=== Ingestion Pipeline Completed Successfully ===")

if __name__ == "__main__":
    # Ensure UTF-8 output on Windows consoles
    sys.stdout.reconfigure(encoding='utf-8')
    
    pdf_folder = "data/pdfs"
    processed_folder = "data/processed"
    vector_db_path = os.getenv("VECTOR_DB_PATH", "vector_db/")
    embed_model = os.getenv("EMBEDDING_MODEL", "mxbai-embed-large")
    
    run_ingestion_pipeline(pdf_folder, processed_folder, vector_db_path, embed_model)
