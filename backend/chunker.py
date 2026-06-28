import os
import re
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_text(file_path, chunk_size=None, chunk_overlap=None):
    """
    Reads a processed .txt file, parses it page-by-page using the 
    '--- Page X ---' delimiters, and returns a list of Document objects 
    where each page represents its own independent chunk.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    filename = os.path.basename(file_path)
    # Map back to original PDF source filename
    source_name = os.path.splitext(filename)[0] + ".pdf"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    # Split content by the page headers: --- Page (\d+) ---
    page_splits = re.split(r'--- Page (\d+) ---\n', content)
    
    # The first element is anything before the first header (usually empty)
    # The remaining elements are pairs of (page_num, page_content)
    documents = []
    
    for i in range(1, len(page_splits), 2):
        page_num = int(page_splits[i])
        page_text = page_splits[i+1].strip()
        
        if not page_text:
            continue
            
        doc = Document(
            page_content=page_text,
            metadata={
                "source": source_name,
                "page": page_num,
                "chunk_id": f"{source_name}_p{page_num}"
            }
        )
        documents.append(doc)
            
    return documents

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    processed_dir = "data/processed"
    txt_files = [f for f in os.listdir(processed_dir) if f.endswith(".txt")]
    
    if not txt_files:
        print("No processed text files found in data/processed/")
    else:
        test_file = os.path.join(processed_dir, txt_files[0])
        print(f"Testing chunking on file: '{test_file}'")
        
        # Load parameters from environment variables
        c_size = int(os.getenv("CHUNK_SIZE", 500))
        c_overlap = int(os.getenv("CHUNK_OVERLAP", 100))
        print(f"Configured Chunk Size: {c_size}, Overlap: {c_overlap}\n")
        
        chunks = chunk_text(test_file, c_size, c_overlap)
        print(f"Generated {len(chunks)} chunks.\n")
        
        # Print the first 3 chunks to verify content and metadata
        for idx, chunk in enumerate(chunks[:3]):
            print(f"--- Chunk {idx + 1} ---")
            print(f"Metadata: {chunk.metadata}")
            print(f"Content Preview (first 150 chars):\n{chunk.page_content[:150]}...")
            print("-" * 30 + "\n")
