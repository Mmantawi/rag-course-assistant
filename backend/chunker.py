import os
import re
import unicodedata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def clean_control_chars(text: str) -> str:
    """Removes Unicode control, format, and private-use characters except whitespace."""
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\r", "\t")
    )

def create_parent_documents(pages_dict, document_id, filename, chat_id=None):
    """
    Splits the extracted page texts into logical parent documents based on headings.
    If no headings are detected on a page, the entire page is treated as a parent document.
    """
    # Heuristics to detect section headings or slide titles
    heading_regex = re.compile(
        r'^('
        r'\d+(\.\d+)*\s+[A-Z].*|'  # Numbers like "1. Introduction" or "1.1 Background"
        r'[A-Z\s0-9:.-]{4,65}:?|'  # ALL CAPS headers (max 65 chars)
        r'(Section|Chapter|Lecture|Part|Module)\s+\d+.*' # Section 1, Lecture 1 etc.
        r')$'
    )
    
    parent_docs = []
    
    for page_num, text in pages_dict.items():
        lines = text.split('\n')
        sections = []
        current_heading = None
        current_content = []
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
                
            # If line matches heading regex and is reasonably short
            if heading_regex.match(stripped) and len(stripped) < 70:
                if current_content:
                    sections.append({
                        "title": current_heading or "Introduction",
                        "content": "\n".join(current_content).strip()
                    })
                current_heading = stripped
                current_content = [line]
            else:
                current_content.append(line)
                
        # Save trailing section
        if current_content:
            sections.append({
                "title": current_heading or "Slide Content",
                "content": "\n".join(current_content).strip()
            })
            
        # If no headings exist or only 1 section with the default name
        if not sections or (len(sections) == 1 and sections[0]["title"] == "Slide Content"):
            parent_id = f"{filename}_p{page_num}"
            parent_docs.append({
                "parent_id": parent_id,
                "document_id": str(document_id),
                "page": page_num,
                "title": "Page Content",
                "content": text.strip(),
                "filename": filename,
                "chat_id": str(chat_id) if chat_id else None
            })
        else:
            for sec_idx, sec in enumerate(sections):
                parent_id = f"{filename}_p{page_num}_s{sec_idx}"
                parent_docs.append({
                    "parent_id": parent_id,
                    "document_id": str(document_id),
                    "page": page_num,
                    "title": sec["title"],
                    "content": sec["content"],
                    "filename": filename,
                    "chat_id": str(chat_id) if chat_id else None
                })
                
    return parent_docs

def create_child_chunks(parent_docs, chunk_size=500, chunk_overlap=100):
    """
    Splits each parent document into child chunks:
    - If length < 1000 characters: splits by paragraphs.
    - Otherwise: uses RecursiveCharacterTextSplitter.
    """
    child_docs = []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    for p in parent_docs:
        parent_id = p["parent_id"]
        document_id = p["document_id"]
        page = p["page"]
        title = p["title"]
        text = p["content"]
        filename = p["filename"]
        
        chunks = []
        if len(text) < 1000:
            # Paragraph split
            paragraphs = [para.strip() for para in re.split(r'\n\s*\n', text) if para.strip()]
            if not paragraphs:
                paragraphs = [text]
            chunks = paragraphs
        else:
            # Standard chunk splitter
            chunks = text_splitter.split_text(text)
            
        chat_id = p.get("chat_id")
        for chunk_idx, chunk_text in enumerate(chunks):
            child_id = f"{parent_id}_c{chunk_idx}"
            
            doc = Document(
                page_content=chunk_text,
                metadata={
                    "child_id": child_id,
                    "parent_id": parent_id,
                    "document_id": document_id,
                    "page": page,
                    "section": title,
                    "chunk_index": chunk_idx,
                    "filename": filename,
                    "source": filename,       # Backward compatibility with existing citations/source lists
                    "chunk_id": child_id,      # Unique ID for database fit/deduplications
                    "chat_id": chat_id
                }
            )
            child_docs.append(doc)
            
    return child_docs

def chunk_text(file_path, chunk_size=500, chunk_overlap=100):
    """
    Fallback method for backwards compatibility with original standalone tests.
    Reads page headers and chunks the plain text page by page.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    filename = os.path.basename(file_path)
    source_name = os.path.splitext(filename)[0] + ".pdf"
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        
    page_splits = re.split(r'--- Page (\d+) ---\n', content)
    pages_dict = {}
    
    for i in range(1, len(page_splits), 2):
        page_num = int(page_splits[i])
        page_text = page_splits[i+1].strip()
        cleaned_text = clean_control_chars(page_text)
        if cleaned_text:
            pages_dict[page_num] = cleaned_text
            
    # Mock a dummy document ID for testing
    import uuid
    dummy_doc_id = uuid.uuid4()
    
    parent_docs = create_parent_documents(pages_dict, dummy_doc_id, source_name)
    return create_child_chunks(parent_docs, chunk_size, chunk_overlap)
