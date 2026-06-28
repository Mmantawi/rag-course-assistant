import os
import sys
import json
import shutil
import asyncio
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add root folder to sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import configurations
from backend.config import (
    LLM_MODEL,
    EMBEDDING_MODEL,
    VECTOR_DB_PATH,
    PDF_FOLDER,
    PROCESSED_FOLDER,
    TOP_K
)

# Import RAG pipeline & ingestion entrypoints
from backend.pipeline import run_rag_pipeline
from backend.ingestion import run_ingestion_pipeline
from backend.retriever import retrieve_relevant_chunks
from backend.prompts import RAG_PROMPT

try:
    from langchain_ollama import ChatOllama
except ImportError:
    from langchain_community.chat_models import ChatOllama

# Initialize FastAPI App
app = FastAPI(
    title="RAG Course Assistant API",
    description="Backend service for PDF ingestion and Retrieval-Augmented Generation.",
    version="1.0.0"
)

# Enable CORS for frontend Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins in local dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define schemas
class ChatRequest(BaseModel):
    question: str
    model_choice: str = "local"

@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Returns the health status and current model configuration of the backend."""
    return {
        "status": "healthy",
        "llm_model": LLM_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "vector_db_path": VECTOR_DB_PATH
    }

@app.get("/documents", status_code=status.HTTP_200_OK)
async def list_documents():
    """Lists the filenames of all PDF documents currently stored and indexed."""
    if not os.path.exists(PDF_FOLDER):
        return {"documents": []}
    
    pdfs = [f for f in os.listdir(PDF_FOLDER) if f.endswith(".pdf")]
    return {"documents": pdfs}

@app.get("/pdf/{filename}")
async def get_pdf(filename: str):
    """Serves a raw PDF file from the local storage folder to display it in browser tab."""
    file_path = os.path.join(PDF_FOLDER, filename)
    if not os.path.exists(file_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PDF file '{filename}' not found."
        )
    return FileResponse(file_path, media_type="application/pdf")

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(files: list[UploadFile] = File(...)):
    """
    Accepts one or more PDF files, saves them to the data/pdfs/ directory,
    and runs the text extraction, chunking, and database ingestion pipeline.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")
        
    os.makedirs(PDF_FOLDER, exist_ok=True)
    saved_files = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"File '{file.filename}' is not a PDF. Only PDF files are supported."
            )
            
        file_path = os.path.join(PDF_FOLDER, file.filename)
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            saved_files.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to save file '{file.filename}': {e}"
            )
            
    print(f"\n[API] Ingesting newly uploaded files: {saved_files}...")
    try:
        # Run the full ingestion pipeline to parse, chunk, embed, and store
        run_ingestion_pipeline(PDF_FOLDER, PROCESSED_FOLDER, VECTOR_DB_PATH, EMBEDDING_MODEL)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document ingestion failed: {e}"
        )
        
    return {
        "message": f"Successfully uploaded and processed {len(saved_files)} document(s).",
        "processed_files": saved_files
    }

@app.post("/chat")
async def chat(request: ChatRequest):
    """
    Executes the static (non-streaming) RAG pipeline for the given question.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")
        
    try:
        result = run_rag_pipeline(
            question=request.question,
            db_path=VECTOR_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            llm_model=request.model_choice,
            top_k=TOP_K
        )
        return result
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate RAG response: {e}"
        )

@app.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Executes the streaming RAG pipeline for the given question.
    Yields JSON lines (SSE-like):
    1. First line yields list of source chunks.
    2. Subsequent lines yield LLM text chunks as they generate.
    """
    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    async def event_generator():
        try:
            # 1. Retrieve relevant chunks first
            retrieved_results = retrieve_relevant_chunks(
                question=request.question,
                db_path=VECTOR_DB_PATH,
                embedding_model=EMBEDDING_MODEL,
                top_k=TOP_K
            )
            
            # 2. Extract contents and sources metadata
            context_snippets = []
            sources = []
            
            for doc, score in retrieved_results:
                context_snippets.append(doc.page_content)
                sources.append({
                    "source": doc.metadata.get("source", "Unknown"),
                    "page": doc.metadata.get("page", "Unknown"),
                    "content": doc.page_content,
                    "score": float(score),
                    "chunk_id": doc.metadata.get("chunk_id", "Unknown"),
                    "vector_score": float(doc.metadata.get("vector_score", 1.0)),
                    "bm25_score": float(doc.metadata.get("bm25_score", 0.0)),
                    "rrf_score": float(doc.metadata.get("rrf_score", 0.0))
                })
            
            # Send sources metadata as the first event chunk
            yield json.dumps({"type": "sources", "data": sources}) + "\n"
            await asyncio.sleep(0.001)  # Release event loop
            
            # 3. Stream from selected model
            combined_context = "\n\n".join(context_snippets)
            formatted_prompt = RAG_PROMPT.format(context=combined_context, question=request.question)
            
            from backend.generator import get_llm
            llm = get_llm(request.model_choice)
            
            # Async stream LLM tokens
            async for chunk in llm.astream(formatted_prompt):
                yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
                await asyncio.sleep(0.001)
                
        except ValueError as ve:
            yield json.dumps({"type": "error", "data": str(ve)}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": f"Stream error: {str(e)}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/documents")
async def delete_documents():
    """
    Deletes all uploaded PDFs, processed text files, and resets the vector database.
    """
    errors = []
    
    # 1. Clear directories
    for folder in [PDF_FOLDER, PROCESSED_FOLDER]:
        if os.path.exists(folder):
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                except Exception as e:
                    errors.append(f"Failed to delete '{file_path}': {e}")
                    
    # 2. Delete Chroma Collection to wipe database
    try:
        from langchain_ollama import OllamaEmbeddings
        from langchain_community.vectorstores import Chroma
        
        embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
        vector_store = Chroma(
            collection_name="pdf_rag_collection",
            embedding_function=embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        vector_store.delete_collection()
    except Exception as e:
        errors.append(f"Failed to reset ChromaDB collection: {e}")
        
    # 3. Delete BM25 index file if it exists
    bm25_path = os.path.join(VECTOR_DB_PATH, "bm25_index.pkl")
    if os.path.exists(bm25_path):
        try:
            os.remove(bm25_path)
        except Exception as e:
            errors.append(f"Failed to delete BM25 index: {e}")
        
    if errors:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"message": "Partial deletion failures occurred.", "errors": errors}
        )
        
    return {"message": "All documents successfully deleted and database index reset."}
