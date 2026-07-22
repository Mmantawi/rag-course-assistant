import os
import sys
import json
import shutil
import asyncio
import datetime
import uuid
import jwt
import bcrypt
from typing import Optional, List
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

# Add root folder to sys.path to resolve imports cleanly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import database connection
from backend.database import get_db, SessionLocal, Base, engine
from backend.models import User, Chat, Message, UploadedDocument

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
    description="Backend service for PDF ingestion and Retrieval-Augmented Generation with PostgreSQL.",
    version="2.0.0"
)

# Enable CORS for frontend Streamlit access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# JWT configurations
JWT_SECRET = os.getenv("JWT_SECRET", "super-secret-rag-chatbot-key-2026")
JWT_ALGORITHM = "HS256"
security = HTTPBearer()

# Helper Functions
def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

def verify_password(password: str, hashed_password: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed_password.encode("utf-8"))

def create_access_token(user_id: uuid.UUID) -> str:
    payload = {
        "user_id": str(user_id),
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7)
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

def get_current_user_id(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload["user_id"]
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials / Token expired",
            headers={"WWW-Authenticate": "Bearer"},
        )

# Pydantic Schemas
class UserRegister(BaseModel):
    username: str
    email: EmailStr
    password: str

class UserLogin(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    username: str

class ChatCreate(BaseModel):
    title: Optional[str] = None

class ChatRequest(BaseModel):
    chat_id: str
    question: str
    model_choice: str = "local"

# --- Background Task for Document Ingestion ---
def process_ingestion_background(saved_filenames: List[str], user_id_str: str):
    db = SessionLocal()
    user_uuid = uuid.UUID(user_id_str)
    try:
        # Mark all as processing
        db.query(UploadedDocument).filter(
            UploadedDocument.filename.in_(saved_filenames),
            UploadedDocument.user_id == user_uuid
        ).update({"status": "processing"}, synchronize_session=False)
        db.commit()

        # Run ingestion
        run_ingestion_pipeline(PDF_FOLDER, PROCESSED_FOLDER, VECTOR_DB_PATH, EMBEDDING_MODEL)

        # Mark all as ready
        db.query(UploadedDocument).filter(
            UploadedDocument.filename.in_(saved_filenames),
            UploadedDocument.user_id == user_uuid
        ).update({"status": "ready"}, synchronize_session=False)
        db.commit()
    except Exception as e:
        print(f"[ERROR] Background document ingestion failed: {e}", file=sys.stderr)
        db.query(UploadedDocument).filter(
            UploadedDocument.filename.in_(saved_filenames),
            UploadedDocument.user_id == user_uuid
        ).update({"status": "failed"}, synchronize_session=False)
        db.commit()
    finally:
        db.close()

# --- Authentication Routes ---
@app.post("/auth/register", status_code=status.HTTP_201_CREATED)
async def register(payload: UserRegister, db: Session = Depends(get_db)):
    # Check if user already exists
    existing_username = db.query(User).filter(User.username == payload.username).first()
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    existing_email = db.query(User).filter(User.email == payload.email).first()
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")

    new_user = User(
        username=payload.username,
        email=payload.email,
        password_hash=hash_password(payload.password)
    )
    db.add(new_user)
    db.commit()
    return {"message": "User registered successfully"}

@app.post("/auth/login", response_model=TokenResponse)
async def login(payload: UserLogin, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == payload.username).first()
    if not user or not verify_password(payload.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")

    user.last_login = datetime.datetime.utcnow()
    db.commit()

    token = create_access_token(user.id)
    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username
    }

@app.get("/auth/me")
async def get_me(db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    user = db.query(User).filter(User.id == uuid.UUID(current_user_id)).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "id": str(user.id),
        "username": user.username,
        "email": user.email,
        "created_at": user.created_at.isoformat()
    }

# --- Chat & History Routes ---
@app.get("/chats")
async def list_chats(db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    chats = db.query(Chat).filter(Chat.user_id == uuid.UUID(current_user_id)).order_by(Chat.updated_at.desc()).all()
    return [
        {
            "id": str(c.id),
            "title": c.title,
            "created_at": c.created_at.isoformat(),
            "updated_at": c.updated_at.isoformat()
        } for c in chats
    ]

@app.post("/chats", status_code=status.HTTP_201_CREATED)
async def create_chat(payload: ChatCreate, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    title = payload.title or "New Chat"
    new_chat = Chat(
        user_id=uuid.UUID(current_user_id),
        title=title
    )
    db.add(new_chat)
    db.commit()
    db.refresh(new_chat)
    return {
        "id": str(new_chat.id),
        "title": new_chat.title,
        "created_at": new_chat.created_at.isoformat()
    }

@app.delete("/chats/{chat_id}")
async def delete_chat(chat_id: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    chat = db.query(Chat).filter(Chat.id == uuid.UUID(chat_id), Chat.user_id == uuid.UUID(current_user_id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    db.delete(chat)
    db.commit()
    return {"message": "Chat deleted successfully"}

@app.get("/chats/{chat_id}/messages")
async def get_chat_messages(chat_id: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    chat = db.query(Chat).filter(Chat.id == uuid.UUID(chat_id), Chat.user_id == uuid.UUID(current_user_id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")
    
    return [
        {
            "id": str(m.id),
            "content": m.content,
            "role": m.role,
            "created_at": m.created_at.isoformat()
        } for m in chat.messages
    ]

# --- Core RAG & Ingestion Endpoints ---
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
async def list_documents(db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    """Lists all PDF documents currently stored and indexed in PostgreSQL for the current user."""
    docs = db.query(UploadedDocument).filter(UploadedDocument.user_id == uuid.UUID(current_user_id)).order_by(UploadedDocument.uploaded_at.desc()).all()
    return {
        "documents": [
            {
                "id": str(d.id),
                "filename": d.filename,
                "storage_path": d.storage_path,
                "file_size": d.file_size,
                "status": d.status,
                "uploaded_at": d.uploaded_at.isoformat()
            } for d in docs
        ]
    }

@app.get("/pdf/{filename}")
async def get_pdf(filename: str, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    """Serves a raw PDF file from the local storage folder to display it in browser tab."""
    # Authenticate file access
    doc = db.query(UploadedDocument).filter(
        UploadedDocument.filename == filename,
        UploadedDocument.user_id == uuid.UUID(current_user_id)
    ).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found or unauthorized access."
        )

    if not os.path.exists(doc.storage_path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Physical file missing on server."
        )
    return FileResponse(doc.storage_path, media_type="application/pdf")

@app.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_documents(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user_id: str = Depends(get_current_user_id)
):
    """
    Accepts one or more PDF files, saves them physically on disk,
    adds records in PostgreSQL with status 'uploaded', and triggers background ingestion.
    """
    if not files:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No files uploaded.")
        
    os.makedirs(PDF_FOLDER, exist_ok=True)
    saved_filenames = []
    
    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail=f"File '{file.filename}' is not a PDF."
            )
            
        file_path = os.path.join(PDF_FOLDER, file.filename)
        try:
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            
            # Create metadata record in DB if not already present
            existing_doc = db.query(UploadedDocument).filter(
                UploadedDocument.filename == file.filename,
                UploadedDocument.user_id == uuid.UUID(current_user_id)
            ).first()
            
            if not existing_doc:
                new_doc = UploadedDocument(
                    user_id=uuid.UUID(current_user_id),
                    filename=file.filename,
                    storage_path=file_path,
                    file_size=len(content),
                    status="uploaded"
                )
                db.add(new_doc)
            else:
                existing_doc.status = "uploaded"
                existing_doc.file_size = len(content)
                existing_doc.uploaded_at = datetime.datetime.utcnow()
                
            db.commit()
            saved_filenames.append(file.filename)
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to process file '{file.filename}': {e}"
            )
            
    print(f"\n[API] Ingesting files in background for user {current_user_id}: {saved_filenames}...")
    background_tasks.add_task(process_ingestion_background, saved_filenames, current_user_id)
        
    return {
        "message": f"Successfully uploaded {len(saved_filenames)} document(s). Processing in background.",
        "processed_files": saved_filenames
    }

@app.post("/chat")
async def chat(request: ChatRequest, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    """
    Executes the static (non-streaming) RAG pipeline, updates database history.
    """
    chat_uuid = uuid.UUID(request.chat_id)
    chat = db.query(Chat).filter(Chat.id == chat_uuid, Chat.user_id == uuid.UUID(current_user_id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")
        
    # Save user message
    user_msg = Message(chat_id=chat_uuid, content=request.question, role="user")
    db.add(user_msg)
    db.commit()

    try:
        result = run_rag_pipeline(
            question=request.question,
            db_path=VECTOR_DB_PATH,
            embedding_model=EMBEDDING_MODEL,
            llm_model=request.model_choice,
            top_k=TOP_K
        )
        
        # Save assistant response
        assistant_msg = Message(chat_id=chat_uuid, content=result["answer"], role="assistant")
        db.add(assistant_msg)
        chat.updated_at = datetime.datetime.utcnow()
        db.commit()
        
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
async def chat_stream(request: ChatRequest, db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    """
    Executes the streaming RAG pipeline, logging history to database on completion.
    """
    chat_uuid = uuid.UUID(request.chat_id)
    chat = db.query(Chat).filter(Chat.id == chat_uuid, Chat.user_id == uuid.UUID(current_user_id)).first()
    if not chat:
        raise HTTPException(status_code=404, detail="Chat session not found")

    if not request.question.strip():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question cannot be empty.")

    # Save user message
    user_msg = Message(chat_id=chat_uuid, content=request.question, role="user")
    db.add(user_msg)
    db.commit()

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
            await asyncio.sleep(0.001)
            
            # 3. Stream from selected model
            combined_context = "\n\n".join(context_snippets)
            formatted_prompt = RAG_PROMPT.format(context=combined_context, question=request.question)
            
            from backend.generator import get_llm
            llm = get_llm(request.model_choice)
            
            full_response = ""
            async for chunk in llm.astream(formatted_prompt):
                full_response += chunk.content
                yield json.dumps({"type": "token", "data": chunk.content}) + "\n"
            
            # Save assistant response to DB
            if full_response.strip():
                db_bg = SessionLocal()
                try:
                    assistant_msg = Message(chat_id=chat_uuid, content=full_response, role="assistant")
                    db_bg.add(assistant_msg)
                    db_bg.query(Chat).filter(Chat.id == chat_uuid).update({"updated_at": datetime.datetime.utcnow()})
                    db_bg.commit()
                finally:
                    db_bg.close()
                
        except ValueError as ve:
            yield json.dumps({"type": "error", "data": str(ve)}) + "\n"
        except Exception as e:
            yield json.dumps({"type": "error", "data": f"Stream error: {str(e)}"}) + "\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.delete("/documents")
async def delete_documents(db: Session = Depends(get_db), current_user_id: str = Depends(get_current_user_id)):
    """
    Deletes all uploaded PDFs, processed text files, and resets the vector database, removing metadata records.
    """
    errors = []
    
    # 1. Clear files on disk
    user_uuid = uuid.UUID(current_user_id)
    user_docs = db.query(UploadedDocument).filter(UploadedDocument.user_id == user_uuid).all()
    
    for doc in user_docs:
        # Delete raw PDF
        if os.path.exists(doc.storage_path):
            try:
                os.remove(doc.storage_path)
            except Exception as e:
                errors.append(f"Failed to delete PDF '{doc.storage_path}': {e}")
                
        # Delete corresponding processed txt
        txt_filename = os.path.splitext(doc.filename)[0] + ".txt"
        txt_path = os.path.join(PROCESSED_FOLDER, txt_filename)
        if os.path.exists(txt_path):
            try:
                os.remove(txt_path)
            except Exception as e:
                errors.append(f"Failed to delete processed file '{txt_path}': {e}")

    # Remove records from PostgreSQL
    try:
        db.query(UploadedDocument).filter(UploadedDocument.user_id == user_uuid).delete()
        db.commit()
    except Exception as e:
        errors.append(f"Failed to delete document records from PostgreSQL: {e}")
        
    # 2. Delete Chroma Collection if no files remain
    total_remaining_docs = db.query(UploadedDocument).count()
    if total_remaining_docs == 0:
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
            
        # Delete BM25 index file if it exists
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
