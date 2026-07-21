import os
import json
import requests
import streamlit as st

# Set page configurations
st.set_page_config(
    page_title="RAG Course Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Import custom UI components
from components import inject_theme_css, render_source_card, render_sidebar_doc

# Backend API URLs
API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
HEALTH_URL = f"{API_BASE_URL}/health"
UPLOAD_URL = f"{API_BASE_URL}/upload"
CHAT_STREAM_URL = f"{API_BASE_URL}/chat/stream"
DOCS_URL = f"{API_BASE_URL}/documents"

# Initialize Session State variables
if "theme" not in st.session_state:
    st.session_state["theme"] = "dark"
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_sources" not in st.session_state:
    st.session_state["last_sources"] = []
if "sources_history" not in st.session_state:
    st.session_state["sources_history"] = []
if "active_source_index" not in st.session_state:
    st.session_state["active_source_index"] = 0
if "uploader_version" not in st.session_state:
    st.session_state["uploader_version"] = 0

# Inject CSS based on the chosen theme
inject_theme_css(st.session_state["theme"])

# --- Helper functions ---
def fetch_documents():
    """Fetches the list of indexed PDFs from the backend API."""
    try:
        r = requests.get(DOCS_URL, timeout=5)
        if r.status_code == 200:
            return r.json().get("documents", [])
    except Exception as e:
        print(f"Error fetching documents: {e}")
    return []

def get_storage_stats():
    """Calculates the number of PDF documents and the total disk space occupied in MB."""
    import os
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(project_root, "data", "pdfs")
    
    if not os.path.exists(pdf_dir):
        return 0, 0.0
        
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    num_files = len(pdf_files)
    
    total_bytes = 0
    for f in pdf_files:
        path = os.path.join(pdf_dir, f)
        if os.path.isfile(path):
            total_bytes += os.path.getsize(path)
            
    total_mb = total_bytes / (1024 * 1024)
    return num_files, total_mb

# --- Main App Layout ---

# Header columns: Title on left, Light/Dark Switch on top right
header_left, header_right = st.columns([8, 2])

with header_left:
    st.markdown("# RAG Course Assistant")
    st.write("Ask natural language questions about your uploaded PDF course materials.")

with header_right:
    # Theme toggle switch
    is_light = st.toggle(
        "Light Mode", 
        value=(st.session_state["theme"] == "light"),
        help="Switch between Light and Dark visual styles."
    )
    new_theme = "light" if is_light else "dark"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

# Check Backend API Server Status with auto-retry
import time

backend_online = False
max_retries = 20
retry_delay = 1.0

# Show a transient status indicator while waiting for startup
with st.spinner("Connecting to Backend API Server..."):
    for attempt in range(max_retries):
        try:
            health_check = requests.get(HEALTH_URL, timeout=2)
            if health_check.status_code == 200:
                backend_online = True
                break
        except Exception:
            pass
        time.sleep(retry_delay)

if not backend_online:
    st.error("⚠️ **Backend API Server is Offline!** Please ensure you have started the backend API server (`python main.py`).")
    if st.button("🔄 Retry Connection"):
        st.rerun()
    st.stop()

st.markdown("---")

# 3-Column layout: Document Manager (Left), Chat Window (Middle), Sources Panel (Right)
doc_col, chat_col, sources_col = st.columns([1.2, 2.0, 1.4], gap="medium")

# --- Left Column: Document Manager ---
with doc_col:
    st.markdown('<div class="panel-header">📁 Document Manager</div>', unsafe_allow_html=True)
    
    with st.container(height=650):
        # Display storage size and document count stats
        num_docs, size_mb = get_storage_stats()
        st.markdown(
            f"""
            <div style="background-color: var(--card-bg); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; margin-bottom: 16px; display: flex; justify-content: space-around; text-align: center;">
                <div>
                    <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Indexed Files</div>
                    <div style="font-size: 18px; font-weight: bold; color: var(--accent); margin-top: 4px;">{num_docs}</div>
                </div>
                <div style="border-left: 1px solid var(--border-color); height: 35px; align-self: center;"></div>
                <div>
                    <div style="font-size: 11px; color: var(--text-secondary); text-transform: uppercase; font-weight: 600; letter-spacing: 0.5px;">Space Occupied</div>
                    <div style="font-size: 18px; font-weight: bold; color: var(--accent); margin-top: 4px;">{size_mb:.2f} MB</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        st.write("Upload PDF slides to index them in the local vector store.")
        
        # LLM Model Selector
        st.markdown("### 🤖 LLM Model Selection")
        selected_model = st.selectbox(
            "Select LLM Provider",
            options=["Local (Ollama)", "Gemini API", "Groq API"],
            index=0,
            help="Choose between running models locally or using cloud provider APIs (requires API keys in your .env)."
        )
        
        # Map choice to backend key
        model_mapping = {
            "Local (Ollama)": "local",
            "Gemini API": "gemini",
            "Groq API": "groq"
        }
        st.session_state["model_choice"] = model_mapping[selected_model]
        
        st.markdown("---")

        
        # PDF File Uploader
        uploaded_files = st.file_uploader(
            "Select PDF slides",
            type=["pdf"],
            accept_multiple_files=True,
            key=f"pdf_uploader_{st.session_state['uploader_version']}",
            label_visibility="collapsed"
        )
        
        if uploaded_files:
            if st.button("🚀 Ingest Documents", use_container_width=True):
                # Formulate multipart request
                files_payload = []
                for file in uploaded_files:
                    # Read file bytes and append to upload list
                    files_payload.append(
                        ("files", (file.name, file.getvalue(), "application/pdf"))
                    )
                    
                with st.spinner("Processing slides (Extracting, Chunking, Embedding)..."):
                    try:
                        r = requests.post(UPLOAD_URL, files=files_payload, timeout=600)
                        if r.status_code == 201:
                            st.success(f"Ingested {len(uploaded_files)} PDF(s) successfully!")
                            st.session_state["uploader_version"] += 1  # Clear uploader widget
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {r.json().get('detail', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Could not connect to backend: {e}")
                        
        st.markdown("---")
        
        # Display Indexed files
        st.markdown("### 🗄️ Indexed Slides")
        indexed_pdfs = fetch_documents()
        
        if not indexed_pdfs:
            st.info("No documents indexed yet. Upload PDF files to start.")
        else:
            for doc in indexed_pdfs:
                render_sidebar_doc(doc)
                
        st.markdown("---")
        
        # Reset/Delete Database Button
        if st.button("🗑️ Reset Vector Store", type="primary", use_container_width=True):
            with st.spinner("Clearing indexes..."):
                try:
                    r = requests.delete(DOCS_URL, timeout=15)
                    if r.status_code == 200:
                        st.session_state["messages"] = []
                        st.session_state["last_sources"] = []
                        st.session_state["sources_history"] = []
                        st.session_state["active_source_index"] = 0
                        st.session_state["uploader_version"] += 1  # Clear uploader widget
                        st.success("Database successfully reset!")
                        st.rerun()
                    else:
                        st.error("Failed to clear database.")
                except Exception as e:
                    st.error(f"Error connecting to backend: {e}")

# --- Middle Column: Chat Window ---
with chat_col:
    st.markdown('<div class="panel-header">💬 Chat Assistant</div>', unsafe_allow_html=True)
    
    # Scrollable container for chat history (height 560 to leave room for input)
    chat_container = st.container(height=560)
    
    with chat_container:
        # Render Chat Messages History with highlighting for the active question
        for idx, msg in enumerate(st.session_state["messages"]):
            is_active = False
            if msg["role"] == "user" and st.session_state.get("sources_history"):
                active_idx = st.session_state["active_source_index"]
                if active_idx < len(st.session_state["sources_history"]):
                    is_active = (idx == st.session_state["sources_history"][active_idx]["msg_index"])
                    
            with st.chat_message(msg["role"]):
                if is_active:
                    st.markdown(f'<div class="active-question-highlight">{msg["content"]}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(msg["content"])
                
    # Capture User Question Input outside the scrollable container (fixed at bottom)
    user_query = st.chat_input("Ask a question about the slides...")
    
    if user_query:
        # 1. Append user message to history
        st.session_state["messages"].append({"role": "user", "content": user_query})
        
        # 2. Show the user message and start assistant streaming inside the container
        with chat_container:
            # Render user query instantly inside the container
            with st.chat_message("user"):
                st.markdown(user_query)
                
            # Now stream assistant response inside the container
            with st.chat_message("assistant"):
                answer_placeholder = st.empty()
                full_response = ""
                
                try:
                    # Call streaming endpoint
                    r = requests.post(
                        CHAT_STREAM_URL,
                        json={
                            "question": user_query,
                            "model_choice": st.session_state.get("model_choice", "local")
                        },
                        stream=True,
                        timeout=(10, None)  # 10s connect, infinite read timeout
                    )
                    
                    # Check for successful connection
                    if r.status_code != 200:
                        st.error("Error communicating with RAG chat pipeline.")
                    else:
                        # Iterate through the Server-Sent Event stream
                        for line in r.iter_lines():
                            if line:
                                data = json.loads(line.decode('utf-8'))
                                
                                if data["type"] == "sources":
                                    sources = data["data"]
                                    st.session_state["last_sources"] = sources
                                    # Record this in sources history
                                    st.session_state["sources_history"].append({
                                        "question": user_query,
                                        "sources": sources,
                                        "msg_index": len(st.session_state["messages"]) - 1
                                    })
                                    st.session_state["active_source_index"] = len(st.session_state["sources_history"]) - 1
                                elif data["type"] == "token":
                                    full_response += data["data"]
                                    answer_placeholder.markdown(full_response + "▌")
                                elif data["type"] == "error":
                                    st.error(data["data"])
                                    
                except Exception as e:
                    st.error(f"Failed to fetch streaming response: {e}")
                    
            # Post-stream cleanup and state updates (always run even if partial response / stream interrupted)
            if full_response:
                answer_placeholder.markdown(full_response)
                st.session_state["messages"].append(
                    {"role": "assistant", "content": full_response}
                )
            if full_response or st.session_state.get("last_sources"):
                st.rerun()

# --- Right Column: Slide Citations Panel ---
with sources_col:
    st.markdown('<div class="panel-header">💡 Retrieved Slides & Context</div>', unsafe_allow_html=True)
    
    with st.container(height=650):
        if not st.session_state.get("sources_history"):
            st.info("No sources retrieved yet. Ask a question to see matching slide references here.")
        else:
            history = st.session_state["sources_history"]
            active_idx = st.session_state["active_source_index"]
            total = len(history)
            
            # Clamp active_idx safely
            active_idx = max(0, min(active_idx, total - 1))
            st.session_state["active_source_index"] = active_idx
            
            # Navigator controls (1/5, prev/next buttons)
            nav_col1, nav_col2, nav_col3 = st.columns([1, 2, 1])
            
            with nav_col1:
                # Previous button
                if st.button("◀", key="prev_source", use_container_width=True, disabled=(active_idx == 0)):
                    st.session_state["active_source_index"] = active_idx - 1
                    st.rerun()
                    
            with nav_col2:
                # Current state label
                st.markdown(
                    f"<div style='text-align: center; font-weight: bold; font-size: 16px; margin-top: 4px;'>"
                    f"Question {active_idx + 1} / {total}"
                    f"</div>",
                    unsafe_allow_html=True
                )
                
            with nav_col3:
                # Next button
                if st.button("▶", key="next_source", use_container_width=True, disabled=(active_idx == total - 1)):
                    st.session_state["active_source_index"] = active_idx + 1
                    st.rerun()
                    
            st.markdown("---")
            
            # Render sources for active index
            active_sources = history[active_idx]["sources"]
            
            # Filter sources to only include those with match percentage >= 10%
            filtered_sources = []
            for src in active_sources:
                score = src.get("score", 1.0)
                match_percentage = max(0, min(100, int((1.0 - score) * 100)))
                if match_percentage >= 10:
                    filtered_sources.append(src)
                    
            if not filtered_sources:
                st.info("There are no sources")
            else:
                # Iterate and render source cards
                for src in filtered_sources:
                    render_source_card(
                        source_name=src["source"],
                        page_num=src["page"],
                        snippet=src["content"],
                        score=src["score"],
                        chunk_id=src.get("chunk_id", "Unknown"),
                        vector_score=src.get("vector_score", 1.0),
                        bm25_score=src.get("bm25_score", 0.0),
                        rrf_score=src.get("rrf_score", 0.0)
                    )

