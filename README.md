# RAG Course Assistant

A multi-model Retrieval-Augmented Generation (RAG) assistant for PDF course documents and lecture slides. The application features a modular FastAPI backend server that handles document extraction, ingestion, and streaming chat generation, paired with a modern Streamlit frontend UI for interactive, real-time querying.

It is designed to support both local execution (using Ollama) and powerful commercial model APIs (Google Gemini and Groq), giving you absolute flexibility over your computing resources and privacy.

---

## 🌟 Key Features

- **Hybrid Retrieval System**: Combines dense semantic vector search (using Chroma DB) with sparse keyword search (using BM25) and fuses the results using **Reciprocal Rank Fusion (RRF)** for optimal document matching.
- **Dynamic Query Analysis**: Automatically analyzes user questions to determine if they need a database query (e.g. retrieval of slide content) or a direct conversational response, speeding up interactions.
- **Multi-Model Orchestration**: Supports local models running via **Ollama** (e.g. `llama3.2` and `mxbai-embed-large`) or external APIs such as **Google Gemini** (`gemini-1.5-flash`) and **Groq** (`llama-3.3-70b-versatile`).
- **Real-Time Streaming**: Implements Server-Sent Events (SSE) on the backend to stream responses token-by-token directly to the frontend for a fast, responsive user experience.
- **Polished Streamlit Interface**:
  - Interactive Dark/Light mode theme switches.
  - Document sidebar listing indexed PDFs and total disk usage metrics.
  - Expandable, detailed **Source Reference Cards** displaying exact page numbers, snippet text, and debug metrics (Vector Distance, BM25 Score, and RRF ranking).

---

## 📂 Repository Structure

```text
rag-chatbot/
├── backend/                  # FastAPI API Service & RAG Pipeline
│   ├── api.py                # FastAPI routes, file upload handles, and SSE streaming
│   ├── config.py             # Configuration parsing from .env
│   ├── pipeline.py           # RAG orchestrator (retrieval + prompt + LLM)
│   ├── ingestion.py          # PDF loader, chunking, and database injection 
│   ├── retriever.py          # Hybrid Retriever (Chroma Vector + BM25 pickle + RRF)
│   ├── query_analysis.py     # Classifies query intent and extracts search terms
│   ├── chunker.py            # Text segmentation strategies
│   ├── pdf_loader.py         # PyMuPDF-based text and layout extractor
│   ├── embedding.py          # Vector embedding model hooks (Ollama)
│   ├── generator.py          # LLM response generation wrapper (Ollama/Gemini/Groq)
│   ├── bm25.py               # BM25 indexing and query utilities
│   └── prompts.py            # System instructions and prompt templates
├── frontend/                 # Interactive Web UI
│   ├── streamlit_app.py      # App layout, health loops, chat storage, and API stream hooks
│   ├── components.py         # Styled HTML components for source reference cards
│   └── styles.css            # Custom CSS styles (themes, cards, layout refinements)
├── data/                     # Local Storage Folder
│   ├── pdfs/                 # Raw PDF slides (ignored in Git except .gitkeep)
│   └── processed/            # Extracted plain text files (ignored in Git except .gitkeep)
├── vector_db/                # Chroma database store (ignored in Git except .gitkeep)
├── sandbox/                  # Script playgrounds for isolated component testing
├── main.py                   # FastAPI backend startup script
├── run.bat                   # Batch script to launch backend and frontend simultaneously
├── requirements.txt          # Python project dependencies
├── .env.example              # Template configuration file
└── README.md                 # Project documentation
```

---

## 🛠️ Prerequisites

- **Python**: version `3.9` to `3.11` recommended.
- **Ollama** (optional, for local model support): [Download & Install Ollama](https://ollama.com/).
- **API Keys** (optional, if using cloud providers):
  - Google Gemini API Key from [Google AI Studio](https://aistudio.google.com/).
  - Groq API Key from [Groq Console](https://console.groq.com/).

---

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Set Up a Virtual Environment
Using Python's built-in `venv`:
```bash
python -m venv .venv

# On Windows (Command Prompt)
.venv\Scripts\activate.bat

# On Windows (PowerShell)
.venv\Scripts\Activate.ps1

# On macOS/Linux
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to a new `.env` file:
```bash
copy .env.example .env
```
Open `.env` and fill in your settings. If you want to use cloud APIs, add your `GEMINI_API_KEY` or `GROQ_API_KEY`. If you are running locally, make sure Ollama is active.

### 5. Install Local Ollama Models (If running locally)
Ensure Ollama is running in the background, and download the models:
```bash
# Pull LLM model
ollama pull llama3.2

# Pull Embedding model
ollama pull mxbai-embed-large
```

---

## 🏃 Running the Application

### Option A: Windows Quick Start (Launch both together)
Double-click `run.bat` or run it from command line:
```cmd
run.bat
```
This launches the FastAPI API in one console tab and the Streamlit app in another.

### Option B: Manual Launch
1. **Start Backend Server**:
   ```bash
   python main.py
   ```
   The backend API will run at `http://127.0.0.1:8000` with interactive API docs available at `http://127.0.0.1:8000/docs`.

2. **Start Frontend Streamlit UI**:
   ```bash
   streamlit run frontend/streamlit_app.py
   ```
   The UI will open automatically in your browser at `http://localhost:8501`.

---

## 🔌 API Endpoints Summary

The FastAPI backend exposes the following primary endpoints:

- `GET /health`: Returns the health status and current LLM model configuration.
- `GET /documents`: Lists all PDF filenames currently stored and indexed in the database.
- `POST /upload`: Uploads one or more PDF files, extracts the text, segments chunks, and builds the Chroma and BM25 indexes.
- `GET /pdf/{filename}`: Serves a raw PDF from local storage to display in the frontend viewer tab.
- `POST /chat/stream`: Streams RAG response content using Server-Sent Events (SSE).
