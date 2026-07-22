import os
import streamlit as st

def inject_theme_css(theme):
    """
    Injects custom CSS tokens (theme variables) and the global stylesheet into the page header.
    Supports dynamic real-time client theme toggling.
    """
    css_path = os.path.join(os.path.dirname(__file__), "styles.css")
    
    if not os.path.exists(css_path):
        print(f"[WARNING] styles.css not found at '{css_path}'")
        return
        
    with open(css_path, "r", encoding="utf-8") as f:
        global_css = f.read()
        
    # Define color palettes for light and dark themes
    if theme == "light":
        theme_vars = """
        :root {
            --bg-primary: #f9fafb;
            --bg-secondary: #f3f4f6;
            --text-primary: #111827;
            --text-secondary: #4b5563;
            --accent: #4f46e5; /* indigo */
            --card-bg: #ffffff;
            --border-color: #e5e7eb;
            --chat-user-bg: #e0e7ff;
            --chat-bot-bg: #ffffff;
        }
        """
    else:  # Dark mode
        theme_vars = """
        :root {
            --bg-primary: #0f0f12;
            --bg-secondary: #17171d;
            --text-primary: #f3f4f6;
            --text-secondary: #9ca3af;
            --accent: #6366f1; /* violet */
            --card-bg: #1e1e24;
            --border-color: #2e2e38;
            --chat-user-bg: #312e81; /* deep indigo */
            --chat-bot-bg: #1e1e24;
        }
        """
        
    # Inject styling block
    custom_style = f"<style>{theme_vars}\n{global_css}</style>"
    st.markdown(custom_style, unsafe_allow_html=True)

def render_source_card(source_name, page_num, snippet, score, chunk_id="Unknown", vector_score=1.0, bm25_score=0.0, rrf_score=0.0):
    """Renders a styled HTML card for a RAG source chunk with a deep link to the PDF page."""
    # Clean up name: strip extensions (.pdf, .txt, etc.)
    clean_name = os.path.splitext(source_name)[0]
    if clean_name.endswith(('.pdf', '.txt')):
        clean_name = os.path.splitext(clean_name)[0]
        
    # Convert distance score to a visual match percentage
    match_percentage = max(0, min(100, int((1.0 - score) * 100)))
    
    # URL to backend file response server with PDF page hash fragment
    api_base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
    pdf_url = f"{api_base_url}/pdf/{source_name}#page={page_num}"
    
    card_html = f"""
    <div class="source-card">
        <div class="source-header">
            <span class="source-title">{clean_name}</span>
            <span class="source-score">{match_percentage}% Match</span>
        </div>
        <div class="source-meta">
            <span>Page {page_num}</span>
        </div>
        <div class="source-snippet">{snippet.strip()}</div>
        <div class="source-debug-info" style="display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; font-size: 11px; border-top: 1px dashed var(--border-color); padding-top: 8px; color: var(--text-primary);">
            <div style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                <strong>Vector Dist:</strong> {vector_score:.4f}
            </div>
            <div style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                <strong>BM25:</strong> {bm25_score:.2f}
            </div>
            <div style="background: var(--bg-secondary); padding: 4px 8px; border-radius: 4px; border: 1px solid var(--border-color);">
                <strong>RRF Score:</strong> {rrf_score:.4f}
            </div>
            <div style="width: 100%; color: var(--text-secondary); margin-top: 4px; font-family: monospace; word-break: break-all;">
                ID: {chunk_id}
            </div>
        </div>
        <a href="{pdf_url}" target="_blank" class="pdf-link-btn">
            Open Slide Page ↗
        </a>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def render_sidebar_doc(doc):
    """Renders a styled item for a document currently stored in the index, supporting status details."""
    if isinstance(doc, dict):
        filename = doc.get("filename", "Unknown")
        status = doc.get("status", "ready")
    else:
        filename = doc
        status = "ready"

    # Clean filename for display in sidebar
    clean_name = os.path.splitext(filename)[0]
    
    status_icon = "⏳" if status == "processing" else ("✅" if status == "ready" else ("❌" if status == "failed" else "📄"))
    
    doc_html = f"""
    <div class="sidebar-doc-item" style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 8px; padding: 6px 10px; background: var(--bg-secondary); border-radius: 6px; border: 1px solid var(--border-color);">
        <div style="display: flex; align-items: center; gap: 8px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">
            <span>📄</span>
            <span style="font-size: 13px; font-weight: 500; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="{filename}">{clean_name}</span>
        </div>
        <span style="font-size: 12px; margin-left: 6px;" title="Status: {status}">{status_icon}</span>
    </div>
    """
    st.markdown(doc_html, unsafe_allow_html=True)

