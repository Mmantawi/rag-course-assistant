import os
import sys
import unicodedata
import fitz  # PyMuPDF

def clean_control_chars(text: str) -> str:
    """Removes Unicode control, format, and private-use characters except whitespace."""
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\r", "\t")
    )

def extract_text_from_pdf(file_path):
    """Extracts all plain text from the specified PDF file page-by-page."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
        
    doc = fitz.open(file_path)
    extracted_pages = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_text = page.get_text()
        # Clean control characters before storing
        cleaned_text = clean_control_chars(page_text)
        extracted_pages.append(f"--- Page {page_num + 1} ---\n{cleaned_text.strip()}")
        
    doc.close()
    return "\n\n".join(extracted_pages)

'''def process_all_pdfs(input_dir, output_dir):
    """Processes all PDFs in input_dir and saves extracted text to output_dir."""
    if not os.path.exists(input_dir):
        print(f"[ERROR] Input directory '{input_dir}' does not exist.")
        sys.exit(1)
        
    os.makedirs(output_dir, exist_ok=True)
    
    pdfs = [f for f in os.listdir(input_dir) if f.endswith(".pdf")]
    if not pdfs:
        print(f"No PDF files found in '{input_dir}'.")
        return
        
    print(f"Found {len(pdfs)} PDF(s) to process.\n")
    
    for pdf_name in pdfs:
        input_path = os.path.join(input_dir, pdf_name)
        
        # Determine the output txt file name
        txt_name = os.path.splitext(pdf_name)[0] + ".txt"
        output_path = os.path.join(output_dir, txt_name)
        
        print(f"Processing '{pdf_name}'...")
        try:
            text_content = extract_text_from_pdf(input_path)
            
            # Save extracted text to a .txt file in UTF-8
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(text_content)
                
            print(f"  -> Saved text to '{output_path}'")
        except Exception as e:
            print(f"  -> [ERROR] Failed to process '{pdf_name}': {e}", file=sys.stderr)
'''
if __name__ == "__main__":
    # Reconfigure stdout to support UTF-8 characters on Windows console
    sys.stdout.reconfigure(encoding='utf-8')
    
    input_folder = "data/pdfs"
    output_folder = "data/processed"
    
    #process_all_pdfs(input_folder, output_folder)
    print("\nAll PDF processing completed!")
