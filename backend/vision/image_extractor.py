import os
import fitz  # PyMuPDF
import unicodedata

def clean_control_chars(text: str) -> str:
    """Removes Unicode control, format, and private-use characters except whitespace."""
    if not text:
        return ""
    return "".join(
        ch for ch in text
        if not unicodedata.category(ch).startswith("C") or ch in ("\n", "\r", "\t")
    )

def extract_images_from_pdf(pdf_path, output_folder, document_id, max_images_per_page=5):
    """
    Extracts every image from each page of the PDF.
    Saves image binaries locally in output_folder.
    Returns a list of image metadata dicts containing bounding boxes and paths.
    """
    doc = fitz.open(pdf_path)
    filename = os.path.basename(pdf_path)
    os.makedirs(output_folder, exist_ok=True)
    
    extracted_images = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        image_list = page.get_images(full=True)
        
        # Limit max images per page to avoid overloading model
        for img_idx, img_info in enumerate(image_list[:max_images_per_page]):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                
                # Create a unique filename on disk
                img_filename = f"{os.path.splitext(filename)[0]}_p{page_num+1}_img{img_idx}.{image_ext}"
                img_path = os.path.join(output_folder, img_filename)
                
                # Save physical image
                with open(img_path, "wb") as f:
                    f.write(image_bytes)
                    
                # Bounding box extraction if available
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    r = rects[0]
                    bbox = [float(r.x0), float(r.y0), float(r.x1), float(r.y1)]
                    
                extracted_images.append({
                    "document_id": document_id,
                    "page_number": page_num + 1,
                    "image_number": img_idx + 1,
                    "image_path": img_path,
                    "bounding_box": bbox,
                    "filename": img_filename
                })
            except Exception as e:
                print(f"[WARNING] Failed to extract image xref {xref} from page {page_num+1}: {e}")
                
    doc.close()
    return extracted_images
