import os
import sys
import time
import uuid
import traceback
from dotenv import load_dotenv

# Load configurations
load_dotenv()

# Import config constants
from backend.config import (
    VISION_PROVIDER,
    VISION_MODEL,
    IMAGE_OUTPUT_FOLDER,
    ENABLE_MULTIMODAL,
    MAX_IMAGES_PER_PAGE,
    CHUNK_SIZE,
    CHUNK_OVERLAP
)

# Import DB and models
from backend.database import SessionLocal
from backend.models import UploadedDocument, UploadedImage
from backend.parent_store import ParentStore

# Import text extraction, chunking and embedding helpers
from backend.pdf_loader import extract_text_from_pdf
from backend.chunker import clean_control_chars, create_parent_documents, create_child_chunks
from backend.embedding import embed_and_store_documents

# Import vision modules
from backend.vision.image_extractor import extract_images_from_pdf
from backend.vision.caption_generator import generate_image_captions

def run_ingestion_pipeline(pdf_dir, processed_dir, db_path, embedding_model):
    """
    Orchestrates the modular Multimodal RAG ingestion pipeline:
    1. Scan PDF source directory.
    2. Retrieve Document ID and Status from database to skip already indexed files.
    3. Extract page-by-page text.
    4. If multimodal enabled, extract slide images and run VLM descriptions.
    5. Save image details and captions to PostgreSQL database.
    6. Merge figure captions with slide text on a page-by-page basis.
    7. Split pages into logical Parent Documents, saving them to ParentStore.
    8. Segment parents into Child Chunks.
    9. Embed child chunks, index in ChromaDB and fit the BM25 search index.
    """
    print("\n==================================================")
    print("=== Starting Multimodal Ingestion Pipeline ===")
    print("==================================================")
    
    start_pipeline_time = time.time()
    
    # 1. Validation and Directory setup
    if not os.path.exists(pdf_dir):
        print(f"[ERROR] PDF source directory '{pdf_dir}' does not exist.")
        return
        
    os.makedirs(processed_dir, exist_ok=True)
    os.makedirs(IMAGE_OUTPUT_FOLDER, exist_ok=True)
    
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith(".pdf")]
    if not pdf_files:
        print(f"No PDF files found in '{pdf_dir}'. Ingestion completed (empty).")
        return
        
    print(f"Found {len(pdf_files)} PDF(s) in source folder.")
    
    db = SessionLocal()
    parent_store = ParentStore(db_path)
    
    all_children_to_embed = []
    
    try:
        for pdf_name in pdf_files:
            pdf_path = os.path.join(pdf_dir, pdf_name)
            
            # Find DB record
            doc_record = db.query(UploadedDocument).filter(UploadedDocument.filename == pdf_name).first()
            if not doc_record:
                print(f"[INFO] Skipping file '{pdf_name}' - no database upload record found.")
                continue
                
            document_uuid = doc_record.id
            status = doc_record.status
            chat_id = doc_record.chat_id
            
            # If document is already fully loaded and not marked for processing/uploaded, skip
            if status == "ready":
                print(f"[INFO] Skipping file '{pdf_name}' - already indexed and ready.")
                continue
                
            print(f"\n>>> Processing Ingestion: '{pdf_name}' (ID: {document_uuid})")
            
            # Extract plain text page-by-page from PDF
            print("    [1/6] Extracting text...")
            try:
                raw_text_content = extract_text_from_pdf(pdf_path)
            except Exception as e:
                print(f"          [ERROR] Text extraction failed: {e}", file=sys.stderr)
                doc_record.status = "failed"
                db.commit()
                continue
                
            # Parse the extracted string into page-by-page mapping
            # This splits by '--- Page X ---'
            import re
            page_splits = re.split(r'--- Page (\d+) ---\n', raw_text_content)
            pages_dict = {}
            for i in range(1, len(page_splits), 2):
                p_num = int(page_splits[i])
                p_text = page_splits[i+1].strip()
                pages_dict[p_num] = clean_control_chars(p_text)
                
            # Multi-Modal Vision Processing
            total_images_count = 0
            vlm_total_latency = 0.0
            
            if ENABLE_MULTIMODAL:
                print("    [2/6] Extracting page images...")
                img_start = time.time()
                try:
                    # Extract page images
                    extracted_images = extract_images_from_pdf(
                        pdf_path=pdf_path,
                        output_folder=IMAGE_OUTPUT_FOLDER,
                        document_id=document_uuid,
                        max_images_per_page=MAX_IMAGES_PER_PAGE
                    )
                    total_images_count = len(extracted_images)
                    print(f"          Extracted {total_images_count} image(s) from slides.")
                    
                    if total_images_count > 0:
                        # Generate semantic captions via Qwen2.5-VL (or configured provider)
                        captioned_images = generate_image_captions(
                            extracted_images=extracted_images,
                            provider=VISION_PROVIDER,
                            model=VISION_MODEL
                        )
                        
                        # Save captions and details to PostgreSQL
                        print("    [3/6] Saving image metadata to database...")
                        for img in captioned_images:
                            vlm_total_latency += img.get("latency", 0.0)
                            
                            # Check if entry already exists in DB
                            existing_img = db.query(UploadedImage).filter(
                                UploadedImage.document_id == document_uuid,
                                UploadedImage.page_number == img["page_number"],
                                UploadedImage.image_number == img["image_number"]
                            ).first()
                            
                            if not existing_img:
                                db_img = UploadedImage(
                                    document_id=document_uuid,
                                    page_number=img["page_number"],
                                    image_number=img["image_number"],
                                    image_path=img["image_path"],
                                    caption=img["caption"]
                                )
                                db.add(db_img)
                            else:
                                existing_img.image_path = img["image_path"]
                                existing_img.caption = img["caption"]
                                
                        db.commit()
                        
                        # Merge captions back into page text
                        print("    [4/6] Merging image captions into slide page text...")
                        # Group captioned details by page number
                        from collections import defaultdict
                        captions_by_page = defaultdict(list)
                        for img in captioned_images:
                            captions_by_page[img["page_number"]].append(img["caption"])
                            
                        for page_number, captions in captions_by_page.items():
                            if page_number in pages_dict:
                                text_suffix = "\n\n--- Slide Figure Descriptions ---\n"
                                text_suffix += "\n\n".join([f"Figure Description: {cap}" for cap in captions])
                                pages_dict[page_number] += text_suffix
                except Exception as ve:
                    print(f"          [WARNING] Multimodal vision phase failed: {ve}", file=sys.stderr)
                    traceback.print_exc()
            else:
                print("    [2/6] Multimodal vision phase is disabled. Skipping image extraction.")
                
            # Log vision statistics
            if total_images_count > 0:
                print(f"          Vision Latency Metrics:")
                print(f"          - Extracted Images: {total_images_count}")
                print(f"          - Total VLM Latency: {vlm_total_latency:.2f} seconds")
                print(f"          - Avg VLM Latency per image: {(vlm_total_latency / total_images_count):.2f} seconds")
                
            # Create Parent Documents
            print("    [5/6] Creating parent documents & parsing headings...")
            parent_docs = create_parent_documents(
                pages_dict=pages_dict,
                document_id=document_uuid,
                filename=pdf_name,
                chat_id=chat_id
            )
            
            # Save parent texts to disk-backed ParentStore
            parent_store.add_parents(parent_docs)
            print(f"          Saved {len(parent_docs)} Parent Document(s) to ParentStore.")
            
            # Create Child Chunks
            child_chunks = create_child_chunks(
                parent_docs=parent_docs,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP
            )
            print(f"          Generated {len(child_chunks)} Child Chunk(s).")
            all_children_to_embed.extend(child_chunks)
            
            # Write page texts to txt files in data/processed/ for compatibility
            txt_name = os.path.splitext(pdf_name)[0] + ".txt"
            txt_path = os.path.join(processed_dir, txt_name)
            
            # Reconstruct string with headers
            merged_content = ""
            for p_num, p_text in pages_dict.items():
                merged_content += f"--- Page {p_num} ---\n{p_text}\n\n"
                
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(merged_content.strip())
                
    except Exception as e:
        print(f"[CRITICAL ERROR] Ingestion loop crashed: {e}", file=sys.stderr)
        traceback.print_exc()
    finally:
        db.close()
        
    # 10. Embed child chunks and save in ChromaDB / BM25
    if all_children_to_embed:
        print(f"\n[6/6] Generating embeddings for {len(all_children_to_embed)} Child Chunk(s)...")
        embed_start_time = time.time()
        try:
            embed_and_store_documents(
                documents=all_children_to_embed,
                db_path=db_path,
                embedding_model=embedding_model
            )
            embed_latency = time.time() - embed_start_time
            print(f"      -> Success! Chroma DB write & BM25 indexing took {embed_latency:.2f} seconds.")
        except Exception as e:
            print(f"      -> [ERROR] Failed to write embeddings to vector database: {e}", file=sys.stderr)
            traceback.print_exc()
            
    pipeline_latency = time.time() - start_pipeline_time
    print("\n==================================================")
    print(f"=== Ingestion Pipeline Completed in {pipeline_latency:.2f}s ===")
    print("==================================================\n")
