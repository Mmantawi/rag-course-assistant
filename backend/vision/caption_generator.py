import time
import traceback
from backend.vision.vision_service import VisionService

# Prompt instructing the VLM to produce a detailed technical/educational description of the slide figure
TECHNICAL_VISION_PROMPT = """Analyze the provided slide image and generate a detailed technical and educational description.
Focus on extracting the technical meaning, educational concepts, and structural data shown in the figure rather than its artistic appearance.

Your description should explicitly cover:
1. The type of image (e.g., flow chart, block diagram, graph, table, screenshot, diagram, photo).
2. Key entities, objects, or UI elements visible.
3. Readable texts, labels, abbreviations, headers, titles, or numbers.
4. Relationships, arrows, direction of flow, and dependencies between items.
5. Mathematical formulas, equations, or code blocks shown.
6. The core educational concept or technical message this graphic illustrates.

Keep the language concise, academic, and structured. Make sure it is optimized to match text queries in a search index."""

def generate_image_captions(extracted_images, provider=None, model=None):
    """
    Orchestrates description generation for a list of extracted image dicts.
    Applies Qwen2.5-VL or other VLMs via VisionService.
    Catches errors per image, ensuring the overall ingestion pipeline does not fail.
    Returns a list of image dicts populated with the generated "caption" string.
    """
    if not extracted_images:
        return []
        
    print(f"\n[Vision] Starting VLM description generation for {len(extracted_images)} image(s)...")
    print(f"         Provider: {provider or 'default'}, Model: {model or 'default'}")
    
    vision_service = VisionService(provider=provider, model=model)
    captioned_images = []
    
    for img in extracted_images:
        path = img["image_path"]
        page = img["page_number"]
        num = img["image_number"]
        
        print(f"         - Captioning page {page} image #{num} ({os.path.basename(path)})...")
        start_time = time.time()
        try:
            caption = vision_service.generate_caption(path, TECHNICAL_VISION_PROMPT)
            latency = time.time() - start_time
            print(f"           -> Success in {latency:.2f}s!")
            
            # Save caption and stats
            img_copy = img.copy()
            img_copy["caption"] = caption
            img_copy["latency"] = latency
            captioned_images.append(img_copy)
        except Exception as e:
            latency = time.time() - start_time
            print(f"           -> [WARNING] Captioning failed for {os.path.basename(path)} after {latency:.2f}s: {e}")
            traceback.print_exc()
            
            # Append empty caption so text ingestion still runs normally
            img_copy = img.copy()
            img_copy["caption"] = f"[Image extraction failed or uncaptioned. Error: {e}]"
            img_copy["latency"] = latency
            captioned_images.append(img_copy)
            
    return captioned_images

import os # Ensure os is imported inside the file scope
