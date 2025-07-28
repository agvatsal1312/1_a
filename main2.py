import os
import json
import cv2
import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from ultralytics import YOLO
from rapidocr_onnxruntime import RapidOCR
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import gc
from typing import List, Dict, Any, Optional

# Initialize OCR globally
reader = RapidOCR()
# Model will be loaded in process_pdf_with_yolo function to avoid global loading issues

def render_page(pdf_path, page_number, dpi):
    """Render a single PDF page to image with memory cleanup"""
    try:
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img
    except Exception as e:
        print(f"Error rendering page {page_number}: {e}")
        return None

def convert_pdf_to_images(pdf_path, dpi=100, max_workers=None):
    """Convert PDF to images with parallel processing and memory management"""
    try:
        doc = fitz.open(pdf_path)
        total_pages = len(doc)
        doc.close()

        # Limit to 50 pages as per challenge requirements
        if total_pages > 50:
            print(f"Warning: PDF has {total_pages} pages, limiting to first 50 pages")
            total_pages = 50

        # Use ThreadPoolExecutor for I/O bound operations
        with ThreadPoolExecutor(max_workers=max_workers or min(total_pages, 4)) as executor:
            futures = [executor.submit(render_page, pdf_path, i, dpi) for i in range(total_pages)]
            images = []
            for future in futures:
                img = future.result()
                if img is not None:
                    images.append(img)
        
        return images
    except Exception as e:
        print(f"Error converting PDF to images: {e}")
        return []

def extract_text_from_bbox(image, bbox, confidence_threshold=0.5):
    """Extract text from bounding box with early filtering"""
    try:
        x1, y1, x2, y2 = map(int, bbox)
        
        # Validate bbox coordinates
        if x1 >= x2 or y1 >= y2 or x1 < 0 or y1 < 0:
            return ""
        
        # Ensure bbox is within image bounds
        h, w = image.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        if x1 >= x2 or y1 >= y2:
            return ""
        
        cropped = image[y1:y2, x1:x2]
        
        # Skip very small regions
        if cropped.shape[0] < 10 or cropped.shape[1] < 10:
            return ""
        
        results = reader(cropped)
        if results and results[0]:
            text = ' '.join([item[1] for item in results[0] if item[2] > confidence_threshold])
        else:
            text = ""
        return text.strip()
    except Exception as e:
        print(f"Error extracting text from bbox: {e}")
        return ""

def run_yolo_on_image(model, image, confidence_threshold=0.4):
    """Run YOLO detection with confidence filtering"""
    try:
        results = model(image, conf=confidence_threshold)
        return results[0]
    except Exception as e:
        print(f"Error running YOLO on image: {e}")
        return None

def assign_heading_levels(detections: List[Dict], page_width: int) -> List[Dict]:
    """Assign heading levels with optimized logic"""
    if not detections:
        return []

    # Pre-calculate max size once
    max_size = max(d["size"] for d in detections) if detections else 1

    for i, det in enumerate(detections):
        size_ratio = det["size"] / max_size
        x1, y1, x2, y2 = det["bbox"]
        center_x = (x1 + x2) / 2
        gap_above = y1 - detections[i - 1]["bbox"][3] if i > 0 else 9999
        text = det["content"]
        
        # Skip empty text early
        if not text.strip():
            continue
            
        level = None

        # Optimized heading level assignment
        # Numbering heuristic (most reliable)
        if re.match(r'^\d+\.\d+\.\d+', text):
            level = "H3"
        elif re.match(r'^\d+\.\d+', text):
            level = "H2"
        elif re.match(r'^\d+\.', text):
            level = "H1"
        # Font size-based heuristic
        elif size_ratio > 0.85:
            level = "H1"
        elif size_ratio > 0.65:
            level = "H2"
        else:
            level = "H3"

        # Spacing heuristic (override if significant gap)
        if gap_above > 80:
            level = "H1"

        det["heading_level"] = level

    return detections

def process_pdf_with_yolo(pdf_path: str, yolo_model_path: str) -> Dict[str, Any]:
    """Main processing function with optimizations"""
    try:
        # Load model (moved from global to avoid Docker issues)
        model = YOLO(yolo_model_path)
        
        # Convert PDF to images
        images = convert_pdf_to_images(pdf_path)
        if not images:
            return {"title": "", "outline": []}

        outline = []
        title = ""

        for page_number, pil_img in enumerate(images, start=1):
            # Convert PIL to OpenCV format
            cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            page_width = cv_img.shape[1]
            
            # Run YOLO detection
            result = run_yolo_on_image(model, cv_img)
            if result is None:
                continue

            # Extract detections with early filtering
            detections = []
            for box in result.boxes:
                cls_id = int(box.cls[0].item())
                label = model.names[cls_id]
                conf = box.conf[0]

                # Check for both Section-header and Title classes
                if label in ["Section-header", "Title"] and conf > 0.4:
                    bbox = box.xyxy[0].tolist()
                    text = extract_text_from_bbox(cv_img, bbox)
                    
                    # Skip empty text early
                    if not text.strip():
                        continue
                        
                    size = bbox[3] - bbox[1]

                    detections.append({
                        "page_number": page_number,
                        "bbox": bbox,
                        "content": text,
                        "size": size,
                        "class": label  # Store the class for title detection
                    })

            # Sort and assign heading levels
            if detections:
                detections.sort(key=lambda x: x["bbox"][1])
                detections = assign_heading_levels(detections, page_width)

                # Extract title from first page (look for Title class specifically)
                if page_number == 1 and detections:
                    # Try to find a Title class detection first
                    title_detection = None
                    for det in detections:
                        if det.get("class") == "Title":
                            title_detection = det
                            break
                    
                    # If no Title class found, use first detection as title
                    if title_detection:
                        title = title_detection["content"]
                        detections.remove(title_detection)
                    else:
                        title = detections[0]["content"]
                        detections.pop(0)

                # Add to outline (already filtered for empty text)
                for det in detections:
                    if det["content"].strip():
                        outline.append({
                            "level": det["heading_level"],
                            "text": det["content"],
                            "page": det["page_number"]
                        })

            # Memory cleanup
            del cv_img
            gc.collect()

        return {
            "title": title,
            "outline": outline
        }
        
    except Exception as e:
        print(f"Error processing PDF: {e}")
        return {"title": "", "outline": []}
    finally:
        # Clean up model to free memory
        if 'model' in locals():
            del model
            gc.collect()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract document outline with H1/H2/H3 headings using YOLO + RapidOCR")
    parser.add_argument("--pdf_path", type=str, default="sample.pdf", help="Path to input PDF")
    parser.add_argument("--yolo_model_path", type=str, default="model.pt", help="Path to trained YOLO model")
    parser.add_argument("--output_json", type=str, default="outline2.json", help="Path to save JSON output")
    parser.add_argument("--dpi", type=int, default=100, help="DPI for PDF rendering")
    parser.add_argument("--max_workers", type=int, default=None, help="Maximum number of worker processes")
    args = parser.parse_args()

    output = process_pdf_with_yolo(args.pdf_path, args.yolo_model_path)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"✅ Outline JSON saved to {args.output_json}")
    print(f"📊 Extracted {len(output['outline'])} headings")
    if output['title']:
        print(f"📄 Document title: {output['title']}")