import os
import cv2
import json
import time
import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count
from tqdm import tqdm


def convert_pdf_to_images(pdf_path, dpi=150):
    """Converts PDF pages to PIL images using PyMuPDF."""
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def _process_single_page(args):
    """Worker function to process one page with YOLO and OCR."""
    page_number, pil_img, yolo_model_path = args
    start_time = time.time()

    # Initialize YOLO + OCR inside subprocess
    model = YOLO(yolo_model_path)
    reader = easyocr.Reader(['en'], gpu=False)

    # Convert PIL to OpenCV
    cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    result = model(cv_img)[0]

    page_results = []
    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        label = model.names[cls_id]

        if label in ["Title", "Section-header"]:
            bbox = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, bbox)
            cropped = cv_img[y1:y2, x1:x2]
            ocr_result = reader.readtext(cropped)
            text = ' '.join([res[1] for res in ocr_result]).strip()

            page_results.append({
                "class": label,
                "page_number": page_number,
                "bounding_box": {
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                },
                "content": text
            })

    end_time = time.time()
    print(f"✅ Page {page_number} processed in {end_time - start_time:.2f} seconds.")
    return page_results


def process_pdf_with_yolo(pdf_path, yolo_model_path, use_parallel=True):
    """Main pipeline that processes all pages of a PDF using YOLO + OCR."""
    print(f"📄 Converting PDF to images: {pdf_path}")
    images = convert_pdf_to_images(pdf_path)
    args_list = [(i + 1, img, yolo_model_path) for i, img in enumerate(images)]

    results = []
    if use_parallel:
        print(f"🚀 Running inference with {min(cpu_count(), len(images))} parallel workers (CPU only)...")
        with Pool(processes=min(cpu_count(), len(images))) as pool:
            with tqdm(total=len(images), desc="Processing pages", ncols=80) as pbar:
                for page_result in pool.imap_unordered(_process_single_page, args_list):
                    results.extend(page_result)
                    pbar.update(1)
    else:
        print("⚙️ Running sequentially...")
        for args in tqdm(args_list, desc="Processing pages", ncols=80):
            result = _process_single_page(args)
            results.extend(result)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PDF Title and Section Header Extractor using YOLO + EasyOCR")
    parser.add_argument("pdf_path", type=str, help="Path to input PDF")
    parser.add_argument("yolo_model_path", type=str, help="Path to trained YOLO model (e.g., model.pt)")
    parser.add_argument("--output_json", type=str, default="result.json", help="Path to save output JSON")
    parser.add_argument("--no_parallel", action="store_true", help="Disable multiprocessing (for debug)")
    total_start_time = time.time()
    args = parser.parse_args()

    output = process_pdf_with_yolo(
        pdf_path=args.pdf_path,
        yolo_model_path=args.yolo_model_path,
        use_parallel=not args.no_parallel
    )

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    total_end_time = time.time()
    total_duration = total_end_time - total_start_time
    print(f"\n🎉 Output saved to: {args.output_json}")
    print(f"⏱️ Total processing time: {total_duration:.2f} seconds.")
