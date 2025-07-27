import os
import json
import cv2
import pytesseract
# from pdf2image import convert_from_path
import fitz  # PyMuPDF
from PIL import Image

from ultralytics import YOLO
from PIL import Image
import easyocr
# Optional: Set tesseract path if not in environment PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def convert_pdf_to_images(pdf_path, dpi=300):
#     return convert_from_path(pdf_path, dpi=dpi)

def convert_pdf_to_images(pdf_path, dpi=150):
    doc = fitz.open(pdf_path)
    images = []
    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


# def extract_text_from_bbox(image, bbox):
#     x1, y1, x2, y2 = map(int, bbox)
#     cropped = image[y1:y2, x1:x2]
#     gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
#     text = pytesseract.image_to_string(gray)
#     return text.strip()
reader = easyocr.Reader(['en'], gpu=False)  # Load once globally

def extract_text_from_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, bbox)
    cropped = image[y1:y2, x1:x2]
    results = reader.readtext(cropped)
    text = ' '.join([res[1] for res in results])
    return text.strip()

def run_yolo_on_image(model, image):
    results = model(image)
    return results[0]

def process_pdf_with_yolo(pdf_path, yolo_model_path):
    images = convert_pdf_to_images(pdf_path)
    model = YOLO(yolo_model_path)

    results_json = []
    for page_number, pil_img in enumerate(images, start=1):
        # Convert to OpenCV image
        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        result = run_yolo_on_image(model, cv_img)

        for box in result.boxes:
            cls_id = int(box.cls[0].item())
            label = model.names[cls_id]
            if label in ["Title", "Section-header"]:
                bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                text = extract_text_from_bbox(cv_img, bbox)
                results_json.append({
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

    return results_json

if __name__ == "__main__":
    import argparse
    import numpy as np

    parser = argparse.ArgumentParser(description="Extract Titles and Section Headers from a PDF using YOLO")
    parser.add_argument("pdf_path", type=str,default="sample.pdf" , help="Path to input PDF")
    parser.add_argument("yolo_model_path", type=str,default="model.pt" , help="Path to trained YOLO model (e.g., .pt file)")
    parser.add_argument("--output_json", type=str, default="output_2.json", help="Path to save JSON output")
    args = parser.parse_args()

    output = process_pdf_with_yolo(args.pdf_path, args.yolo_model_path)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print(f"Saved output JSON to {args.output_json}")
