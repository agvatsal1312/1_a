# Adobe India Hackathon - Round1_a: Document Outline Extraction

## Approach

This solution uses a two-stage approach to extract structured document outlines from PDFs:

1. **Object Detection**: YOLO model trained to detect "Section-header" regions in PDF pages
2. **Text Extraction**: RapidOCR for accurate text extraction from detected regions
3. **Heading Classification**: Intelligent heuristics to assign H1/H2/H3 levels based on:
   - Font size ratios
   - Text positioning and alignment
   - Numbering patterns (1., 1.1., 1.1.1.)
   - Spacing between sections

## Models and Libraries Used

### Core Dependencies:
- **YOLO (Ultralytics)**: Object detection for finding heading regions
- **RapidOCR**: Fast and accurate text extraction from detected regions
- **PyMuPDF (fitz)**: PDF to image conversion
- **OpenCV**: Image processing and format conversion
- **PIL**: Image handling

### Model Details:
- **YOLO Model**: Custom trained model (`model.pt`) for detecting "Section-header" regions
- **Model Size**: ~10MB (well under 200MB limit)
- **Execution Time**: Optimized for ≤10 seconds on 50-page PDFs

## Features

✅ **AMD64 Architecture**: Compatible with specified platform requirements  
✅ **CPU-Only**: No GPU dependencies  
✅ **Offline Operation**: No internet calls required  
✅ **Multi-Page Support**: Handles PDFs up to 50 pages  
✅ **Robust Heading Detection**: Uses multiple heuristics for accurate level assignment  
✅ **Performance Optimized**: Memory-efficient processing with garbage collection  

## How to Build and Run

### Prerequisites
- Docker installed on your system
- AMD64 architecture support

### Build the Docker Image
```bash
docker build --platform linux/amd64 -t adobe-hackathon:round1a .
```

### Run the Solution
```bash
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  adobe-hackathon:round1a
```

### Expected Behavior
1. Place PDF files in the `input/` directory
2. Run the Docker container
3. Find corresponding JSON files in the `output/` directory
4. Each JSON contains:
   ```json
   {
     "title": "Document Title",
     "outline": [
       {"level": "H1", "text": "Introduction", "page": 1},
       {"level": "H2", "text": "Background", "page": 2},
       {"level": "H3", "text": "History", "page": 3}
     ]
   }
   ```

## Technical Implementation

### Key Optimizations:
- **Parallel Processing**: ThreadPoolExecutor for PDF page rendering
- **Memory Management**: Garbage collection and early filtering
- **Confidence Filtering**: Only high-confidence OCR results included
- **Early Exit**: Skip empty or invalid regions

### Heading Level Assignment Logic:
1. **Size-based**: Larger text → higher heading level
2. **Position-based**: Centered text → H1
3. **Numbering-based**: 1. → H1, 1.1. → H2, 1.1.1. → H3
4. **Spacing-based**: Large gaps → H1

## Constraints Compliance

- ✅ **Model Size**: ≤200MB (actual: ~10MB)
- ✅ **Execution Time**: ≤10 seconds for 50-page PDFs
- ✅ **Architecture**: AMD64 compatible
- ✅ **Network**: No internet access required
- ✅ **Runtime**: CPU-only, optimized for 8 CPUs, 16GB RAM

## Testing

The solution has been tested with:
- Various PDF formats and layouts
- Different heading styles and structures
- Multi-page documents
- Complex document layouts

## File Structure

```
/
├── Dockerfile              # Container definition
├── entrypoint.py           # Main execution script
├── main2.py               # Core processing logic
├── model.pt               # Trained YOLO model
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── input/                 # PDF input directory (mounted)
└── output/                # JSON output directory (mounted)
``` 