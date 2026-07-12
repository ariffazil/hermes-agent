---
name: ocr
description: Use when extracting text from images or scanned documents -- receipts, screenshots, whiteboard photos, or any image-only text. Handles TIFF, PNG, JPEG, BMP, WEBP, PDF pages via pytesseract + Tesseract. Not for born-digital PDFs (use text extraction instead).
version: 1.0.0
author: Hermes Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [ocr, text-extraction, image-processing, tesseract, receipts, screenshots]
    related_skills: [note-taking, document-processing]
---

# OCR — Text Extraction from Images

## Overview

Extract text from images using Tesseract OCR via pytesseract. Works on any image format Tesseract supports (PNG, JPEG, TIFF, BMP, WEBP) and scanned PDFs. Returns per-word bounding boxes and confidence scores when you need granular control.

**Tesseract is already installed** at `/usr/bin/tesseract` and pytesseract is in the venv — no additional setup needed.

## When to Use

- Receipt or invoice text extraction
- Screenshot text extraction (e.g. tweet, document, UI)
- Whiteboard photo text capture
- Scanned document OCR (PDF or image)
- Any image-only text that needs to become searchable

## When NOT to Use

- Born-digital PDFs with embedded text — extract directly without OCR
- Large-scale server OCR at volume — use native Tesseract binary or cloud services
- Real-time video OCR — too slow, wrong tool class
- Non-English scripts without tessdata installed — check `tesseract --list-langs` first

## Quick Start

### Basic text extraction

```python
from hermes_agent import tool
from pytesseract import image_to_string
from PIL import Image

@tool
def ocr_extract(image_path: str) -> str:
    """Extract text from an image file using Tesseract OCR."""
    img = Image.open(image_path)
    text = image_to_string(img)
    return text.strip()
```

### With bounding boxes and confidence

```python
import pytesseract

# Get detailed data
data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

# Each word with bbox + confidence
for i, word in enumerate(data['text']):
    if word.strip():
        print(f"{word} | conf:{data['conf'][i]} | bbox:({data['left'][i]},{data['top'][i]})")
```

### Multi-language OCR

```python
# Set language(s) — check available: tesseract --list-langs
text = pytesseract.image_to_string(img, lang='eng+msa')  # English + Malay
```

### PDF / scanned document

```python
import pytesseract
from PDFDocument import PDFDocument  # or PyMuPDF

# For scanned PDFs, render each page as image then OCR
doc = PDFDocument.open("scanned.pdf")
for page_num in range(len(doc)):
    page = doc[page_num]
    pix = page.get_pixmap(matrix=fit.Scale(300/72))  # 300 DPI
    img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
    text = pytesseract.image_to_string(img, lang='eng')
    print(f"--- Page {page_num + 1} ---")
    print(text)
```

## Tesseract Language Codes

| Code  | Language         |
|-------|-----------------|
| eng   | English         |
| msa   | Malay           |
| chi_sim | Simplified Chinese |
| chi_tra | Traditional Chinese |
| ara   | Arabic          |
| hin   | Hindi           |
| jpn   | Japanese        |
| kor   | Korean          |
| tha   | Thai            |

Check all installed: `tesseract --list-langs`

## DPI and Image Quality

Tesseract expects **300 DPI minimum** for good results. Photos from phones are usually fine. Screenshots at native resolution are usually fine.

For low-DPI images:
```python
# Upscale before OCR
img = Image.open(image_path)
w, h = img.size
img = img.resize((w * 2, h * 2), Image.LANCZOS)
text = pytesseract.image_to_string(img)
```

## Preprocessing Tips

| Problem | Fix |
|---------|-----|
| Dark image | `ImageOps.autocontrast(img)` or convert to grayscale first |
| Skewed text | `pytesseract.image_to_osd()` to detect angle, then rotate |
| Noise | `ImageFilter.MedianFilter(size=3)` before OCR |
| Large image | Resize to max 4000px on longest side before OCR |

## Common Pitfalls

1. **Wrong language selected** — if text isn't in English, specify `lang='...'` or get garbled output
2. **DPI too low** — screenshots at 96 DPI produce poor results; scale up 2-3x first
3. **Born-digital PDF** — OCR on text PDFs gives garbage; extract text directly instead
4. **Installed tessdata** — run `tesseract --list-langs` to confirm the language pack exists before using it
5. **PDF page as image** — PyMuPDF/PIL render at 300+ DPI for best results

## Verification Checklist

- [ ] Image opens and is readable by PIL (`Image.open` succeeds)
- [ ] `tesseract --list-langs` confirms language pack is installed
- [ ] Text output is non-empty for a test image
- [ ] Confidence scores are returned when using `image_to_data`
- [ ] For PDFs: page rendered at ≥300 DPI before OCR
