#!/usr/bin/env python3
"""OCR from images using Tesseract (pytesseract). Lightweight, offline, 100+ languages.

Usage:
    python extract_tesseract.py image.png                    # Plain text
    python extract_tesseract.py image.png --json             # Per-word with confidence + bounding boxes
    python extract_tesseract.py image.png --lang msa+eng     # Malay + English
    python extract_tesseract.py image.png --psm 6            # Page segmentation mode (6=block, 3=auto)
    python extract_tesseract.py image.png --dpi 300          # Override DPI for better accuracy
    python extract_tesseract.py receipt.jpg --json --lang eng # Receipt OCR with word-level data

Requires: pip install pytesseract Pillow  (tesseract-ocr must be installed on system)
"""
import sys
import json
import argparse
from pathlib import Path

SUPPORTED = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp", ".gif"}

def ocr_text(path, lang="eng", psm=None, dpi=None):
    from PIL import Image
    import pytesseract
    img = Image.open(path)
    config = ""
    if psm:
        config += f"--psm {psm} "
    if dpi:
        config += f"--dpi {dpi} "
    text = pytesseract.image_to_string(img, lang=lang, config=config.strip())
    return text.strip()

def ocr_json(path, lang="eng", psm=None, dpi=None):
    from PIL import Image
    import pytesseract
    img = Image.open(path)
    config = ""
    if psm:
        config += f"--psm {psm} "
    if dpi:
        config += f"--dpi {dpi} "
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT, config=config.strip())
    words = []
    for i in range(len(data["text"])):
        text = data["text"][i].strip()
        conf = int(data["conf"][i])
        if text and conf > 0:
            words.append({
                "text": text,
                "confidence": conf,
                "bbox": {
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i]
                },
                "block": data["block_num"][i],
                "par": data["par_num"][i],
                "line": data["line_num"][i],
                "word": data["word_num"][i]
            })
    avg_conf = sum(w["confidence"] for w in words) / len(words) if words else 0
    return {
        "file": str(path),
        "lang": lang,
        "word_count": len(words),
        "avg_confidence": round(avg_conf, 1),
        "full_text": " ".join(w["text"] for w in words),
        "words": words
    }

def main():
    parser = argparse.ArgumentParser(description="OCR images with Tesseract")
    parser.add_argument("image", help="Path to image file")
    parser.add_argument("--json", action="store_true", help="Per-word JSON with confidence + bounding boxes")
    parser.add_argument("--lang", default="eng", help="Language(s), e.g. eng, msa+eng, chi_sim+eng (default: eng)")
    parser.add_argument("--psm", type=int, help="Page segmentation mode (3=auto, 6=block, 7=line, 8=word)")
    parser.add_argument("--dpi", type=int, help="Override DPI for better accuracy on low-res images")
    args = parser.parse_args()

    path = Path(args.image)
    if not path.exists():
        print(f"Error: {path} not found", file=sys.stderr)
        sys.exit(1)
    if path.suffix.lower() not in SUPPORTED:
        print(f"Warning: {path.suffix} may not be supported. Trying anyway.", file=sys.stderr)

    if args.json:
        result = ocr_json(path, lang=args.lang, psm=args.psm, dpi=args.dpi)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        text = ocr_text(path, lang=args.lang, psm=args.psm, dpi=args.dpi)
        print(text)

if __name__ == "__main__":
    main()
