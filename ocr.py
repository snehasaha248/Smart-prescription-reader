"""
ocr.py
The image-to-text converter. Reads text from a prescription image.

Two text-recognition engines:
  "vision"    - Apple Vision (built into macOS). Reads HANDWRITING and phone photos well.
  "tesseract" - Tesseract OCR. Works on any computer, good for PRINTED text only.

Both return the same format, so the rest of the project does not care which one was used:
  {"text": "...", "lines": [{"text": "...", "words": [(word, confidence 0-100), ...],
                            "box": (x, y, width, height)}, ...]}
  The box is the line's position on the page as fractions (0-1), measured from the TOP-left corner.
"""
import shutil
import sys
from pathlib import Path

import cv2
import pytesseract

# Apple Vision only exists on macOS (needs the pyobjc-framework-Vision package).
try:
    import Vision
    from Foundation import NSData
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False

# Find the tesseract program: first next to this Python (our micromamba env), then on PATH.
_local = Path(sys.executable).parent / "tesseract"
pytesseract.pytesseract.tesseract_cmd = str(_local) if _local.exists() else (shutil.which("tesseract") or "tesseract")

# --psm 6 = "treat the image as one block of text" (works well for prescriptions)
TESSERACT_CONFIG = "--oem 3 --psm 6"


def available_engines():
    return (["vision"] if VISION_AVAILABLE else []) + ["tesseract"]


def read_text(image, engine="tesseract"):
    """Run OCR on an OpenCV image with the chosen engine."""
    if engine == "vision" and VISION_AVAILABLE:
        lines = _read_vision(image)
    else:
        lines = _read_tesseract(image)
    return {"text": "\n".join(line["text"] for line in lines), "lines": lines}


def _read_tesseract(image):
    data = pytesseract.image_to_data(image, config=TESSERACT_CONFIG,
                                     output_type=pytesseract.Output.DICT)
    height, width = image.shape[:2]
    lines = {}
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        # Tesseract numbers each block, paragraph and line; together they identify a line.
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append((word, float(data["conf"][i]), data["left"][i] / width,
                                          data["top"][i] / height, data["width"][i] / width,
                                          data["height"][i] / height))
    result = []
    for key in sorted(lines):
        words = lines[key]
        x = min(w[2] for w in words)
        y = min(w[3] for w in words)
        box = (x, y, max(w[2] + w[4] for w in words) - x, max(w[3] + w[5] for w in words) - y)
        result.append({"text": " ".join(w[0] for w in words),
                       "words": [(w[0], w[1]) for w in words], "box": box})
    return result


def _read_vision(image):
    # Vision takes image file bytes, so encode the OpenCV image as PNG first.
    png = cv2.imencode(".png", image)[1].tobytes()
    handler = Vision.VNImageRequestHandler.alloc().initWithData_options_(
        NSData.dataWithBytes_length_(png, len(png)), None)
    request = Vision.VNRecognizeTextRequest.alloc().init()
    request.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
    # Language correction would "fix" medicine names into English words, so it is switched off.
    request.setUsesLanguageCorrection_(False)
    request.setRecognitionLanguages_(["en-US"])  # stops messy letters being read as Cyrillic etc.
    handler.performRequests_error_([request], None)

    result = []
    for observation in request.results() or []:
        best = observation.topCandidates_(1)[0]
        text = str(best.string()).strip()
        if not text:
            continue
        b = observation.boundingBox()  # Vision measures from the BOTTOM-left, so flip y
        box = (b.origin.x, 1 - b.origin.y - b.size.height, b.size.width, b.size.height)
        confidence = float(best.confidence()) * 100  # Vision gives one score per line
        result.append({"text": text, "words": [(w, confidence) for w in text.split()], "box": box})
    # Top-to-bottom, then left-to-right.
    result.sort(key=lambda line: (round(line["box"][1] / 0.01), line["box"][0]))
    return _merge_pieces(result)


def _merge_pieces(lines):
    """
    Vision sometimes splits one written line into pieces ("Tab" | "Qutipin 200mg").
    Join pieces that sit on the same row and close together; notes far away stay separate.
    """
    merged = []
    for line in sorted(lines, key=lambda l: l["box"][0]):  # left to right
        x, y, w, h = line["box"]
        for other in merged:
            ox, oy, ow, oh = other["box"]
            overlap = min(y + h, oy + oh) - max(y, oy)  # shared height
            gap = x - (ox + ow)                          # space between the two pieces
            if overlap > 0.5 * min(h, oh) and -0.02 < gap < 0.08:
                other["text"] += " " + line["text"]
                other["words"] += line["words"]
                top, bottom = min(y, oy), max(y + h, oy + oh)
                other["box"] = (ox, top, x + w - ox, bottom - top)
                break
        else:
            merged.append(dict(line))
    merged.sort(key=lambda line: (round(line["box"][1] / 0.01), line["box"][0]))
    return merged


def text_to_result(text):
    """Turn typed/edited text back into OCR format (a person checked it, so confidence = 100)."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {"text": "\n".join(lines),
            "lines": [{"text": line, "words": [(w, 100.0) for w in line.split()],
                       "box": (0.0, i / max(len(lines), 1), 1.0, 1 / max(len(lines), 1))}
                      for i, line in enumerate(lines)]}


if __name__ == "__main__":
    # Quick test: python ocr.py [image path]
    path = sys.argv[1] if len(sys.argv) > 1 else "data/samples/rx_47.png"
    image = cv2.imread(path)
    for engine in available_engines():
        print(f"===== {engine} =====")
        print(read_text(image, engine)["text"])
