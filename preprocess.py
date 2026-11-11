"""
preprocess.py
Cleans a prescription photo so the OCR engine can read it more easily.

Steps: grayscale -> enlarge if small -> fix uneven lighting -> remove noise
       -> black & white (threshold) -> straighten (deskew)
"""
import cv2
import numpy as np


def load_image(file_bytes):
    """Turn uploaded file bytes (from Streamlit or open(...).read()) into an OpenCV image."""
    array = np.frombuffer(file_bytes, dtype=np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)


def to_gray(image):
    """Colour -> grayscale. OCR only needs brightness, not colour."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def deskew(binary):
    """Rotate the image so the text lines are horizontal."""
    # Coordinates of all dark (text) pixels.
    coords = np.column_stack(np.where(binary < 128)).astype(np.float32)
    if len(coords) < 50:
        return binary
    # The smallest rectangle around the text tells us how tilted it is.
    angle = cv2.minAreaRect(coords[:, ::-1])[-1]
    if angle > 45:
        angle -= 90
    if abs(angle) < 0.3 or abs(angle) > 15:  # too small to matter, or unreliable
        return binary
    h, w = binary.shape
    matrix = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(binary, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def preprocess(image):
    """Full cleaning pipeline. Input: OpenCV image. Output: clean black & white image."""
    gray = to_gray(image)

    # 1. Small photos are hard to read, so enlarge them.
    if gray.shape[1] < 1200:
        scale = 1200 / gray.shape[1]
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Fix shadows: divide the image by a blurred copy of itself (its "background").
    background = cv2.GaussianBlur(gray, (0, 0), sigmaX=25)
    even = cv2.divide(gray, background, scale=255)

    # 3. Remove grain/noise while keeping letter edges sharp.
    denoised = cv2.fastNlMeansDenoising(even, h=15)

    # 4. Black text on a white page. Otsu picks the best cut-off automatically.
    _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 5. Straighten tilted photos.
    return deskew(binary)


if __name__ == "__main__":
    # Quick test: python preprocess.py  -> saves results/preprocessed_example.png
    import os
    os.makedirs("results", exist_ok=True)
    original = cv2.imread("data/samples/rx_20.png")
    clean = preprocess(original)
    cv2.imwrite("results/preprocessed_example.png", clean)
    print("Original size:", original.shape, "-> cleaned size:", clean.shape)
    print("Saved results/preprocessed_example.png")
