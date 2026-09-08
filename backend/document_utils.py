import io
import cv2
import numpy as np

def robust_binarize(img_gray: np.ndarray) -> np.ndarray:
    """Binarizes gray signature images cleanly handling lighting variations."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(img_gray)
    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def encode_png(img_mat: np.ndarray) -> bytes:
    """Encodes a numpy image matrix to raw PNG byte stream."""
    success, buffer = cv2.imencode(".png", img_mat)
    if not success:
        raise ValueError("Failed to encode image to PNG.")
    return buffer.tobytes()

def auto_extract_signature(img_bytes: bytes, top_k: int = 5):
    """
    Extracts candidate signature regions from document captures using
    contour aspect ratios and density filtering.
    """
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode document image.")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Contrast enhance & binarize
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    norm = clahe.apply(gray)
    binary = cv2.adaptiveThreshold(
        norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 12
    )

    # Dilate horizontally to connect signature cursive components
    rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3))
    dilated = cv2.dilate(binary, rect_kernel, iterations=2)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    min_area = (h * w) * 0.005  # At least 0.5% of total page
    max_area = (h * w) * 0.35   # At most 35% of total page

    for c in contours:
        x, y, cw, ch = cv2.boundingRect(c)
        area = cw * ch
        aspect = cw / max(1, ch)

        # Filter for typical signature bounding shapes (wider than tall)
        if min_area <= area <= max_area and 1.2 <= aspect <= 7.0:
            pad = 10
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + cw + pad)
            y2 = min(h, y + ch + pad)

            crop = gray[y1:y2, x1:x2]
            density = np.mean(crop < 200)

            candidates.append({
                "crop": crop,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": round(float(density * 100), 2)
            })

    candidates.sort(key=lambda item: item["score"], reverse=True)
    if not candidates:
        # Fallback to center crop if no distinct region passes heuristic
        cx1, cy1 = int(w * 0.2), int(h * 0.4)
        cx2, cy2 = int(w * 0.8), int(h * 0.8)
        candidates.append({
            "crop": gray[cy1:cy2, cx1:cx2],
            "bbox": [cx1, cy1, cx2 - cx1, cy2 - cy1],
            "score": 50.0
        })

    return gray, candidates[:top_k]