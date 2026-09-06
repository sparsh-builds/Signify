import cv2
import numpy as np
from document_utils import robust_binarize

def remove_noise(binary_img: np.ndarray) -> np.ndarray:
    """
    Vectorized morphological filtering to strip isolated salt-and-pepper noise
    in milliseconds rather than using a nested Python pixel loop.
    """
    # Invert to ink=255 for morphological processing
    inv = 255 - binary_img
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # Opening removes tiny isolated background specks
    opened = cv2.morphologyEx(inv, cv2.MORPH_OPEN, kernel)
    return 255 - opened

def centralize_signature(binary_img: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(255 - binary_img)
    if coords is None:
        return binary_img
    x, y, w, h = cv2.boundingRect(coords)
    sig_crop = binary_img[y:y+h, x:x+w]
    
    canvas = np.ones_like(binary_img) * 255
    start_y = (binary_img.shape[0] - h) // 2
    start_x = (binary_img.shape[1] - w) // 2
    canvas[start_y:start_y+h, start_x:start_x+w] = sig_crop
    return canvas

def rotate_to_horizontal(binary_img: np.ndarray) -> np.ndarray:
    coords = cv2.findNonZero(255 - binary_img)
    if coords is None or len(coords) < 2:
        return binary_img
    
    rect = cv2.minAreaRect(coords)
    angle = rect[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
        
    (h, w) = binary_img.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(binary_img, M, (w, h), borderValue=255)
    return rotated

def full_preprocessing(img_bytes: bytes, target_size=(64, 64), mode: str = "scan") -> tuple[np.ndarray, np.ndarray]:
    nparr = np.frombuffer(img_bytes, np.uint8)
    gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError("Could not decode image from provided bytes.")

    # 1. Downscale large smartphone photos for fast processing
    max_dim = 1200
    h_orig, w_orig = gray.shape
    if max(h_orig, w_orig) > max_dim:
        scale = max_dim / float(max(h_orig, w_orig))
        gray = cv2.resize(gray, (int(w_orig * scale), int(h_orig * scale)), interpolation=cv2.INTER_AREA)

    # 2. Auto-rotate sideways/vertical phone photos
    if gray.shape[0] > gray.shape[1]:
        gray = cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)

    # 3. Binarization
    if mode == "photo":
        binary = 255 - robust_binarize(gray)
    else:
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 4. Scale check & padding
    coords = cv2.findNonZero(255 - binary)
    if coords is not None:
        _, _, w, h = cv2.boundingRect(coords)
        if w > binary.shape[1] * 0.75 or h > binary.shape[0] * 0.75:
            scale = 1.0 / np.sqrt(2)
            binary = cv2.resize(binary, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            pad_y = (gray.shape[0] - binary.shape[0]) // 2
            pad_x = (gray.shape[1] - binary.shape[1]) // 2
            binary = cv2.copyMakeBorder(binary, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=255)
            
    cleaned = remove_noise(binary)
    centered = centralize_signature(cleaned)
    rotated = rotate_to_horizontal(centered)
    
    coords_final = cv2.findNonZero(255 - rotated)
    if coords_final is not None:
        x, y, w, h = cv2.boundingRect(coords_final)
        cropped = rotated[y:y+h, x:x+w]
    else:
        cropped = rotated
        
    cnn_input = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return rotated, cnn_input