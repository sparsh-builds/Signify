import cv2
import numpy as np
import base64

def generate_pressure_heatmap(binary_img: np.ndarray) -> np.ndarray:
    """
    Simulates writing pressure using distance transforms on stroke widths.
    Thick, slow strokes -> high distance -> Warm colors (Red/Orange)
    Thin, fluid strokes -> low distance -> Cool colors (Cyan/Blue)
    """
    inv = 255 - binary_img
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    
    # Normalize to 0 - 255
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    # Apply JET/INFERNO colormap for thermal pressure rendering
    heatmap = cv2.applyColorMap(dist_norm, cv2.COLORMAP_JET)
    
    # Mask background back to dark theme slate
    heatmap[inv == 0] = [15, 23, 42]  # #0f172a
    return heatmap

def generate_stroke_diff_overlay(ref_img: np.ndarray, test_img: np.ndarray) -> np.ndarray:
    """
    Overlays candidate signature over reference:
      - GREEN: Perfect stroke alignment
      - RED: Unnatural deviations / forged additions
      - AMBER: Subtle natural intra-writer variance
    """
    ref_inv = (255 - ref_img) > 0
    test_inv = (255 - test_img) > 0
    
    h, w = ref_img.shape
    diff_rgb = np.full((h, w, 3), 15, dtype=np.uint8)  # Slate dark canvas
    
    # Matching pixels (Green)
    matched = ref_inv & test_inv
    diff_rgb[matched] = [50, 205, 50]  # Forest / Lime Green
    
    # Divergent forged pixels (Red / Amber)
    forged_strokes = test_inv & (~ref_inv)
    diff_rgb[forged_strokes] = [30, 30, 230]  # Bright Red
    
    # Reference missed pixels (Muted Blue/Gray)
    missed_ref = ref_inv & (~test_inv)
    diff_rgb[missed_ref] = [180, 100, 50]
    
    return diff_rgb

def img_to_base64(img_bgr: np.ndarray) -> str:
    _, buf = cv2.imencode(".png", img_bgr)
    return base64.b64encode(buf.tobytes()).decode("ascii")

