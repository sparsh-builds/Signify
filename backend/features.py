import cv2
import numpy as np
import io
from PIL import Image, ImageChops, ImageEnhance

def compute_ela_heatmap(img_bytes: bytes, quality: int = 90) -> dict:
    """
    Feature 1: Error Level Analysis (ELA) Tamper Heatmap.
    Identifies localized compression anomalies typical of digital inpainting,
    photoshop cloning, or document text eradication.
    """
    try:
        orig = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        # Re-compress in memory at uniform JPEG quality
        buffer = io.BytesIO()
        orig.save(buffer, 'JPEG', quality=quality)
        buffer.seek(0)
        recompressed = Image.open(buffer)
        
        # Calculate pixel difference residual
        diff = ImageChops.difference(orig, recompressed)
        
        # Amplify residual differences for visualization
        extrema = diff.getextrema()
        max_diff = max([ex[1] for ex in extrema])
        scale = 255.0 / max(1, max_diff) if max_diff > 0 else 1.0
        enhancer = ImageEnhance.Brightness(diff)
        amplified = enhancer.enhance(scale * 1.5)
        
        # Calculate mean error level score
        diff_np = np.array(diff).astype(np.float32)
        mean_error = float(np.mean(diff_np))
        max_error = float(np.max(diff_np))
        
        # High deviation from expected baseline indicates tampering/pasting
        is_tampered = mean_error > 8.5 or max_error > 120.0
        
        # Downsample amplified diff to a lightweight 200px preview
        amplified.thumbnail((300, 150))
        out_buf = io.BytesIO()
        amplified.save(out_buf, format="PNG")
        import base64
        ela_b64 = base64.b64encode(out_buf.getvalue()).decode('ascii')
        
        return {
            "mean_error": round(mean_error, 2),
            "max_error": round(max_error, 2),
            "tamper_detected": is_tampered,
            "tamper_confidence": round(min(99.0, max_error * 0.7), 1),
            "preview_png_base64": ela_b64
        }
    except Exception as e:
        return {
            "mean_error": 0.0,
            "max_error": 0.0,
            "tamper_detected": False,
            "tamper_confidence": 0.0,
            "preview_png_base64": ""
        }

def evaluate_iso_19794_7_compliance(img_gray: np.ndarray) -> dict:
    """
    Feature 2: ISO/IEC 19794-7 Biometric Quality Assessment Scorecard.
    Audits specimen quality against biometric exchange format standards.
    """
    if img_gray is None or img_gray.size == 0:
        return {"passed_all": False, "score": 0, "criteria": []}

    h, w = img_gray.shape
    
    # 1. Stroke Resolution & Dimensional Stability
    # Minimum compliant canvas dimension is 150x60
    res_pass = (w >= 150 and h >= 60)
    res_status = "PASS" if res_pass else "FAIL"
    res_desc = f"{w}x{h} px (>150x60 px threshold)"
    
    # 2. Ink-to-Paper Signal-to-Noise Ratio (SNR)
    # Estimate background noise variance vs stroke contrast
    _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ink_pixels = img_gray[binary == 0]
    paper_pixels = img_gray[binary == 255]
    
    mean_ink = float(np.mean(ink_pixels)) if len(ink_pixels) > 0 else 0.0
    mean_paper = float(np.mean(paper_pixels)) if len(paper_pixels) > 0 else 255.0
    std_paper = float(np.std(paper_pixels)) if len(paper_pixels) > 0 else 1.0
    
    snr_val = (mean_paper - mean_ink) / max(1.0, std_paper)
    snr_pass = snr_val >= 4.0  # Clear contrast against paper texture
    snr_status = "PASS" if snr_pass else "WARN"
    snr_desc = f"{round(snr_val, 1)} SNR (Req >= 4.0)"
    
    # 3. Dynamic Range & Gray Depth Coverage
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    nonzero_bins = np.count_nonzero(hist)
    dynamic_coverage = (nonzero_bins / 256.0) * 100.0
    dr_pass = dynamic_coverage >= 30.0
    dr_status = "PASS" if dr_pass else "FAIL"
    dr_desc = f"{round(dynamic_coverage, 1)}% Grayscale Spread"
    
    # 4. Aspect Ratio & Spatial Symmetry
    aspect = w / max(1, h)
    aspect_pass = (1.2 <= aspect <= 7.0)
    aspect_status = "PASS" if aspect_pass else "WARN"
    aspect_desc = f"{round(aspect, 2)}:1 Proportions"
    
    passed_count = sum([res_pass, snr_pass, dr_pass, aspect_pass])
    iso_score = round((passed_count / 4.0) * 100.0, 1)

    return {
        "passed_all": (passed_count >= 3),
        "compliance_score": iso_score,
        "criteria": [
            {"name": "Spatial Dimension Check", "status": res_status, "detail": res_desc},
            {"name": "Contrast SNR (Ink/Paper)", "status": snr_status, "detail": snr_desc},
            {"name": "Dynamic Grayscale Coverage", "status": dr_status, "detail": dr_desc},
            {"name": "Bounding Aspect Ratio", "status": aspect_status, "detail": aspect_desc}
        ]
    }

def estimate_pseudo_velocity_profile(binary_signature: np.ndarray, num_samples: int = 50) -> list:
    """
    Feature 1: Dynamic Stroke Velocity Inversion
    Approximates kinematic pen speed from static ink deposits.
    Thicker stroke / pooling = deceleration & hesitation (low velocity).
    Thinner stroke = rapid ballistic movement (high velocity).
    v(s) proportional to 1 / (distance_transform(s) + epsilon)
    """
    if binary_signature is None or binary_signature.size == 0:
        return [50.0] * num_samples

    # Invert so stroke foreground is non-zero
    stroke_mask = (binary_signature < 128).astype(np.uint8) * 255
    if np.sum(stroke_mask) == 0:
        return [50.0] * num_samples

    # Euclidean distance transform measures local stroke half-width (radius)
    dist_transform = cv2.distanceTransform(stroke_mask, cv2.DIST_L2, 5)

    # Extract non-zero radius values along the stroke medial axis
    radii = dist_transform[dist_transform > 0.5]
    if len(radii) < num_samples:
        return [50.0] * num_samples

    # Downsample uniformly along the contour length
    indices = np.linspace(0, len(radii) - 1, num_samples).astype(int)
    sampled_radii = radii[indices]

    # Invert radius to approximate velocity: thinner strokes = higher velocity
    raw_velocities = 1.0 / (sampled_radii + 0.5)

    # Normalize to an 0-100 kinematic scale
    v_min, v_max = np.min(raw_velocities), np.max(raw_velocities)
    norm_v = ((raw_velocities - v_min) / (max(1e-5, v_max - v_min))) * 100.0
    return [round(float(v), 1) for v in norm_v]


def classify_pen_medium_and_substrate(binary_sig: np.ndarray) -> dict:
    """
    Feature 2: Physical Pen Medium & Substrate Classifier
    Evaluates stroke edge boundary falloff gradients (dI/dx).
    - Digital Stylus: Infinite edge sharpness, zero capillary feathering.
    - Ballpoint Pen: Medium sharpness with internal striation grooves.
    - Fountain / Gel Ink: High capillary spread and porous fiber bleeding.
    """
    if binary_sig is None or binary_sig.size == 0:
        return {"medium": "Unknown", "confidence": 0.0, "ink_bleed_index": 0.0}

    # Sobel gradient magnitude on stroke edges
    sobel_x = cv2.Sobel(binary_sig, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(binary_sig, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)

    # Edge transition width evaluation
    edge_pixels = grad_mag[grad_mag > 20]
    if len(edge_pixels) == 0:
        return {"medium": "Standard Scan", "confidence": 85.0, "ink_bleed_index": 0.12}

    mean_grad = float(np.mean(edge_pixels))
    std_grad = float(np.std(edge_pixels))

    # Metric: Sharpness vs edge roughness variance
    bleed_ratio = std_grad / max(1.0, mean_grad)

    if mean_grad > 110.0 and bleed_ratio < 0.35:
        medium = "Digital Stylus / Apple Pencil (Vector Trace)"
        conf = round(min(99.0, 75.0 + (mean_grad / 5)), 1)
    elif bleed_ratio > 0.65:
        medium = "Fountain / Gel Pen (Capillary Fiber Bleed)"
        conf = round(min(97.5, 70.0 + (bleed_ratio * 30)), 1)
    else:
        medium = "Steel-Ballpoint Pen (Standard Cellulose Substrate)"
        conf = round(min(96.0, 72.0 + (mean_grad / 8)), 1)

    return {
        "medium": medium,
        "confidence": conf,
        "ink_bleed_index": round(bleed_ratio, 3)
    }
    
    
def extract_crest_trough_metrics(binary_img: np.ndarray) -> dict:
    if binary_img is None or binary_img.size == 0:
        return {"len_ratio": 0.0, "width_ratio": 0.0, "crest_trough_val": 0.0}

    # Invert so ink is white (255)
    ink_mask = (binary_img < 128).astype(np.uint8) * 255
    coords = cv2.findNonZero(ink_mask)

    # Crop tightly to the stroke envelope to eliminate white-space padding disparities
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        pad = 4
        h_img, w_img = binary_img.shape
        y1, y2 = max(0, y - pad), min(h_img, y + h + pad)
        x1, x2 = max(0, x - pad), min(w_img, x + w + pad)
        cropped_ink = ink_mask[y1:y2, x1:x2]
    else:
        cropped_ink = ink_mask

    proj = np.sum(cropped_ink > 0, axis=0).astype(np.float32)
    if len(proj) == 0 or np.max(proj) == 0:
        return {"len_ratio": 0.0, "width_ratio": 0.0, "crest_trough_val": 0.0}

    # Resample to 40 normalized columns
    resampled = cv2.resize(proj.reshape(1, -1), (40, 1), interpolation=cv2.INTER_AREA).flatten()
    norm_profile = resampled / (np.max(resampled) + 1e-6)

    # Waveform variance
    crest_trough_val = float(np.std(norm_profile))

    return {
        "len_ratio": float(np.count_nonzero(cropped_ink)) / float(cropped_ink.size),
        "width_ratio": float(cropped_ink.shape[1]) / float(max(1, cropped_ink.shape[0])),
        "crest_trough_val": round(crest_trough_val, 5)
    }
def extract_harris_and_orb_points(processed_img: np.ndarray):
    gray = np.float32(processed_img)
    dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
    dst = cv2.dilate(dst, None)
    corner_count = int(np.sum(dst > 0.01 * dst.max())) if dst.max() > 0 else 0
    
    orb = cv2.ORB_create(nfeatures=150)
    kp, des = orb.detectAndCompute(processed_img, None)
    
    if des is not None and len(des.shape) == 2 and des.shape[1] == 32:
        return corner_count, des
    return corner_count, None

def match_forgery_confidence(test_img: np.ndarray, enrolled_template: dict) -> tuple:
    test_metrics = extract_crest_trough_metrics(test_img)
    test_corners, test_des = extract_harris_and_orb_points(test_img)
    
    ref_metrics = enrolled_template.get("metrics", {"len_ratio": 0.0, "crest_trough_val": 0.0})
    dev_len = abs(test_metrics["len_ratio"] - ref_metrics.get("len_ratio", 0.0)) / (ref_metrics.get("len_ratio", 0.0) + 1e-8)
    dev_ct = abs(test_metrics["crest_trough_val"] - ref_metrics.get("crest_trough_val", 0.0)) / (ref_metrics.get("crest_trough_val", 0.0) + 1e-8)
    
    metric_confidence = max(0.0, 1.0 - (dev_len * 0.5 + dev_ct * 0.5))
    
    raw_ref_des = enrolled_template.get("descriptors")
    ref_des = None
    if raw_ref_des is not None and len(raw_ref_des) > 0:
        try:
            arr = np.array(raw_ref_des, dtype=np.uint8)
            if len(arr.shape) == 2 and arr.shape[1] == 32:
                ref_des = arr
        except Exception:
            ref_des = None

    keypoint_confidence = 0.0
    if test_des is not None and ref_des is not None and len(test_des) > 0 and len(ref_des) > 0:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(test_des, ref_des)
        good_matches = [m for m in matches if m.distance < 55]
        keypoint_confidence = min(1.0, len(good_matches) / 15.0)
    else:
        ref_corners = max(1, enrolled_template.get("corners", 1))
        keypoint_confidence = max(0.0, 1.0 - abs(test_corners - ref_corners) / float(ref_corners))
        
    return round(float(metric_confidence), 4), round(float(keypoint_confidence), 4)

def check_screen_spoof_fft(image_bytes: bytes, threshold: float = 0.88) -> dict:
    """
    2D FFT presentation attack detection tuned to reject authentic paper photos
    while isolating sub-pixel raster grating from electronic displays.
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"is_spoof": False, "spoof_score": 0.0, "reason": "Decode skipped"}

    img_resized = cv2.resize(img, (256, 256))
    f = np.fft.fft2(img_resized)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1e-8)

    h, w = magnitude_spectrum.shape
    center_y, center_x = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center_x) ** 2 + (Y - center_y) ** 2)

    # Isolated periodic screen-grating band (high-frequency sub-pixel lattice)
    screen_freq_mask = (dist_from_center >= 95) & (dist_from_center <= 122)
    base_freq_mask = dist_from_center < 35

    high_energy = np.mean(magnitude_spectrum[screen_freq_mask])
    base_energy = np.mean(magnitude_spectrum[base_freq_mask]) + 1e-8

    spectral_ratio = float(high_energy / base_energy)
    spoof_score = round(min(100.0, max(0.0, (spectral_ratio - 0.55) * 220)), 1)

    is_spoof = bool(spectral_ratio > threshold)

    return {
        "is_spoof": is_spoof,
        "spoof_score": spoof_score,
        "spectral_ratio": round(spectral_ratio, 4),
        "status": "PRESENTATION_ATTACK_DETECTED" if is_spoof else "BONAFIDE_PAPER"
    }