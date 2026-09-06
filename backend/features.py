import cv2
import numpy as np

def extract_crest_trough_metrics(processed_img: np.ndarray) -> dict:
    inv = (255 - processed_img) > 0
    total_area = float(np.sum(inv)) + 1e-8
    
    coords = cv2.findNonZero((255 - processed_img).astype(np.uint8))
    if coords is None:
        return {"len_ratio": 0.0, "width_ratio": 0.0, "crest_trough_val": 0.0}
        
    x, y, w, h = cv2.boundingRect(coords)
    len_to_space = float(w) / total_area
    width_to_space = float(h) / total_area
    
    vertical_proj = np.sum(inv, axis=0).astype(np.float32)
    diffs = np.diff(vertical_proj)
    crests = np.where((diffs[:-1] > 0) & (diffs[1:] < 0))[0]
    troughs = np.where((diffs[:-1] < 0) & (diffs[1:] > 0))[0]
    
    relative_dist_sum = 0.0
    min_len = min(len(crests), len(troughs))
    for i in range(min_len):
        relative_dist_sum += abs(float(crests[i]) - float(troughs[i]))
        
    crest_trough_param = float(relative_dist_sum) / total_area
    
    return {
        "len_ratio": float(len_to_space),
        "width_ratio": float(width_to_space),
        "crest_trough_val": float(crest_trough_param)
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