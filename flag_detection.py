"""
Shared flag detection module for leader identification.
Used by both test_flag_detection.py and beartrap_analysis_parallel.py.
"""

import re
from pathlib import Path
import cv2
import numpy as np

BASE_DIR = Path(__file__).parent
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"

def _compute_color_weight_map(roi_color):
    """Precompute color weight map to bias matches toward the specific #6cc5a0 color only."""
    try:
        hsv_roi = cv2.cvtColor(roi_color, cv2.COLOR_BGR2HSV)
        # Specific #6cc5a0 band with dilation
        target = np.uint8([[[160, 197, 108]]])  # BGR
        target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0, 0]
        h0, s0, v0 = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
        H = hsv_roi[:, :, 0].astype(np.int16)
        S = hsv_roi[:, :, 1].astype(np.int16)
        V = hsv_roi[:, :, 2].astype(np.int16)
        dH = np.abs(H - h0)
        dH = np.minimum(dH, 180 - dH)
        h_tol, s_tol, v_tol = 30, 120, 120
        mask6c = (dH <= h_tol) & (np.abs(S - s0) <= s_tol) & (np.abs(V - v0) <= v_tol)
        mask6c_u8 = mask6c.astype(np.uint8) * 255
        kernel = np.ones((3, 3), np.uint8)
        mask6c_dil = cv2.dilate(mask6c_u8, kernel, iterations=1)
        # Weight ONLY by specific color presence
        m6 = (mask6c_dil.astype(np.float32) / 255.0)
        return m6
    except Exception:
        return None

def _color_ratios_near_hit(roi_color, hit_loc, win_wh):
    """Compute #6cc5a0 color ratio near the matched hit location (no generic green)."""
    color6c_ratio = 0.0
    try:
        x_hit, y_hit = hit_loc
        bw, bh = win_wh
        # Sample a slightly offset patch (up-left) to better cover the flag ink
        offsets = [(0, 0), (-2, -2), (-2, 0), (0, -2), (2, 2), (4, 2), (2, 4)]
        color6c_vals = []
        for dx, dy in offsets:
            xs = x_hit + dx
            ys = y_hit + dy
            x2 = min(roi_color.shape[1], xs + bw)
            y2 = min(roi_color.shape[0], ys + bh)
            if xs < 0 or ys < 0 or x2 <= xs or y2 <= ys:
                continue
            patch = roi_color[ys:y2, xs:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            # Skip generic green computation (not used)
            # Specific #6cc5a0
            target = np.uint8([[[160, 197, 108]]])  # BGR for #6cc5a0
            target_hsv = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)[0, 0]
            h0, s0, v0 = int(target_hsv[0]), int(target_hsv[1]), int(target_hsv[2])
            H = hsv[:, :, 0].astype(np.int16)
            S = hsv[:, :, 1].astype(np.int16)
            V = hsv[:, :, 2].astype(np.int16)
            dH = np.abs(H - h0)
            dH = np.minimum(dH, 180 - dH)
            h_tol, s_tol, v_tol = 30, 120, 120
            mask6c = (dH <= h_tol) & (np.abs(S - s0) <= s_tol) & (np.abs(V - v0) <= v_tol)
            mask6c_u8 = mask6c.astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            mask6c_dil = cv2.dilate(mask6c_u8, kernel, iterations=1)
            color6c_vals.append(float(mask6c_dil.mean()) if mask6c_dil.size else 0.0)
        # generic green ratio intentionally unused
        if color6c_vals:
            color6c_ratio = float(np.median(color6c_vals))
    except Exception:
        pass
    return color6c_ratio

def _white_ratio_near_hit(roi_color, hit_loc, win_wh):
    """Compute near-white ratio around the matched hit location.
    Uses HSV thresholds (low saturation, high value) with slight dilation.
    """
    white_ratio = 0.0
    try:
        x_hit, y_hit = hit_loc
        bw, bh = win_wh
        offsets = [(0, 0), (-2, -2), (-2, 0), (0, -2), (2, 2), (4, 2), (2, 4)]
        vals = []
        for dx, dy in offsets:
            xs = x_hit + dx
            ys = y_hit + dy
            x2 = min(roi_color.shape[1], xs + bw)
            y2 = min(roi_color.shape[0], ys + bh)
            if xs < 0 or ys < 0 or x2 <= xs or y2 <= ys:
                continue
            patch = roi_color[ys:y2, xs:x2]
            if patch.size == 0:
                continue
            hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
            S = hsv[:, :, 1]
            V = hsv[:, :, 2]
            # near-white: very low saturation and high value
            mask = (S <= 35) & (V >= 225)
            mask_u8 = mask.astype(np.uint8)
            mask_dil = cv2.dilate(mask_u8, np.ones((3, 3), np.uint8), iterations=1)
            vals.append(float(mask_dil.mean()) if mask_dil.size else 0.0)
        if vals:
            white_ratio = float(np.median(vals))
    except Exception:
        pass
    return white_ratio

def _detect_in_zone(zone_gray, zone_color, zone_name="zone"):
    """
    Helper to run template matching and color analysis on a specific zone.
    Returns dict with all metrics for logging.
    """
    flag_template = cv2.imread(str(FLAG_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if flag_template is None or zone_gray.size == 0:
        return {
            "zone": zone_name,
            "shape_score": 0.0,
            "color6c_ratio": 0.0,
            "white_ratio": 0.0,
            "hit_loc": None,
            "hit_wh": None,
        }
    
    best_val = 0.0
    best_loc = None
    best_wh = None
    
    # Precompute color weight map
    color_weight_map = _compute_color_weight_map(zone_color) if zone_color is not None and zone_color.size > 0 else None
    
    # Use only scale 1.0 (no zoom) to keep consistent template size
    h, w = flag_template.shape
    new_h, new_w = h, w
    if new_h > 0 and new_w > 0 and new_h <= zone_gray.shape[0] and new_w <= zone_gray.shape[1]:
        res1 = cv2.matchTemplate(zone_gray, flag_template, cv2.TM_CCOEFF_NORMED)
        _, mask = cv2.threshold(flag_template, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        try:
            res2 = cv2.matchTemplate(zone_gray, flag_template, cv2.TM_CCORR_NORMED, mask=mask)
            res_shape = 0.5 * res1 + 0.5 * res2
        except Exception:
            res_shape = res1
        
        # Build weighted map for localization only (do not use to define shape score)
        res_weighted = res_shape
        if color_weight_map is not None:
            try:
                S = cv2.integral(color_weight_map)
                Hh, Ww = color_weight_map.shape[:2]
                sum_map = S[new_h:Hh+1, new_w:Ww+1] - S[new_h:Hh+1, 0:Ww+1-new_w] - S[0:Hh+1-new_h, new_w:Ww+1] + S[0:Hh+1-new_h, 0:Ww+1-new_w]
                wavg = sum_map / float(new_h * new_w)
                wavg = np.clip(wavg.astype(np.float32), 0.0, 1.0)
                res_weighted = res_shape * (1.0 + 0.8 * wavg)
            except Exception:
                pass
        
        # Select location from weighted map, but measure shape from unweighted map
        _, _, _, max_loc = cv2.minMaxLoc(res_weighted)
        best_loc = max_loc
        best_val = float(res_shape[max_loc[1], max_loc[0]])
        best_wh = (new_w, new_h)
    
    if best_loc is None or not np.isfinite(best_val):
        best_val = 0.0
        best_loc = None
    
    shape_score = float(best_val)
    
    # No uniqueness or combined score; TL50-only metrics
    
    # Color ratios
    color6c_ratio = 0.0
    white_ratio = 0.0
    if best_loc is not None and best_wh is not None and zone_color is not None and zone_color.size > 0:
        color6c_ratio = _color_ratios_near_hit(zone_color, best_loc, best_wh)
        white_ratio = _white_ratio_near_hit(zone_color, best_loc, best_wh)
    
    # No combined score; return raw metrics only
    
    return {
        "zone": zone_name,
        "shape_score": float(shape_score),
        "color6c_ratio": float(color6c_ratio),
        "white_ratio": float(white_ratio),
        "hit_loc": best_loc,
        "hit_wh": best_wh,
    }

def detect_flag_in_row(row_color_img, tokens=None, save_roi_to=None, save_hit_to=None, save_avatar_to=None, save_tl50_to=None, verbose=True):
    """
    Shared flag detection used by both test helper and parallel analysis.
    
    Args:
        row_color_img: BGR row image
        tokens: Optional list of OCR tokens [{"text": str, "x": int, "y": int, "conf": float}]
                Used to compute ROI bounds between rank and name.
        save_roi_to: Optional Path to save the ROI crop for debugging
        verbose: Whether to print debug logs (default True for backward compatibility)
    
    Returns:
        dict with keys (TL50-only schema):
            - has_flag: bool
            - shape_score: float
            - green_score: float  (#6cc5a0 ratio)
            - white_score: float  (near-white ratio)
            - final_score: float  (average of the three)
            - tl50: dict with {x,y,w,h} or None
    """
    if row_color_img is None or row_color_img.size == 0:
        return {"has_flag": False, "flag_conf": 0.0}
    
    try:
        row_gray = cv2.cvtColor(row_color_img, cv2.COLOR_BGR2GRAY)
    except Exception:
        row_gray = row_color_img if len(row_color_img.shape) == 2 else cv2.cvtColor(row_color_img, cv2.COLOR_BGR2GRAY)

    h_row, w_row = row_gray.shape[:2]
    
    # Determine ROI bounds from tokens (rank left, name left)
    x_name_left = None
    x_rank = None
    if tokens:
        # Name candidates: leftmost non-UI token with decent confidence
        name_candidates = []
        for t in tokens:
            text = str(t.get("text", "")).strip()
            conf = float(t.get("conf", 0))
            if not text or conf < 0.4:
                continue
            if re.fullmatch(r"\d[\d\s\.,kKmM]*", text):
                continue
            if re.search(r"(points\s+de|de\s+d[ée]g[âa]ts|rapport\s+de|d[ée]g[âa]ts|ints|oi)", text, re.IGNORECASE):
                continue
            name_candidates.append(t)
        if name_candidates:
            x_name_left = min(int(t.get("x", 0)) for t in name_candidates)
        
        # Rank number at left (1-20)
        for tok in tokens:
            text = str(tok.get("text", ""))
            conf = float(tok.get("conf", 0))
            x = int(tok.get("x", 0))
            if re.fullmatch(r"\d{1,2}", text) and conf >= 0.5 and x < w_row * 0.25:
                try:
                    v = int(text)
                except Exception:
                    v = 99
                if 1 <= v <= 20:
                    x_rank = x if x_rank is None else min(x_rank, x)
    
    # Fallbacks if tokens not provided or not found
    if x_name_left is None:
        x_name_left = int(w_row * 0.35)
    if x_rank is None:
        x_rank = int(w_row * 0.05)

    # Horizontal band between rank and name, using full height for better contour detection
    band_x1 = max(0, int(x_rank + w_row * 0.01))  # Shift further left, closer to rank
    band_x2 = min(w_row, int(x_name_left - w_row * 0.02))
    if band_x2 <= band_x1:
        band_x1, band_x2 = 0, min(w_row, int(w_row * 0.30))
    band_gray = row_gray[:, band_x1:band_x2]  # Full height
    band_color = row_color_img[:, band_x1:band_x2]  # Full height

    # Detect avatar region within the band to find top-left corner
    avatar_rect = None
    try:
        edges = cv2.Canny(band_gray, 60, 180)
        edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_score = -1.0
        best_contour_x = 0
        best_contour_y = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w < 40 or h < 40:
                continue
            # Prefer squarish contours, use them to locate avatar top-left
            aspect = min(w, h) / max(w, h)
            if aspect < 0.6:
                continue
            area = w * h
            top_bias = max(0.0, 1.0 - (y / max(1.0, h_row * 0.6)))
            score = aspect * area * (0.8 + 0.2 * top_bias)
            if score > best_score:
                best_score = score
                best_contour_x = x
                best_contour_y = y
        # Create 300x300 square anchored at detected top-left
        avatar_rect = (best_contour_x, best_contour_y, 300, 300)
    except Exception:
        avatar_rect = None
    if avatar_rect is None:
        # Fallback: 300x300 square anchored at left of band
        avatar_rect = (0, 0, 300, 300)

    ax, ay, aw, ah = avatar_rect
    # Absolute coords in row
    av_x1 = band_x1 + ax
    av_y1 = ay
    av_x2 = min(w_row, av_x1 + aw)
    av_y2 = min(h_row, av_y1 + ah)

    # No ROI concepts anymore; keep only optional avatar/TL50 crops
    if save_avatar_to is not None:
        try:
            av_crop = row_color_img[av_y1:av_y2, av_x1:av_x2]
            if av_crop.size > 0:
                cv2.imwrite(str(save_avatar_to), av_crop)
        except Exception:
            pass
    tl_rect = None
    tl_gray = None
    tl_color = None
    try:
        tl_w = min(60, av_x2 - av_x1)
        tl_h = min(80, av_y2 - av_y1)
        tl_rect = (av_x1, av_y1, tl_w, tl_h)
        if tl_w > 0 and tl_h > 0:
            tl_color = row_color_img[av_y1:av_y1 + tl_h, av_x1:av_x1 + tl_w]
            tl_gray = cv2.cvtColor(tl_color, cv2.COLOR_BGR2GRAY) if tl_color.size > 0 else None
            if save_tl50_to is not None and tl_color.size > 0:
                cv2.imwrite(str(save_tl50_to), tl_color)
    except Exception:
        pass

    # TL50-only detection and logging
    tl_result = None
    if tl_gray is not None and tl_color is not None and tl_gray.size > 0:
        if verbose:
            print(f"\n=== TL50 (Top-Left 60x80) ===")
            print(f"TL50 position: x={tl_rect[0]}, y={tl_rect[1]}, w={tl_rect[2]}, h={tl_rect[3]}")
        tl_result = _detect_in_zone(tl_gray, tl_color, zone_name="tl50")
        if verbose:
            print(f"- Flag image similarity: {tl_result['shape_score']:.4f}")
            print(f"- Color #6cc5a0 ratio:   {tl_result['color6c_ratio']:.4f}")
            print(f"- Color ffffff ratio:    {tl_result['white_ratio']:.4f}")
    
    # Decision based solely on TL50 check (shape + green + white)
    if tl_result is None:
        # Minimal payload only; no legacy ROI/hit/uniqueness/border fields
        return {
            "has_flag": False,
            "shape_score": 0.0,
            "green_score": 0.0,
            "white_score": 0.0,
            "final_score": 0.0,
            "tl50": {"x": int(tl_rect[0]), "y": int(tl_rect[1]), "w": int(tl_rect[2]), "h": int(tl_rect[3])} if tl_rect else None,
        }

    shape_score = tl_result['shape_score']
    color6c_ratio = tl_result['color6c_ratio']
    white_ratio = tl_result['white_ratio']
    final_score = (shape_score + color6c_ratio + white_ratio) / 3.0
    left_bias_ok = True  # by design TL50 is top-left

    if verbose:
        print(f"\n=== DECISION (TL50 ONLY) ===")
        print(f"Thresholds: shape>=0.42, green>=0.50, white>=0.50")
    
    thr_shape = 0.42
    min_green_major = 0.50
    min_white_major = 0.40
    shape_ok = shape_score >= thr_shape
    green_ok = color6c_ratio >= min_green_major
    white_ok = white_ratio >= min_white_major
    has_flag = shape_ok and green_ok and white_ok and left_bias_ok
    if verbose:
        print(f"- Shape score {shape_score:.4f} >= {thr_shape}: {'OK' if shape_ok else 'FAIL'}")
        print(f"- Green ratio  {color6c_ratio:.4f} >= {min_green_major}: {'OK' if green_ok else 'FAIL'}")
        print(f"- White ratio  {white_ratio:.4f} >= {min_white_major}: {'OK' if white_ok else 'FAIL'}")
        print(f"- Left bias: {left_bias_ok}")
        print(f">>> HAS_FLAG: {has_flag}")

    # No hit box saving or return anymore

    return {
        "has_flag": bool(has_flag),
        "shape_score": float(shape_score),
        "green_score": float(color6c_ratio),
        "white_score": float(white_ratio),
        "final_score": float(final_score),
        "tl50": {"x": int(tl_rect[0]), "y": int(tl_rect[1]), "w": int(tl_rect[2]), "h": int(tl_rect[3])} if tl_rect else None,
    }

def detect_flag_in_row_file(image_path: Path, save_roi_to=None, save_hit_to=None, save_avatar_to=None, save_tl50_to=None):
    """
    Convenience wrapper for file-based detection (used by tests).
    Performs OCR internally to extract tokens for ROI calculation.
    """
    import easyocr
    
    img = cv2.imread(str(image_path))
    if img is None:
        return {"has_flag": False, "flag_conf": 0.0}
    
    # Perform OCR to get tokens
    reader = easyocr.Reader(["fr", "en"], gpu=False, verbose=False)
    row_texts_detail = reader.readtext(str(image_path), detail=1)
    tokens = []
    for box, text, conf in row_texts_detail:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        tokens.append({"text": text.strip(), "conf": conf, "x": min(xs), "y": sum(ys) / len(ys)})
    
    return detect_flag_in_row(img, tokens=tokens, save_roi_to=save_roi_to, save_hit_to=save_hit_to, save_avatar_to=save_avatar_to, save_tl50_to=save_tl50_to)
