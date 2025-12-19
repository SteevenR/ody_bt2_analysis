"""Image preprocessing and calibration utilities"""
import cv2
import re
from pathlib import Path
from .config import RANK_TEMPLATES


def preprocess_for_ocr(img):
    """Preprocess image for better OCR results."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 2
    gray_up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    th = cv2.medianBlur(th, 3)
    return gray_up, th, scale


def detect_rank_icons_by_template(image_path: Path, rank_templates: dict[int, Path]):
    """Detect rank icon positions using template matching."""
    img = cv2.imread(str(image_path))
    if img is None:
        return {}
    
    # Check which templates exist
    existing_templates = {r: p for r, p in rank_templates.items() if p.exists()}
    if not existing_templates:
        return {}
    
    scale = 2
    img_up = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    detected = {}
    
    for rank, tpl_path in existing_templates.items():
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        tpl_up = cv2.resize(tpl, (tpl.shape[1] * scale, tpl.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        best_pos = None
        best_val = 0
        # Multi-scale matching for robustness
        for tpl_scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            th, tw = int(tpl_up.shape[0] * tpl_scale), int(tpl_up.shape[1] * tpl_scale)
            if th <= 0 or tw <= 0 or th > gray_up.shape[0] or tw > gray_up.shape[1]:
                continue
            tpl_scaled = cv2.resize(tpl_up, (tw, th), interpolation=cv2.INTER_CUBIC)
            res = cv2.matchTemplate(gray_up, tpl_scaled, cv2.TM_CCOEFF_NORMED)
            if res.size == 0:
                continue
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_pos = (max_loc[0] + tw // 2, max_loc[1] + th // 2)
        if best_pos and best_val >= 0.5:
            y_center = best_pos[1] / scale
            x_center = best_pos[0] / scale
            detected[rank] = (y_center, x_center)
    
    return detected


def calibrate_line_height_from_dot_zero(dot_zero_file: Path, rank_templates: dict):
    """Calibrate line height from .0 file using rank icons."""
    detected_ranks = detect_rank_icons_by_template(dot_zero_file, rank_templates)
    if len(detected_ranks) < 2:
        return None, None, None
    
    sorted_ranks = sorted(detected_ranks.items())
    diffs = []
    for i in range(1, len(sorted_ranks)):
        y_diff = sorted_ranks[i][1][0] - sorted_ranks[i-1][1][0]
        diffs.append(y_diff)
    
    if not diffs:
        return None, None, None
    
    line_height = sum(diffs) / len(diffs)
    rank_x_positions = [x for (y, x) in detected_ranks.values()]
    rank_x_col = sum(rank_x_positions) / len(rank_x_positions)
    rank_y_positions = {r: y for r, (y, x) in detected_ranks.items()}
    
    if 1 not in rank_y_positions and 2 in rank_y_positions:
        rank_y_positions[1] = rank_y_positions[2] - line_height
    if 3 not in rank_y_positions and 2 in rank_y_positions:
        rank_y_positions[3] = rank_y_positions[2] + line_height
    
    return line_height, rank_x_col, rank_y_positions


def calibrate_for_total_two(reader, total_two_file: Path, line_height_from_one: float, debug_mode=False):
    """For total.2, find the position of rank 8 and extrapolate for ranks 9, 10."""
    img = cv2.imread(str(total_two_file))
    if img is None:
        return None, None
    
    img_up, _, scale = preprocess_for_ocr(img)
    h, w = img_up.shape[:2]
    
    # Scan top-left area for rank number "8"
    scan_height = int(line_height_from_one * scale * 2)
    scan_width = int(w * 0.2)
    scan_img = img_up[:scan_height, :scan_width]
    
    boxes = reader.readtext(scan_img, detail=1)
    rank8_y = None
    for box, text, conf in boxes:
        if re.search(r"\b8\b", text) and conf > 0.5:
            ys = [p[1] for p in box]
            rank8_y = sum(ys) / len(ys)
            break
    
    if rank8_y is None:
        return None, None
    
    # Convert rank8_y back to original pixel coordinates
    rank8_y_orig = rank8_y / scale
    
    # Extrapolate positions for ranks 9 and 10
    rank_y_positions = {
        8: rank8_y_orig,
        9: rank8_y_orig + line_height_from_one,
        10: rank8_y_orig + 2 * line_height_from_one,
    }
    
    rank_x_col = scan_width / 2 / scale
    return rank_x_col, rank_y_positions


def find_first_rank_ocr(reader, img_up, line_height_px: float, expected_first_rank: int):
    """Find the position of the first rank using OCR."""
    h, w = img_up.shape[:2]
    scan_height = int(line_height_px * 2)
    scan_img = img_up[:scan_height, :int(w * 0.2)]
    boxes = reader.readtext(scan_img, detail=1)
    
    for box, text, conf in boxes:
        if re.search(rf"\b{expected_first_rank}\b", text) and conf > 0.5:
            ys = [p[1] for p in box]
            rank_y = sum(ys) / len(ys)
            xs = [p[0] for p in box]
            rank_x = sum(xs) / len(xs)
            return rank_y, rank_x
    
    return None, None
