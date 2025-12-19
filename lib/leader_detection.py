"""Leader detection using flag template matching"""
import cv2
from pathlib import Path


def find_leader_from_flag(reader, image_path: Path, flag_template_path: Path):
    """Find leader name by detecting flag icon and nearby text.
    
    Returns: (leader_name, confidence) or (None, 0) if not found
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return None, 0
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(str(flag_template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None, 0
    
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    
    if max_val < 0.6:
        return None, max_val
    
    x_flag, y_flag = max_loc
    
    try:
        ocr_data = reader.readtext(img, detail=1)
    except Exception as e:
        return None, 0
    
    best_match = None
    best_dist = float("inf")
    
    for box, text, conf in ocr_data:
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_text = min(xs)
        y_text = sum(ys) / len(ys)
        
        # Flag should be to the left of the name
        if x_text < x_flag:
            continue
        
        dist = abs(y_text - y_flag)
        if dist < best_dist:
            best_dist = dist
            best_match = text.strip()
    
    return best_match, max_val


def detect_leader_fallback(participants: list) -> str:
    """Fallback: leader is rank 1 player."""
    if not participants:
        return None
    # Sort by rank to ensure rank 1 is first
    sorted_participants = sorted(participants, key=lambda p: p.get("rank", 999))
    return sorted_participants[0]["name"]


def detect_leader_with_tiebreaker(flagged_participants: list) -> str:
    """Detect leader from flagged participants with tie-breaking logic.
    
    When multiple participants have flags (OCR errors), prefer:
    1. Higher damage score
    2. Earlier page number (more reliable)
    3. Lower rank number
    """
    if not flagged_participants:
        return None
    
    if len(flagged_participants) == 1:
        return flagged_participants[0]["name"]
    
    # Sort by: damage desc, page asc, rank asc
    best = max(
        flagged_participants,
        key=lambda p: (
            p.get("damage", 0),
            -p.get("page_number", 999),
            -p.get("rank", 999)
        )
    )
    
    return best["name"]
