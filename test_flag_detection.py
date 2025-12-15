"""
Test flag detection for leader identification.
Usage: python test_flag_detection.py <path_to_image>
Example: python test_flag_detection.py beartrap_data/2025-12-11/24.0.png
"""

import sys
from pathlib import Path
import cv2
import easyocr
import re
import numpy as np
from flag_detection import detect_flag_in_row_file

BASE_DIR = Path(__file__).parent
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"

def _detect_flag_in_row(image_path: Path):
    print(f"\n{'='*80}")
    print(f"Row-mode flag detection: {image_path}")
    print('='*80)

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return None

    h_row, w_row = img.shape[:2]
    print(f"Row size: {w_row}x{h_row} pixels")

    template = cv2.imread(str(FLAG_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"[ERROR] Cannot read flag template: {FLAG_TEMPLATE_PATH}")
        return None
    print(f"Flag template: {template.shape[1]}x{template.shape[0]}")

    # Use shared detection module
    debug_dir = Path("data/debug_roi_tests")
    debug_dir.mkdir(parents=True, exist_ok=True)
    roi_debug_path = debug_dir / f"{image_path.stem}_roi.png"
    hit_debug_path = debug_dir / f"{image_path.stem}_hit.png"
    av_debug_path = debug_dir / f"{image_path.stem}_avatar.png"
    tl_debug_path = debug_dir / f"{image_path.stem}_tl50.png"
    
    result = detect_flag_in_row_file(image_path, save_roi_to=roi_debug_path, save_hit_to=hit_debug_path, save_avatar_to=av_debug_path, save_tl50_to=tl_debug_path)
    
    # Concise summary aligned with TL50-only checks
    print("\n=== SUMMARY (TL50 ONLY) ===")
    flag_conf = float(result.get('shape_score', 0.0))
    green = float(result.get('green_score', 0.0))
    white = float(result.get('white_score', 0.0))
    tl = result.get('tl50')
    if tl:
        print(f"TL50: x={tl['x']}, y={tl['y']}, w={tl['w']}, h={tl['h']}")
    print(f"Flag similarity: {flag_conf:.4f}")
    print(f"Green #6cc5a0:   {green:.4f}")
    print(f"White ffffff:    {white:.4f}")
    final_score = float(result.get('final_score', (flag_conf + green + white) / 3.0))
    detected = bool(result.get('has_flag', False))
    print(f"Final score (avg shape+green+white): {final_score:.4f}")
    print(f"Decision: {'DETECTED' if detected else 'REJECTED'}")
    return detected


def test_flag_detection(image_path: Path):
    print(f"\n{'='*80}")
    print(f"Testing flag detection: {image_path}")
    print('='*80)
    
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Load flag template
    template = cv2.imread(str(FLAG_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print(f"[ERROR] Cannot read flag template: {FLAG_TEMPLATE_PATH}")
        return
    
    print(f"Flag template size: {template.shape[1]}x{template.shape[0]} pixels")
    
    # If this looks like a row image (e.g., debug_rows) use row-mode detection
    if "debug_rows" in str(image_path).replace("\\", "/").lower() or img.shape[0] < img.shape[1] / 2:
        return _detect_flag_in_row(image_path)

    # Page-level fallback (less reliable; prefer row-mode)
    print("[INFO] Page-level detection is noisy; prefer row images.")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    print(f"Match confidence (page): {max_val:.4f}")
    if max_val >= 0.7:
        print("✓ Possible flag on page")
    else:
        print("⚠️  No reliable flag on page")
    return max_val >= 0.7

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_flag_detection.py <path_to_image>")
        print("Example: python test_flag_detection.py beartrap_data/2025-12-11/24.0.png")
        sys.exit(1)
    
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)
    
    test_flag_detection(img_path)

if __name__ == "__main__":
    main()
