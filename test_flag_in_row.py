"""
Test flag detection on a single row image.
Usage: python test_flag_in_row.py <path_to_row_image>
Example: python test_flag_in_row.py data/debug_rows/24.0/rank02_Buffy.png
"""

import sys
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).parent
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"

def test_flag_in_row(image_path: Path):
    print(f"\n{'='*80}")
    print(f"Testing flag in row: {image_path}")
    print('='*80)
    
    row_img = cv2.imread(str(image_path))
    if row_img is None:
        print(f"[ERROR] Cannot read image: {image_path}")
        return
    
    print(f"Row image size: {row_img.shape[1]}x{row_img.shape[0]} pixels")
    
    flag_template = cv2.imread(str(FLAG_TEMPLATE_PATH), cv2.IMREAD_GRAYSCALE)
    if flag_template is None:
        print(f"[ERROR] Cannot read flag template: {FLAG_TEMPLATE_PATH}")
        return
    
    print(f"Flag template size: {flag_template.shape[1]}x{flag_template.shape[0]} pixels")
    
    row_gray = cv2.cvtColor(row_img, cv2.COLOR_BGR2GRAY) if len(row_img.shape) == 3 else row_img
    
    # Try multiple scales
    print(f"\n--- TEMPLATE MATCHING ---")
    best_val = 0
    best_scale = 1.0
    best_loc = None
    
    for scale in [0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5, 2.0]:
        h, w = flag_template.shape
        new_h, new_w = int(h * scale), int(w * scale)
        if new_h <= 0 or new_w <= 0 or new_h > row_gray.shape[0] or new_w > row_gray.shape[1]:
            continue
        
        scaled_template = cv2.resize(flag_template, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        res = cv2.matchTemplate(row_gray, scaled_template, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(res)
        
        print(f"  Scale {scale:.1f}: confidence={max_val:.4f} at ({max_loc[0]}, {max_loc[1]})")
        
        if max_val > best_val:
            best_val = max_val
            best_scale = scale
            best_loc = max_loc
    
    print(f"\n--- RESULT ---")
    print(f"Best match: scale={best_scale:.1f}, confidence={best_val:.4f}, position=({best_loc[0]}, {best_loc[1]})")
    print(f"Threshold: 0.6")
    
    if best_val >= 0.6:
        print(f"✓ FLAG DETECTED!")
    else:
        print(f"✗ Flag not detected (confidence {best_val:.4f} < 0.6)")
    
    print(f"{'='*80}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_flag_in_row.py <path_to_row_image>")
        print("Example: python test_flag_in_row.py data/debug_rows/24.0/rank02_Buffy.png")
        sys.exit(1)
    
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)
    
    test_flag_in_row(img_path)

if __name__ == "__main__":
    main()
