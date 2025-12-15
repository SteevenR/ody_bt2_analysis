"""
Test script to analyze a single beartrap screenshot file with detailed OCR output.
Usage: python test_single_file.py <path_to_image>
Example: python test_single_file.py beartrap_data/2025-12-11/21.1.png
"""

import sys
import json
from pathlib import Path
import easyocr
import cv2
import re
import unicodedata

# Import functions from main script
BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from beartrap_analysis_parallel import (
    init_reader,
    preprocess_for_ocr,
    clean_extracted_name,
    normalize_damage,
    _normalize_key,
    _fuzzy_normalize_key,
    load_translations_store,
    resolve_with_translation,
    DATA_OUTPUT_DIR
)

def analyze_single_row_image(reader, img_path: Path):
    """Analyze a single row image with detailed OCR output."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {img_path}")
    print('='*80)
    
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return
    
    print(f"Image size: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Preprocess
    gray_up, bin_img, scale = preprocess_for_ocr(img)
    
    # OCR with detail
    row_texts_detail = reader.readtext(img, detail=1)
    
    print(f"\n--- RAW OCR OUTPUT ({len(row_texts_detail)} tokens) ---")
    tokens = []
    for i, (box, text, conf) in enumerate(row_texts_detail):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        x_pos = min(xs)
        y_pos = sum(ys) / len(ys)
        tokens.append({"text": text.strip(), "conf": conf, "x": x_pos, "y": y_pos})
        print(f"  [{i}] '{text}' | conf={conf:.2f} | x={x_pos:.0f} | y={y_pos:.0f}")
    
    # Extract name
    print(f"\n--- NAME EXTRACTION ---")
    name_candidates = []
    for t in tokens:
        text = t["text"].strip()
        if not text:
            continue
        if re.fullmatch(r"\d[\d\s\.,kKmM]*", text):
            print(f"  ✗ '{text}' (number)")
            continue
        if re.search(r"(points\s+de|de\s+d[ée]g[âa]ts|rapport\s+de|d[ée]g[âa]ts|ints|oi)", text, re.IGNORECASE):
            print(f"  ✗ '{text}' (UI text)")
            continue
        if t["conf"] < 0.4:
            print(f"  ✗ '{text}' (low confidence {t['conf']:.2f})")
            continue
        print(f"  ✓ '{text}'")
        name_candidates.append(t)
    
    name_candidates.sort(key=lambda t: t["x"])
    raw_name = " ".join([t["text"] for t in name_candidates]).strip()
    name = clean_extracted_name(raw_name)
    
    print(f"\nRaw name: '{raw_name}'")
    print(f"Cleaned name: '{name}'")
    
    # Filter UI text
    if name and re.search(r"rapport\s*(de|da)?\s*combat", name, re.IGNORECASE):
        print(f"⚠️  UI text detected, filtering out")
        name = ""
    
    # Extract damage
    print(f"\n--- DAMAGE EXTRACTION ---")
    all_numbers = []
    for tok in tokens:
        nums = re.findall(r"\d[\d\s\.,kKmM]*", tok["text"])
        for num_str in nums:
            val = normalize_damage(num_str)
            # Only keep numbers > 100 (real damage) or rank numbers (< 20) on left side
            if val > 100:
                print(f"  ✓ {num_str} → {val:,} (damage)")
                all_numbers.append((val, num_str))
            elif 0 < val <= 20 and tok["x"] < img.shape[1] * 0.15:
                print(f"  ✓ {num_str} → {val} (rank indicator)")
                all_numbers.append((val, num_str))
            else:
                print(f"  ✗ {num_str} → {val} (filtered: too small or not on left)")
    
    damage_numbers = [v for v in all_numbers if v[0] > 100]
    damage = max([v for v, _ in damage_numbers]) if damage_numbers else 0
    
    print(f"\nFinal damage: {damage:,}")
    
    # Translation resolution
    if name:
        print(f"\n--- TRANSLATION RESOLUTION ---")
        trans_store = load_translations_store(DATA_OUTPUT_DIR / "player_translations.json")
        
        k_normal = _normalize_key(name)
        k_fuzzy = _fuzzy_normalize_key(name)
        
        print(f"Original: '{name}'")
        print(f"Normalized key: '{k_normal}'")
        print(f"Fuzzy key: '{k_fuzzy}'")
        
        pid, cname, lang, matched_by, added = resolve_with_translation(trans_store, name)
        
        print(f"\nResolution:")
        print(f"  Player ID: {pid}")
        print(f"  Canonical name: {cname}")
        print(f"  Language: {lang}")
        print(f"  Matched by: {matched_by}")
        print(f"  New player: {added}")
    
    print(f"\n{'='*80}\n")

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_single_file.py <path_to_image>")
        print("Example: python test_single_file.py beartrap_data/2025-12-11/21.1.png")
        sys.exit(1)
    
    img_path = Path(sys.argv[1])
    if not img_path.exists():
        print(f"[ERROR] File not found: {img_path}")
        sys.exit(1)
    
    print("[INFO] Initializing OCR reader...")
    reader = init_reader()
    
    analyze_single_row_image(reader, img_path)

if __name__ == "__main__":
    main()
