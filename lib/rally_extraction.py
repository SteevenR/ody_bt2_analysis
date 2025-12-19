"""Rally extraction - complete participant extraction from multi-page screenshots"""
import cv2
import json
import re
from pathlib import Path
from .image_processing import preprocess_for_ocr, calibrate_line_height_from_dot_zero, calibrate_for_total_two, find_first_rank_ocr
from .text_utils import clean_extracted_name, normalize_damage
from .config import RANK_TEMPLATES, DATA_OUTPUT_DIR
from .ocr_rank_extraction import extract_rank_from_tokens
from .segment_new import segment_lines_dynamic

# Import flag detection from root
import sys
BASE_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(BASE_DIR))
from flag_detection import detect_flag_in_row


def segment_lines_by_rank_positions(img_up, line_height_px, rank_y_positions: dict, img_width, page_stem: str, reader=None, is_totals=False, debug_mode=False, event_id="", rally_id=0):
    """Segment image into lines based on rank positions."""
    h, w = img_up.shape[:2]
    half_height = int(line_height_px / 2)
    lines = []
    
    # Debug: save segmentation visualization
    if debug_mode:
        debug_dir = DATA_OUTPUT_DIR / "debug_segments"
        debug_dir.mkdir(parents=True, exist_ok=True)
    
    if page_stem.endswith(".0"):
        rank_1_y = rank_y_positions.get(1, 0)
        rank_y_positions[4] = rank_y_positions.get(3, 0) + line_height_px
        for rank in [1, 2, 3, 4]:
            y_center = rank_y_positions.get(rank, rank_1_y + (rank - 1) * line_height_px)
            y1 = max(0, int(y_center - half_height))
            y2 = min(h, int(y_center + half_height))
            row_img = img_up[y1:y2, 0:w]
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
                # Debug: save segment for .0 page too
                if debug_mode:
                    debug_dir = DATA_OUTPUT_DIR / "debug_segments"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    seg_file = debug_dir / f"{event_id}_rally{rally_id}_{page_stem}_rank{rank:02d}.png"
                    # Upscale small segments for readability
                    seg = row_img
                    min_h = 256
                    scale_dbg = 1 if seg.shape[0] >= min_h else max(2, int(round(min_h / max(1, seg.shape[0]))))
                    if scale_dbg > 1:
                        seg = cv2.resize(seg, (seg.shape[1] * scale_dbg, seg.shape[0] * scale_dbg), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(seg_file), seg)
    elif page_stem.endswith(".2"):
        # Totals .2 uses pre-calculated ranks (8-10). Rallies .2 should start at rank 12 via OCR.
        if is_totals:
            for rank in sorted(rank_y_positions.keys()):
                y_center = rank_y_positions[rank]
                y1 = max(0, int(y_center - half_height))
                y2 = min(h, int(y_center + half_height))
                row_img = img_up[y1:y2, 0:w]
                if row_img.size > 0:
                    lines.append((y1, y2, row_img, rank))
        else:
            start_rank = 12
            num_ranks = 4
            # For small cropped pages (.2), derive a dynamic line height from image height
            dynamic_lh = max(1, img_up.shape[0] / num_ranks)
            current_y = 0
            for idx in range(num_ranks):
                rank = start_rank + idx
                y1 = max(0, int(current_y))
                y2 = min(h, int(current_y + dynamic_lh))
                row_img = img_up[y1:y2, 0:w]
                if row_img.size > 0:
                    lines.append((y1, y2, row_img, rank))
                    # Debug: save segment (upscale small segments)
                    if debug_mode:
                        debug_dir = DATA_OUTPUT_DIR / "debug_segments"
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        seg_file = debug_dir / f"{event_id}_rally{rally_id}_{page_stem}_rank{rank:02d}.png"
                        seg = row_img
                        min_h = 256
                        scale_dbg = 1 if seg.shape[0] >= min_h else max(2, int(round(min_h / max(1, seg.shape[0]))))
                        if scale_dbg > 1:
                            seg = cv2.resize(seg, (seg.shape[1] * scale_dbg, seg.shape[0] * scale_dbg), interpolation=cv2.INTER_CUBIC)
                        cv2.imwrite(str(seg_file), seg)
                current_y += dynamic_lh
    else:
        if page_stem.endswith(".1"):
            start_rank = 5
            num_ranks = 7 if is_totals else 7
        else:
            start_rank = 1
            num_ranks = 4
        first_rank_y = None
        if reader:
            first_rank_y_val, _ = find_first_rank_ocr(reader, img_up, line_height_px, start_rank)
            first_rank_y = first_rank_y_val
        current_y = first_rank_y - half_height if first_rank_y is not None else 0
        
        if debug_mode and page_stem.endswith(".1"):
            print(f"[DEBUG] .1 segmentation: first_rank_y={first_rank_y}, current_y={current_y}, line_height={line_height_px}, num_ranks={num_ranks}")
        
        for idx in range(num_ranks):
            rank = start_rank + idx
            y1 = max(0, int(current_y))
            y2 = min(h, int(current_y + line_height_px))
            
            if debug_mode and page_stem.endswith(".1"):
                print(f"[DEBUG]   rank {rank}: y1={y1}, y2={y2}, height={y2-y1}")
            
            row_img = img_up[y1:y2, 0:w]
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
                # Debug: save segment
                if debug_mode:
                    debug_dir = DATA_OUTPUT_DIR / "debug_segments"
                    debug_dir.mkdir(parents=True, exist_ok=True)
                    seg_file = debug_dir / f"{event_id}_rally{rally_id}_{page_stem}_rank{rank:02d}.png"
                    seg = row_img
                    min_h = 256
                    scale_dbg = 1 if seg.shape[0] >= min_h else max(2, int(round(min_h / max(1, seg.shape[0]))))
                    if scale_dbg > 1:
                        seg = cv2.resize(seg, (seg.shape[1] * scale_dbg, seg.shape[0] * scale_dbg), interpolation=cv2.INTER_CUBIC)
                    cv2.imwrite(str(seg_file), seg)
            current_y += line_height_px
    
    # Debug: save all segments summary
    if debug_mode and (page_stem.endswith(".2") or page_stem.endswith(".0")):
        print(f"[DEBUG] {page_stem}: saved {len(lines)} segment images to {debug_dir}")
        for idx, (y1, y2, _, rank) in enumerate(lines):
            print(f"  - rank {rank}: y1={y1}, y2={y2}, height={y2-y1}")
    
    return lines


def extract_rally_participants(reader, page_files: list[Path], ocr_logger=None, event_id="", rally_id=0, debug_mode=False):
    """Extract participants from rally pages (rally or totals).
    
    Args:
        reader: EasyOCR reader instance
        page_files: List of image files (.0, .1, .2)
        ocr_logger: Optional OCR logger
        event_id: Event identifier
        rally_id: Rally number (-1 for totals)
        debug_mode: Enable verbose output
    
    Returns:
        List of participant dicts with name, damage, rank, has_flag, etc.
    """
    dot_zero_file = None
    dot_one_file = None
    dot_two_file = None
    
    for f in page_files:
        if f.stem.endswith(".0"):
            dot_zero_file = f
        elif f.stem.endswith(".1"):
            dot_one_file = f
        elif f.stem.endswith(".2"):
            dot_two_file = f

    is_totals = rally_id == -1

    # Calibration: rallies prefer .0; totals prefer .1
    line_height = None
    rank_x_col = None
    rank_y_positions = None

    if not is_totals and dot_zero_file is not None:
        line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_zero_file, RANK_TEMPLATES)
        if line_height is None and dot_one_file is not None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for {dot_zero_file.name}; falling back to {dot_one_file.name}")
            line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_one_file, RANK_TEMPLATES)
        if line_height is None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for both .0 and .1 - aborting rally parsing")
            return []
        if debug_mode:
            src = dot_zero_file.name if dot_zero_file else dot_one_file.name
            print(f"[DEBUG] Calibration OK from {src}: line_height={line_height:.1f}, positions={rank_y_positions}")
    elif dot_one_file is not None:
        line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_one_file, RANK_TEMPLATES)
        if line_height is None and dot_zero_file is not None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for {dot_one_file.name}; falling back to {dot_zero_file.name}")
            line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_zero_file, RANK_TEMPLATES)
        if line_height is None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for both .1 and .0 - aborting totals parsing")
            return []
        if debug_mode:
            src = dot_one_file.name if dot_one_file else dot_zero_file.name
            print(f"[DEBUG] Calibration OK from {src}: line_height={line_height:.1f}, positions={rank_y_positions}")
    else:
        if debug_mode:
            print(f"[DEBUG] No .0 or .1 file found")
        return []
    
    participants = []
    flag_metrics_log = DATA_OUTPUT_DIR / "flag_metrics.jsonl"
    
    for page_idx, f in enumerate(sorted(page_files)):
        # For totals (.1/.2), skip .0; for rallies, process .0/.1/.2
        if is_totals:
            if f.stem.endswith(".0"):
                if debug_mode:
                    print(f"[DEBUG] Skipping {f.stem} (no player data in .0)")
                continue
            if not (f.stem.endswith(".1") or f.stem.endswith(".2")):
                continue
        else:
            if not (f.stem.endswith(".0") or f.stem.endswith(".1") or f.stem.endswith(".2")):
                continue
        
        # For totals .2, adapt the calibration
        lh = line_height
        ry = rank_y_positions
        if is_totals and f.stem.endswith(".2"):
            rx_adapted, ry_adapted = calibrate_for_total_two(reader, f, line_height, debug_mode=debug_mode)
            if ry_adapted is not None:
                ry = ry_adapted
                if debug_mode:
                    print(f"[DEBUG] Adapted calibration for .2: rank_8_y={ry[8]:.1f}")
            else:
                if debug_mode:
                    print(f"[DEBUG] Failed to adapt calibration for .2")
                continue
        
        img = cv2.imread(str(f))
        if img is None:
            continue
        
        img_up, bin_img, scale = preprocess_for_ocr(img)
        img_up_color = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        
        if debug_mode:
            print(f"[DEBUG] {f.stem}: Original height={img.shape[0]}, Upsampled height={img_up.shape[0]}, scale={scale}")
        
        lh_scaled = lh * scale
        scaled_rank_y = {r: y * scale for r, y in ry.items()}
        
        # Use new dynamic segmentation: segment all lines until end of image
        lines = segment_lines_dynamic(img_up, lh_scaled, scaled_rank_y, img_up.shape[1], f.stem, reader, is_totals=is_totals, debug_mode=debug_mode, event_id=event_id, rally_id=rally_id)
        
        # Note: no more fixed expected counts; we segment all available lines
        # Don't truncate, OCR will determine which ones are valid later
        
        processed_rows = 0
        for y1, y2, row_img, detected_rank in lines:
            row_color_img = img_up_color[y1:y2, 0:img_up.shape[1]]
            row_gray = cv2.cvtColor(row_color_img, cv2.COLOR_BGR2GRAY)
            row_texts_detail = reader.readtext(row_gray, detail=1)
            
            if debug_mode:
                print(f"[DEBUG-RAW] {f.stem} rank {detected_rank}: {len(row_texts_detail)} tokens found")
                for box, text, conf in row_texts_detail:
                    xs = [p[0] for p in box]
                    print(f"  - '{text}' (conf={conf:.2f}) x_min={min(xs):.0f}")
            
            tokens = []
            for box, text, conf in row_texts_detail:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                tokens.append({"text": text.strip(), "conf": conf, "x": min(xs), "y": sum(ys) / len(ys)})
            
            # Extract actual rank from OCR (not from segment position)
            actual_rank = extract_rank_from_tokens(tokens)
            if actual_rank is None:
                actual_rank = detected_rank  # Fallback to detected rank if OCR extraction fails
            
            name_candidates = []
            for t in tokens:
                text = t["text"].strip()
                if not text:
                    continue
                if re.fullmatch(r"\d[\d\s\.,kKmM]*", text):
                    continue
                if re.search(r"(points\s+de|de\s+d[ée]g[âa]ts|rapport\s+de|d[ée]g[âa]ts|ints|oi)", text, re.IGNORECASE):
                    continue
                if t["conf"] < 0.4:
                    continue
                name_candidates.append(t)
            
            name_candidates.sort(key=lambda t: t["x"])
            name = clean_extracted_name(" ".join([t["text"] for t in name_candidates]).strip())
            
            if name and re.search(r"rapport\s*(de|da)?\s*combat", name, re.IGNORECASE):
                name = ""
            
            all_numbers = []
            for tok in tokens:
                nums = re.findall(r"\d[\d\s\.,kKmM]*", tok["text"])
                for num_str in nums:
                    val = normalize_damage(num_str)
                    if val > 100 or (0 < val <= 20 and tok["x"] < row_img.shape[1] * 0.15):
                        all_numbers.append((val, num_str))
            
            damage_numbers = [v for v in all_numbers if v[0] > 100]
            damage = max([v for v, _ in damage_numbers]) if damage_numbers else 0
            
            if debug_mode and (name or damage > 0):
                print(f"[DEBUG-OCR] {f.stem} rank {actual_rank}: name='{name}' damage={damage:,}")
            
            # Save debug segment with actual OCR-extracted rank
            if debug_mode:
                debug_dir = DATA_OUTPUT_DIR / "debug_segments"
                debug_dir.mkdir(parents=True, exist_ok=True)
                seg_file = debug_dir / f"{event_id}_rally{rally_id}_{f.stem}_rank{actual_rank:02d}.png"
                # Upscale small segments for readability
                seg = row_img
                min_h = 256
                scale_dbg = 1 if seg.shape[0] >= min_h else max(2, int(round(min_h / max(1, seg.shape[0]))))
                if scale_dbg > 1:
                    seg = cv2.resize(seg, (seg.shape[1] * scale_dbg, seg.shape[0] * scale_dbg), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(str(seg_file), seg)
            
            # Flag detection
            flag_result = detect_flag_in_row(row_color_img, tokens=tokens, verbose=debug_mode)
            has_flag = bool(flag_result.get("has_flag", False))
            flag_conf = float(flag_result.get("final_score", flag_result.get("shape_score", 0.0)))
            
            if debug_mode:
                try:
                    metrics_entry = {
                        "event": event_id,
                        "rally": rally_id,
                        "page": f.stem,
                        "file": f.name,
                        "rank": int(actual_rank),
                        "name": name,
                        "shape_score": float(flag_result.get("shape_score", 0.0)),
                        "green_score": float(flag_result.get("green_score", 0.0)),
                        "white_score": float(flag_result.get("white_score", 0.0)),
                        "final_score": float(flag_result.get("final_score", 0.0)),
                        "has_flag": bool(has_flag),
                    }
                    with open(flag_metrics_log, "a", encoding="utf-8") as fm:
                        fm.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            
            raw_ocr = " ".join([t["text"] for t in tokens])
            if ocr_logger:
                ocr_logger.log_extraction(event_id=event_id, rally_id=rally_id, filename=f.name,
                                          rank=actual_rank, raw_ocr_tokens=raw_ocr, extracted_name=name,
                                          damage=damage, all_tokens=tokens)
            
            # For .2 pages, allow extraction even without damage if name is found (they may be on separate lines)
            # For other pages, require both name and damage
            should_extract = False
            if f.stem.endswith(".2") and not is_totals:
                should_extract = (name or damage > 0)
            else:
                should_extract = (name and damage > 0)
            
            if should_extract:
                participants.append({
                    "name": name,
                    "damage": damage,
                    "rank": actual_rank,
                    "raw_line": raw_ocr if all_numbers else "",
                    "source_file": str(f),
                    "page_number": page_idx,
                    "has_flag": has_flag,
                    "final_score": flag_conf
                })
            processed_rows += 1

        # No more fallback needed with dynamic segmentation
        # All lines are now segmented dynamically based on calibrated line height
    
    return participants
