import json
import re
import unicodedata
import time
import os
from pathlib import Path
from multiprocessing import Pool, cpu_count
import easyocr
import cv2
import numpy as np
import hashlib
from flag_detection import detect_flag_in_row

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ============================
# CONFIG
# ============================

BASE_DIR = Path(__file__).parent
DATA_INPUT_DIR = BASE_DIR / "beartrap_data"
DATA_OUTPUT_DIR = BASE_DIR / "data"
DATA_OUTPUT_FILE = DATA_OUTPUT_DIR / "beartrap.json"
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"
ALIASES_FILE = DATA_OUTPUT_DIR / "player_aliases.json"
OCR_LOG_FILE = DATA_OUTPUT_DIR / "ocr_extraction_log.jsonl"
DAMAGE_AGG_LOG_FILE = DATA_OUTPUT_DIR / "damage_aggregation_log.jsonl"

DATA_OUTPUT_DIR.mkdir(exist_ok=True)
RANK_TEMPLATES = {
    1: BASE_DIR / "assets" / "rank1.png",
    2: BASE_DIR / "assets" / "rank2.png",
    3: BASE_DIR / "assets" / "rank3.png",
}

# Multiprocessing config
NUM_WORKERS = 4
GPU_AVAILABLE = False
GPU_STATUS = "CPU-only"

# ============================
# GPU DETECTION & INITIALIZATION
# ============================

def init_gpu():
    """Detect and initialize GPU (ROCm for AMD)."""
    global GPU_AVAILABLE, GPU_STATUS
    if not TORCH_AVAILABLE:
        GPU_STATUS = "CPU-only (torch not installed)"
        print(f"[GPU] ✗ {GPU_STATUS}")
        return False
    try:
        cuda_ok = torch.cuda.is_available()
        torch_ver = getattr(getattr(torch, 'version', None), '__version__', None)
        torch_ver_str = f"torch {torch_ver}" if torch_ver else "torch (version unknown)"
        if cuda_ok:
            try:
                device_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            except Exception:
                device_names = ["Unknown GPU"]
            GPU_AVAILABLE = True
            GPU_STATUS = f"CUDA/ROCm enabled ({', '.join(device_names)}) | {torch_ver_str}"
            print(f"[GPU] ✓ {GPU_STATUS}")
            return True
        else:
            GPU_STATUS = f"CPU-only (no CUDA/ROCm detected) | {torch_ver_str}"
            print(f"[GPU] ✗ {GPU_STATUS}")
            return False
    except Exception as e:
        GPU_STATUS = f"CPU-only (error: {e})"
        print(f"[GPU] ✗ {GPU_STATUS}")
        return False


def init_reader_gpu():
    """Initialize EasyOCR Reader configured for GPU if available."""
    try:
        print(f"[OCR] Initializing EasyOCR Reader; gpu={GPU_AVAILABLE}")
        reader = easyocr.Reader(["fr", "en", "es", "ru", "ko"], gpu=GPU_AVAILABLE, verbose=False)
        print("[OCR] Reader ready")
        return reader
    except (ValueError, RuntimeError) as e:
        print(f"[OCR] Fallback to FR/EN: {e}")
        reader = easyocr.Reader(["fr", "en"], gpu=GPU_AVAILABLE, verbose=False)
        print("[OCR] Reader ready (fallback)")
        return reader

def init_reader():
    """Initialize OCR reader."""
    return init_reader_gpu()

# ============================
# DAMAGE AGGREGATION LOGGING
# ============================

class DamageAggregationLogger:
    """Logs detailed damage contributions per player for auditing."""
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.player_contributions = {}
    def add_contribution(self, player_id: str, canonical_name: str,
                         event_id: str, rally_id: int, filename: str,
                         damage: int, rank: int):
        if player_id not in self.player_contributions:
            self.player_contributions[player_id] = {
                "canonical_name": canonical_name,
                "contributions": []
            }
        self.player_contributions[player_id]["contributions"].append({
            "event": event_id,
            "rally_id": rally_id,
            "file": filename,
            "rank": rank,
            "damage": damage
        })
    def save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            for player_id, data in sorted(self.player_contributions.items()):
                damages = [c["damage"] for c in data["contributions"]]
                total_damage = sum(damages)
                entry = {
                    "player_id": player_id,
                    "canonical_name": data["canonical_name"],
                    "total_contributions": len(data["contributions"]),
                    "total_damage": total_damage,
                    "damage_breakdown": f"sum([{', '.join(str(d) for d in sorted(damages, reverse=True))}]) = {total_damage}",
                    "contributions": data["contributions"],
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] Damage aggregation log saved to {self.log_path} ({len(self.player_contributions)} players)")

# ============================
# OCR EXTRACTION LOGGING
# ============================

class OCRLogger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        self.entries = []
    def log_extraction(self, event_id: str, rally_id: int, filename: str,
                       rank: int, raw_ocr_tokens: str, extracted_name: str,
                       damage: int, all_tokens: list):
        entry = {
            "event": event_id,
            "rally": rally_id,
            "file": filename,
            "rank": rank,
            "raw_ocr": raw_ocr_tokens,
            "extracted_name": extracted_name,
            "damage": damage,
            "tokens": [{"text": t["text"], "x": int(t["x"]), "y": int(t["y"]), "conf": round(t["conf"], 2)} for t in all_tokens]
        }
        self.entries.append(entry)
    def save(self):
        with open(self.log_path, "w", encoding="utf-8") as f:
            for entry in self.entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        print(f"[INFO] OCR extraction logs saved to {self.log_path} ({len(self.entries)} extractions)")

# ============================
# ALIAS & TRANSLATIONS (helpers copied)
# ============================

def _normalize_key(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _fuzzy_normalize_key(name: str) -> str:
    """Normalize key with fuzzy character substitution for OCR errors."""
    if not name:
        return ""
    s = str(name).strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # Apply fuzzy replacements BEFORE removing special chars
    s = s.replace("]", "1").replace("|", "1").replace("o", "0")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def load_translations_store(path: Path) -> dict:
    SUPPORTED_LANGUAGES = ["fr", "en", "es", "zh", "ru", "ko"]
    PRIMARY_LANGUAGE = "fr"
    if not path.exists():
        return {
            "languages": {lang: lang for lang in SUPPORTED_LANGUAGES},
            "primary_language": PRIMARY_LANGUAGE,
            "players": {},
            "alias_to_id": {},
        }
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {
            "languages": {lang: lang for lang in SUPPORTED_LANGUAGES},
            "primary_language": PRIMARY_LANGUAGE,
            "players": {},
            "alias_to_id": {},
        }

def save_translations_store(path: Path, store: dict):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    tmp.replace(path)

def detect_language_from_text(text: str) -> str:
    if not text:
        return "en"
    t = text[:50]
    if re.search(r"[\u4E00-\u9FFF]", t):
        return "zh"
    if re.search(r"[\uAC00-\uD7AF]", t):
        return "ko"
    if re.search(r"[\u0400-\u04FF]", t):
        return "ru"
    return "en"

def add_player_translation(trans_store: dict, player_id: str, original_name: str, language: str):
    SUPPORTED_LANGUAGES = ["fr", "en", "es", "zh", "ru", "ko"]
    if player_id not in trans_store["players"]:
        trans_store["players"][player_id] = {
            "names_by_language": {lang: "" for lang in SUPPORTED_LANGUAGES},
            "transliterations_by_language": {lang: None for lang in SUPPORTED_LANGUAGES},
            "language_detected": language,
            "aliases": [original_name],
            "pending_review": True,
        }
    player = trans_store["players"][player_id]
    if not player["names_by_language"].get(language):
        player["names_by_language"][language] = original_name
    player["language_detected"] = language
    if original_name not in player.get("aliases", []):
        player["aliases"].append(original_name)

def get_name_for_language(trans_store: dict, player_id: str, language: str) -> str:
    player = trans_store["players"].get(player_id)
    if not player:
        return player_id
    names = player.get("names_by_language", {})
    if language in names and names[language]:
        return names[language]
    detected = player.get("language_detected", "en")
    if detected in names and names[detected]:
        return names[detected]
    if "fr" in names and names["fr"]:
        return names["fr"]
    for k, v in names.items():
        if v:
            return v
    return player_id

def resolve_with_translation(trans_store: dict, observed_name: str):
    if not observed_name:
        return None, None, None, "", False
    language = detect_language_from_text(observed_name)
    k = _normalize_key(observed_name)
    if k in trans_store.get("alias_to_id", {}):
        pid = trans_store["alias_to_id"][k]
        cname = get_name_for_language(trans_store, pid, language)
        return pid, cname, language, "alias", False
    # Try exact match first
    for pid, player in trans_store.get("players", {}).items():
        for v in player.get("names_by_language", {}).values():
            if v and (observed_name == v or _normalize_key(v) == k):
                trans_store.setdefault("alias_to_id", {})[k] = pid
                if observed_name not in player.get("aliases", []):
                    player.setdefault("aliases", []).append(observed_name)
                cname = get_name_for_language(trans_store, pid, language)
                return pid, cname, language, "exact", False
    # Fuzzy match for OCR errors (e.g., ] vs 1, O vs 0)
    k_fuzzy = _fuzzy_normalize_key(observed_name)
    if k_fuzzy != k and k_fuzzy in trans_store.get("alias_to_id", {}):
        pid = trans_store["alias_to_id"][k_fuzzy]
        trans_store.setdefault("alias_to_id", {})[k] = pid
        cname = get_name_for_language(trans_store, pid, language)
        return pid, cname, language, "fuzzy", False
    # Create new player
    pid = f"pl_{k.replace(' ', '_')}" if k else f"pl_{abs(hash(observed_name))}"
    add_player_translation(trans_store, pid, observed_name, language)
    trans_store.setdefault("alias_to_id", {})[k] = pid
    cname = get_name_for_language(trans_store, pid, language)
    return pid, cname, language, "auto", True

# ============================
# OCR / SEGMENTATION HELPERS (subset from main script)
# ============================

def clean_extracted_name(name: str) -> str:
    if not name:
        return name
    name = re.sub(r"^[Rr]\s+", "", name)
    for pat in [r"points\s+de\s+d[ée]g[âa]ts", r"rapport\s+de\s+combat", r"d[ée]g[âa]ts"]:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[:;.,]+$", "", name)
    return re.sub(r"\s+", " ", name).strip()

def normalize_damage(value_str: str) -> int:
    if not value_str:
        return 0
    s = str(value_str).strip()
    s = re.sub(r"Points de D[eé]g[âa]ts\s*[:\s]*", "", s, flags=re.IGNORECASE)
    s = s.replace(" ", "").replace(",", ".")
    m = re.search(r"(\d[\d\s\.,]*[km]?)", s, re.IGNORECASE)
    if not m:
        return 0
    ds = m.group(1)
    suf = re.search(r"([km])$", ds, re.IGNORECASE)
    if suf:
        ds = ds[:-1]
    try:
        n = float(ds)
    except ValueError:
        return 0
    if suf and suf.group(1).lower() == "m":
        n *= 1_000_000
    elif suf and suf.group(1).lower() == "k":
        n *= 1_000
    return int(n)

def ocr_image_lines(reader, image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Unreadable image: {image_path}")
        return []
    try:
        return reader.readtext(img, detail=0)
    except Exception as e:
        print(f"[WARN] OCR error on {image_path}: {e}")
        return []

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scale = 2
    gray_up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    th = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    th = cv2.medianBlur(th, 3)
    return gray_up, th, scale

# Abridged calibration/segmentation functions from existing script
# (Using same logic to keep output compatibility)

def detect_rank_icons_by_template(image_path: Path, rank_templates: dict[int, Path]):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[DEBUG] Cannot read image: {image_path}")
        return {}
    
    # Check which templates exist
    existing_templates = {r: p for r, p in rank_templates.items() if p.exists()}
    if not existing_templates:
        print(f"[DEBUG] No rank templates found in {list(rank_templates.values())}")
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

def calibrate_line_height_from_dot_zero(dot_zero_file: Path):
    detected_ranks = detect_rank_icons_by_template(dot_zero_file, RANK_TEMPLATES)
    if len(detected_ranks) < 2:
        print(f"[DEBUG] Template matching failed: detected_ranks={detected_ranks} (need at least 2)")
        return None, None, None
    sorted_ranks = sorted(detected_ranks.items())
    diffs = []
    for i in range(1, len(sorted_ranks)):
        dy = sorted_ranks[i][1][0] - sorted_ranks[i-1][1][0]
        if dy > 5:
            diffs.append(dy)
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

def calibrate_for_total_two(reader, total_two_file: Path, line_height_from_one: float, debug_mode=False) -> tuple:
    """For total.2, find the position of rank 8 and extrapolate for ranks 9, 10.
    Returns positions in ORIGINAL pixel coordinates (before upsampling)."""
    img = cv2.imread(str(total_two_file))
    if img is None:
        return None, None, None
    
    if debug_mode:
        print(f"[DEBUG] {total_two_file.stem}: image height={img.shape[0]}, width={img.shape[1]}")
    
    img_up, _, scale = preprocess_for_ocr(img)
    h, w = img_up.shape[:2]
    
    if debug_mode:
        print(f"[DEBUG] {total_two_file.stem}: upsampled height={h}, scale={scale}")
        print(f"[DEBUG] {total_two_file.stem}: line_height_from_one={line_height_from_one:.1f}")
    
    # Scan top-left area for rank number "8"
    scan_height = int(line_height_from_one * scale * 2)
    scan_width = int(w * 0.2)
    scan_img = img_up[:scan_height, :scan_width]
    
    boxes = reader.readtext(scan_img, detail=1)
    rank8_y = None
    for box, text, conf in boxes:
        t = text.strip()
        if t == "8" and conf >= 0.5:
            ys = [p[1] for p in box]
            rank8_y = sum(ys) / len(ys)
            break
    
    if rank8_y is None:
        if debug_mode:
            print(f"[DEBUG] {total_two_file.stem}: Could not find rank 8")
        return None, None, None
    
    # Convert rank8_y back to original pixel coordinates
    rank8_y_orig = rank8_y / scale
    
    if debug_mode:
        print(f"[DEBUG] {total_two_file.stem}: Found rank 8 at y={rank8_y:.1f} (upsampled), {rank8_y_orig:.1f} (original)")
    
    # Extrapolate positions for ranks 9 and 10 using line height (in original coords)
    rank_y_positions = {
        8: rank8_y_orig,
        9: rank8_y_orig + line_height_from_one,
        10: rank8_y_orig + 2 * line_height_from_one,
    }
    
    if debug_mode:
        print(f"[DEBUG] {total_two_file.stem}: Extrapolated positions (orig): 8={rank8_y_orig:.1f}, 9={rank8_y_orig + line_height_from_one:.1f}, 10={rank8_y_orig + 2*line_height_from_one:.1f}")
    
    rank_x_col = scan_width / 2 / scale  # Convert back to original coords
    return line_height_from_one, rank_x_col, rank_y_positions

def find_first_rank_ocr(reader, img_up, line_height_px: float, expected_first_rank: int):
    h, w = img_up.shape[:2]
    scan_height = int(line_height_px * 2)
    scan_img = img_up[:scan_height, :int(w * 0.2)]
    boxes = reader.readtext(scan_img, detail=1)
    for box, text, conf in boxes:
        t = text.strip()
        if re.fullmatch(r"\d{1,2}", t) and conf >= 0.5:
            try:
                rank = int(t)
                if rank == expected_first_rank:
                    ys = [p[1] for p in box]
                    return sum(ys) / len(ys)
            except Exception:
                pass
    return None

def segment_lines_by_rank_positions(img_up, line_height_px, rank_y_positions: dict, img_width, page_stem: str, reader=None):
    h, w = img_up.shape[:2]
    half_height = int(line_height_px / 2)
    lines = []
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
    elif page_stem.endswith(".2"):
        # For total.2: use directly the pre-calculated rank_y_positions
        for rank in sorted(rank_y_positions.keys()):
            y_center = rank_y_positions[rank]
            y1 = max(0, int(y_center - half_height))
            y2 = min(h, int(y_center + half_height))
            row_img = img_up[y1:y2, 0:w]
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
    else:
        # For total.1 and others: find first rank via OCR then continue
        if page_stem.endswith(".1"):
            start_rank = 5
            num_ranks = 7
        else:
            start_rank = 1
            num_ranks = 4
        first_rank_y = None
        if reader:
            first_rank_y = find_first_rank_ocr(reader, img_up, line_height_px, start_rank)
        current_y = first_rank_y - half_height if first_rank_y is not None else 0
        for idx in range(num_ranks):
            rank = start_rank + idx
            y1 = max(0, int(current_y))
            y2 = min(h, int(current_y + line_height_px))
            row_img = img_up[y1:y2, 0:w]
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
            current_y += line_height_px
    return lines

def extract_rally_participants_from_pages_v2(reader, page_files: list[Path], ocr_logger=None, event_id="", rally_id=0, debug_mode=False):
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
        line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_zero_file)
        if line_height is None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for {dot_zero_file.name} - line_height is None")
            return []
        if debug_mode:
            print(f"[DEBUG] Calibration OK from .0: line_height={line_height:.1f}, positions={rank_y_positions}")
    elif dot_one_file is not None:
        line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_one_file)
        if line_height is None:
            if debug_mode:
                print(f"[DEBUG] Calibration failed for {dot_one_file.name} - line_height is None")
            return []
        if debug_mode:
            print(f"[DEBUG] Calibration OK from .1: line_height={line_height:.1f}, positions={rank_y_positions}")
    elif dot_zero_file is None:
        # No .0 and no .1 - nothing to process
        if debug_mode:
            print(f"[DEBUG] No .0 or .1 file found")
        return []
    else:
        # Only .0 file and totals mode: nothing to extract
        if debug_mode:
            print(f"[DEBUG] Only .0 file found - skipping (no player data in total.0)")
        return []
    # Debug directories disabled to avoid file creation during pipeline
    debug_dir = None
    debug_roi_dir = None
    participants = []
    # Prepare optional flag metrics log for cross-checking with tests
    flag_metrics_log = DATA_OUTPUT_DIR / "flag_metrics.jsonl"
    
    for f in sorted(page_files):
        # For totals (.1/.2), skip .0; for rallies, process .0/.1/.2
        if is_totals:
            if f.stem.endswith(".0"):
                if debug_mode:
                    print(f"[DEBUG] Skipping {f.stem} (no player data in .0)")
                continue
            if not (f.stem.endswith(".1") or f.stem.endswith(".2")):
                continue
        else:
            # Rally pages: accept .0/.1/.2
            if not (f.stem.endswith(".0") or f.stem.endswith(".1") or f.stem.endswith(".2")):
                continue
        
        # For totals .2, adapt the calibration to use ranks 8-10; for rallies, keep same calibration
        lh = line_height
        ry = rank_y_positions
        if is_totals and f.stem.endswith(".2"):
            lh_adapted, rx_adapted, ry_adapted = calibrate_for_total_two(reader, f, line_height, debug_mode=debug_mode)
            if ry_adapted is not None:
                lh = lh_adapted
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
        # Keep a color-upsampled version for color-based checks
        img_up_color = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        page_debug_dir = None
        
        if debug_mode:
            print(f"[DEBUG] {f.stem}: Original image height={img.shape[0]}, Upsampled height={img_up.shape[0]}, scale={scale}")
            print(f"[DEBUG] {f.stem}: Using line_height={lh:.1f}, rank_y_positions={ry}")
        
        # Use calibration-based segmentation for .1 and .2
        lh_scaled = lh * scale
        scaled_rank_y = {r: y * scale for r, y in ry.items()}
        
        if debug_mode:
            print(f"[DEBUG] {f.stem}: Scaled line_height={lh_scaled:.1f}, scaled_rank_y={scaled_rank_y}")
        
        lines = segment_lines_by_rank_positions(img_up, lh_scaled, scaled_rank_y, img_up.shape[1], f.stem, reader)
        
        expected_map = {".0": 4, ".1": 7, ".2": 3}
        expected = None
        for suffix, cnt in expected_map.items():
            if f.stem.endswith(suffix):
                expected = cnt
                break
        
        if debug_mode:
            print(f"[DEBUG] {f.stem}: segmented {len(lines)} lines (expected {expected})")
        
        if expected is not None and len(lines) > expected:
            lines = lines[:expected]
        for y1, y2, row_img, detected_rank in lines:
            # Corresponding color row from color-upsampled page
            row_color_img = img_up_color[y1:y2, 0:img_up.shape[1]]
            # Ensure row grayscale is derived from color row to keep alignment
            row_gray = cv2.cvtColor(row_color_img, cv2.COLOR_BGR2GRAY)
            row_texts_detail = reader.readtext(row_gray, detail=1)
            
            # Debug: print raw OCR tokens
            if debug_mode:
                print(f"[DEBUG-RAW] {f.stem} rank {detected_rank}: {len(row_texts_detail)} tokens found")
                for box, text, conf in row_texts_detail:
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    x_min, x_max = min(xs), max(xs)
                    y_min, y_max = min(ys), max(ys)
                    print(f"  - '{text}' (conf={conf:.2f}) box=({x_min:.0f},{y_min:.0f})-({x_max:.0f},{y_max:.0f})")
            
            tokens = []
            for box, text, conf in row_texts_detail:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                tokens.append({"text": text.strip(), "conf": conf, "x": min(xs), "y": sum(ys) / len(ys)})
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
            # Filter out UI text that's not a player name
            if name and re.search(r"rapport\s*(de|da)?\s*combat", name, re.IGNORECASE):
                name = ""
            all_numbers = []
            for tok in tokens:
                nums = re.findall(r"\d[\d\s\.,kKmM]*", tok["text"])
                for num_str in nums:
                    val = normalize_damage(num_str)
                    # Only keep numbers > 100 (real damage) or rank numbers (< 20) on left side
                    if val > 100 or (0 < val <= 20 and tok["x"] < row_img.shape[1] * 0.15):
                        all_numbers.append((val, num_str))
            # Filter out rank numbers from damage candidates
            damage_numbers = [v for v in all_numbers if v[0] > 100]
            damage = max([v for v, _ in damage_numbers]) if damage_numbers else 0
            
            # Debug logging for each row
            if debug_mode and (name or damage > 0):
                print(f"[DEBUG-OCR] {f.stem} rank {detected_rank}: name='{name}' damage={damage}")
            
            # Skip writing debug row or ROI images
            roi_debug_path = None
            
            # Check for leader flag in this row using shared detection module
            has_flag = False
            flag_conf = 0.0
            # Prepare debug paths for hit, avatar, and TL50
            # No debug directories or image outputs
            hit_debug_path = None
            av_debug_path = None
            tl_debug_path = None
            flag_result = detect_flag_in_row(row_color_img, tokens=tokens, save_roi_to=roi_debug_path, save_hit_to=hit_debug_path, save_avatar_to=av_debug_path, save_tl50_to=tl_debug_path, verbose=debug_mode)
            has_flag = bool(flag_result.get("has_flag", False))
            # Use TL50 scoring fields; prefer final_score then shape_score
            flag_conf = float(flag_result.get("final_score", flag_result.get("shape_score", 0.0)))
            
            # Write flag metrics only in debug mode
            if debug_mode:
                try:
                    metrics_entry = {
                        "event": event_id,
                        "rally": rally_id,
                        "page": f.stem,
                        "file": f.name,
                        "rank": int(detected_rank),
                        "name": name,
                        "shape_score": float(flag_result.get("shape_score", 0.0)),
                        "green_score": float(flag_result.get("green_score", 0.0)),
                        "white_score": float(flag_result.get("white_score", 0.0)),
                        "final_score": float(flag_result.get("final_score", 0.0)),
                        "has_flag": bool(has_flag),
                        "tl50": flag_result.get("tl50")
                    }
                    with open(flag_metrics_log, "a", encoding="utf-8") as fm:
                        fm.write(json.dumps(metrics_entry, ensure_ascii=False) + "\n")
                except Exception:
                    pass
            
            raw_ocr = " ".join([t["text"] for t in tokens])
            if ocr_logger:
                ocr_logger.log_extraction(event_id=event_id, rally_id=rally_id, filename=f.name,
                                          rank=detected_rank, raw_ocr_tokens=raw_ocr, extracted_name=name,
                                          damage=damage, all_tokens=tokens)
            if name and damage > 0:
                participants.append({
                    "name": name,
                    "damage": damage,
                    "rank": detected_rank,
                    "raw_line": raw_ocr if all_numbers else "",
                    "source_file": f.name,
                    "has_flag": has_flag,
                    "final_score": flag_conf
                })
    return participants

def find_leader_from_flag(reader, image_path: Path, flag_template_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(str(flag_template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        return None
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.6:
        return None
    x_flag, y_flag = max_loc
    try:
        ocr_data = reader.readtext(str(image_path))
    except Exception as e:
        print(f"[WARN] OCR error finding leader: {e}")
        return None
    best_match = None
    best_dist = float("inf")
    for box, text, conf in ocr_data:
        # Filter out UI text and low confidence
        if conf < 0.5:
            continue
        if re.search(r"(rapport|combat|points|dégâts|degats|piège|chasse|attaque|ours)", text, re.IGNORECASE):
            continue
        y_text = box[0][1]
        dist = abs(y_text - y_flag)
        if dist < best_dist:
            best_dist = dist
            best_match = text
    return best_match

def detect_leader_fallback(participants):
    """Fallback: leader is rank 1 player."""
    if not participants:
        return None
    # Sort by rank to ensure rank 1 is first
    sorted_participants = sorted(participants, key=lambda p: p.get("rank", 999))
    return sorted_participants[0]["name"]

# ============================
# MAIN (Parallel)
# ============================

def compute_event_source_hash(day_dir: Path) -> str:
    parts = []
    for f in sorted(day_dir.iterdir(), key=lambda p: p.name):
        if f.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue
        try:
            st = f.stat()
            parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
        except Exception:
            parts.append(f.name)
    digest = hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()
    return digest

def process_single_rally(args):
    rally_id, files, date_str, debug_mode = args
    print(f"[WORKER] PID={os.getpid()} starting rally {rally_id}; files={len(files)}")
    reader = init_reader()
    t0 = time.time()
    participants_raw = extract_rally_participants_from_pages_v2(reader, files, ocr_logger=None, event_id=date_str, rally_id=rally_id)
    t_ocr = time.time() - t0
    print(f"[WORKER] PID={os.getpid()} rally {rally_id} OCR done in {t_ocr:.2f}s; participants_raw={len(participants_raw) if participants_raw else 0}")
    if not participants_raw:
        print(f"[WORKER] PID={os.getpid()} rally {rally_id} no participants parsed")
        return None
    participants_unique = []
    seen = set()
    leader_name_obs = None
    flagged = []
    for p in participants_raw:
        key = (p["name"], p["damage"]) 
        if key in seen:
            continue
        seen.add(key)
        participants_unique.append(p)
        if p.get("has_flag", False):
            flagged.append(p)
    # Resolve leader with tie-breakers: close scores prefer earlier page and lower rank
    if flagged:
        # Primary: highest confidence (use new TL50 scores)
        def _flag_score(fp: dict) -> float:
            return float(fp.get("final_score", fp.get("shape_score", fp.get("flag_conf", 0.0))))
        best_flag_conf = max(_flag_score(fp) for fp in flagged)
        margin = 0.05
        eligible = [fp for fp in flagged if _flag_score(fp) >= best_flag_conf - margin]
        def page_index_of(src: str) -> tuple:
            # Extract like '1.0' -> (1,0); fall back high numbers if missing
            m = re.search(r"(\d+)\.(\d+)", src or "")
            if m:
                return (int(m.group(1)), int(m.group(2)))
            return (9999, 9999)
        eligible.sort(key=lambda fp: (page_index_of(str(fp.get("source_file", ""))), int(fp.get("rank", 9999))))
        leader_name_obs = eligible[0]["name"]
    
    if not leader_name_obs:
        print(f"[WARN] Rally {rally_id}: No leader flag detected - needs manual review")
    
    print(f"[WORKER] PID={os.getpid()} rally {rally_id} finished; unique participants={len(participants_unique)}; leader='{leader_name_obs}'")
    return {"rally_id": rally_id, "leader_name_obs": leader_name_obs, "participants": participants_unique, "files": [str(f.name) for f in files]}


def parse_top_players_from_total_pages(reader, total_pages: list[Path], trans_store: dict, event_id: str, ocr_logger=None, debug_mode=False):
    """Parse top players AND totals from total.0/1/2 pages using the rally extractor for consistency.
    
    Returns: (players_list, trans_added, alliance_damage, rally_count)
    where alliance_damage and rally_count are extracted from the total.0 page if available.
    """
    participants_raw = extract_rally_participants_from_pages_v2(
        reader, total_pages, ocr_logger=ocr_logger, event_id=event_id, rally_id=-1, debug_mode=debug_mode
    )
    if not participants_raw:
        return [], False, 0, 0

    seen = set()
    participants_unique = []
    for p in participants_raw:
        key = (p.get("name"), p.get("damage"))
        if key in seen:
            continue
        seen.add(key)
        participants_unique.append(p)

    resolved_players = []
    trans_added = False
    for p in sorted(participants_unique, key=lambda x: x.get("damage", 0), reverse=True):
        pid, cname, lang, matched_by, added = resolve_with_translation(trans_store, p.get("name"))
        if added:
            trans_added = True
        resolved_players.append({
            "id": pid,
            "name": cname,
            "name_original": p.get("name"),
            "matched_by": matched_by,
            "language_detected": lang,
            "damage": p.get("damage", 0),
            "rank": p.get("rank"),
        })

    # Extract totals from total.0 page if available
    alliance_damage = 0
    rally_count_total = 0
    
    dot_zero_file = None
    for f in total_pages:
        if f.stem.endswith(".0"):
            dot_zero_file = f
            break
    
    if dot_zero_file:
        # OCR the total.0 file to extract totals from header/footer
        total_lines = ocr_image_lines(reader, dot_zero_file)
        if debug_mode:
            print(f"[DEBUG] OCR'd {dot_zero_file.name}: {len(total_lines)} lines found")
        for i, line in enumerate(total_lines):
            line_lower = str(line).lower()
            if debug_mode and ("ralliement" in line_lower or "dégât" in line_lower):
                print(f"[DEBUG]   Line {i}: '{line}'")
            
            # Find rally count (usually near "ralliement" label)
            if "ralliement" in line_lower and i + 1 < len(total_lines):
                next_line = str(total_lines[i + 1])
                val = normalize_damage(next_line)
                if debug_mode:
                    print(f"[DEBUG]   Found 'ralliement', next line: '{next_line}' -> val={val}")
                if val > 0 and val <= 100:  # rally count should be reasonable (< 100)
                    rally_count_total = val
                    if debug_mode:
                        print(f"[DEBUG]   -> Accepted as rally_count_total={val}")
            
            # Find total damage - look for "alliance" specifically
            if alliance_damage == 0 and ("alliance" in line_lower or "dégâts totaux" in line_lower):
                # Try to extract from the current line itself
                val = normalize_damage(str(line))
                if val > 1000:
                    alliance_damage = val
                    if debug_mode:
                        print(f"[DEBUG]   Found alliance damage in same line with value {val}")
                # If not found in current line, try next line
                elif i + 1 < len(total_lines):
                    next_line = str(total_lines[i + 1])
                    val = normalize_damage(next_line)
                    if debug_mode:
                        print(f"[DEBUG]   Checking next line: '{next_line}' -> val={val}")
                    if val > 1000:  # alliance damage should be large
                        alliance_damage = val
                        if debug_mode:
                            print(f"[DEBUG]   -> Accepted as alliance_damage={val}")
    else:
        if debug_mode:
            print(f"[DEBUG] No .0 file found")

    return resolved_players, trans_added, alliance_damage, rally_count_total

def totals_only_mode(reader, trans_store: dict, debug_mode=False):
    """Refresh all totals and top players from total.* pages (skip rally OCR)."""
    print("[INFO] ===== TOTALS ONLY MODE (skip rallies) =====")
    if not DATA_OUTPUT_FILE.exists():
        print("[WARN] No existing beartrap.json found. Run full analysis first.")
        return
    
    with open(DATA_OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    totals_log = DATA_OUTPUT_DIR / "totals_log.jsonl"
    if totals_log.exists():
        totals_log.unlink()
    
    # Initialize OCR logger if in debug mode
    ocr_logger = None
    if debug_mode:
        ocr_logger = OCRLogger(OCR_LOG_FILE)
        print("[DEBUG] OCR logging ENABLED for totals extraction")
    else:
        print("[INFO] Debug mode OFF - no OCR logging")
    
    for event in data.get("events", []):
        event_date = event.get("id") or event.get("date", "")
        day_dir = DATA_INPUT_DIR / event_date
        if not day_dir.exists():
            print(f"[WARN] Directory not found: {day_dir}")
            continue
        
        # Collect total.0/1/2 files
        total_pages = [
            f for f in day_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            and re.match(r"total\.\d+$", f.stem)
        ]
        
        alliance_damage = 0
        rally_count_total = 0
        top_players_resolved = []
        
        if total_pages:
            # Use consolidated parser to get both totals and players
            top_players_resolved, _, dmg_from_total, rally_from_total = parse_top_players_from_total_pages(
                reader,
                sorted(total_pages, key=lambda p: p.stem),
                trans_store,
                event_id=event_date,
                ocr_logger=ocr_logger,
                debug_mode=debug_mode,
            )
            alliance_damage = dmg_from_total
            rally_count_total = rally_from_total
            
            if debug_mode:
                print(f"[DEBUG] {event_date}: Found {len(total_pages)} total.* files, extracted {len(top_players_resolved)} players")
            
            if debug_mode and top_players_resolved:
                for i, p in enumerate(top_players_resolved[:5], 1):
                    print(f"  [{i}] {p.get('name')} (id={p.get('id')}) - {p.get('damage')} damage")
        else:
            if debug_mode:
                print(f"[DEBUG] {event_date}: No total.* files found")
        
        # Fallback to total.png if totals not found
        if alliance_damage == 0 and rally_count_total == 0:
            total_file = day_dir / "total.png"
            if total_file.exists():
                total_lines = ocr_image_lines(reader, total_file)
                for i, line in enumerate(total_lines):
                    line_lower = str(line).lower()
                    if "ralliement" in line_lower and i + 1 < len(total_lines):
                        next_line = str(total_lines[i + 1])
                        val = normalize_damage(next_line)
                        if val > 0 and val <= 100:
                            rally_count_total = val
                    if "dégât" in line_lower and i + 1 < len(total_lines):
                        next_line = str(total_lines[i + 1])
                        val = normalize_damage(next_line)
                        if val > 1000:
                            alliance_damage = val
        
        # Update event
        event["alliance_total_damage"] = alliance_damage
        event["rally_count_total"] = rally_count_total
        
        if top_players_resolved:
            event["top_totals"] = {"players": top_players_resolved, "source": "total_pages"}
        
        # Log detected totals
        log_entry = {
            "event_date": event_date,
            "detected_alliance_damage": alliance_damage,
            "detected_rally_count": rally_count_total,
            "detected_top_players": [
                {"id": p.get("id"), "name": p.get("name"), "damage": p.get("damage")}
                for p in top_players_resolved[:10]
            ] if top_players_resolved else [],
        }
        with open(totals_log, "a", encoding="utf-8") as tl:
            tl.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
        
        print(f"[TOTALS] {event_date}: alliance_damage={alliance_damage}, rally_count={rally_count_total}, top_players={len(top_players_resolved)}")
    
    with open(DATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    if debug_mode and ocr_logger:
        ocr_logger.save()
    
    print("[INFO] Totals refreshed and saved.")
    print(f"[INFO] Totals log: {totals_log}")

def main(totals_only=False, debug_mode=False):
    
    start_total = time.time()
    print("[INFO] ===== BeartrapAnalysis PARALLEL (GPU-Optimized) =====")
    init_gpu()
    print(f"[INFO] GPU Status: {GPU_STATUS}")
    print(f"[INFO] Workers: {NUM_WORKERS}")

    reader_main = init_reader()
    
    # Load translations store for all modes
    trans_store = load_translations_store(DATA_OUTPUT_DIR / "player_translations.json")
    
    # Totals-only mode (refresh totals from total.* pages, skip rallies)
    if totals_only:
        totals_only_mode(reader_main, trans_store, debug_mode=debug_mode)
        return
    
    trans_modified = False
    ocr_logger = OCRLogger(OCR_LOG_FILE)
    dmg_logger = DamageAggregationLogger(DAMAGE_AGG_LOG_FILE)

    # Load existing data for incremental updates
    if DATA_OUTPUT_FILE.exists():
        with open(DATA_OUTPUT_FILE, "r", encoding="utf-8") as f:
            existing_json = json.load(f)
        existing_events = {ev["id"]: ev for ev in existing_json.get("events", [])}
    else:
        existing_json = {"events": []}
        existing_events = {}

    new_events = []

    for day_dir in sorted(DATA_INPUT_DIR.iterdir()):
        if not day_dir.is_dir():
            continue

        date_str = day_dir.name
        current_hash = compute_event_source_hash(day_dir)
        prev_event = existing_events.get(date_str)
        if prev_event is not None and prev_event.get("source_hash") == current_hash:
            print(f"[INFO] Event {date_str} unchanged (hash match), skipping.")
            continue

        # Initialize totals
        alliance_damage = 0
        rally_count_total = 0
        
        # Attempt to read totals and top players from total.0/1/2 pages (consolidated approach)
        total_pages = [
            f for f in day_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            and re.match(r"total\.\d+$", f.stem)
        ]
        top_players_resolved = []
        totals_from_pages = False
        
        if total_pages:
            # Consolidated parsing: get both totals AND players from total.0/1/2
            top_players_resolved, added, dmg_from_total, rally_from_total = parse_top_players_from_total_pages(
                reader_main,
                sorted(total_pages, key=lambda p: p.stem),
                trans_store,
                event_id=date_str,
            )
            if added:
                trans_modified = True
            
            # Use totals extracted from total.0 if they were found
            if dmg_from_total > 0:
                alliance_damage = dmg_from_total
                totals_from_pages = True
            if rally_from_total > 0:
                rally_count_total = rally_from_total
                totals_from_pages = True
        
        # Fallback to total.png only if totals not found in total.0/1/2
        if not totals_from_pages:
            total_file = day_dir / "total.png"
            if total_file.exists():
                total_lines = ocr_image_lines(reader_main, total_file)
                # total_lines should have label + value pairs
                # Look for patterns: "Ralliements" or "Dégâts Totaux" labels
                for i, line in enumerate(total_lines):
                    line_lower = str(line).lower()
                    # Find rally count (usually near "ralliement" label)
                    if "ralliement" in line_lower and i + 1 < len(total_lines):
                        next_line = str(total_lines[i + 1])
                        val = normalize_damage(next_line)
                        if val > 0 and val <= 100:
                            rally_count_total = val
                    # Find total damage (usually near "dégât" label)
                    if "dégât" in line_lower and i + 1 < len(total_lines):
                        next_line = str(total_lines[i + 1])
                        val = normalize_damage(next_line)
                        if val > 1000:
                            alliance_damage = val

        event = {
            "id": date_str,
            "date": date_str,
            "name": "Bear Trap",
            "alliance_total_damage": alliance_damage,
            "rally_count_total": rally_count_total,
            "rallies": [],
            "source_hash": current_hash,
        }

        if top_players_resolved:
            event["top_totals"] = {"players": top_players_resolved, "source": "total_pages"}

        rally_groups = {}
        for f in day_dir.iterdir():
            if f.name.lower() == "total.png":
                continue
            if f.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            m = re.match(r"(\d+)\.(\d+)", f.stem)
            if not m:
                continue
            rally_id = int(m.group(1))
            rally_groups.setdefault(rally_id, []).append(f)

        work_items = [(rally_id, sorted(files), date_str, debug_mode) for rally_id, files in sorted(rally_groups.items())]

        if work_items:
            with Pool(processes=NUM_WORKERS) as pool:
                results = pool.map(process_single_rally, work_items)
        else:
            results = []

        for res in results:
            if not res:
                continue
            rally_id = res["rally_id"]
            leader_name_obs = res["leader_name_obs"]
            participants_unique = res["participants"]
            leader_pid, leader_cname, leader_lang, leader_match, added = resolve_with_translation(trans_store, leader_name_obs)
            if added:
                trans_modified = True
            participants_resolved = []
            for p in participants_unique:
                pid, cname, lang, matched_by, added = resolve_with_translation(trans_store, p["name"])
                if added:
                    trans_modified = True
                participants_resolved.append({
                    "canonical_id": pid,
                    "name": cname,
                    "name_original": p["name"],
                    "matched_by": matched_by,
                    "language_detected": lang,
                    "damage": p["damage"],
                    "is_leader": (cname == leader_cname)
                })
                file_ref = p.get("source_file", f"rally_{rally_id}")
                dmg_logger.add_contribution(pid, cname, date_str, rally_id, file_ref, p["damage"], p.get("rank", 0))
            rally = {
                "id": f"{date_str}-rally-{rally_id}",
                "leader": leader_cname,
                "participants": participants_resolved,
            }
            event["rallies"].append(rally)

        players_map = {}
        id_to_name = {}

        # Prefer totals from total.* pages when available
        for tp in top_players_resolved:
            pid = tp.get("id") or f"pl_{_normalize_key(tp.get('name') or '')}"
            dmg = tp.get("damage", 0)
            players_map[pid] = dmg
            id_to_name.setdefault(pid, tp.get("name", pid))

        # Add rally-based sums for players not present in totals
        for r in event["rallies"]:
            for p in r["participants"]:
                pid = p.get("canonical_id") or f"pl_{_normalize_key(p['name'])}"
                if pid in players_map:
                    id_to_name.setdefault(pid, p["name"])
                    continue
                players_map.setdefault(pid, 0)
                players_map[pid] += p["damage"]
                id_to_name.setdefault(pid, p["name"])

        event["players"] = [
            {"id": pid, "name": id_to_name.get(pid, pid), "total_damage": dmg}
            for pid, dmg in sorted(players_map.items(), key=lambda kv: kv[1], reverse=True)
        ]
        if not event.get("alliance_total_damage"):
            event["alliance_total_damage"] = sum(p["total_damage"] for p in event["players"])
        # Keep rally count from total.png (do not override)
        new_events.append(event)
    # Merge + save
    if new_events:
        for ev in new_events:
            existing_events[ev["id"]] = ev
        existing_json["events"] = [existing_events[k] for k in sorted(existing_events.keys())]
    with open(DATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_json, f, indent=2, ensure_ascii=False)
    if trans_modified or not (DATA_OUTPUT_DIR / "player_translations.json").exists():
        save_translations_store(DATA_OUTPUT_DIR / "player_translations.json", trans_store)
    try:
        ocr_logger.save()
    except Exception:
        pass
    try:
        dmg_logger.save()
    except Exception:
        pass
    total_time = time.time() - start_total
    print(f"[STATS] Total time: {total_time:.2f}s; Workers: {NUM_WORKERS}; GPU: {GPU_STATUS}")

if __name__ == "__main__":
    import sys
    totals_only = "--totals-only" in sys.argv
    debug_logs = "--debug" in sys.argv
    print(f"[INFO] Starting BeartrapAnalysis with totals_only={totals_only}, debug_mode={debug_logs}")
    main(totals_only=totals_only, debug_mode=debug_logs)
