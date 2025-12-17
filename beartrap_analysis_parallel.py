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
        return {}
    scale = 2
    img_up = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    gray_up = cv2.cvtColor(img_up, cv2.COLOR_BGR2GRAY)
    detected = {}
    for rank, tpl_path in rank_templates.items():
        if not tpl_path.exists():
            continue
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
    else:
        if page_stem.endswith(".1"):
            start_rank = 5
            num_ranks = 7
        elif page_stem.endswith(".2"):
            start_rank = 12
            num_ranks = 1
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

def extract_rally_participants_from_pages_v2(reader, page_files: list[Path], ocr_logger=None, event_id="", rally_id=0):
    dot_zero_file = None
    for f in page_files:
        if f.stem.endswith(".0"):
            dot_zero_file = f
            break
    if dot_zero_file is None:
        return []
    line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(dot_zero_file)
    if line_height is None:
        return []
    # Debug directories disabled to avoid file creation during pipeline
    debug_dir = None
    debug_roi_dir = None
    participants = []
    # Prepare optional flag metrics log for cross-checking with tests
    flag_metrics_log = DATA_OUTPUT_DIR / "flag_metrics.jsonl"
    for f in sorted(page_files):
        img = cv2.imread(str(f))
        if img is None:
            continue
        img_up, bin_img, scale = preprocess_for_ocr(img)
        # Keep a color-upsampled version for color-based checks
        img_up_color = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        page_debug_dir = None
        lh_scaled = line_height * scale
        scaled_rank_y = {r: y * scale for r, y in rank_y_positions.items()}
        lines = segment_lines_by_rank_positions(img_up, lh_scaled, scaled_rank_y, img_up.shape[1], f.stem, reader)
        expected_map = {".0": 4, ".1": 7, ".2": 1}
        expected = None
        for suffix, cnt in expected_map.items():
            if f.stem.endswith(suffix):
                expected = cnt
                break
        if expected is not None and len(lines) > expected:
            lines = lines[:expected]
        for y1, y2, row_img, detected_rank in lines:
            # Corresponding color row from color-upsampled page
            row_color_img = img_up_color[y1:y2, 0:img_up.shape[1]]
            # Ensure row grayscale is derived from color row to keep alignment
            row_gray = cv2.cvtColor(row_color_img, cv2.COLOR_BGR2GRAY)
            row_texts_detail = reader.readtext(row_gray, detail=1)
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
            flag_result = detect_flag_in_row(row_color_img, tokens=tokens, save_roi_to=roi_debug_path, save_hit_to=hit_debug_path, save_avatar_to=av_debug_path, save_tl50_to=tl_debug_path)
            has_flag = bool(flag_result.get("has_flag", False))
            # Use TL50 scoring fields; prefer final_score then shape_score
            flag_conf = float(flag_result.get("final_score", flag_result.get("shape_score", 0.0)))
            
            # Write a compact metrics entry for this row to help debug divergences
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
    rally_id, files, date_str = args
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


def parse_top_players_from_total_pages(reader, total_pages: list[Path], trans_store: dict, event_id: str):
    """Parse top players from total.0/1/2 pages using the rally extractor for consistency."""
    participants_raw = extract_rally_participants_from_pages_v2(
        reader, total_pages, ocr_logger=None, event_id=event_id, rally_id=-1
    )
    if not participants_raw:
        return [], False

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
            "source_file": p.get("source_file"),
        })

    return resolved_players, trans_added

def recalc_totals_mode(reader):
    """Quick mode: only recalculate alliance_total_damage and rally_count_total from total.png files."""
    print("[INFO] ===== RECALC TOTALS ONLY MODE =====")
    if not DATA_OUTPUT_FILE.exists():
        print("[WARN] No existing beartrap.json found. Run full analysis first.")
        return
    
    with open(DATA_OUTPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    for event in data.get("events", []):
        event_date = event.get("id") or event.get("date", "")
        day_dir = DATA_INPUT_DIR / event_date
        if not day_dir.exists():
            print(f"[WARN] Directory not found: {day_dir}")
            continue
        
        total_file = day_dir / "total.png"
        if not total_file.exists():
            print(f"[WARN] total.png not found for {event_date}")
            continue
        
        # Parse totals from total.png. OCR can shuffle lines, so look around keywords and fall back to best candidates.
        alliance_damage = 0
        rally_count = 0
        total_lines = [str(t) for t in ocr_image_lines(reader, total_file) if str(t).strip()]

        # Pre-collect numeric candidates with their line index
        numeric_candidates = []
        for idx, line in enumerate(total_lines):
            val = normalize_damage(line)
            if val > 0:
                numeric_candidates.append((idx, val, line))

        def find_number_near_keyword(keywords, prefer_small=False, window=2):
            hits = []
            for i, line in enumerate(total_lines):
                line_lower = line.lower()
                if any(k in line_lower for k in keywords):
                    for j in range(0, window + 1):
                        if i + j < len(total_lines):
                            val = normalize_damage(total_lines[i + j])
                            if val > 0:
                                hits.append(val)
            if prefer_small:
                hits = [v for v in hits if v <= 100]
                if hits:
                    return min(hits)
            else:
                hits = [v for v in hits if v >= 1_000]
                if hits:
                    return max(hits)
            return 0

        rally_count = find_number_near_keyword(["ralliement"], prefer_small=True)
        if rally_count == 0:
            small_numbers = [v for _, v, _ in numeric_candidates if v <= 100]
            if small_numbers:
                rally_count = min(small_numbers)

        alliance_damage = find_number_near_keyword(["degat", "degats", "dégat", "dégâts", "total"], prefer_small=False)
        if alliance_damage == 0:
            large_numbers = [v for _, v, _ in numeric_candidates if v >= 1_000]
            if large_numbers:
                alliance_damage = max(large_numbers)
        
        event["alliance_total_damage"] = alliance_damage
        event["rally_count_total"] = rally_count
        print(f"[RECALC] {event_date}: alliance_damage={alliance_damage}, rally_count={rally_count}")
    
    with open(DATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print("[INFO] Totals recalculated and saved.")

def main(recalc_totals_only=False):
    start_total = time.time()
    print("[INFO] ===== BeartrapAnalysis PARALLEL (GPU-Optimized) =====")
    init_gpu()
    print(f"[INFO] GPU Status: {GPU_STATUS}")
    print(f"[INFO] Workers: {NUM_WORKERS}")

    reader_main = init_reader()
    
    # Quick totals recalc mode
    if recalc_totals_only:
        recalc_totals_mode(reader_main)
        return
    
    trans_store = load_translations_store(DATA_OUTPUT_DIR / "player_translations.json")
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

        # Attempt to read totals from total.png
        alliance_damage = 0
        rally_count_total = 0
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
                    if val > 0:
                        rally_count_total = val
                # Find total damage (usually near "dégât" label)
                if "dégât" in line_lower and i + 1 < len(total_lines):
                    next_line = str(total_lines[i + 1])
                    val = normalize_damage(next_line)
                    if val > 0:
                        alliance_damage = val

        # Attempt to read top players from total.0/1/2 pages (top 10)
        total_pages = [
            f for f in day_dir.iterdir()
            if f.is_file()
            and f.suffix.lower() in [".png", ".jpg", ".jpeg"]
            and re.match(r"total\.\d+$", f.stem)
        ]
        top_players_resolved = []
        if total_pages:
            top_players_resolved, added = parse_top_players_from_total_pages(
                reader_main,
                sorted(total_pages, key=lambda p: p.stem),
                trans_store,
                event_id=date_str,
            )
            if added:
                trans_modified = True

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

        work_items = [(rally_id, sorted(files), date_str) for rally_id, files in sorted(rally_groups.items())]

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
    recalc_only = "--recalc-totals" in sys.argv
    main(recalc_totals_only=recalc_only)
