import json
import re
import unicodedata
from pathlib import Path
import easyocr
import cv2
import numpy as np


# ============================
# CONFIG
# ============================

BASE_DIR = Path(__file__).parent
DATA_INPUT_DIR = BASE_DIR / "beartrap_data"
DATA_OUTPUT_DIR = BASE_DIR / "data"
DATA_OUTPUT_FILE = DATA_OUTPUT_DIR / "beartrap.json"
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"
ALIASES_FILE = DATA_OUTPUT_DIR / "player_aliases.json"

DATA_OUTPUT_DIR.mkdir(exist_ok=True)
RANK_TEMPLATES = {
    1: BASE_DIR / "assets" / "rank1.png",
    2: BASE_DIR / "assets" / "rank2.png",
    3: BASE_DIR / "assets" / "rank3.png",
}


# ============================
# ALIAS MAPPING
# ============================

def _normalize_key(name: str) -> str:
    if not name:
        return ""
    s = str(name).strip().lower()
    # remove accents
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    # collapse whitespace and punctuation except spaces
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _new_alias_store():
    return {"players": {}, "alias_to_id": {}}

def load_alias_store(path: Path) -> dict:
    if not path.exists():
        return _new_alias_store()
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict) or "players" not in data or "alias_to_id" not in data:
            return _new_alias_store()
        return data
    except Exception:
        return _new_alias_store()

def save_alias_store(path: Path, store: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)

def ensure_player(store: dict, canonical_name: str, pending: bool = True) -> str:
    # create deterministic id from normalized name to keep files readable
    key = _normalize_key(canonical_name)
    player_id = f"pl_{key.replace(' ', '_')}" if key else f"pl_{abs(hash(canonical_name))}"
    if player_id not in store["players"]:
        store["players"][player_id] = {
            "name": canonical_name,
            "aliases": [],
            "pending_review": pending,
        }
    return player_id

def link_alias(store: dict, alias: str, player_id: str):
    k = _normalize_key(alias)
    if not k:
        return
    store["alias_to_id"][k] = player_id
    p = store["players"].get(player_id)
    if p is not None:
        if alias not in p["aliases"]:
            p["aliases"].append(alias)

def resolve_canonical(store: dict, observed_name: str):
    """
    Returns tuple: (player_id, canonical_name, matched_by, added)
    matched_by in {"alias", "exact", "auto"}
    added indicates store was modified (auto-added)
    """
    if not observed_name:
        return None, None, "", False
    k = _normalize_key(observed_name)
    # alias match
    if k in store["alias_to_id"]:
        pid = store["alias_to_id"][k]
        cname = store["players"].get(pid, {}).get("name", observed_name)
        return pid, cname, "alias", False
    # exact canonical name match
    for pid, meta in store["players"].items():
        if meta.get("name") == observed_name:
            link_alias(store, observed_name, pid)
            return pid, meta.get("name"), "exact", False
    # auto-add new canonical
    pid = ensure_player(store, observed_name, pending=True)
    link_alias(store, observed_name, pid)
    return pid, observed_name, "auto", True


# ============================
# TRANSLATION / MULTILINGUAL MANAGEMENT
# ============================

TRANSLATIONS_FILE = DATA_OUTPUT_DIR / "player_translations.json"
SUPPORTED_LANGUAGES = ["fr", "en", "es", "zh", "ru", "ko"]
PRIMARY_LANGUAGE = "fr"

def transliterate_cyrillic(text: str) -> str:
    """Translittère le cyrillique (Russe) en caractères latins."""
    mapping = {
        'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D',
        'Е': 'E', 'Ё': 'Yo', 'Ж': 'Zh', 'З': 'Z', 'И': 'I',
        'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M', 'Н': 'N',
        'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T',
        'У': 'U', 'Ф': 'F', 'Х': 'H', 'Ц': 'Ts', 'Ч': 'Ch',
        'Ш': 'Sh', 'Щ': 'Sch', 'Ъ': '', 'Ы': 'Y', 'Ь': '',
        'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd',
        'е': 'e', 'ё': 'yo', 'ж': 'zh', 'з': 'z', 'и': 'i',
        'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm', 'н': 'n',
        'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't',
        'у': 'u', 'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch',
        'ш': 'sh', 'щ': 'sch', 'ъ': '', 'ы': 'y', 'ь': '',
        'э': 'e', 'ю': 'yu', 'я': 'ya',
    }
    result = ""
    for ch in text:
        result += mapping.get(ch, ch)
    return result

def transliterate_cjk_pinyin_simple(text: str) -> str:
    """
    Translittération très simplifiée pour CJK.
    Pour une vraie translittération, il faudrait une bibliothèque comme 'pinyin'.
    Ici on retourne juste une approximation romaji/pinyin/romanization basique.
    """
    # Pour la démo, on retourne juste une version latin-approximée
    # En production, installer 'pinyin' library pour le chinois
    return text  # placeholder

def transliterate_korean_hangul(text: str) -> str:
    """Translittération très basique du Coréen (Hangul)."""
    # Hangul to Romanization - très simplifié
    # En production, utiliser une vraie bibliothèque
    return text  # placeholder

def transliterate_non_latin(text: str, language: str) -> str:
    """
    Retourne une translittération phonétique du texte non-latin.
    Returns: (original_text, transliterated_text) ou juste text si déjà latin.
    """
    if language == "ru":
        return transliterate_cyrillic(text)
    elif language == "zh":
        # Besoin de 'pinyin' library - fallback simple
        return text
    elif language == "ko":
        # Besoin de vraie romanization - fallback simple
        return text
    return text

def has_non_latin_chars(text: str) -> bool:
    """Détecte si le texte contient des caractères non-latins."""
    for ch in text:
        code = ord(ch)
        # Latin étendu (0-0x024F) + espaces/ponctuation
        if code > 0x024F and code not in [0x0020, 0x0027, 0x002D]:  # space, apostrophe, dash
            return True
    return False

def load_translations_store(path: Path) -> dict:
    """Charge le fichier de traductions des joueurs."""
    if not path.exists():
        return {
            "languages": {lang: lang for lang in SUPPORTED_LANGUAGES},
            "primary_language": PRIMARY_LANGUAGE,
            "players": {},
            "alias_to_id": {},
        }
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"[WARN] Erreur lors de la lecture des traductions: {e}, création d'un nouveau store")
        return {
            "languages": {lang: lang for lang in SUPPORTED_LANGUAGES},
            "primary_language": PRIMARY_LANGUAGE,
            "players": {},
            "alias_to_id": {},
        }

def save_translations_store(path: Path, store: dict):
    """Sauvegarde le fichier de traductions."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)

def detect_language_from_text(text: str) -> str:
    """
    Détecte la langue d'un texte basée sur les caractères Unicode.
    Retourne le code langue, fallback "en".
    """
    if not text:
        return PRIMARY_LANGUAGE
    
    text_sample = text[:50]
    
    # Caractères CJK (Chinois)
    if re.search(r'[\u4E00-\u9FFF]', text_sample):
        return "zh"
    
    # Hangul (Coréen)
    if re.search(r'[\uAC00-\uD7AF]', text_sample):
        return "ko"
    
    # Cyrillique (Russe)
    if re.search(r'[\u0400-\u04FF]', text_sample):
        return "ru"
    
    # Latin = EN/FR/ES (ambigus, on laisse en "en" par défaut)
    return "en"

def add_player_translation(trans_store: dict, player_id: str, 
                          original_name: str, language: str):
    """
    Ajoute un joueur au store de traductions.
    Stocke le nom original + translittération si non-latin.
    
    Args:
        trans_store: dictionnaire de traductions
        player_id: id canonique (pl_...)
        original_name: nom tel qu'observé
        language: code langue détecté
    """
    if player_id not in trans_store["players"]:
        # Initialiser avec noms vides pour toutes les langues
        trans_store["players"][player_id] = {
            "names_by_language": {lang: "" for lang in SUPPORTED_LANGUAGES},
            "transliterations_by_language": {lang: None for lang in SUPPORTED_LANGUAGES},
            "language_detected": language,
            "aliases": [original_name],
            "pending_review": True,
        }
    
    player = trans_store["players"][player_id]
    
    # Mettre à jour le nom pour la langue détectée
    player["names_by_language"][language] = original_name
    player["language_detected"] = language
    
    # Générer translittération si non-latin
    if has_non_latin_chars(original_name) and not player["transliterations_by_language"][language]:
        trans = transliterate_non_latin(original_name, language)
        if trans and trans != original_name:
            player["transliterations_by_language"][language] = trans
    
    # Ajouter aux aliases
    if original_name not in player.get("aliases", []):
        player["aliases"].append(original_name)

def get_name_for_language(trans_store: dict, player_id: str, language: str) -> str:
    """
    Récupère le nom d'un joueur dans une langue donnée.
    Fallback sur la langue détectée si la traduction n'existe pas.
    """
    player = trans_store["players"].get(player_id)
    if not player:
        return player_id
    
    names = player.get("names_by_language", {})
    
    # Essayer la langue demandée
    if language in names and names[language]:
        name = names[language]
        trans = player.get("transliterations_by_language", {}).get(language)
        if trans and trans != name:
            return f"{name} ({trans})"
        return name
    
    # Fallback sur la langue détectée
    detected_lang = player.get("language_detected", "en")
    if detected_lang in names and names[detected_lang]:
        return names[detected_lang]
    
    # Fallback sur le français (langue primaire)
    if "fr" in names and names["fr"]:
        return names["fr"]
    
    # Fallback sur n'importe quelle langue non-vide
    for lang in SUPPORTED_LANGUAGES:
        if lang in names and names[lang]:
            return names[lang]
    
    return player_id

def resolve_with_translation(trans_store: dict, observed_name: str):
    """
    Résout un nom observé en utilisant le store de traductions.
    
    Returns:
        tuple: (player_id, canonical_name, language_detected, matched_by, added)
    """
    if not observed_name:
        return None, None, None, "", False
    
    detected_lang = detect_language_from_text(observed_name)
    k = _normalize_key(observed_name)
    
    # Alias match (normalized key)
    if k in trans_store["alias_to_id"]:
        pid = trans_store["alias_to_id"][k]
        player = trans_store["players"].get(pid, {})
        canonical_name = get_name_for_language(trans_store, pid, detected_lang)
        return pid, canonical_name, detected_lang, "alias", False
    
    # Exact name match
    for pid, player in trans_store["players"].items():
        names = player.get("names_by_language", {})
        if observed_name in names.values():
            # Link alias
            if k and k not in trans_store["alias_to_id"]:
                trans_store["alias_to_id"][k] = pid
            if observed_name not in player.get("aliases", []):
                player["aliases"].append(observed_name)
            canonical_name = get_name_for_language(trans_store, pid, detected_lang)
            return pid, canonical_name, detected_lang, "exact", False
    
    # Auto-add new player
    pid = f"pl_{k.replace(' ', '_')}" if k else f"pl_{abs(hash(observed_name))}"
    add_player_translation(trans_store, pid, observed_name, detected_lang)
    if k:
        trans_store["alias_to_id"][k] = pid
    canonical_name = get_name_for_language(trans_store, pid, detected_lang)
    return pid, canonical_name, detected_lang, "auto", True

def get_display_name(trans_store: dict, player_id: str) -> str:
    """
    Récupère le nom d'affichage d'un joueur (langue primaire).
    """
    return get_name_for_language(trans_store, player_id, PRIMARY_LANGUAGE)


# ============================
# OCR HELPERS
# ============================

def ocr_image_boxes(reader, image_path: Path):
    """
    Retourne les résultats OCR bruts pour une image :
    liste de tuples (box, text, conf)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Image illisible : {image_path}")
        return []
    data = reader.readtext(img, detail=1)
    # print(f"[DEBUG] OCR results for {image_path}:")
    # for box, text, conf in data:
        # print(f" Box: {box},  Text: '{text}', Confidence: {conf:.2f}")
    return data


def ocr_image_lines(reader, image_path: Path):
    """
    Version simple: juste les textes (pour total.png)
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[WARN] Image illisible : {image_path}")
        return []
    data = reader.readtext(img, detail=0)
    return data


# ============================
# IMAGE PREPROCESS & ROW SEGMENTATION
# ============================

def preprocess_for_ocr(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # upscale to help OCR on small UI text
    scale = 2
    gray_up = cv2.resize(gray, (gray.shape[1] * scale, gray.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
    # adaptive threshold to enhance text
    th = cv2.adaptiveThreshold(gray_up, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
    # light denoise
    th = cv2.medianBlur(th, 3)
    return gray_up, th, scale


def find_leaderboard_roi(binary_img):
    # Use vertical projection to detect the dense text block area
    cols_sum = np.sum(binary_img // 255, axis=0)
    # Smooth projection
    smooth = cv2.blur(cols_sum.astype(np.float32), (31, 1))
    thresh = np.max(smooth) * 0.2
    mask = smooth > thresh
    if not np.any(mask):
        # fallback: whole image
        h, w = binary_img.shape
        return (0, 0, w, h)
    xs = np.where(mask)[0]
    x1, x2 = int(xs[0]), int(xs[-1])
    # Y bounds via horizontal projection within [x1,x2]
    sub = binary_img[:, x1:x2]
    rows_sum = np.sum(sub // 255, axis=1)
    smooth_r = cv2.blur(rows_sum.astype(np.float32), (1, 31))
    thr_r = np.max(smooth_r) * 0.2
    mask_r = smooth_r > thr_r
    if not np.any(mask_r):
        h, _ = binary_img.shape
        return (x1, 0, x2 - x1, h)
    ys = np.where(mask_r)[0]
    y1, y2 = int(ys[0]), int(ys[-1])
    return (x1, y1, x2 - x1, y2 - y1)


def segment_rows(binary_img_roi):
    # Horizontal projection to detect text bands separated by gaps
    rows_sum = np.sum(binary_img_roi // 255, axis=1)
    # Normalize and smooth
    rows_sum = rows_sum.astype(np.float32)
    rows_sum = cv2.blur(rows_sum, (1, 21))
    # Identify gaps where projection is near zero
    maxv = np.max(rows_sum)
    gap_thresh = maxv * 0.15
    is_text = rows_sum > gap_thresh
    bands = []
    in_band = False
    start = 0
    for y, val in enumerate(is_text):
        if val and not in_band:
            in_band = True
            start = y
        elif not val and in_band:
            in_band = False
            end = y
            if end - start > 15:  # minimum height
                bands.append((start, end))
    if in_band:
        end = len(is_text) - 1
        if end - start > 15:
            bands.append((start, end))
    return bands


def crop_rows(img_up, roi_rect, bands):
    x, y, w, h = roi_rect
    crops = []
    for (y1, y2) in bands:
        cy1 = y + y1
        cy2 = y + y2
        row_img = img_up[cy1:cy2, x:x + w]
        if row_img.size == 0:
            continue
        crops.append(((x, cy1, w, cy2 - cy1), row_img))
    return crops


# ============================
# PARSING TOTAL
# ============================

def normalize_damage(value_str: str) -> int:
    """
    Convert damage string (e.g., '7 687 755', '1.2M', '8 7342 282') to integer.
    Handles French separators (space, comma, period) and k/m suffixes.
    """
    if not value_str:
        return 0
    
    s = str(value_str).strip()
    
    # Remove "Points de Dégâts" label if present
    s = re.sub(r"Points de D[eé]g[âa]ts\s*[:\s]*", "", s, flags=re.IGNORECASE)
    s = s.strip()
    
    # Extract the FIRST numeric token (chain of digits/spaces/commas/periods + optional k/m)
    # This should be one complete damage value
    match = re.search(r"(\d[\d\s\.,]*[km]?)", s, re.IGNORECASE)
    if not match:
        return 0
    
    damage_str = match.group(1).strip()
    
    # Remove spaces (French separator) and normalize decimal separator to period
    damage_str = damage_str.replace(" ", "").replace(",", ".")
    
    # Extract suffix (k or m)
    suffix_match = re.search(r"([km])$", damage_str, re.IGNORECASE)
    suffix = suffix_match.group(1).lower() if suffix_match else None
    
    # Remove suffix and parse as number
    if suffix:
        damage_str = damage_str[:-1]
    
    try:
        number = float(damage_str)
    except ValueError:
        return 0
    
    # Apply multiplier
    if suffix == "m":
        number *= 1_000_000
    elif suffix == "k":
        number *= 1_000
    
    return int(number)


def extract_total_stats(lines):
    """
    Parse total.png à partir de lignes simples (sans boxes).
    """
    alliance_damage = None
    rally_count = None
    event_name = None

    # Track indices of label lines
    rall_idx = None
    dmg_idx = None

    for i, line in enumerate(lines):
        # Nom d'event (non critique)
        m = re.search(r"\[(.*?)\]", line)
        if m:
            event_name = m.group(1)

        # Dégâts totaux (label may be separate from numeric)
        if ("Dégâts Totaux" in line) or ("Degats Totaux" in line) or ("Dégâts Totaux de" in line):
            dmg_idx = i
            m = re.search(r"(\d[\d\s\.,MKmk]+)", line)
            if m:
                alliance_damage = normalize_damage(m.group(1))

        # Ralliements (label may be separate from numeric)
        if "Ralliements" in line:
            rall_idx = i
            m = re.search(r"Ralliements\s*:?\s*(\d+)", line)
            if m:
                rally_count = int(m.group(1))

    # If not found inline, search the next few lines after labels
    if rally_count is None and rall_idx is not None:
        for j in range(rall_idx + 1, min(rall_idx + 4, len(lines))):
            m = re.search(r"(\d{1,3})", lines[j])
            if m:
                rally_count = int(m.group(1))
                break

    if alliance_damage is None and dmg_idx is not None:
        for j in range(dmg_idx + 1, min(dmg_idx + 4, len(lines))):
            m = re.search(r"(\d[\d\s\.,MKmk]+)", lines[j])
            if m:
                alliance_damage = normalize_damage(m.group(1))
                break

    # Fallbacks: try to infer from all lines
    if rally_count is None:
        # Look for a standalone small integer anywhere
        for line in lines:
            m = re.findall(r"\b(\d{1,3})\b", line)
            for n in m:
                val = int(n)
                if 1 <= val <= 999:
                    rally_count = val
                    break
            if rally_count is not None:
                break

    if alliance_damage is None:
        # Find the largest plausible damage number in the page
        candidates = []
        for line in lines:
            for num in re.findall(r"\d[\d\s\.,MKmk]+", line):
                candidates.append(normalize_damage(num))
        if candidates:
            alliance_damage = max(candidates)

    return event_name, alliance_damage, rally_count


# ============================
# RANK / LINES RECONSTRUCTION
# ============================

def is_rank_text(text: str) -> bool:
    """
    Détermine si un fragment OCR ressemble à un rang (1, 2, 3, "00", "10", "E", etc.)
    """
    t = text.strip()
    if re.fullmatch(r"\d{1,2}", t):
        return True
    # Artefacts éventuels, à ajuster si tu vois des patterns récurrents
    if t in ["E", "e"]:
        return True
    return False


def extract_ranks_from_boxes(all_boxes):
    """
    À partir de toutes les boxes (box, text, conf) d'un rally :
    retourne une liste de dicts {rank, y_center}
    pour les rangs détectés.
    """
    ranks = []
    for box, text, conf in all_boxes:
        if not is_rank_text(text):
            continue
        # box: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
        ys = [p[1] for p in box]
        y_center = sum(ys) / len(ys)
        try:
            r = int(text)
        except ValueError:
            # ex: "E" → tu peux mapper sur 5 si tu veux
            continue
        ranks.append({"rank": r, "y": y_center})

    ranks.sort(key=lambda r: r["y"])
    return ranks


def estimate_line_height_from_ranks(ranks):
    """
    À partir des rangs détectés (rang + y), estime la hauteur moyenne
    entre deux lignes de rangs successives.
    """
    if len(ranks) < 2:
        return None

    diffs = []
    for i in range(1, len(ranks)):
        dy = ranks[i]["y"] - ranks[i-1]["y"]
        if dy > 5:  # on ignore les très petits écarts
            diffs.append(dy)

    if not diffs:
        return None

    # moyenne simple
    line_height = sum(diffs) / len(diffs)
    return line_height


def build_visual_lines_from_ranks(ranks, line_height, max_extra_lines=2):
    """
    Construit une map ligne_visuelle -> y,
    en extrapolant les rangs manquants au-dessus et en dessous.

    Ex: rangs trouvés: 2 et 3
    → on reconstruit y1, y2, y3, y4, etc.
    """
    if not ranks or line_height is None:
        return {}

    # on part des rangs trouvés
    min_rank = min(r["rank"] for r in ranks)
    max_rank = max(r["rank"] for r in ranks)

    # map rang -> y
    rank_to_y = {r["rank"]: r["y"] for r in ranks}

    # extrapoler vers le haut
    for r in range(min_rank - 1, max(min_rank - max_extra_lines, 0), -1):
        if r + 1 in rank_to_y:
            rank_to_y[r] = rank_to_y[r+1] - line_height

    # extrapoler vers le bas
    for r in range(max_rank + 1, max_rank + max_extra_lines + 1):
        if r - 1 in rank_to_y:
            rank_to_y[r] = rank_to_y[r-1] + line_height

    # on retourne trié par rang
    visual_lines = dict(sorted(rank_to_y.items(), key=lambda kv: kv[0]))
    return visual_lines


def group_boxes_by_visual_lines(all_boxes, visual_lines, y_tolerance_factor=0.4):
    """
    Associe à chaque rang (ligne visuelle) la liste des fragments OCR
    dont le y_center est proche de la y_target de la ligne.
    """
    rank_to_items = {rank: [] for rank in visual_lines.keys()}

    if not visual_lines:
        return rank_to_items

    # une estimation de tolérance en Y
    # ex: si line_height = 150 → tolérance ~60
    # tu peux affiner ensuite
    ranks_sorted = sorted(visual_lines.items(), key=lambda kv: kv[0])
    if len(ranks_sorted) >= 2:
        approx_line_height = ranks_sorted[1][1] - ranks_sorted[0][1]
    else:
        approx_line_height = 40  # fallback

    y_tol = abs(approx_line_height) * y_tolerance_factor
    if y_tol < 10:
        y_tol = 10

    for box, text, conf in all_boxes:
        ys = [p[1] for p in box]
        y_center = sum(ys) / len(ys)

        # trouver le rang dont la y_target est la plus proche
        best_rank = None
        best_dy = None
        for rank, y_target in visual_lines.items():
            dy = abs(y_center - y_target)
            if best_dy is None or dy < best_dy:
                best_dy = dy
                best_rank = rank

        if best_rank is not None and best_dy <= y_tol:
            xs = [p[0] for p in box]
            x_min = min(xs)
            rank_to_items[best_rank].append((x_min, text))

    # tri par x à l'intérieur de chaque ligne
    for rank, items in rank_to_items.items():
        items.sort(key=lambda t: t[0])

    return rank_to_items


# ============================
# PARSING D'UNE LIGNE DE CLASSEMENT
# ============================

def parse_player_from_line_items(items):
    """
    items: liste de (x, text) pour une ligne (rang donné).
    On essaie d'en extraire (name, damage) si possible.
    Hypothèse: 'Points de Dégâts' apparaît, avec un nombre à côté.
    """
    if not items:
        return None

    line_text = " ".join([t for (_, t) in items])
    if "Points de" not in line_text:
        return None

    # pseudo = avant 'Points de'
    # dégâts = le nombre après
    parts = line_text.split("Points de", 1)
    left = parts[0].strip()
    right = parts[1].strip()

    # nettoyer le pseudo: enlever un éventuel rang au début
    left = re.sub(r"^\d+\s*", "", left).strip()
    left = re.sub(r"^[Ee]\s*", "", left).strip()

    if not left:
        return None

    dmg = normalize_damage(right)
    if dmg <= 0:
        return None

    return {
        "name": left,
        "damage": dmg,
        "raw_line": line_text,
    }


def parse_player_from_row(reader, row_img):
    # OCR the single row; try detail=0, fallback to detail=1 concat
    texts = reader.readtext(row_img, detail=0)
    if not texts:
        parts = reader.readtext(row_img, detail=1)
        texts = [t for _, t, _ in parts]
        if not texts:
            return None
    line = " ".join(texts)
    # Heuristics: split around labels and pick rightmost numeric
    if "Points de" in line:
        parts = line.split("Points de", 1)
        left = parts[0].strip()
        right = parts[1].strip()
    else:
        # try English
        m = re.search(r"(.*?)(\d[\d\s\.,kKmM]+)\s*$", line)
        if not m:
            return None
        left = m.group(1).strip()
        right = m.group(2).strip()

    left = re.sub(r"^\d+\s*", "", left).strip()
    left = re.sub(r"^[Ee]\s*", "", left).strip()
    if not left:
        return None

    dmg = normalize_damage(right)
    if dmg <= 0:
        # try to find last numeric token in line
        nums = re.findall(r"\d[\d\s\.,kKmM]+", line)
        if nums:
            dmg = normalize_damage(nums[-1])
    if dmg <= 0:
        return None

    return {"name": left, "damage": dmg, "raw_line": line}


def extract_rally_participants_from_boxes(all_boxes):
    """
    Flux complet pour un rally:
    1) extraire les rangs
    2) estimer la hauteur de ligne
    3) reconstruire les lignes visuelles
    4) regrouper les boxes par ligne
    5) parser chaque ligne en joueur
    """
    ranks = extract_ranks_from_boxes(all_boxes)
    if not ranks:
        print("[WARN] Aucun rang détecté dans ce rally.")
        return []

    line_height = estimate_line_height_from_ranks(ranks)
    if line_height is None:
        print("[WARN] Impossible d'estimer la hauteur des lignes.")
        return []

    visual_lines = build_visual_lines_from_ranks(ranks, line_height, max_extra_lines=2)

    rank_to_items = group_boxes_by_visual_lines(all_boxes, visual_lines)

    participants = []
    for rank, items in rank_to_items.items():
        player = parse_player_from_line_items(items)
        if player:
            participants.append(player)

    print(f"[INFO] Participants extraits: {len(participants)}")
    for p in participants:
        print(f"  - {p['name']} : {p['damage']} dmg (ligne brute: '{p['raw_line']}')")

    return participants


def detect_rank_icons_by_template(image_path: Path, rank_templates: dict[int, Path]):
    """
    Detect ranks 1, 2, 3 using template matching on upscaled image.
    Returns: dict {rank: (y_center, x_center)} for detected ranks
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"[CALIB] Cannot read {image_path}")
        return {}
    
    # Upscale for better template matching
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
        
        # Upscale template too
        tpl_up = cv2.resize(tpl, (tpl.shape[1] * scale, tpl.shape[0] * scale), interpolation=cv2.INTER_CUBIC)
        
        best_pos = None
        best_val = 0
        
        # Try multiple scales for robustness
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
            # Unscale back to original coordinates
            y_center = best_pos[1] / scale
            x_center = best_pos[0] / scale
            detected[rank] = (y_center, x_center)
            print(f"[CALIB] Rank {rank} detected at Y={y_center:.1f}, X={x_center:.1f} (score: {best_val:.2f})")
    
    return detected


def calibrate_line_height_from_dot_zero(reader, dot_zero_file: Path):
    """
    Calibrate line height and rank column position from *.0 file using template matching.
    Detects ranks 1, 2, 3 icons, computes line_height from Y differences.
    Returns: (line_height, rank_x_col, rank_y_positions)
    """
    detected_ranks = detect_rank_icons_by_template(dot_zero_file, RANK_TEMPLATES)
    print(f"[CALIB] {dot_zero_file.name}: detected ranks {sorted(detected_ranks.keys())}")
    
    if len(detected_ranks) < 2:
        print(f"[CALIB] Not enough ranks to calibrate; need at least 2.")
        return None, None, None
    
    # Calculate line_height from consecutive detected ranks
    sorted_ranks = sorted(detected_ranks.items())
    diffs = []
    for i in range(1, len(sorted_ranks)):
        dy = sorted_ranks[i][1][0] - sorted_ranks[i-1][1][0]
        if dy > 5:
            diffs.append(dy)
    
    if not diffs:
        print(f"[CALIB] Cannot compute line height from detected ranks.")
        return None, None, None
    
    line_height = sum(diffs) / len(diffs)
    print(f"[CALIB] Calculated line_height = {line_height:.1f} px")
    
    # Get the X column of ranks (average of detected rank X positions)
    rank_x_positions = [x for (y, x) in detected_ranks.values()]
    rank_x_col = sum(rank_x_positions) / len(rank_x_positions)
    print(f"[CALIB] Rank column X = {rank_x_col:.1f} px")
    
    # Build rank_y_positions from detected ranks
    rank_y_positions = {r: y for r, (y, x) in detected_ranks.items()}
    
    # If rank 1 is missing, estimate it above rank 2
    if 1 not in rank_y_positions:
        if 2 in rank_y_positions:
            rank_y_positions[1] = rank_y_positions[2] - line_height
            print(f"[CALIB] Rank 1 missing; estimated Y = {rank_y_positions[1]:.1f}")
    
    # If rank 3 is missing, estimate it below rank 2
    if 3 not in rank_y_positions and 2 in rank_y_positions:
        rank_y_positions[3] = rank_y_positions[2] + line_height
        print(f"[CALIB] Rank 3 estimated Y = {rank_y_positions[3]:.1f}")
    
    return line_height, rank_x_col, rank_y_positions


def find_first_rank_ocr(reader, img_up, line_height_px: float, expected_first_rank: int):
    """
    Find the Y position of the first rank on this page via OCR.
    Scans the top portion of the image for numeric rank in left column.
    Returns: Y position of first rank, or None if not found.
    """
    h, w = img_up.shape[:2]
    scan_height = int(line_height_px * 2)  # Scan first 2 lines
    scan_img = img_up[:scan_height, :int(w * 0.2)]  # Left 20% only
    
    boxes = reader.readtext(scan_img, detail=1)
    for box, text, conf in boxes:
        t = text.strip()
        if re.fullmatch(r"\d{1,2}", t) and conf >= 0.5:
            try:
                rank = int(t)
                if rank == expected_first_rank:
                    ys = [p[1] for p in box]
                    y_center = sum(ys) / len(ys)
                    print(f"[SEGM] Found rank {rank} at Y={y_center:.1f}")
                    return y_center
            except:
                pass
    
    return None


def segment_lines_by_rank_positions(img_up, line_height_px, rank_y_positions: dict, img_width, page_stem: str, reader=None):
    """
    Segment image into lines.
    For page *.0 (ranks 1-4): center each line on detected rank_y_positions.
    For other pages (*.1, *.2): detect first rank position via OCR, then use regular grid.
    
    Args:
        img_up: upscaled image
        line_height_px: height of each line (in upscaled coordinates)
        rank_y_positions: dict {rank: y_center} from calibration (only used for *.0)
        img_width: width of image for cropping
        page_stem: page filename stem (e.g., "1.0", "1.1", "1.2")
        reader: OCR reader (needed for other pages to find first rank)
    
    Returns: list of (y1, y2, row_img, rank) tuples where y1/y2 are the row bounds.
    """
    h, w = img_up.shape[:2]
    half_height = int(line_height_px / 2)
    lines = []
    
    if page_stem.endswith(".0"):
        # Page 1.0: segment centered on detected ranks 1, 2, 3
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
        # Pages 1.1, 1.2: detect first rank via OCR, then use regular grid
        if page_stem.endswith(".1"):
            start_rank = 5
            num_ranks = 7
        elif page_stem.endswith(".2"):
            start_rank = 12
            num_ranks = 1
        else:
            start_rank = 1
            num_ranks = 4
        
        # Try to find first rank position via OCR
        first_rank_y = None
        if reader:
            first_rank_y = find_first_rank_ocr(reader, img_up, line_height_px, start_rank)
        
        # If found, center first line on it; otherwise start from top
        if first_rank_y is not None:
            current_y = first_rank_y - half_height
        else:
            print(f"[SEGM] Could not find rank {start_rank} via OCR; starting from top")
            current_y = 0
        
        # Regular grid segmentation
        for idx in range(num_ranks):
            rank = start_rank + idx
            y1 = max(0, int(current_y))
            y2 = min(h, int(current_y + line_height_px))
            row_img = img_up[y1:y2, 0:w]
            if row_img.size > 0:
                lines.append((y1, y2, row_img, rank))
            current_y += line_height_px
    
    return lines


def extract_rally_participants_from_pages_v2(reader, page_files: list[Path]):
    """
    Four-phase extraction:
    Phase 1: Calibrate line_height and rank positions from *.0 file using template matching
    Phase 2: For all files, segment using calibrated line_height (center each line on rank Y)
    Phase 3: OCR each line to extract rank, name, damage; reconstruct names by tokens
    Phase 4: Build statistics
    """
    # Phase 1: Calibrate from *.0 file
    dot_zero_file = None
    for f in page_files:
        if f.stem.endswith(".0"):
            dot_zero_file = f
            break
    
    if dot_zero_file is None:
        print("[PHASE1] No *.0 file found; cannot calibrate.")
        return []
    
    line_height, rank_x_col, rank_y_positions = calibrate_line_height_from_dot_zero(reader, dot_zero_file)
    if line_height is None:
        print("[PHASE1] Failed to calibrate line height.")
        return []
    
    print(f"[PHASE1] Calibration complete: line_height={line_height:.1f}, rank_x_col={rank_x_col:.1f}")
    print(f"[PHASE1] Rank Y positions: {rank_y_positions}")
    
    debug_dir = DATA_OUTPUT_DIR / "debug_rows"
    debug_dir.mkdir(exist_ok=True)
    
    participants = []
    
    # Phase 2 & 3: For each page, segment and extract participants
    for f in sorted(page_files):
        img = cv2.imread(str(f))
        if img is None:
            print(f"[WARN] Cannot read {f}")
            continue
        
        img_up, bin_img, scale = preprocess_for_ocr(img)
        page_debug_dir = debug_dir / f.stem
        page_debug_dir.mkdir(exist_ok=True)
        
        print(f"\n[PHASE2] Processing {f.name} with scale={scale}")
        
        # Phase 2: Segment image lines centered on detected rank Y positions
        lh_scaled = line_height * scale
        scaled_rank_y = {r: y * scale for r, y in rank_y_positions.items()}
        lines = segment_lines_by_rank_positions(img_up, lh_scaled, scaled_rank_y, img_up.shape[1], f.stem, reader)
        
        # Enforce expected row counts per page
        expected_map = {".0": 4, ".1": 7, ".2": 1}
        expected = None
        for suffix, cnt in expected_map.items():
            if f.stem.endswith(suffix):
                expected = cnt
                break
        
        if expected is not None and len(lines) > expected:
            lines = lines[:expected]
        
        print(f"[PHASE2] {f.name}: {len(lines)} lignes segmentées (expected {expected})")
        
        # Phase 3: OCR each line and extract participant data
        for y1, y2, row_img, detected_rank in lines:
            row_texts_detail = reader.readtext(row_img, detail=1)
            tokens = []
            for box, text, conf in row_texts_detail:
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                tokens.append({
                    "text": text.strip(),
                    "conf": conf,
                    "x": min(xs),
                    "y": sum(ys) / len(ys)
                })
            
            # The rank from detected_rank is already set from segmentation
            rank = detected_rank
            
            # Group tokens by Y position (same Y = same line within the row)
            # Use larger tolerance (~20px) to handle slight vertical variations
            y_groups = {}
            for tok in tokens:
                y_key = round(tok["y"] / 20) * 20  # Group by ~20px bands
                if y_key not in y_groups:
                    y_groups[y_key] = []
                y_groups[y_key].append(tok)
            
            if not y_groups:
                continue
            
            # Find main Y groups: exclude groups that are mostly digits/labels
            # Name group = has non-numeric tokens, doesn't start with rank number
            name_groups = []
            dmg_group = []
            
            for y_key in sorted(y_groups.keys()):
                group = y_groups[y_key]
                has_text = any(not re.fullmatch(r"[\d\s\.,kKmM]*", t["text"]) for t in group)
                has_points_label = any("Points de" in t["text"] for t in group)
                has_left_number = any(t["x"] <= row_img.shape[1] * 0.15 and re.fullmatch(r"\d{1,2}", t["text"]) for t in group)
                
                # Name is a group with text that doesn't start with rank number
                if has_text and not (has_left_number and len(group) == 1):
                    name_groups.append(group)
                
                # Damage group has Points de label or is mostly numeric
                if has_points_label or (len(group) > 0 and not has_text):
                    dmg_group.extend(group)
            
            # Reconstruct name: concatenate all name groups, sorted by X
            name_tokens = []
            for group in name_groups:
                name_tokens.extend([t for t in group if "Points de" not in t["text"] and t["conf"] >= 0.3])
            
            name_tokens_sorted = sorted([t for t in name_tokens if not (t["x"] <= row_img.shape[1] * 0.15 and re.fullmatch(r"\d{1,2}", t["text"]))], key=lambda t: t["x"])
            
            # Filter out damage-like patterns at the end (formatted numbers with spaces)
            name_parts = []
            for t in name_tokens_sorted:
                # Skip tokens that look like damage (pure numbers with spaces/dots/commas)
                if re.fullmatch(r"\d[\d\s\.,kKmM]*", t["text"]):
                    continue
                name_parts.append(t["text"])
            
            name = " ".join(name_parts).strip()
            
            # Extract damage: find all numeric values, exclude rank numbers (1-20), take largest
            all_numbers = []
            for tok in tokens:
                # Skip rank column for now, collect all numbers
                nums = re.findall(r"\d[\d\s\.,kKmM]*", tok["text"])
                for num_str in nums:
                    # Normalize to check value
                    val = normalize_damage(num_str)
                    # Exclude rank numbers (1-20) and tiny values
                    if 100 < val or (val > 0 and val <= 20 and tok["x"] > row_img.shape[1] * 0.15):
                        all_numbers.append((val, num_str))
            
            # Take the largest damage value
            if all_numbers:
                all_numbers.sort(key=lambda x: x[0], reverse=True)
                damage = all_numbers[0][0]
            else:
                damage = 0
            
            # Save debug crop
            debug_path = page_debug_dir / f"rank{rank:02d}_{name[:15]}.png"
            cv2.imwrite(str(debug_path), row_img)
            
            print(f"[PHASE3] {f.name}::rank{rank} -> name='{name}', dmg={damage}")
            
            if name and damage > 0:
                participants.append({
                    "name": name,
                    "damage": damage,
                    "rank": rank,
                    "raw_line": " ".join([t["text"] for t in tokens])
                })
    
    print(f"\n[PHASE3] Total participants extracted: {len(participants)}")
    return participants


# ============================
# LEADER DETECTION
# ============================

def detect_flag_position(image_path: Path, flag_template_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template = cv2.imread(str(flag_template_path), cv2.IMREAD_GRAYSCALE)
    if template is None:
        print("[WARN] Template flag introuvable.")
        return None
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)
    if max_val < 0.6:
        return None
    return max_loc  # (x, y)


def find_leader_from_flag(reader, image_path: Path, flag_template_path: Path):
    flag_pos = detect_flag_position(image_path, flag_template_path)
    if not flag_pos:
        return None
    x_flag, y_flag = flag_pos
    ocr_data = reader.readtext(str(image_path))
    best_match = None
    best_dist = float("inf")
    for box, text, conf in ocr_data:
        y_text = box[0][1]
        dist = abs(y_text - y_flag)
        if dist < best_dist:
            best_dist = dist
            best_match = text
    return best_match


def detect_rank_icons(image_path: Path, rank_templates: dict[int, Path]):
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    found = []
    for rank, tpl_path in rank_templates.items():
        if not tpl_path.exists():
            continue
        tpl = cv2.imread(str(tpl_path), cv2.IMREAD_GRAYSCALE)
        if tpl is None:
            continue
        best_pos = None
        best_val = 0
        for scale in [0.75, 0.9, 1.0, 1.1, 1.25, 1.5]:
            th, tw = int(tpl.shape[0] * scale), int(tpl.shape[1] * scale)
            if th <= 5 or tw <= 5:
                continue
            tpl_s = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_LINEAR)
            res = cv2.matchTemplate(gray, tpl_s, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(res)
            if max_val > best_val:
                best_val = max_val
                best_pos = (max_loc[0] + tw // 2, max_loc[1] + th // 2)
        if best_pos and best_val >= 0.6:
            found.append((rank, best_pos[0], best_pos[1], best_val))

    if not found:
        return None
    found.sort(key=lambda t: t[2])
    return found


def detect_leader_fallback(participants):
    """
    Si on ne trouve pas via le flag, on prend le premier du classement.
    """
    if not participants:
        return None
    return participants[0]["name"]


# ============================
# MAIN
# ============================

def main():
    print("[INFO] Initialisation OCR...")
    # Utiliser seulement les langues disponibles dans EasyOCR
    # Fallback à fr/en si zh/ru/ko/es non disponibles
    try:
        reader = easyocr.Reader(["fr", "en", "es", "zh", "ru", "ko"])
    except (ValueError, RuntimeError) as e:
        print(f"[WARN] Certaines langues non disponibles, fallback à FR/EN: {e}")
        reader = easyocr.Reader(["fr", "en"])

    # Alias store (deprecated, kept for backward compatibility)
    alias_store = load_alias_store(ALIASES_FILE)
    alias_modified = False
    
    # Translations store (nouveau système multilingue)
    trans_store = load_translations_store(TRANSLATIONS_FILE)
    trans_modified = False

    # Charger JSON existant (idempotence)
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
        if date_str in existing_events:
            print(f"[INFO] Event {date_str} déjà présent, skip.")
            continue

        print(f"\n[INFO] Traitement du dossier {date_str}")

        # ----- TOTAL -----
        total_file = day_dir / "total.png"
        if not total_file.exists():
            print("[WARN] Aucun total.png trouvé, skip.")
            continue

        total_lines = ocr_image_lines(reader, total_file)
        print("=== OCR TOTAL DEBUG ===")
        for l in total_lines:
            print(l)
        print("=== END OCR TOTAL DEBUG ===")

        event_name, alliance_damage, rally_count = extract_total_stats(total_lines)

        event = {
            "id": date_str,
            "date": date_str,
            # On se fiche du nom d'événement; fixer à une valeur par défaut
            "name": "Bear Trap",
            "alliance_total_damage": alliance_damage,
            "rally_count_total": rally_count,
            "rallies": []
        }

        # ----- RALLIES -----
        rally_groups = {}
        for f in day_dir.iterdir():
            if f.name == "total.png":
                continue
            if f.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
                continue
            m = re.match(r"(\d+)\.(\d+)", f.stem)
            if not m:
                continue
            rally_id = int(m.group(1))
            rally_groups.setdefault(rally_id, []).append(f)

        for rally_id, files in sorted(rally_groups.items()):
            print(f"[INFO] Ralliement {rally_id} : {len(files)} fichiers")

            leader_name_obs = None
            # detect leader from first page if possible
            for f in sorted(files):
                leader_name_obs = find_leader_from_flag(reader, f, FLAG_TEMPLATE_PATH)
                if leader_name_obs:
                    break

            # new approach: unified line height calibration + segmentation
            participants_raw = extract_rally_participants_from_pages_v2(reader, files)

            if not participants_raw:
                print(f"[WARN] Aucun participant détecté pour ralliement {rally_id}")
                continue

            # dédoublonnage simple name+damage
            participants_unique = []
            seen = set()
            for p in participants_raw:
                key = (p["name"], p["damage"]) 
                if key in seen:
                    continue
                seen.add(key)
                participants_unique.append(p)

            if not leader_name_obs:
                leader_name_obs = detect_leader_fallback(participants_unique)

            # Resolve leader canonical identity using translations
            leader_pid, leader_cname, leader_lang, leader_match, added = resolve_with_translation(trans_store, leader_name_obs)
            if added:
                trans_modified = True

            # Resolve participants to canonical identities with translations
            participants_resolved = []
            for p in participants_unique:
                pid, cname, lang, matched_by, added = resolve_with_translation(trans_store, p["name"])
                if added:
                    trans_modified = True
                participants_resolved.append({
                    "canonical_id": pid,
                    "name": cname,  # canonical name used downstream
                    "name_original": p["name"],
                    "matched_by": matched_by,
                    "language_detected": lang,
                    "damage": p["damage"],
                    "is_leader": (cname == leader_cname)
                })

            rally = {
                "id": f"{date_str}-rally-{rally_id}",
                "leader": leader_cname,
                "participants": participants_resolved,
            }

            event["rallies"].append(rally)

        # Aggregate players per event by canonical id for web consumption
        players_map = {}  # pid -> total_damage
        id_to_name = {}
        for r in event["rallies"]:
            for p in r["participants"]:
                pid = p.get("canonical_id")
                if not pid:
                    # fallback on original name if no id (shouldn't happen)
                    pid = f"pl_{_normalize_key(p['name'])}"
                players_map.setdefault(pid, 0)
                players_map[pid] += p["damage"]
                # prefer canonical name
                if pid not in id_to_name:
                    id_to_name[pid] = p["name"]

        event["players"] = [
            {"id": pid, "name": id_to_name.get(pid, pid), "total_damage": dmg}
            for pid, dmg in sorted(players_map.items(), key=lambda kv: kv[1], reverse=True)
        ]

        new_events.append(event)
    existing_json["events"].extend(new_events)

    print(f"\n[INFO] Sauvegarde dans {DATA_OUTPUT_FILE}")
    with open(DATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_json, f, indent=2, ensure_ascii=False)

    # Persist aliases if changed or file missing (backward compat)
    try:
        if alias_modified or not ALIASES_FILE.exists():
            print(f"[INFO] Mise à jour des alias dans {ALIASES_FILE}")
            save_alias_store(ALIASES_FILE, alias_store)
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder les alias: {e}")

    # Persist translations if changed or file missing
    try:
        if trans_modified or not TRANSLATIONS_FILE.exists():
            print(f"[INFO] Mise à jour des traductions dans {TRANSLATIONS_FILE}")
            save_translations_store(TRANSLATIONS_FILE, trans_store)
    except Exception as e:
        print(f"[WARN] Impossible de sauvegarder les traductions: {e}")

    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()