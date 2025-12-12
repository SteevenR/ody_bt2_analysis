import json
import re
from pathlib import Path
import easyocr
import cv2


# ============================
# CONFIG
# ============================

BASE_DIR = Path(__file__).parent
DATA_INPUT_DIR = BASE_DIR / "beartrap_data"
DATA_OUTPUT_DIR = BASE_DIR / "data"
DATA_OUTPUT_FILE = DATA_OUTPUT_DIR / "beartrap.json"
FLAG_TEMPLATE_PATH = BASE_DIR / "assets" / "flag.png"

DATA_OUTPUT_DIR.mkdir(exist_ok=True)


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
# PARSING TOTAL
# ============================

def normalize_damage(value_str: str) -> int:
    s = value_str.replace(" ", "").replace(",", ".").lower()
    match = re.match(r"(\d+(\.\d+)?)([mk])?$", s)
    if not match:
        return 0
    number = float(match.group(1))
    suffix = match.group(3)
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

    for line in lines:
        # Nom d'event
        m = re.search(r"\[(.*?)\]", line)
        if m:
            candidate = m.group(1)
            # on prend le dernier, en pratique [Piège de Chasse 2]
            event_name = candidate

        # Dégâts totaux
        if "Dégâts Totaux" in line or "Degats Totaux" in line:
            m = re.search(r"(\d[\d\s\.MKmk]+)", line)
            if m:
                alliance_damage = normalize_damage(m.group(1))

        # Ralliements
        m = re.search(r"Ralliements\s*:\s*(\d+)", line)
        if m:
            rally_count = int(m.group(1))

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
    reader = easyocr.Reader(["fr", "en"])

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
            "name": event_name or "Bear Trap",
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

            all_boxes = []
            leader_name = None

            for f in sorted(files):
                boxes = ocr_image_boxes(reader, f)
                all_boxes.extend(boxes)

                if not leader_name:
                    leader_name = find_leader_from_flag(reader, f, FLAG_TEMPLATE_PATH)

            participants_raw = extract_rally_participants_from_boxes(all_boxes)

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

            if not leader_name:
                leader_name = detect_leader_fallback(participants_unique)

            rally = {
                "id": f"{date_str}-rally-{rally_id}",
                "leader": leader_name,
                "participants": [
                    {
                        "name": p["name"],
                        "damage": p["damage"],
                        "is_leader": (p["name"] == leader_name)
                    }
                    for p in participants_unique
                ]
            }

            event["rallies"].append(rally)

        new_events.append(event)

    existing_json["events"].extend(new_events)

    print(f"\n[INFO] Sauvegarde dans {DATA_OUTPUT_FILE}")
    with open(DATA_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(existing_json, f, indent=2, ensure_ascii=False)

    print("[INFO] Terminé.")


if __name__ == "__main__":
    main()