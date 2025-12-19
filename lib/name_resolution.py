"""Player name resolution and translation management"""
import json
from pathlib import Path
from .text_utils import _normalize_key, _fuzzy_normalize_key, detect_language_from_text

SUPPORTED_LANGUAGES = ["fr", "en", "es", "zh", "ru", "ko"]
PRIMARY_LANGUAGE = "fr"


def load_translations_store(path: Path) -> dict:
    """Load player translations store."""
    if not path.exists():
        return {
            "players": {},
            "alias_to_id": {},
            "primary_language": PRIMARY_LANGUAGE,
            "supported_languages": SUPPORTED_LANGUAGES
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"players": {}, "alias_to_id": {}, "primary_language": PRIMARY_LANGUAGE, "supported_languages": SUPPORTED_LANGUAGES}


def save_translations_store(path: Path, store: dict):
    """Save player translations store."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, ensure_ascii=False)
    tmp.replace(path)


def add_player_translation(trans_store: dict, player_id: str, original_name: str, language: str):
    """Add player translation to store."""
    if player_id not in trans_store["players"]:
        trans_store["players"][player_id] = {
            "canonical_id": player_id,
            "names_by_language": {},
            "language_detected": language,
            "aliases": []
        }
    player = trans_store["players"][player_id]
    if not player["names_by_language"].get(language):
        player["names_by_language"][language] = original_name
    player["language_detected"] = language
    if original_name not in player.get("aliases", []):
        player.setdefault("aliases", []).append(original_name)


def get_name_for_language(trans_store: dict, player_id: str, language: str) -> str:
    """Get player name for specific language."""
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
    """Resolve observed name to canonical player ID and name.
    
    Returns: (player_id, canonical_name, language, matched_by, was_added)
    """
    if not observed_name:
        return None, None, None, None, False
    
    language = detect_language_from_text(observed_name)
    k = _normalize_key(observed_name)
    
    if k in trans_store.get("alias_to_id", {}):
        pid = trans_store["alias_to_id"][k]
        cname = get_name_for_language(trans_store, pid, language)
        return pid, cname, language, "alias", False
    
    # Try exact match first
    for pid, player in trans_store.get("players", {}).items():
        for lang, name in player.get("names_by_language", {}).items():
            if name and _normalize_key(name) == k:
                trans_store.setdefault("alias_to_id", {})[k] = pid
                cname = get_name_for_language(trans_store, pid, language)
                return pid, cname, language, "exact", False
    
    # Fuzzy match for OCR errors
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
    return pid, cname, language, "new", True
