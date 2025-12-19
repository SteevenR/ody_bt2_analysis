"""Text processing utilities for names and damage values"""
import re
import unicodedata


def clean_extracted_name(name: str) -> str:
    """Clean up extracted player name."""
    if not name:
        return ""
    name = re.sub(r"^[Rr]\s+", "", name)
    for pat in [r"points\s+de\s+d[ée]g[âa]ts", r"rapport\s+de\s+combat", r"d[ée]g[âa]ts"]:
        name = re.sub(pat, "", name, flags=re.IGNORECASE)
    name = re.sub(r"[:;.,]+$", "", name)
    return re.sub(r"\s+", " ", name).strip()


def normalize_damage(value_str: str) -> int:
    """Convert damage string to integer (handles K/M suffixes)."""
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
        n = float(ds.replace(" ", "").replace(",", "."))
    except ValueError:
        return 0
    if suf and suf.group(1).lower() == "m":
        n *= 1_000_000
    elif suf and suf.group(1).lower() == "k":
        n *= 1_000
    return int(n)


def _normalize_key(name: str) -> str:
    """Normalize player name for matching."""
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


def detect_language_from_text(text: str) -> str:
    """Detect language from text sample."""
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
