#!/usr/bin/env python3
"""
Test simple du système multilingue de translittération
"""

import re

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

def detect_language(text: str) -> str:
    """Détecte la langue basée sur les caractères Unicode."""
    if re.search(r'[\u4E00-\u9FFF]', text):
        return "zh"
    if re.search(r'[\uAC00-\uD7AF]', text):
        return "ko"
    if re.search(r'[\u0400-\u04FF]', text):
        return "ru"
    return "en"

def has_non_latin_chars(text: str) -> bool:
    """Détecte si le texte contient des caractères non-latins."""
    for ch in text:
        code = ord(ch)
        if code > 0x024F and code not in [0x0020, 0x0027, 0x002D]:
            return True
    return False

# Tests
test_cases = [
    "Buffy",              # EN
    "José",               # ES (latin avec accent)
    "Баффи",              # RU
    "巴菲",                # ZH
    "버피",                # KO
    "Жан-Клод",           # RU multi-word
]

print("=== Test du système multilingue ===\n")

for name in test_cases:
    lang = detect_language(name)
    has_non_latin = has_non_latin_chars(name)
    
    trans = None
    if lang == "ru" and has_non_latin:
        trans = transliterate_cyrillic(name)
    
    display = f"{name} ({trans})" if trans and trans != name else name
    
    print(f"Nom: {name:15} | Langue: {lang:2} | Non-latin: {str(has_non_latin):5} | Affichage: {display}")

print("\n=== ✓ Tests complétés ===")
