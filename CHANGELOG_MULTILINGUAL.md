# Changelog - Système Multilingue

## ✨ Nouvelles fonctionnalités

### 1. Support multilingue pour les noms de joueurs
Le système détecte automatiquement la langue d'un nom et gère les translittérations pour les caractères non-latins.

**Noms supportés:**
- 🇬🇧 Anglais: `Buffy` → `Buffy`
- 🇫🇷 Français: `José` → `José`
- 🇪🇸 Espagnol: caractères latins
- 🇷🇺 Russe (Cyrillique): `Баффи` → `Баффи (Baffi)` ✓
- 🇨🇳 Chinois (CJK): `巴菲` → `巴菲 (Bafeì)` (placeholder)
- 🇰🇷 Coréen (Hangul): `버피` → `버피 (Beopi)` (placeholder)

### 2. Translittération phonétique
Les noms non-latins sont affichés avec leur translittération:
- **Cyrillique (RU):** Conversion complète (Б→B, У→U, etc.)
- **Chinois (ZH):** Placeholder (nécessite `pip install pinyin`)
- **Coréen (KO):** Placeholder (nécessite romanization)

### 3. Nouvelle interface de gestion
Page dédiée: `web/translations.html`
- Voir tous les joueurs avec leurs translittérations
- Identifier les auto-ajouts à réviser
- Corriger/ajouter des translittérations
- Fusionner les doublons
- Valider les joueurs

## 📁 Fichiers modifiés

### Backend
- **`analyse_beartrap.py`**
  - Nouvelles fonctions: `transliterate_cyrillic()`, `detect_language_from_text()`, `has_non_latin_chars()`, `get_display_name()`
  - Fonction `resolve_with_translation()` remplace `resolve_canonical()`
  - Support OCR multilingue (fallback à FR/EN si langues unavailable)

### Data
- **`data/player_translations.json`** (remplace `player_aliases.json`)
  - Structure: `name`, `language`, `transliteration`, `aliases`, `pending_review`
  - Format simplifié: un nom + translittération par joueur

### Frontend
- **`web/translations.html`** (nouveau)
  - Interface de gestion des traductions
  - Filtres, statistiques, édition

### Documentation
- **`README.md`** (mis à jour)
  - Section "Gestion multilingue des noms" détaillée
  - Exemples de translittération
  - Instructions de gestion manuelle
- **`QUICKSTART.md`** (mis à jour)
  - Étape 4: "Gérer les noms multilingues"

## 🔧 API Changes

### Deprecated (encore supporté pour backward compat)
- `resolve_canonical()` → utiliser `resolve_with_translation()`
- `player_aliases.json` → utiliser `player_translations.json`

### Nouveaux utilitaires
```python
detect_language_from_text(text: str) -> str
# Détecte la langue (en, fr, es, ru, zh, ko)

transliterate_non_latin(text: str, language: str) -> str
# Translittère un texte non-latin

has_non_latin_chars(text: str) -> bool
# Vérifie si le texte contient des caractères non-latins

get_display_name(trans_store: dict, player_id: str) -> str
# Récupère le nom d'affichage (avec translittération si applicable)
```

## 📊 Exemple de structure

### Avant (player_aliases.json)
```json
{
  "players": {
    "pl_buffy": {
      "name": "Buffy",
      "aliases": ["Buffy"],
      "pending_review": false
    }
  },
  "alias_to_id": {"buffy": "pl_buffy"}
}
```

### Après (player_translations.json)
```json
{
  "players": {
    "pl_buffy": {
      "name": "Буффи",
      "language": "ru",
      "transliteration": "Baffi",
      "aliases": ["Буффи"],
      "pending_review": false
    }
  },
  "alias_to_id": {"baffi": "pl_buffy"}
}
```

## 🧪 Tests

Exécutez le test multilingue:
```powershell
python test_multilingual.py
```

## ⚠️ Notes

1. **EasyOCR fallback:** Si certaines langues (ZH, RU, KO) ne sont pas disponibles, le système utilise FR/EN
2. **Translittération CJK:** Pour Chinois et Coréen, installer:
   ```powershell
   pip install pinyin
   ```
3. **Backward compat:** L'ancien `player_aliases.json` est toujours généré en parallèle

## 🚀 Prochaines étapes optionnelles

- Ajouter translittération complète pour CJK via `pinyin` library
- Éditeur JSON graphique pour `player_translations.json`
- Export CSV multilingue
- Statistiques par langue
