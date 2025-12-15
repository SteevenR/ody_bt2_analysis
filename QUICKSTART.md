# Quick Start

## 1. Préparer les données
Placez vos screenshots dans `beartrap_data/YYYY-MM-DD/`:
```
beartrap_data/2025-12-11/
  1.0.png      (4 joueurs)
  1.1.png      (7 joueurs)
  1.2.png      (1 joueur)
  total.png    (totaux alliance)
```

## 2. Analyser
```powershell
python .\analyse_beartrap.py
```

Sorties:
- `data/beartrap.json` (événements + ralliements + joueurs)
- `data/player_translations.json` (noms multilingues avec translittérations)

## 3. Visualiser
```powershell
python -m http.server 8000
# Puis ouvrir http://localhost:8000/web/index.html
```

## 4. Gérer les noms multilingues
Les noms des joueurs sont conservés tels qu'observés:
- **Noms latins** (Buffy, José, etc.) → affichés tels quels
- **Noms non-latins** (Баффи, 巴菲, 버피) → affichés avec translittération phonétique (Baffi, etc.)

Allez sur: `http://localhost:8000/web/translations.html`

Pour:
- Voir tous les joueurs avec leurs translittérations
- Identifier les auto-ajouts à réviser (`pending_review`)
- Corriger/ajouter des translittérations manuellement
- Fusionner des doublons
- Valider les joueurs

Édition manuelle: `data/player_translations.json` (format JSON)

Relancez l'analyse pour mettre à jour.

## Commandes rapides
```powershell
# Extraire seulement
python .\analyse_beartrap.py

# Servir localement
python -m http.server 8000

# Arrêter le serveur
Ctrl+C

# Tester la translittération
python test_multilingual.py
```

Plus de détails: voir `README.md`
