# Configuration pour GitHub Pages

Ce dossier `/web` est servi comme site GitHub Pages.

## URLs
- **Page d'accueil**: `https://[votre-username].github.io/ody_bt2_analysis/`
- **Dashboard**: `https://[votre-username].github.io/ody_bt2_analysis/web/index.html`

## Données
Les fichiers de données (`beartrap.json`, `player_translations.json`, `ui_translations.json`) sont servis depuis le dossier `/data/` à la racine.

## Déploiement automatique
Chaque modification des fichiers suivants déclenche un redéploiement:
- `/beartrap_data/**` (nouvelles données screenshots)
- `/data/beartrap.json` (données analysées)
- `player_aliases.json` ou `player_translations.json` (mapping des joueurs)
- `analyse_beartrap.py` (script d'analyse)

## Déploiement manuel
Via GitHub Actions: cliquer sur "Run workflow" pour forcer un redéploiement.
