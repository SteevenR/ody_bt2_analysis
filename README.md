# Bear Trap Analysis

Analyse automatique des résultats Bear Trap à partir de screenshots et export JSON + page web interactive.

## Pré-requis
- Python 3.10+
- ``pip install -r requirements.txt``

## Entrées attendues
Placez vos images dans des sous-dossiers par date dans ``beartrap_data/``:

```
beartrap_data/
  2025-12-11/
    1.0.png   # 4 lignes
    1.1.png   # 7 lignes
    1.2.png   # 1 ligne
    total.png # totaux alliance + nombre de ralliements
  2025-12-12/
    1.0.png
    1.1.png
    1.2.png
    # Bear Trap Event Analysis Dashboard
    Système d'analyse et de visualisation des événements "Bear Trap" avec extraction OCR, reconnaissance multilingue des joueurs, et dashboard interactif.

    ## 🎯 Vue d'ensemble

    Le système capture des screenshots des résultats Bear Trap, extrait automatiquement les données (noms, dégâts, classements), et génère un dashboard web pour visualiser:
    - Statistiques par événement et ralliement
    - Classements des joueurs
    - Évolution des dégâts dans le temps
    - Noms des joueurs en 6 langues (FR, EN, ES, ZH, RU, KO)
- ``total.png`` doit contenir les labels "Ralliements" et "Dégâts Totaux de l'Alliance" pour extraction correcte.
    ## 📁 Structure du projet
```powershell
python .\analyse_beartrap.py
```
      YYYY-MM-DD/                  # Dossier par date d'événement
## Gestion multilingue des noms

Le système détecte automatiquement la langue d'un nom de joueur et gère les translittérations pour les caractères non-latins.

    data/                          # Données générées (sortie OCR)
      beartrap.json               # Tous les événements/rallies/participants
      player_translations.json     # Noms en plusieurs langues + translittération
      player_aliases.json          # Mapping noms → IDs (legacy)
      ui_translations.json         # Interface en 6 langues
      ocr_extraction_log.jsonl     # Log détaillé de chaque extraction OCR
    web/                           # Dashboard interactif
      index.html, app.js, style.css
    assets/                        # Templates pour détection rang
      rank1.png, rank2.png, rank3.png, flag.png
    .github/workflows/deploy.yml  # CI/CD automatique
### Structure des noms

    ## 🐛 Déboguer les erreurs OCR

    Fichier: `data/ocr_extraction_log.jsonl` — chaque ligne contient:
    ```json
    {
      "event": "2025-12-15",
      "rally": 3,
      "file": "3.1.png",
      "rank": 7,
      "raw_ocr": "The 1 True 7 Points de Dégâts : 2410859 King",
      "extracted_name": "The 1 True King",
      "damage": 2410859,
      "tokens": [
        {"text": "The 1 True", "x": 625, "y": 128, "conf": 0.89},
        {"text": "King", "x": 919, "y": 132, "conf": 1.0}
      ]
    }
    ```

    **Interpréter les erreurs:**
    - Nom contient labels → vérifier les tokens non-filtrés
    - Dégâts incorrect → vérifier le plus grand nombre dans les tokens
    - Confiance basse (< 0.4) → token peut être ignoré

    ## 🔧 Éditer les noms manuellement

    Fichier: `data/player_translations.json`

    ```json
    {
      "players": {
        "pl_buffy": {
          "names_by_language": {
            "fr": "Buffy",
            "en": "Buffy",
            "es": "Buffy",
            "ru": "Буффи",
            "zh": "巴菲",
            "ko": "버피"
          },
          "transliterations_by_language": {
            "ru": "Buffi"
          },
          "language_detected": "en",
          "aliases": ["Buffy", "Buff"],
          "pending_review": false
        }
      }
    }
    ```

    **Après édition:** committer et pousser vers GitHub — le workflow redéploiera automatiquement.

    ## 🔄 CI/CD Workflow

    **Fichier:** `.github/workflows/deploy.yml`

    **Triggers:**
    - Push vers `main` + changement dans `beartrap_data/`, `data/`, ou `beartrap_analysis_parallel.py`
    - Manuel: Actions → "Run workflow"

    **Étapes:**
    1. Run `python beartrap_analysis_parallel.py`
    2. Commit `data/*.json`, `player_translations.json` back to `main`
    3. Copy `web/*` + `data/*` to `deploy/`
    4. Push to `gh-pages` branch
    5. GitHub Pages déploie automatiquement

    **Configuration GitHub Pages:**
    - Settings → Pages → select `gh-pages` branch
    - URL: `https://SteevenR.github.io/ody_bt2_analysis/`

    ## 📋 Checklist post-amélioration

    Après une modification du code (extraction, nettoyage, etc.):

    - [ ] Tester localement: `python beartrap_analysis_parallel.py`
    - [ ] Inspecter `data/ocr_extraction_log.jsonl` pour les erreurs
    - [ ] Vérifier `data/player_translations.json` pour les nouveaux joueurs
    - [ ] Ouvrir dashboard local: `python -m http.server 8000`
    - [ ] Visiter `http://localhost:8000` et tester les langues
    - [ ] Pousser vers GitHub
    - [ ] Attendre le workflow (~5 min)
    - [ ] Visiter GitHub Pages et vérifier les changes

    ## 📚 Ressources

    - **EasyOCR:** https://github.com/JaidedAI/EasyOCR
    - **OpenCV:** https://opencv.org/
    - **Chart.js:** https://www.chartjs.org/

    ---

    **Version:** 2.0  
    **Dernière maj:** 2025-12-15
- **Problèmes multilingues:** Vérifiez ``data/player_translations.json``; un nouveau joueur peut être auto-ajouté avec ``pending_review: true``. Utilisez ``web/translations.html`` pour réviser, puis relancez.

## Fichiers clés

- ``beartrap_analysis_parallel.py``  Pipeline complet (parallèle: calibration, segmentation, OCR, JSON multilingue).
- ``web/index.html``  Page web avec sections interactives.
- ``web/translations.html``  Gestion des noms multilingues des joueurs.
- ``web/app.js``  Chargement JSON, agrégations, graphes Chart.js.
- ``web/style.css``  Styles (dark theme).
- ``data/beartrap.json``  Résultat JSON final (événements, ralliements, participants, joueurs).
- ``data/player_translations.json``  Mapping des joueurs avec translittérations (éditable manuellement).
- ``assets/rank1.png``, ``rank2.png``, ``rank3.png``  Templates pour détection des rangs.
- ``assets/flag.png``  Template pour détection du leader (non critique).
