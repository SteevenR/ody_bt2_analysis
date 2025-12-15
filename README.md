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
    total.png
```

Contraintes:
- ``1.0`` détecte les rangs par template matching (icônes 1, 2, 3) et découpe centré sur l'icône de rang.
- ``1.1`` et ``1.2`` segmentés sur une grille à hauteur calibrée.
- ``total.png`` doit contenir les labels "Ralliements" et "Dégâts Totaux de l'Alliance" pour extraction correcte.

## Lancer l'analyse
```powershell
python .\analyse_beartrap.py
```

Sorties:
- ``data/beartrap.json``: tous les événements/ralliements/joueurs avec ``alliance_total_damage`` et ``rally_count_total``.
- ``data/player_translations.json``: mapping des joueurs (support 6 langues: FR, EN, ES, ZH, RU, KO).
- ``data/debug_rows/``: crops de debug par rang pour contrôle visuel (optionnel).

## Gestion multilingue des noms

Le système détecte automatiquement la langue d'un nom de joueur et gère les translittérations pour les caractères non-latins.

### Structure des noms
- **Noms latins** (EN, FR, ES): conservés tels quels
  - `Buffy`  `Buffy`
  - `José`  `José`

- **Noms non-latins** (ZH, RU, KO): conservés + translittération phonétique
  - `巴菲` (chinois)  `巴菲 (Bafeì)`
  - `Баффи` (russe)  `Баффи (Buffi)`
  - `버피` (coréen)  `버피 (Beopi)`

### Fichier: ``data/player_translations.json``

```json
{
  "languages": {
    "fr": "Français",
    "en": "English",
    "es": "Español",
    "zh": "中文",
    "ru": "Русский",
    "ko": "한국어"
  },
  "primary_language": "fr",
  "players": {
    "pl_buffy": {
      "name": "Буффи",
      "language": "ru",
      "transliteration": "Buffi",
      "aliases": ["Буффи", "Buff"],
      "pending_review": false
    }
  },
  "alias_to_id": {
    "buffi": "pl_buffy"
  }
}
```

### Fonctionnement
1. **Détection de langue**: L'OCR analyse les caractères Unicode du nom
   - CJK (U+4E00U+9FFF)  Chinois
   - Hangul (U+AC00U+D7AF)  Coréen
   - Cyrillique (U+0400U+04FF)  Russe
   - Sinon  Latin (EN/FR/ES)

2. **Translittération**: Si le nom contient des caractères non-latins, on génère une approximation phonétique
   - Cyrillique  conversion simple (БB, УU, etc.)
   - Chinois/Coréen  placeholder (nécessite bibliothèque pinyin/romanization)

3. **Résolution**: Un nom observé est résolu en:
   - Cherchant dans les aliases normalisés
   - Cherchant dans les noms exacts
   - Créant automatiquement un nouveau joueur (`pending_review: true`)

4. **Affichage**: `Nom (Translittération)` si non-latin, sinon juste `Nom`

### Gestion manuelle
Utilisez ``web/translations.html`` pour:
- Voir tous les joueurs avec leurs traductions
- Identifier les auto-ajouts à réviser (`pending_review: true`)
- Ajouter/corriger des translittérations manquantes
- Fusionner des doublons (deux ids pour la même personne)
- Valider en passant ``pending_review`` à ``false``

Après modification, relancez l'analyse pour mettre à jour les agrégations.

## Web  Dashboard statistiques
Visualisez les données via une page web interactive servie localement.

**Pour visualiser en local:**
```powershell
python -m http.server 8000
```
Puis ouvrez: ``http://localhost:8000/web/index.html``

 N'ouvrez pas ``web/index.html`` directement en ``file://`` (double-clic) car les navigateurs bloquent le chargement de fichiers JSON locaux (politique CORS).

**Sections du dashboard:**

1. **Dernier événement**  Résumé et classement des joueurs pour l'événement le plus récent.

2. **Évolution des événements**  Multi-axe (dégâts totaux, ralliements, joueurs présents) avec toggles pour afficher/masquer chaque série.

3. **Évolution des joueurs (Top N)**  Courbes de dégâts cumulés pour les N meilleurs joueurs (top 5, 10, 15 sélectionnable).

4. **Leader  Totaux des ralliements**  Courbe de dégâts générés par les ralliements d'un leader, avec option pour voir les points par ralliement.

5. **Joueur  Meilleur ralliement par événement**  Tableau et graphe montrant le meilleur ralliement de chaque joueur par événement.

6. **Historique des événements**  Liste simple des dates et comptages.

7. **Gestion des traductions**  Page dédiée pour gérer les noms multilingues des joueurs.

## Notes techniques

- **Calibration (phase 1):** Détecte les rangs 1, 2, 3 sur ``1.0.png`` via template matching sur les assets (rank1.png, rank2.png, rank3.png). Calcule la hauteur de ligne et centrage.
- **Segmentation (phase 2):** Découpe ``1.0`` centré sur les rangs détectés; ``1.1`` et ``1.2`` en grille régulière ancrée au premier rang trouvé via OCR.
- **Extraction (phase 3):** OCR par ligne avec groupage des tokens par bande Y pour reconstruire les noms; extraction des dégâts par label "Points de Dégâts" ou plus grand nombre numérique.
- **Totaux (phase 4):** Parse ``total.png`` pour récupérer ``alliance_total_damage`` et ``rally_count_total``; cherche les plus grands nombres plausibles en cas de bruit.
- **Détection de langue**: Analyse les plages Unicode pour identifier la langue du nom.
- **Translittération**: Conversion basique pour le Cyrillique; placeholders pour CJK (nécessite dépendances externes).

## Dépannage

- **Si ``total.png`` est bruité:** le parseur cherche le plus grand nombre plausible pour les dégâts et un entier raisonnable (1999) pour les ralliements. Vérifiez que les labels "Ralliements" et "Dégâts Totaux de l'Alliance" sont visibles.
- **Crops debug:** consultez ``data/debug_rows/<event>/<page>/rankXX_*.png`` pour inspecter chaque ligne extraite.
- **Tendances vides sur le dashboard:** avec un seul événement, les graphes d'évolution n'affichent qu'un point. Ajoutez plusieurs dates dans ``beartrap_data/`` pour voir les courbes évolutives.
- **Translittération incomplète:** Pour le Chinois et Coréen, installez ``pip install pinyin`` pour une translittération complète.
- **Problèmes multilingues:** Vérifiez ``data/player_translations.json``; un nouveau joueur peut être auto-ajouté avec ``pending_review: true``. Utilisez ``web/translations.html`` pour réviser, puis relancez.

## Fichiers clés

- ``analyse_beartrap.py``  Pipeline complet (calibration  segmentation  OCR  JSON multilingue).
- ``web/index.html``  Page web avec sections interactives.
- ``web/translations.html``  Gestion des noms multilingues des joueurs.
- ``web/app.js``  Chargement JSON, agrégations, graphes Chart.js.
- ``web/style.css``  Styles (dark theme).
- ``data/beartrap.json``  Résultat JSON final (événements, ralliements, participants, joueurs).
- ``data/player_translations.json``  Mapping des joueurs avec translittérations (éditable manuellement).
- ``assets/rank1.png``, ``rank2.png``, ``rank3.png``  Templates pour détection des rangs.
- ``assets/flag.png``  Template pour détection du leader (non critique).
