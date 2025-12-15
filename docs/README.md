# Bear Trap Analysis

Analyse automatique des résultats Bear Trap à partir de screenshots et export JSON + page web.

## Pré-requis
- Python 3.10+
- `pip install -r requirements.txt`

## Entrées attendues
Placez vos images dans des sous-dossiers par date dans `beartrap_data/`:

```
 beartrap_data/
   2025-12-11/
     1.0.png   # 4 lignes
     1.1.png   # 7 lignes
     1.2.png   # 1 ligne
     total.png # totaux alliance + nombre de ralliements
```

Contraintes:
- `1.0` détecte les rangs par template (1,2,3) et découpe centré sur l’icône de rang.
- `1.1` et `1.2` segmentés sur une grille à hauteur calibrée.

## Lancer l’analyse
```powershell
python .\analyse_beartrap.py
```

Sorties:
- `data/beartrap.json`: tous les événements/ralliements/joueurs + `alliance_total_damage` et `rally_count_total`.
- `data/player_aliases.json`: mapping des identités canoniques (voir ci-dessous).
- `data/debug_rows/`: crops de debug par rang (optionnel).

## Mapping des joueurs (alias → identité canonique)
Le script résout les noms des joueurs vers des identités canoniques et agrège les dégâts par joueur canonique.

Fichier: `data/player_aliases.json` (créé/complété automatiquement)
```json
{
  "players": {
    "pl_buffy": {
      "name": "Buffy",
      "aliases": ["Buffy"],
      "pending_review": false
    }
  },
  "alias_to_id": {
    "buffy": "pl_buffy"
  }
}
```

Principes:
- Normalisation d’alias: minuscules, accents retirés, ponctuation supprimée, espaces compressés.
- Résolution:
  - Si l’alias normalisé existe → on renvoie l’id canonique.
  - Si un joueur canonique a exactement ce nom → on le lie comme alias.
  - Sinon on crée automatiquement un nouveau joueur canonique (`pending_review: true`).
- Les participants exportés sont déjà canoniques: `participants[].name` est le nom canonique; on garde `participants[].name_original`, `canonical_id`, `matched_by`.
- L’agrégation `event.players` utilise `total_damage` par `canonical_id`.

Opérations manuelles courantes:
- Fusionner deux joueurs: choisir l’id cible dans `players`, déplacer les alias de l’autre vers le premier, mettre à jour `alias_to_id` pour pointer vers l’id cible, puis supprimer l’entrée `players` devenue orpheline.
- Renommer un joueur: modifier `players[<id>].name` et éventuellement ses `aliases`.
- Valider un auto-ajout: passer `pending_review` à `false`.

Après modification, relancez l’analyse: les agrégations refléteront le nouveau mapping.

## Web
Les données sont lues depuis `web/index.html` et `web/app.js`.
- Le tableau et le graphe utilisent `event.players[].total_damage` et affichent les noms canoniques.

**Pour visualiser en local:**
```powershell
python -m http.server 8000
```
Puis ouvrez: `http://localhost:8000/web/index.html`

⚠️ N'ouvrez pas `web/index.html` directement en `file://` (double-clic) car les navigateurs bloquent le chargement de fichiers JSON locaux (politique CORS).

## Astuces / Dépannage
- Si `total.png` est bruité, le parseur cherche le plus grand nombre plausible pour les dégâts et un entier raisonnable pour les ralliements.
- Les crops par rang sont dans `data/debug_rows/<page>/` pour aider au contrôle visuel.
