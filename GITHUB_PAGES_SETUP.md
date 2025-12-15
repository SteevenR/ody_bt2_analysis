# Configuration GitHub Pages pour ody_bt2_analysis

## Étapes de configuration

### 1. Activer GitHub Pages

1. Allez sur **Settings** du repository GitHub
2. Allez à **Pages** (menu de gauche)
3. Sous "Source", sélectionnez:
   - **Branch**: `gh-pages`
   - **Folder**: `/ (root)`
4. Cliquez sur **Save**

### 2. Workflow GitHub Actions automatique

Le workflow `.github/workflows/deploy.yml` se charge de tout:

1. **Détecte les changements** dans:
   - `/beartrap_data/**` (nouveaux screenshots)
   - `/data/beartrap.json` (données analysées)
   - `player_aliases.json` ou `player_translations.json` (mapping joueurs)
   - `analyse_beartrap.py` (script d'analyse)

2. **Exécute les actions**:
   - Installe les dépendances Python
   - Exécute `python analyse_beartrap.py` pour générer les données
   - Copie le contenu de `/web` dans un dossier `deploy/`
   - Copie les données JSON générées dans `deploy/data/`
   - Pousse tout vers la branche `gh-pages`

3. **GitHub Pages déploie** automatiquement depuis `gh-pages`

### 3. URLs d'accès

Une fois configuré, votre dashboard sera accessible à:

```
https://SteevenR.github.io/ody_bt2_analysis/
```

## Flux de déploiement

```
1. Vous pushez des changements dans /beartrap_data/ ou modifiez player_aliases.json
   ↓
2. GitHub Actions détecte le changement et lance le workflow
   ↓
3. Python exécute analyse_beartrap.py et génère les fichiers JSON
   ↓
4. Le workflow copie:
   - /web/* → dossier deploy/
   - /data/* → dossier deploy/data/
   ↓
5. Les fichiers sont poussés vers la branche gh-pages
   ↓
6. GitHub Pages détecte le changement sur gh-pages
   ↓
7. Votre site est à jour automatiquement!
```

### 3. URLs d'accès

Une fois configuré, votre dashboard sera accessible à:

```
https://SteevenR.github.io/ody_bt2_analysis/
```

ou directement:

```
https://SteevenR.github.io/ody_bt2_analysis/web/index.html
```

## Structure du déploiement

```
Repository main (branche main)
├── .github/workflows/deploy.yml    (workflow)
├── analyse_beartrap.py             (exécuté par le workflow)
├── beartrap_data/                  (screenshots - source)
├── data/                           (données générées)
│   ├── beartrap.json
│   ├── player_translations.json
│   └── ui_translations.json
└── web/                            (source du site)
    ├── index.html
    ├── app.js
    └── style.css

                  ↓ (workflow)

GitHub Pages (branche gh-pages)
├── index.html                      (copié de /web/)
├── app.js                          (copié de /web/)
├── style.css                       (copié de /web/)
├── data/                           (copié de /data/)
│   ├── beartrap.json
│   ├── player_translations.json
│   └── ui_translations.json
└── .nojekyll                       (empêche Jekyll)

                  ↓

GitHub Pages sert le contenu à:
https://SteevenR.github.io/ody_bt2_analysis/
```

## Déploiement manuel

Pour forcer un redéploiement sans modifier les fichiers:

1. Allez sur **Actions** du repository GitHub
2. Sélectionnez le workflow **"Build & Deploy Dashboard"**
3. Cliquez sur **"Run workflow"** → **"Run workflow"**

Le workflow:
- Relance l'analyse Python
- Reconstruit le site
- Pousse vers la branche `gh-pages`
- GitHub Pages se met à jour automatiquement

## Dépannage

### Le site ne se met pas à jour
1. Vérifiez que la branche `gh-pages` existe: **Code** → **Branches**
2. Vérifiez que GitHub Pages est configuré sur `gh-pages`: **Settings** → **Pages**
3. Vérifiez les logs du workflow: **Actions** → dernier run
4. Attendez quelques secondes et rafraîchissez le navigateur

### Erreur "file not found" sur le site
- Les chemins dans `app.js` utilisent `getDataPath()` qui s'adapte automatiquement
- Vérifiez que `/data/` contient les fichiers JSON
- Vérifiez les logs du workflow pour les erreurs Python

### Erreur de permission du workflow
- Allez sur **Settings** → **Actions** → **General**
- Dans **Workflow permissions**, sélectionnez **"Read and write permissions"** puis **Save**
- Cochez la case **"Allow GitHub Actions to create and approve pull requests"**
- Si le push échoue avec `403`, appliquez ces réglages puis relancez le workflow

### La branche gh-pages n'existe pas
- Lancez manuellement le workflow une fois: **Actions** → **"Run workflow"**
- Cela créera la branche `gh-pages` automatiquement

## Points importants

- **Les screenshots ne sont pas déployés** (trop volumineux) - seules les données JSON l'are
- **Le workflow s'exécute automatiquement** quand vous ajoutez de nouveaux screenshots
- **Les utilisateurs voient toujours les données à jour** sans intervention manuelle
- **Le site est en lecture seule** (pas d'édition en ligne)

## Technologies utilisées

- **GitHub Pages** pour l'hébergement
- **GitHub Actions** pour les déploiements automatiques
- **Chart.js** pour les graphiques
- **Bootstrap/CSS** pour le style
- **Python + OpenCV + EasyOCR** pour l'analyse des screenshots

---

**Mise à jour du déploiement:** 2025-12-15
**Version:** 1.0 - Configuration initiale
