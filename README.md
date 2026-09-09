# Inspection visuelle de faisceaux de câbles

Prototype local avec **YOLOv8-Seg**, OpenCV, MQTT et **FlowFuse Dashboard pour
Node-RED**. Le dashboard affiche la vidéo annotée et le verdict OK/NOK.
Aucun service cloud n'est nécessaire pendant la démonstration.

## Démonstration finale

`python run.py demo` enchaîne **test1.MOV puis test2.MOV, en boucle**. Le modèle
n'est chargé qu'une fois. La connexion MQTT, la fenêtre et les compteurs restent
actifs entre les vidéos ; les indices d'images sont continus.

```text
test1.MOV -> test2.MOV -> test1.MOV -> ...
                 Python / best_v02.pt
                         |
                  MQTT local : 1883
                         |
                Node-RED : vidéo + OK/NOK
```

La vitesse dépend de l'inférence CPU, pas nécessairement des 30 images/s de la
source. `Q` ou `Échap` ferme la fenêtre ; `Ctrl+C` arrête le mode sans fenêtre.

## Installation

Utiliser **Python 3.11, 64 bits**, et Node.js 22 (>= 22.9) ou 24 LTS avec npm. Créer une venv sur chaque
ordinateur : ne pas copier celle du Mac sur Windows. Les versions Python
principales sont fixées d'après l'environnement Mac Intel testé.

Node-RED recommande une version LTS de Node.js ; consulter les
[versions prises en charge](https://nodered.org/docs/faq/node-versions).
Le paquet inclut Node-RED 4.1.15 et FlowFuse Dashboard 1.30.2.
Le verrou npm inclut les correctifs des dépendances `npm` et `qs` via
`overrides` ; il ne modifie pas l'installation Node-RED globale.

Toutes les commandes suivantes s'exécutent **à la racine de ce dossier**.

### Windows : PowerShell ou CMD

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
npm.cmd --prefix nodered ci
```

Ces commandes appellent directement Python dans la venv : aucune activation
ni modification de la politique PowerShell n'est nécessaire. Ne pas utiliser
`source`, qui est une commande Mac/Linux. Si une venv existe avec un autre
Python, la renommer avant d'en créer une nouvelle.

### Mac Intel

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
npm --prefix nodered ci
```

L'installation initiale nécessite Internet. Ensuite, les poids et vidéos
fournis permettent une exécution locale. La combinaison historique PyTorch
2.2.2 vise Python 3.11 sur Mac Intel/Windows x64, pas toutes les plateformes.
SAM 2 s'installe séparément dans Colab, **jamais dans cette venv**.

## Lancement en trois terminaux

Ouvrir trois terminaux à la racine du projet. Sur Mac, remplacer
`.\.venv\Scripts\python.exe` par `.venv/bin/python` et `npm.cmd` par `npm`.

### 1. Courtier MQTT

```powershell
.\.venv\Scripts\python.exe scripts/start_broker.py
```

Le courtier Python écoute uniquement sur `127.0.0.1:1883`. Si Mosquitto ou un
autre courtier fonctionne déjà sur ce port, **ne pas en lancer un deuxième**.
Node-RED n'est pas lui-même le courtier MQTT.

### 2. Dashboard Node-RED

```powershell
npm.cmd --prefix nodered start
```

Ouvrir **http://127.0.0.1:1880/inspection/**.

Cette commande charge directement le flow fourni : **aucun import manuel**
n'est nécessaire. Elle utilise `nodered/runtime/`, séparé du dossier personnel
`.node-red`. Si une autre instance occupe le port 1880, l'arrêter ou utiliser
`npm.cmd --prefix nodered start -- --port 1881`, puis ouvrir le port 1881.

Pour conserver une installation Node-RED existante, installer
`@flowfuse/node-red-dashboard`, puis importer
`nodered/wire_harness_dashboard_flow.json` **une seule fois**, en remplaçant
l'ancien flow d'inspection. Vérifier qu'il reste un seul `ui-base`. Ne pas
supprimer les autres flows personnels. Le projet n'utilise pas l'ancien
dashboard Angular. Voir le
[guide officiel FlowFuse](https://dashboard.flowfuse.com/getting-started.html).

### 3. Analyse des vidéos

```powershell
.\.venv\Scripts\python.exe run.py demo
```

Les masques s'affichent localement et sur Node-RED : connecteur rouge, clip
bleu, câble vert. Python calcule le statut ; Node-RED ne recalcule pas la règle.

Variantes, avec la venv activée ou son chemin Python complet :

```bash
python run.py demo --once
python run.py demo --no-display
python run.py offline --video data/videos/test1.MOV
python run.py offline --video data/videos/test2.MOV
python run.py webcam --camera 0
python run.py esp --stream-url http://192.168.1.50:81/stream
```

`--once` joue les deux vidéos une fois. `--no-display` désactive uniquement
la fenêtre locale, pas MQTT. `--no-mqtt` est réservé aux tests sans dashboard :
**ne pas l'ajouter à la démonstration Node-RED**.

## Règle OK/NOK et configuration

**NOK est prioritaire** si un clip et un connecteur détectés ont des masques
qui se croisent, des boîtes qui se touchent, ou un écart entre les bords des
boîtes inférieur ou égal à **20 pixels**. La confiance minimale est **0,35**.

**OK est le statut par défaut** dans tous les autres cas, même lorsqu'un objet
n'est pas détecté. Aucun INDETERMINE n'est produit. OK ne signifie donc pas
que la conformité physique est confirmée. Les erreurs techniques restent des
erreurs et ne sont pas transformées en OK.

Le câble reste segmenté et affiché mais ne participe plus au verdict.
Le nom des vidéos ne décide jamais du statut. La distance utilise les pixels
de la vidéo originale : la recalibrer pour une autre caméra. Une proximité 2D
ne prouve pas une attache mécanique.

Paramètres principaux dans `config/app.yaml` :

| Paramètre | Valeur de la démonstration |
| --- | --- |
| `source.demo_videos` | `data/videos/test1.MOV`, puis `data/videos/test2.MOV` |
| `detector.yolo_model_path` | `models/best_v02.pt` |
| `validation.mode` | `clip_attachment` |
| `validation.clip_attachment.min_confidence` | `0.35` |
| `validation.clip_attachment.connector_near_px` | `20.0` |

## MQTT et diagnostics

`config/mqtt.json` active MQTT par défaut. Python utilise l'identifiant
`wire-harness-python` et Node-RED `wire-harness-dashboard` : des identifiants
distincts évitent les déconnexions mutuelles.

| Topic | Contenu |
| --- | --- |
| `factory/inspection/status` | Statut, timestamp, source, indice global, détails |
| `factory/inspection/video_stream` | JPEG annoté dans `image_base64`, statut, source, indice global |
| `factory/inspection/events` | Changement de vidéo/statut et événements NOK |
| `factory/inspection/metrics` | Compteurs cumulatifs d'images, pas de pièces |
| `factory/inspection/snapshot` | Capture NOK et chemin local |

Les widgets conservent le format d'origine : `payload.status` et
`payload.image_base64`. Les autres champs sont des diagnostics additionnels.
`relation.decision_basis` distingue `connector_contact` et `default_ok`.
Pour OK par défaut, la confiance vaut 0 : aucune probabilité de conformité
n'est calculée. Le compteur historique `indeterminate_count` reste à 0.

Les dernières images et le dernier statut sont retenus par MQTT et restent
visibles après l'arrêt. Vérifier le timestamp : une image affichée ne prouve
pas que Python tourne encore. Python se reconnecte automatiquement au
courtier ; les images perdues pendant une coupure ne sont pas rejouées.

Si le dashboard reste figé alors que Python publie, vérifier qu'un seul
courtier MQTT sert les deux applications. Deux instances Mosquitto concurrentes
peuvent conserver des connexions sur des courtiers différents malgré le même
port affiché. Reconnecter Node-RED au courtier utilisé par Python, sans ajouter
un deuxième flow ou un deuxième `ui-base`.

Les journaux sont dans `data/logs/` et les captures dans `data/snapshots/`.
L'ancien helper `src/utils/inspection_logger.py` est conservé pour compatibilité ;
l'application utilise la journalisation centralisée `src/inspection_logger.py`.

## Entraînement et annotations

Ouvrir **`ENTRAINEMENT_YOLO.ipynb`** dans Google Colab et charger
`data/datasets/faisceau_seg_essai_actuel.zip`. Le notebook utilise
`yolov8n-seg.pt` avec `0: connector`, `1: clip`, `2: cable`.

Pour préparer d'autres annotations : `ANNOTATION_SAM2_COLAB.ipynb`,
`scripts/annotate_sam2.py` et `ANNOTATION_SAM2.md`. Ces étapes peuvent nécessiter
Internet et un GPU Colab. Les notebooks mentionnent parfois `data/models/` :
**dans cette distribution, placer les nouveaux poids dans `models/`**, puis
adapter `detector.yolo_model_path`.

Le dataset est un **jeu d'essai SAM 2 imparfait**, pas une vérité terrain
industrielle entièrement approuvée. Contrôler les annotations avant de
conclure à une performance réelle du modèle.

## Vérification et limites

```bash
python -m unittest discover -s test -p "test_*.py" -q
python run.py demo --once --no-display
```

Derniers essais individuels avec `best_v02.pt` et les seuils fournis :

| Vidéo | Images | OK par défaut | NOK |
| --- | ---: | ---: | ---: |
| `test1.MOV`, scénario NOK | 126 | 33 | 93 |
| `test2.MOV`, scénario OK | 265 | 256 | 9 |

Les OK de ces essais correspondent à une absence de détection du clip.
**Neuf fausses alertes subsistent sur test2** : les boîtes prédites sont trop
proches. Ces résultats concernent des images, pas une précision industrielle.
Ce prototype n'est ni un contrôle de conformité fiable ni un système de sécurité.

Vérification finale du 9 septembre 2026 depuis cette distribution :

- **94 tests automatisés réussis** et `pip check` sans conflit dans la venv utilisée.
- **391 statuts et 391 JPEG annotés reçus via un vrai abonné MQTT**, aux indices 0 à 390 ; passage à test2 à l'indice 126, sans trou ni remise à zéro.
- Un passage complet donne **289 OK et 102 NOK**, en accord avec les essais individuels ci-dessus.
- Boucle vérifiée sur **405 images** : retour à test1 à l'indice 391, avec compteurs cumulés.
- Dashboard existant vérifié dans le navigateur : images mises à jour, affichage OK et NOK, aucune erreur JavaScript détectée.
- Instance fournie vérifiée sur le port temporaire 1881 : FlowFuse Dashboard chargé depuis les dépendances du projet et réception de la vidéo.
- Courtier Python vérifié avec une vraie connexion MQTT sur un port temporaire ; arrêt propre vérifié.
- `npm audit --omit=dev` : aucune vulnérabilité signalée dans les dépendances verrouillées au moment du contrôle.

Les tests de cette version sont exécutés sur Mac Intel. Les commandes Windows
sont fournies, mais l'exécution Windows complète reste à vérifier sur le PC
de destination. Aucun GPU n'est requis pour la démo CPU.

## Arborescence

```text
run.py
requirements.txt
config/                    # Application, MQTT, courtier, anciennes zones
models/best_v02.pt          # Poids utilisés
data/videos/               # test1.MOV et test2.MOV
data/datasets/              # Archive YOLOv8-Seg pour Colab
data/logs/                  # Journaux ignorés par Git
data/snapshots/             # Captures ignorées par Git
src/                       # Capture, segmentation, validation, MQTT
nodered/                   # Flow, dépendances npm, démarrage local
scripts/                   # Courtier Python, annotation SAM 2
test/                      # Tests automatisés
ENTRAINEMENT_YOLO.ipynb
ANNOTATION_SAM2_COLAB.ipynb
ANNOTATION_SAM2.md
requirements-sam2.txt       # Colab uniquement
```

## Avant le commit

Vérifier que les vidéos et le dataset d'usine peuvent être partagés avec les
destinataires du dépôt. Les poids, vidéos et archive sont inclus ; ni la venv,
ni `node_modules`, ni les credentials Node-RED ne doivent être ajoutés.

```bash
git status --short
git diff --check
git add .
git diff --cached --stat
git commit -m "Finalize YOLOv8-Seg demo with continuous video and MQTT dashboard"
```

Le dépôt reste local jusqu'à un éventuel `git push` explicite.
