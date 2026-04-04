# Prototype d'inspection Wire Harness

Projet Python simple pour tester une inspection visuelle locale avec YOLO, zones de validation, statut `OK / NOK`, et dashboard Node-RED.

Le dossier contient deja :
- le modele de demonstration `data/models/best.pt`
- la video de demonstration `data/videos/demo_wire_harness.mp4`
- le notebook Colab `ENTRAINEMENT_YOLO.ipynb`
- le dataset ZIP pour Colab `data/datasets/wire_harness_demo_colab.zip`
- le flow Node-RED `nodered/wire_harness_dashboard_flow.json`

## Structure

```text
github_ready/
  README.md
  requirements.txt
  run.py
  ENTRAINEMENT_YOLO.ipynb
  config/
  data/
    datasets/
    logs/
    models/
    snapshots/
    videos/
  nodered/
  scripts/
  src/
```

## Windows - demarrage rapide

Prerequis :
- Python 3.11 64-bit
- VS Code
- Optionnel : Node-RED pour le dashboard

### 1. Installation

Dans le dossier du projet, lancer :

```bat
scripts\01_install_windows.bat
```

Ce script exige Python 3.11 et recree `.venv` si un ancien environnement utilise une autre version.

### 2. Ouvrir l'application

Lancer :

```bat
scripts\02_ouvrir_application.bat
```

Un menu simple s'ouvre :
- `1` : video demo
- `2` : webcam PC
- `3` : stream IP / ESP

## Test direct

### Demo locale

```bat
scripts\02_ouvrir_application.bat
```

Puis choisir `1`.

### Webcam

```bat
scripts\02_ouvrir_application.bat
```

Puis choisir `2`.

### Stream IP / ESP32-CAM

```bat
scripts\02_ouvrir_application.bat
```

Puis choisir `3`, puis saisir l'URL du stream, par exemple :

```text
http://192.168.1.50:81/stream
```

## Dashboard Node-RED

Le projet peut publier les resultats en MQTT.

### Broker MQTT local

```bat
scripts\03_demarrer_broker_local.bat
```

### Node-RED

```bat
scripts\04_demarrer_nodered.bat
```

Puis importer :

```text
nodered/wire_harness_dashboard_flow.json
```

## Entrainement Colab

Le notebook Colab est fourni ici :

```text
ENTRAINEMENT_YOLO.ipynb
```

Le dataset ZIP est fourni ici :

```text
data/datasets/wire_harness_demo_colab.zip
```

Workflow simple :
1. ouvrir le notebook dans Google Colab
2. uploader `wire_harness_demo_colab.zip`
3. lancer l'entrainement
4. recuperer `best.pt`
5. remplacer `data/models/best.pt`

## Fichiers utiles

- `run.py` : lanceur principal
- `config/app.yaml` : configuration video, stream, modele
- `config/zones.json` : zones d'inspection
- `src/app_runner.py` : pipeline principal
- `src/detector/yolo_detector.py` : inference YOLO

## GitHub

Pour publier ce dossier sur GitHub :

```bash
git init
git add .
git commit -m "Initial wire harness inspection prototype"
git branch -M main
git remote add origin <url-du-repo>
git push -u origin main
```

## Remarque

Le modele inclus est un modele de demonstration. Pour une utilisation reelle, il faudra le remplacer par un modele entraine sur les vraies donnees de l'atelier.
