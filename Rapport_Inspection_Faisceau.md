# Rapport de Projet — Prototype d'Inspection Visuelle de Faisceaux de Câbles

**Projet :** Inspection automatisée par vision artificielle du positionnement de clips sur connecteurs
**Version :** 1.0
**Date :** Septembre 2026
**Auteur :** _[Nom de l'auteur]_
**Encadrant :** _[Nom de l'encadrant]_
**Établissement :** _[Nom de l'établissement]_

---

## Table des matières

1. [Introduction](#1-introduction)
2. [Technologies utilisées](#2-technologies-utilisées)
3. [Préparation des données et entraînement du modèle](#3-préparation-des-données-et-entraînement-du-modèle)
4. [Logique métier — Algorithme de décision](#4-logique-métier--algorithme-de-décision)
5. [Architecture du système](#5-architecture-du-système)
6. [Résultats et validation](#6-résultats-et-validation)
7. [Limites et perspectives](#7-limites-et-perspectives)
8. [Conclusion](#8-conclusion)
9. [Annexes](#9-annexes)

---

## 1. Introduction

Le présent rapport documente la conception, le développement et la validation d'un **prototype d'inspection visuelle automatisée** destiné au contrôle qualité de faisceaux de câbles en environnement industriel.

### 1.1 Contexte

Dans les chaînes d'assemblage de faisceaux électriques, la vérification du positionnement correct des clips sur les connecteurs constitue une étape critique du contrôle qualité. Un clip mal positionné ou resté accidentellement attaché au connecteur après manipulation représente un défaut d'assemblage susceptible d'engendrer des non-conformités en aval de la production.

### 1.2 Objectif du prototype

L'objectif principal de ce prototype est de **détecter automatiquement si un clip est resté attaché au connecteur** et de remonter un verdict en temps réel :

- **OK** : le clip n'est pas détecté à proximité du connecteur — situation nominale par défaut.
- **NOK** : le clip est détecté en contact ou à proximité immédiate du connecteur — défaut identifié.

Le système repose sur un modèle de segmentation d'instances entraîné par apprentissage profond (YOLOv8-Seg), couplé à une règle géométrique de décision et à une interface de supervision en temps réel via un dashboard industriel.

### 1.3 Périmètre fonctionnel

Le prototype couvre les fonctionnalités suivantes :

- Acquisition vidéo depuis une source hors ligne (fichier vidéo) ou en direct (webcam, caméra IP/ESP32-CAM)
- Détection et segmentation des objets d'intérêt : **connecteur**, **clip**, **câble**
- Application d'une règle métier de décision basée sur la proximité géométrique
- Publication des verdicts et des images annotées en temps réel via le protocole MQTT
- Visualisation sur un dashboard industriel Node-RED / FlowFuse Dashboard 2.0
- Sauvegarde automatique des captures d'écran des défauts détectés (snapshots NOK)

> **[Insérer une photo générale du poste d'assemblage ou du faisceau de câbles ici]**

---

## 2. Technologies utilisées

Le prototype s'appuie sur un ensemble de technologies open-source et d'outils industriels éprouvés, sélectionnés pour leur maturité, leur interopérabilité et leur facilité de déploiement.

### 2.1 Vision par ordinateur et apprentissage profond

| Composant             | Technologie                        | Version        | Rôle                                              |
| --------------------- | ---------------------------------- | -------------- | -------------------------------------------------- |
| Modèle de détection | **YOLOv8-Seg** (Ultralytics) | 8.3.15         | Segmentation d'instances en temps réel            |
| Framework IA          | **PyTorch**                  | 2.2.2          | Moteur d'inférence du modèle                     |
| Traitement d'images   | **OpenCV** (`cv2`)         | ≥ 4.8         | Capture vidéo, transformation d'images, affichage |
| Calcul numérique     | **NumPy**                    | ≥ 1.26, < 2.0 | Manipulation des masques de segmentation           |

### 2.2 Communication et interface

| Composant               | Technologie                                           | Rôle                                        |
| ----------------------- | ----------------------------------------------------- | -------------------------------------------- |
| Protocole de messagerie | **MQTT** (Mosquitto)                            | Transport temps réel des verdicts et images |
| Client MQTT Python      | **paho-mqtt**                                   | Publication des payloads JSON depuis Python  |
| Dashboard industriel    | **Node-RED** + **FlowFuse Dashboard 2.0** | Interface de supervision visuelle            |
| Sérialisation          | **JSON**                                        | Format d'échange entre Python et Node-RED   |

### 2.3 Plateforme d'entraînement et annotation

| Composant                | Technologie                  | Rôle                                                  |
| ------------------------ | ---------------------------- | ------------------------------------------------------ |
| Annotation des images    | **Roboflow**           | Plateforme d'annotation manuelle et gestion du dataset |
| Entraînement du modèle | **Google Colab** (GPU) | Entraînement du modèle YOLOv8-Seg                    |
| Langage principal        | **Python 3.11**        | Développement de l'ensemble du pipeline               |

### 2.4 Configuration et gestion

| Composant                 | Format                          | Rôle                                                       |
| ------------------------- | ------------------------------- | ----------------------------------------------------------- |
| Configuration applicative | **YAML** (`app.yaml`)   | Paramètres du détecteur, de la validation, de l'affichage |
| Configuration MQTT        | **JSON** (`mqtt.json`)  | Paramètres du courtier et des topics                       |
| Configuration des zones   | **JSON** (`zones.json`) | Zones d'inspection (mode hérité)                          |

---

## 3. Préparation des données et entraînement du modèle

### 3.1 Acquisition des vidéos

Les données d'entrée ont été acquises directement sur le poste d'assemblage à partir de vidéos filmées en conditions réelles. Deux vidéos de référence ont été utilisées pour le développement et la validation du prototype :

| Vidéo        | Scénario     | Description                                                              |
| ------------- | ------------- | ------------------------------------------------------------------------ |
| `test1.MOV` | **NOK** | Séquence montrant un clip resté attaché au connecteur (défaut)       |
| `test2.MOV` | **OK**  | Séquence montrant un assemblage conforme (clip détaché du connecteur) |

Ces vidéos constituent les cas de test principaux pour valider le comportement du système face aux deux situations d'inspection : conformité et non-conformité.

> **[Insérer une capture d'écran d'une frame typique de test1.MOV (scénario NOK) ici]**

> **[Insérer une capture d'écran d'une frame typique de test2.MOV (scénario OK) ici]**

### 3.2 Annotation manuelle sur Roboflow

L'annotation des images a été réalisée **manuellement** via la plateforme **Roboflow**. Chaque image extraite des vidéos a été annotée individuellement par l'opérateur, en traçant les contours de segmentation (polygones) autour des trois classes d'objets d'intérêt :

| Classe        | ID | Description            | Couleur d'affichage |
| ------------- | -- | ---------------------- | ------------------- |
| `connector` | 0  | Connecteur électrique | Rouge               |
| `clip`      | 1  | Clip de fixation       | Bleu                |
| `cable`     | 2  | Câble / faisceau      | Vert                |

Le processus d'annotation a suivi les étapes suivantes :

1. **Extraction des frames** : des images représentatives ont été extraites manuellement depuis les vidéos d'entraînement.
2. **Import sur Roboflow** : les images ont été importées dans un projet Roboflow dédié.
3. **Annotation polygonale** : pour chaque image, les contours précis des connecteurs, clips et câbles ont été tracés manuellement à l'aide de l'outil de segmentation polygonale de Roboflow.
4. **Vérification et correction** : chaque annotation a été vérifiée visuellement pour garantir la qualité et la cohérence du dataset.

Au total, **79 images** ont été annotées et constituent le dataset d'entraînement du modèle.

> **[Insérer une capture d'écran de l'interface Roboflow montrant une image annotée ici]**

> **[Insérer une capture d'écran de la vue d'ensemble du dataset sur Roboflow ici]**

### 3.3 Export du dataset au format YOLOv8-Seg

Une fois l'annotation terminée, le dataset a été exporté depuis Roboflow au format **YOLOv8 Segmentation**, qui organise les données selon la structure suivante :

```
dataset/
├── train/
│   ├── images/
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── labels/
│       ├── image_001.txt
│       ├── image_002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── data.yaml
```

Chaque fichier `.txt` de labels contient les annotations de segmentation au format YOLO :

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

Où chaque ligne décrit un polygone de segmentation normalisé (coordonnées entre 0 et 1), précédé de l'identifiant de classe (`0` = connector, `1` = clip, `2` = cable).

Le fichier `data.yaml` généré par Roboflow définit les chemins des ensembles d'entraînement/validation et la correspondance des classes :

```yaml
names:
  0: connector
  1: clip
  2: cable

train: ../train/images
val: ../valid/images
```

### 3.4 Entraînement sur Google Colab

L'entraînement du modèle a été effectué sur **Google Colab** en exploitant l'accélération GPU disponible. Le notebook d'entraînement (`ENTRAINEMENT_YOLO.ipynb`) orchestre les étapes suivantes :

1. **Installation des dépendances** : installation du package `ultralytics` dans l'environnement Colab.
2. **Chargement du dataset** : import du dataset exporté depuis Roboflow.
3. **Configuration de l'entraînement** : paramétrage du modèle de base, du nombre d'époques, de la taille des images et du device GPU.
4. **Lancement de l'entraînement** : fine-tuning d'un modèle pré-entraîné YOLOv8 sur le dataset de segmentation personnalisé.
5. **Export du modèle** : récupération des poids du meilleur modèle (`best.pt`).

#### Paramètres d'entraînement

| Paramètre                      | Valeur                   |
| ------------------------------- | ------------------------ |
| Architecture de base            | YOLOv8n-seg (nano)       |
| Nombre d'images                 | 79                       |
| Tâche                          | Segmentation d'instances |
| Taille d'image                  | 640 × 640 px            |
| Device                          | GPU (Google Colab)       |
| Seuil de confiance (inférence) | 0.35                     |

Le modèle entraîné final (`best_v02.pt`) est déployé dans le répertoire `data/models/` du projet et chargé automatiquement par l'application Python au démarrage.

> **[Insérer la courbe d'entraînement (loss) depuis Google Colab ici]**

> **[Insérer la courbe de mAP (mean Average Precision) depuis Google Colab ici]**

> **[Insérer la matrice de confusion de la validation du modèle ici]**

---

## 4. Logique métier — Algorithme de décision

### 4.1 Principe général

Le modèle YOLOv8-Seg détecte et segmente les objets présents dans l'image, mais **il ne prédit pas directement le verdict OK ou NOK**. Le verdict est calculé par un algorithme de décision dédié, implémenté dans le module `ClipAttachmentValidator`, qui analyse les relations géométriques entre les détections du modèle.

Cette séparation entre la détection (modèle IA) et la décision (règle métier) garantit la traçabilité et la modularité du système : le seuil de décision peut être ajusté indépendamment du modèle.

### 4.2 Règle géométrique de proximité

La règle métier repose sur le principe suivant :

> **Si la distance euclidienne entre les bords des boîtes englobantes du clip et du connecteur est inférieure ou égale à 20 pixels, ou si leurs masques de segmentation présentent une intersection, le verdict est NOK. Dans tous les autres cas, le verdict est OK par défaut.**

#### Paramètres configurables

| Paramètre            | Valeur      | Fichier de configuration                              |
| --------------------- | ----------- | ----------------------------------------------------- |
| `min_confidence`    | 0.35        | `config/app.yaml` → `validation.clip_attachment` |
| `connector_near_px` | 20.0 pixels | `config/app.yaml` → `validation.clip_attachment` |

- La **distance** est mesurée entre les bords des boîtes englobantes (et non entre leurs centres), en pixels de la vidéo originale.
- Le **seuil de confiance** filtre les détections dont le score est inférieur à 0.35, afin d'éliminer les faux positifs.
- Le **câble** est affiché à l'écran mais **ne participe pas au verdict** : seule la relation clip/connecteur détermine le statut.

### 4.3 Détail de l'algorithme de validation

L'algorithme de validation s'exécute image par image selon le diagramme suivant :

```
┌─────────────────────────────────────────────────────────┐
│                   Frame vidéo (image)                   │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│           Inférence YOLOv8-Seg (détection)              │
│   → Masques de segmentation + boîtes englobantes        │
│   → Classes : connector (0), clip (1), cable (2)        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│              Filtrage par seuil de confiance             │
│                  (min_confidence = 0.35)                 │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│        Pour chaque paire (clip, connector) :            │
│                                                         │
│   1. Calcul de la distance euclidienne entre les        │
│      bords des boîtes englobantes                       │
│                                                         │
│   2. Test d'intersection des masques de segmentation    │
│                                                         │
│   Si distance ≤ 20 px  OU  masques en intersection :    │
│       → Verdict = NOK (clip attaché au connecteur)      │
│       → Confiance = min(conf_clip, conf_connector)      │
│                                                         │
│   Sinon (y compris si clip ou connecteur absent) :      │
│       → Verdict = OK par défaut                         │
│       → Confiance = 0 (aucune preuve de conformité)     │
└─────────────────────────────────────────────────────────┘
```

#### Points clés de l'algorithme

1. **NOK immédiat et prioritaire** : dès qu'une paire clip/connecteur est détectée à proximité, le verdict NOK est émis sans délai ni confirmation sur plusieurs images. Même une intersection d'un seul pixel suffit.
2. **OK par défaut** : en l'absence de détection du clip ou du connecteur, ou si le clip est suffisamment éloigné, le verdict est OK par défaut. Cela signifie qu'un OK n'est **pas une confirmation positive** d'assemblage conforme, mais l'absence de condition NOK détectée.
3. **Pas de mémoire inter-images** : aucune décision précédente n'est conservée. Si les détections disparaissent après un NOK, l'image suivante est immédiatement OK par défaut.
4. **Indépendance du câble** : le câble est détecté et affiché pour le contexte visuel, mais son contact avec le clip ou le connecteur ne déclenche ni NOK ni OK.

#### Calcul de la distance entre boîtes

La distance euclidienne entre les bords de deux boîtes englobantes est calculée comme suit :

```
distance = √( max(ax1 - bx2, bx1 - ax2, 0)² + max(ay1 - by2, by1 - ay2, 0)² )
```

Où `(ax1, ay1, ax2, ay2)` et `(bx1, by1, bx2, by2)` sont les coordonnées des boîtes englobantes du clip et du connecteur respectivement. La distance vaut zéro si les boîtes se touchent ou se chevauchent.

### 4.4 Résultats obtenus

Les résultats suivants ont été obtenus lors de l'essai complet avec le modèle `best_v02.pt`, un seuil de confiance de 0.35 et une distance de proximité de 20 pixels :

| Vidéo        | Scénario attendu | Nb images | OK par défaut |          NOK | Indéterminé |
| ------------- | ----------------- | --------: | -------------: | -----------: | ------------: |
| `test1.MOV` | NOK (défaut)     |       126 |             33 | **93** |             0 |
| `test2.MOV` | OK (conforme)     |       265 |  **256** |            9 |             0 |

**Analyse des résultats :**

- Sur `test1` (scénario NOK), **93 images sur 126** (73,8 %) sont correctement classées NOK. Les 33 images OK correspondent aux frames où le clip n'est pas détecté au seuil actuel (début/fin de séquence, occlusion partielle).
- Sur `test2` (scénario OK), **256 images sur 265** (96,6 %) sont correctement classées OK. Les 9 faux NOK (indices 36 à 44) résultent d'une proximité temporaire des boîtes prédites (distance de 0 à 11 pixels), en contradiction avec le scénario OK attendu.

> **[Insérer un graphe montrant l'évolution du verdict (OK/NOK) frame par frame sur test1.MOV ici]**

> **[Insérer un graphe montrant l'évolution du verdict (OK/NOK) frame par frame sur test2.MOV ici]**

---

## 5. Architecture du système

### 5.1 Vue d'ensemble

Le système s'articule autour de trois composants principaux interconnectés :

```
┌──────────────┐       MQTT        ┌──────────────┐       HTTP        ┌──────────────┐
│              │  ──────────────►  │              │  ──────────────►  │              │
│  Application │   JSON payloads   │   Courtier   │    Dashboard      │  Navigateur  │
│    Python    │   (status, video, │    MQTT      │    Node-RED       │     Web      │
│              │    metrics, ...)  │  (Mosquitto) │   (FlowFuse)     │              │
│  YOLOv8-Seg  │                   │ 127.0.0.1    │  127.0.0.1       │ /inspection/ │
│  + OpenCV    │                   │  :1883       │   :1880          │              │
└──────────────┘                   └──────────────┘                   └──────────────┘
       │                                                                     │
       │  Capture vidéo                                         Affichage    │
       │  (fichier / webcam / IP)                              temps réel    │
       ▼                                                                     ▼
 ┌────────────┐                                                 ┌────────────────┐
 │   Source    │                                                 │   Dashboard    │
 │   Vidéo    │                                                 │  OK/NOK + flux │
 └────────────┘                                                 │   vidéo live   │
                                                                └────────────────┘
```

> **[Insérer un schéma d'architecture global du système ici]**

### 5.2 Pipeline de traitement vidéo

L'application Python exécute une **boucle de traitement continue** qui constitue le cœur du pipeline d'inspection. Chaque itération de la boucle traite une frame vidéo selon la séquence suivante :

```
┌─── Boucle principale (while True) ────────────────────────────────────────┐
│                                                                           │
│  1. ACQUISITION      → Lecture d'une frame depuis la source vidéo         │
│                         (VideoSource : fichier, webcam, flux IP)          │
│                                                                           │
│  2. INFÉRENCE        → Prédiction YOLOv8-Seg sur la frame                │
│                         → Masques de segmentation booléens                │
│                         → Boîtes englobantes + classes + confiances       │
│                                                                           │
│  3. VALIDATION       → Règle métier ClipAttachmentValidator              │
│                         → Calcul de distance entre boîtes                 │
│                         → Test d'intersection des masques                 │
│                         → Verdict OK / NOK                                │
│                                                                           │
│  4. VISUALISATION    → Superposition des masques sur la frame             │
│                         (connecteur=rouge, clip=bleu, câble=vert)         │
│                         → Bandeau de statut OK/NOK                        │
│                                                                           │
│  5. PUBLICATION      → Envoi MQTT des payloads JSON :                     │
│                         - status (verdict + confiance + frame)            │
│                         - metrics (compteurs OK/NOK + FPS)                │
│                         - events (changements de statut)                  │
│                         - video_stream (image annotée en Base64)          │
│                         - snapshot (capture NOK si applicable)            │
│                                                                           │
│  6. SAUVEGARDE       → Si NOK : enregistrement du snapshot annoté        │
│                         dans data/snapshots/ (cooldown de 20 frames)      │
│                                                                           │
│  7. AFFICHAGE LOCAL  → Fenêtre OpenCV (optionnel, désactivable)          │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Modes d'acquisition vidéo

Le pipeline supporte quatre modes d'acquisition :

| Mode        | Source                                    | Commande de lancement                                   |
| ----------- | ----------------------------------------- | ------------------------------------------------------- |
| `offline` | Fichier vidéo local (`.MOV`, `.mp4`) | `python run.py offline --video data/videos/test1.MOV` |
| `webcam`  | Webcam USB du PC                          | `python run.py webcam`                                |
| `esp`     | Flux MJPEG d'une caméra IP / ESP32-CAM   | `python run.py esp --stream-url http://IP:81/stream`  |
| `demo`    | Playlist de vidéos en boucle             | `python run.py demo`                                  |

### 5.3 Communication temps réel via MQTT

La communication entre l'application Python et le dashboard Node-RED s'effectue via le protocole **MQTT** (Message Queuing Telemetry Transport), un protocole de messagerie léger spécifiquement conçu pour les applications IoT et industrielles.

#### Architecture MQTT

| Composant                           | Rôle                                         | Adresse                            |
| ----------------------------------- | --------------------------------------------- | ---------------------------------- |
| **Courtier MQTT** (Mosquitto) | Relais central des messages                   | `127.0.0.1:1883`                 |
| **Client Python** (paho-mqtt) | Producteur : publie les verdicts et images    | Client ID :`wire-harness-python` |
| **Node-RED**                  | Consommateur : s'abonne aux topics et affiche | Nœuds MQTT-in                     |

#### Topics MQTT

L'application publie sur cinq topics dédiés :

| Topic                               | Contenu                                          | Retain | QoS |
| ----------------------------------- | ------------------------------------------------ | ------ | --- |
| `factory/inspection/status`       | Verdict OK/NOK, confiance, frame index, détails | Oui    | 0   |
| `factory/inspection/metrics`      | Compteurs OK/NOK, FPS, dernière erreur          | Oui    | 0   |
| `factory/inspection/events`       | Changements de statut, événements NOK          | Non    | 0   |
| `factory/inspection/snapshot`     | Capture d'écran NOK encodée en Base64          | Non    | 0   |
| `factory/inspection/video_stream` | Flux vidéo annoté encodé en Base64            | Oui    | 0   |

#### Exemple de payload `status`

```json
{
  "status": "NOK",
  "confidence": 0.672,
  "mode": "offline",
  "frame_index": 42,
  "source_name": "test1.MOV",
  "timestamp": "2026-09-09T14:23:15.123Z",
  "relation": {
    "rule": "clip_attachment",
    "method": "box_distance_and_mask_intersection",
    "decision_basis": "connector_contact",
    "connector_near_px": 20.0,
    "pairs": [
      {
        "clip_index": 0,
        "target_class": "connector",
        "target_index": 1,
        "box_gap_px": 4.237,
        "masks_intersect": true
      }
    ]
  }
}
```

### 5.4 Dashboard Node-RED / FlowFuse

L'interface de supervision est construite avec **Node-RED** et le composant **FlowFuse Dashboard 2.0**. Le flow Node-RED fourni (`wire_harness_dashboard_flow.json`) se charge de :

1. **S'abonner** aux topics MQTT pour recevoir les verdicts, métriques et flux vidéo.
2. **Afficher** le statut courant (OK/NOK) avec un indicateur visuel clair.
3. **Afficher** le flux vidéo annoté en temps réel (images reçues via MQTT).
4. **Afficher** les compteurs d'images OK, NOK et le FPS.
5. **Lister** l'historique des événements d'inspection de manière compacte.

Le dashboard est accessible via navigateur à l'adresse :

```
http://127.0.0.1:1880/inspection/
```

> **[Insérer une capture d'écran du Dashboard Node-RED affichant un statut OK ici]**

> **[Insérer une capture d'écran du Dashboard Node-RED affichant un statut NOK ici]**

> **[Insérer une capture d'écran du flow Node-RED dans l'éditeur ici]**

### 5.5 Structure du projet

```
wire-harness-vision-inspection/
├── run.py                          # Point d'entrée principal (menu interactif)
├── requirements.txt                # Dépendances Python de base
├── requirements-yolo.txt           # Dépendances YOLOv8 / PyTorch
├── config/
│   ├── app.yaml                    # Configuration applicative (détecteur, validation, affichage)
│   ├── mqtt.json                   # Configuration MQTT (courtier, topics, QoS)
│   └── zones.json                  # Zones d'inspection (mode hérité)
├── data/
│   ├── datasets/                   # Dataset de segmentation (Roboflow export)
│   ├── models/
│   │   └── best_v02.pt            # Modèle YOLOv8-Seg entraîné
│   ├── snapshots/                  # Captures d'écran des défauts NOK
│   ├── videos/
│   │   ├── test1.MOV              # Vidéo de test — scénario NOK
│   │   └── test2.MOV              # Vidéo de test — scénario OK
│   └── logs/                       # Journaux d'exécution
├── nodered/
│   └── wire_harness_dashboard_flow.json  # Flow Node-RED du dashboard
├── scripts/                        # Scripts utilitaires (setup, lancement, broker)
├── src/
│   ├── app_runner.py               # Boucle principale d'inspection
│   ├── video_source.py             # Abstraction des sources vidéo
│   ├── detector/
│   │   ├── yolo_detector.py        # Intégration YOLOv8-Seg (Ultralytics)
│   │   ├── mock_detector.py        # Détecteur de démonstration
│   │   └── anomaly_detector.py     # Détecteur heuristique
│   ├── validation/
│   │   ├── clip_attachment.py      # Règle métier clip/connecteur (OK/NOK)
│   │   └── zone_validator.py       # Validation par zones (mode hérité)
│   ├── messaging/
│   │   └── mqtt_publisher.py       # Client MQTT (publication JSON)
│   ├── ui_payload/                 # Construction des payloads JSON
│   └── utils/                      # Utilitaires (overlay, snapshots, logging)
└── ENTRAINEMENT_YOLO.ipynb         # Notebook Colab d'entraînement du modèle
```

---

## 6. Résultats et validation

### 6.1 Validation fonctionnelle

Le prototype a été validé avec succès sur les critères suivants :

| Critère                                                          | Statut           |
| ----------------------------------------------------------------- | ---------------- |
| Détection et segmentation des 3 classes (connector, clip, cable) | ✅ Opérationnel |
| Verdict OK/NOK basé sur la règle géométrique de proximité    | ✅ Opérationnel |
| Affichage des masques de segmentation en temps réel              | ✅ Opérationnel |
| Publication MQTT des verdicts et métriques                       | ✅ Opérationnel |
| Flux vidéo annoté sur le dashboard Node-RED                     | ✅ Opérationnel |
| Sauvegarde automatique des snapshots NOK                          | ✅ Opérationnel |
| Support multi-sources (fichier, webcam, IP)                       | ✅ Opérationnel |
| Mode démonstration continue (playlist en boucle)                 | ✅ Opérationnel |

### 6.2 Tests automatisés

Le projet inclut une suite de **85 tests automatisés** couvrant :

- La priorité NOK en cas de contact clip/connecteur
- Le retour OK par défaut en l'absence de détections
- Les transitions immédiates de statut
- La sérialisation correcte des messages MQTT
- Le calcul de distance entre boîtes englobantes
- Le test d'intersection des masques de segmentation

### 6.3 Résultats d'inférence

> **[Insérer un masque YOLOv8-Seg superposé sur une frame de test1.MOV (scénario NOK) ici]**

> **[Insérer un masque YOLOv8-Seg superposé sur une frame de test2.MOV (scénario OK) ici]**

> **[Insérer une capture d'écran de la fenêtre OpenCV avec overlay d'inspection ici]**

---

## 7. Limites et perspectives

### 7.1 Limites actuelles

1. **Taille du dataset** : avec 79 images annotées, le modèle reste un prototype expérimental. La robustesse en conditions industrielles variées (éclairage, angle, occultations) n'est pas encore garantie.
2. **Règle géométrique 2D** : la proximité dans l'image (pixels) n'est pas une preuve d'attache mécanique. Les erreurs de masque, la perspective et les occlusions peuvent produire des verdicts erronés.
3. **Faux positifs** : 9 faux NOK observés sur le scénario OK (`test2.MOV`), résultant d'une proximité temporaire des boîtes prédites.
4. **Inférence CPU** : le prototype fonctionne sur CPU, ce qui limite le débit de traitement. L'utilisation d'un GPU dédié accélérerait significativement le pipeline.
5. **Seuil fixe de 20 pixels** : le seuil de proximité est exprimé en pixels de la vidéo originale et doit être ajusté si la résolution ou le point de vue de la caméra change.

### 7.2 Perspectives d'amélioration

1. **Enrichissement du dataset** : annotation d'images supplémentaires issues de conditions variées (éclairage, angles, types de connecteurs).
2. **Déploiement GPU** : migration vers une machine d'inférence équipée d'un GPU (NVIDIA) pour atteindre le temps réel à haute résolution.
3. **Caméra industrielle** : remplacement de la webcam par une caméra industrielle fixe, avec éclairage contrôlé et point de vue calibré.
4. **Règles métier avancées** : intégration de règles de validation supplémentaires (vérification de l'orientation du clip, confirmation sur plusieurs images consécutives).
5. **Intégration PLC/SCADA** : publication des verdicts vers un automate programmable via OPC-UA ou Modbus pour intégration dans la chaîne de production.

---

## 8. Conclusion

Ce projet a permis de développer un **prototype fonctionnel d'inspection visuelle automatisée** capable de détecter en temps réel le positionnement de clips sur des connecteurs de faisceaux de câbles. Le système combine un modèle de segmentation d'instances YOLOv8-Seg, une règle métier de décision géométrique, et une architecture de communication MQTT couplée à un dashboard industriel Node-RED.

Les résultats obtenus démontrent la faisabilité de l'approche : le prototype détecte correctement la majorité des situations NOK (73,8 % sur le scénario de défaut) et OK (96,6 % sur le scénario conforme), avec un nombre limité de faux positifs.

Bien que le modèle actuel, entraîné sur un dataset de 79 images, reste un prototype expérimental nécessitant un enrichissement des données et une validation industrielle approfondie, l'architecture logicielle modulaire et les interfaces de communication standardisées (MQTT, JSON, Node-RED) constituent une base solide pour une évolution vers un système de production.

---

## 9. Annexes

### Annexe A — Commandes de lancement rapide

```bash
# Activation de l'environnement virtuel
source .venv/bin/activate

# Mode démonstration continue (test1 + test2 en boucle)
python run.py demo

# Mode hors ligne — vidéo unique
python run.py offline --video data/videos/test1.MOV

# Mode webcam
python run.py webcam

# Mode caméra IP / ESP32-CAM
python run.py esp --stream-url http://192.168.1.50:81/stream

# Exécution sans MQTT et sans fenêtre (test rapide)
python run.py offline --video data/videos/test1.MOV --no-display --no-mqtt --max-frames 20
```

### Annexe B — Configuration de l'application (`app.yaml`)

```yaml
detector:
  mode: yolo
  yolo_model_path: data/models/best_v02.pt
  yolo:
    confidence_threshold: 0.35
    image_size: 640
    device: cpu

validation:
  mode: clip_attachment
  clip_attachment:
    min_confidence: 0.35
    connector_near_px: 20.0
```

### Annexe C — Configuration MQTT (`mqtt.json`)

```json
{
  "enabled": true,
  "broker": {
    "host": "127.0.0.1",
    "port": 1883,
    "client_id": "wire-harness-python"
  },
  "topics": {
    "status": "factory/inspection/status",
    "metrics": "factory/inspection/metrics",
    "events": "factory/inspection/events",
    "snapshot": "factory/inspection/snapshot",
    "video_stream": "factory/inspection/video_stream"
  }
}
```

### Annexe D — Prérequis techniques

| Composant            | Version requise                  |
| -------------------- | -------------------------------- |
| Python               | 3.11+                            |
| PyTorch              | 2.2.2                            |
| Ultralytics (YOLOv8) | 8.3.15                           |
| NumPy                | ≥ 1.26, < 2.0                   |
| OpenCV               | ≥ 4.8                           |
| paho-mqtt            | ≥ 2.0                           |
| Courtier MQTT        | Mosquitto (ou amqtt en fallback) |
| Node-RED             | + FlowFuse Dashboard 2.0         |
| Google Colab         | GPU (entraînement)              |

---

*Document généré dans le cadre du projet d'inspection visuelle de faisceaux de câbles — Septembre 2026*
