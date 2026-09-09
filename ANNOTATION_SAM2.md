# Annotation des videos avec SAM 2.1

Le travail se fait en trois temps : placer quelques reperes sur le Mac, calculer
les masques dans Colab, puis controler les images et exporter le dataset.
Il n'est pas necessaire de tracer les contours a la main.

Les fichiers de l'application, son modele et `github_ready` ne sont pas modifies
par ce parcours. SAM sert ici a preparer des donnees d'entrainement.

## Classes a utiliser

| Touche / classe | Nom | Partie a segmenter |
| --- | --- | --- |
| 0 | connector | Piece cubique doree et noire uniquement, sans le support blanc/gris |
| 1 | clip | Piece mobile noire, sur le connecteur ou sur le cable selon le montage |
| 2 | cable | Fil flexible noir qui relie les elements |

Un identifiant d'objet correspond a une piece physique. Plusieurs connecteurs
ont la meme classe `0`, mais des identifiants differents.
Cet ordre des classes est celui du nouveau dataset ; il differe du modele demo.

## Regle metier confirmee : clip mobile

Le clip n'est PAS un support fixe sur la table. La regle communiquee pour la
demonstration est la suivante :

- **test1 / NOK** : le clip oublie reste attache au connecteur.
- **test2 / OK** : le clip a ete detache du connecteur puis installe sur le cable.

L'annotation doit decrire la piece la ou elle se trouve reellement, y compris
pendant son deplacement. Le contact avec le connecteur ou le cable n'interdit
pas l'export. La classe reste `1 clip` dans les deux cas. Ne pas transformer
ces classes en OK/NOK, ni attribuer un verdict a chaque image a partir du nom
de la video. L'application d'inspection devra evaluer cette relation entre les
pieces a l'instant pertinent ; cette mise a jour ne modifie pas l'application.

Les anciennes regles `empty_clip_ids`, `empty_clip_frames` et les zones
protegees autour du clip sont supprimees de review/export. Les anciens fichiers
`review_settings.json` peuvent rester sur disque : ils sont ignores et ne sont
plus inclus dans les nouveaux ZIP. Aucun reglage de ces fichiers n'est requis.

## Objet cache, hors champ ou incertain

Dans points, il est maintenant possible de declarer la visibilite d'UN objet
sur une image ou une plage. Cette declaration humaine est distincte des clics
SAM et ne supprime pas les masques bruts. Elle ne choisit pas un statut OK/NOK.

- **H / absent** : objet entierement cache ou hors champ. Son masque exploite
  devient vide et aucune ligne YOLO n'est produite pour lui sur ces images.
  Les autres objets ne sont pas modifies.
- **I / incertain** : objet partiellement visible, flou, ambigu, ou image a ne
  pas utiliser. Toute l'image est exclue de l'export, meme si elle avait deja
  ete approuvee. Aucun objet visible n'est ainsi transforme en fond par erreur.
- **C / annuler** : retirer la declaration H/I sur l'image ou la plage choisie.
  Le masque SAM redevient disponible ; cela ne garantit pas qu'il soit correct.

Un connecteur clairement visible mais immobile n'est pas absent. Si le but est
de ne traiter que la manipulation, utiliser une selection d'images ou la
logique applicative, pas une fausse annotation de fond. Le format YOLO-Seg
documente stocke une classe et des coordonnees par instance, sans champ de
visibilite : l'outil exclut donc l'image entiere pour I plutot que d'inventer
un label "ignore". Voir le [format officiel](https://docs.ultralytics.com/datasets/segment/).

### Sur une plage d'images

1. Sauvegarder et fermer avec Q toute ancienne fenetre points. Rouvrir la
   session pour charger les nouveaux raccourcis :

```bash
.venv/bin/python scripts/annotate_sam2.py points data/annotations_sam2_clip_bas_retour/test2 --max-height 650
```

2. Avec Tab, selectionner **#1 connector** si c'est bien lui a corriger.
   Les couleurs identifient les classes : connector rouge, clip bleu, cable
   vert. Ne pas confondre le numero d'objet avec la classe : connector a la
   classe 0. Ne pas appuyer sur 0 pour selectionner #1 : cela cree un objet.
3. Placer le curseur sur le premier indice concerne, puis appuyer sur **G**.
4. Placer le curseur sur le dernier indice, puis appuyer sur **H** (absence
   certaine) ou **I** (doute). Les deux bornes sont incluses.
5. Verifier l'apercu et le statut de l'objet, puis **Q** pour sauvegarder.

H/I/C sans G agit seulement sur l'image courante. G une seconde fois annule
la selection de plage. Pour annuler une declaration sur une plage, refaire
G au debut, aller a la fin, puis C. Changer d'objet avec Tab annule la plage
en cours, afin de ne pas la reporter par erreur sur un autre objet.
Les indices commencent a 0 : le curseur 40 correspond au texte Image 41/53.
X efface seulement des clics ; D supprime l'objet entier de la sequence.
Ni X, ni D, ni la touche B de review ne sert a declarer une occultation locale.

### Ce qui est conserve

Les declarations sont sauvegardees dans `visibility.json`, lie aux images et
aux identifiants/classes des objets. Les clics et les NPZ bruts restent intacts.
Les nouveaux pack et notebook Colab transmettent ce fichier. Le calcul stocke
aussi une copie dans run.json, afin de recuperer ces declarations si une
ancienne cellule de telechargement oublie le fichier separe.

Il s'agit d'une **correction d'annotation apres SAM**, pas d'un apprentissage
automatique de l'absence : SAM peut encore calculer une sortie brute sur ces
images, mais cette sortie n'est pas utilisee comme label pour l'objet absent.
H/I seuls ne demandent pas de relancer Colab ; changer les clics, oui. A la
reapparition de l'objet, ajouter des reperes sur une image ou il est vraiment
visible si le suivi a derive. Ne pas laisser de H/I sur cette nouvelle image.

Dans review, la vue YOLO initiale applique les absences. Avec V, la vue SAM brut
montre encore les sorties originales pour comparaison. Les pixels supprimes
par H sont visibles en jaune dans la vue differences. L'approbation doit etre
refaite apres modification de H/I ; aucune nouvelle image n'est approuvee
automatiquement. Si TOUS les masques exploites sont vides, B est necessaire
pour confirmer l'image vide. Une image avec I ne peut pas etre approuvee.

Apres les declarations et les corrections de clics, sauvegarder avec Q et
creer une nouvelle archive, sans remplacer l'ancienne :

```bash
.venv/bin/python scripts/annotate_sam2.py pack data/annotations_sam2_clip_bas_retour/test1 data/annotations_sam2_clip_bas_retour/test2 --output data/reperes_sam2_visibilite.zip
```

## Essai en cours : corriger le fond et verifier le clip

Les resultats du dernier calcul sont dans
`data/annotations_sam2_clip_bas_retour/test1` et `test2` (le nom historique
des dossiers ne signifie pas que le clip doit rester en bas de l'image).
Les objets existants sont `#1 connector`, `#2 cable`, `#4 clip`.

1. Fermer les fenetres review avec Q, puis ouvrir points, une video a la fois :

```bash
.venv/bin/python scripts/annotate_sam2.py points data/annotations_sam2_clip_bas_retour/test1 --max-height 650
.venv/bin/python scripts/annotate_sam2.py points data/annotations_sam2_clip_bas_retour/test2 --max-height 650
```

2. Selectionner `#2 cable` avec Tab. Ajouter des clics droits sur les trous de
   la table pris a tort pour le cable. Sur chaque nouvelle image reperee,
   ajouter aussi au moins un clic gauche sur du cable reellement visible.
3. Verifier aussi `#4 clip` avec Tab : les reperes historiques ont ete prepares
   avant cette clarification. Dans test2, ils doivent suivre la piece mobile
   sur le cable, pas rester sur un support voisin du connecteur. Corriger les
   mauvais reperes avec X pour l'image selectionnee, puis replacer les points.
   Ne pas creer un nouvel objet pour chaque position de la meme piece.
4. Q enregistre et ferme. Les anciens masques restent sur disque, mais changer
   les points exige un nouveau calcul et une nouvelle verification humaine.
5. Apres sauvegarde, recreer le ZIP avec le script actuel :

```bash
.venv/bin/python scripts/annotate_sam2.py pack data/annotations_sam2_clip_bas_retour/test1 data/annotations_sam2_clip_bas_retour/test2 --output data/reperes_sam2_visibilite.zip
```

Ouvrir le notebook `ANNOTATION_SAM2_COLAB.ipynb` a jour dans Colab avec un GPU.
Reexecuter la cellule de chargement avec ce nouveau ZIP, puis les suivantes.
Ne pas reutiliser un ancien ZIP qui embarque l'ancienne logique de conversion.
Telecharger `resultats_sam2_clip_mobile.zip`, le placer dans `data/`, puis :

```bash
.venv/bin/python scripts/annotate_sam2.py restore data/resultats_sam2_clip_mobile.zip --output data/annotations_sam2_clip_mobile_retour
.venv/bin/python scripts/annotate_sam2.py review data/annotations_sam2_clip_mobile_retour/test1 --max-height 650
.venv/bin/python scripts/annotate_sam2.py review data/annotations_sam2_clip_mobile_retour/test2 --max-height 650
```

La seule suppression des regles de clip vide ne demande PAS de nouveau calcul
SAM : fermer et relancer review suffit si les points n'ont pas change.
Les anciennes approbations doivent etre refaites avec la nouvelle politique.
Sur les images reperees, un masque qui manque un clic positif ou inclut un
clic negatif reste bloque. Entre ces images, verifier visuellement le suivi.

Apres validation humaine et annotation de toutes les instances visibles :

```bash
.venv/bin/python scripts/annotate_sam2.py export --train data/annotations_sam2_clip_mobile_retour/test1 --val data/annotations_sam2_clip_mobile_retour/test2 --output data/datasets/faisceau_seg_clip_mobile
```

## 1. Ouvrir le terminal du Mac

```bash
cd /Users/bigsur/Documents/wire-harness-vision-inspection
source .venv/bin/activate
python scripts/annotate_sam2.py --help
```

Les operations locales utilisent seulement OpenCV et NumPy, deja presents dans
la venv de l'application. Ne pas installer `requirements-sam2.txt` dans cette
venv : le calcul SAM necessite PyTorch >= 2.5.1, alors que le Mac Intel utilise
actuellement PyTorch 2.2.2. Le notebook installe SAM dans Colab.

Il n'est pas necessaire de lancer MQTT, Node-RED ou `run.py` pendant l'annotation.

## 2. Preparer les images

Cette commande est a executer une seule fois pour ces dossiers :

```bash
python scripts/annotate_sam2.py prepare data/videos/test1.MOV data/videos/test2.MOV
```

Elle cree `data/annotations_sam2/test1` et `data/annotations_sam2/test2`, avec
les images et un fichier `prompts.json` initialement vide.
La cadence par defaut est de 6 images/s, adaptee a un premier essai court.
Les videos originales restent intactes. Pour un mouvement trop rapide,
repreparer a 15 ou 30 images/s dans un autre dossier avec, par exemple,
`--sample-fps 15 --output data/annotations_sam2_15fps`.

Si le dossier existe deja, passer a l'etape suivante. Le script refuse de
l'ecraser pour proteger les reperes. Pour une video mal orientee, preparer une
nouvelle sequence avec `--rotation 90`, `180` ou `270`, avant de poser les clics.

## 3. Placer les reperes

```bash
python scripts/annotate_sam2.py points data/annotations_sam2/test1
```

Cliquer dans la fenetre pour lui donner le focus. Choisir avec le curseur
`Image` une image ou les pieces sont visibles.

1. Appuyer une fois sur `0` pour creer le premier connecteur. Appuyer sur `b`,
   encadrer uniquement le cube dore/noir, puis valider avec Entree.
2. Ajouter un clic gauche sur la piece si utile. Un clic droit sur le support
   blanc/gris indique a SAM qu'il faut exclure cette partie.
3. Pour un autre connecteur, appuyer de nouveau sur `0` et placer ses reperes.
4. Appuyer sur `1` pour creer le clip. Un rectangle serre ou des clics positifs
   sur le petit ergot suffisent comme point de depart.
5. Appuyer sur `2` pour creer le cable. Poser plusieurs clics gauches le long
   du fil, notamment aux courbures. Ajouter des clics droits sur les elements
   voisins que SAM pourrait confondre avec lui.
6. Appuyer sur `s` pour enregistrer, puis sur `q` pour quitter.

Repeter pour la seconde video :

```bash
python scripts/annotate_sam2.py points data/annotations_sam2/test2
```

Commandes utiles dans la fenetre :

| Touche | Action |
| --- | --- |
| Tab | Selectionner un objet deja cree |
| n / p | Image suivante / precedente |
| u | Annuler le dernier clic, ou le rectangle s'il n'y a plus de clics |
| x | Effacer les reperes de l'objet selectionne sur cette image |
| d | Supprimer l'objet selectionne et tous ses reperes |
| s / q | Enregistrer / enregistrer et quitter |

**Pour suivre la meme piece sur une autre image, utiliser Tab, pas 0/1/2.**
Ces chiffres creent de nouveaux objets. L'objet selectionne porte le symbole `>`.

Commencer par quelques images bien visibles ; il n'est pas necessaire de
reperer toutes les images. Chaque image reperee pour un objet doit contenir
au moins un clic positif ou un rectangle. Ne pas inventer une piece absente.
Sur les images finalement retenues, toutes les pieces visibles des classes
cibles doivent etre annotees, y compris un connecteur supplementaire.

A ce stade, seuls les reperes sont visibles. Les masques seront calcules dans
Colab. Si la fenetre est trop haute, ajouter `--max-height 620` a la commande.

## 4. Creer le ZIP pour Colab

Fermer les fenetres de reperage apres sauvegarde, puis lancer :

```bash
python scripts/annotate_sam2.py pack data/annotations_sam2/test1 data/annotations_sam2/test2 --output data/reperes_sam2.zip
```

Le ZIP contient les images, leurs reperes, le script et les dependances de
calcul. Il ne contient ni le projet complet, ni la venv, ni le modele demo.
Pour un nouvel essai, choisir un autre nom de ZIP si celui-ci existe deja.

## 5. Calculer les masques dans Colab

1. Ouvrir `ANNOTATION_SAM2_COLAB.ipynb` dans Google Colab.
2. Choisir **Execution > Modifier le type d'execution > GPU**.
3. Executer les cellules dans l'ordre. Au chargement, choisir
   `data/reperes_sam2.zip`.
4. Laisser le notebook installer SAM et telecharger les poids officiels Tiny.
5. Executer le calcul, regarder les apercus, puis telecharger
   `resultats_sam2.zip` avec la derniere cellule.

Le code SAM est fixe a une revision precise. Le calcul tourne dans un processus
separe pour utiliser les bibliotheques installees, meme si le noyau du notebook
a deja importe d'autres versions. La configuration initiale utilise du float32,
avec les images et l'etat de suivi stockes en RAM pour limiter la memoire GPU.
Les masques sont propages individuellement, vers l'avant et vers l'arriere si
le premier repere n'est pas sur la premiere image.

Les images de l'usine sont envoyees a Colab lorsque le ZIP est charge. Pour
rester entierement local, utiliser une machine compatible dans une venv SAM
separee et appeler la commande `propagate` indiquee plus bas.

## 6. Recuperer et controler les masques sur le Mac

Si le ZIP se trouve dans Telechargements :

```bash
python scripts/annotate_sam2.py restore ~/Downloads/resultats_sam2.zip --output data/annotations_sam2_retour
python scripts/annotate_sam2.py review data/annotations_sam2_retour/test1
python scripts/annotate_sam2.py review data/annotations_sam2_retour/test2
```

Adapter le chemin si le navigateur a renomme le ZIP. L'import cree un dossier
neuf, conserve les reperes initiaux et remet les decisions en attente.

Dans la fenetre de controle :

- `a` : approuver l'image si toutes les pieces visibles sont correctement segmentees.
- `r` : exclure l'image ; `u` : remettre une decision en attente.
- `b` : confirmer une image entierement sans objet, seulement si tous les masques sont vides.
- `v` : alterner entre SAM original, YOLO export et differences. La vue initiale est YOLO.
- `m` : comparer avec l'image sans masque ; `n/p` : changer d'image.
- `j/k` : faire defiler les details si tous les messages ne tiennent pas a l'ecran.
- `q` : enregistrer et quitter.

Un avertissement de masque vide exige de verifier que la piece est reellement
absente ou entierement cachee. Une piece visible oubliee par SAM doit etre
corrigee avant approbation. `A` approuve une ANNOTATION, pas un montage OK :
un clip reste sur le connecteur peut etre une excellente image du cas NOK.
Ne pas supprimer le clip de l'annotation parce que le montage est incorrect.

### Conversion des masques discontinus

Il n'est pas necessaire de relancer Colab pour cette mise a jour. Les commandes
`review` et `export` recalculent la conversion a partir des NPZ existants.
Le champ `quality` d'un ancien `run.json` est historique : il n'est pas utilise
pour decider si la nouvelle conversion est autorisee.

La fonction locale `merge_multi_segment` est une implementation NumPy autonome
du principe de raccords aller-retour utilise par Ultralytics. Les contours
de plusieurs morceaux d'une meme instance restent sur UNE ligne YOLO.
Les contours internes sont aussi parcourus pour conserver les ouvertures.
Ce n'est ni une dilatation, ni une reconstitution du cable sous la main.
Les raccords peuvent toutefois ajouter de fins traits au masque rasterise :
ils doivent etre controles, pas consideres comme du cable reel.

La conversion verifie exactement le texte des sommets normalises relu en
float32 et rasterise a la resolution source. Les coordonnees de centres de
pixels evitent un decalage involontaire par arrondi. L'IoU avec SAM doit etre
d'au moins 0,98. Les composantes parasites de 3 pixels au maximum ne sont
retirees que si leur total ne depasse pas 0,1 % de l'objet. Aucune grosse
composante n'est supprimee pour faciliter l'export. Aucune ouverture interne
ne doit etre remplie par la conversion, meme partiellement.

Dans la vue differences, le magenta montre les pixels ajoutes par YOLO et le
jaune les pixels retires. La touche `A` exige une vue YOLO ou differences
visible. Une ancienne approbation doit etre refaite : l'approbation est liee
aux masques, aux reperes, aux regles et au polygone exact affiches.

### Qualite d'annotation, pas verdict d'assemblage

Aucune zone de la table n'est reservee au clip. Les intersections entre
instances ne declenchent plus de blocage specifique. Le raccord de format
reste visible dans l'apercu des differences, meme s'il passe pres du clip.
Il ne prouve pas que le cable reel est en contact avec cette piece.

Les controles generaux restent actifs pour toutes les classes : fidelite au
masque SAM, preservation de ses ouvertures, contours non degeneres, respect
des clics, integrite des fichiers et validation humaine. Un message
`conversion du contour trop imprecise` indique encore un IoU inferieur a 0,98,
pas un montage NOK. Corriger alors la segmentation (par exemple les trous de
table inclus a tort), sans forcer l'approbation d'un mauvais contour.

Ces verifications n'imposent ni l'emplacement du clip, ni un etat OK/NOK. Les
annotations des deux scenarios sont exportables si leur geometrie est fidele.

## 7. Exporter le dataset YOLOv8-Seg

Apres avoir approuve des images dans les deux sequences :

```bash
python scripts/annotate_sam2.py export --train data/annotations_sam2_retour/test1 --val data/annotations_sam2_retour/test2 --output data/datasets/faisceau_seg_essai
```

Resultat : `data/datasets/faisceau_seg_essai.zip`, contenant `images/train`,
`images/val`, `labels/train`, `labels/val`, `data.yaml` et un manifeste de suivi.
`masks_sam/train` et `masks_sam/val` conservent les NPZ SAM bruts, y compris
leurs eventuelles hallucinations, pour l'audit. `masks_visible/train` et
`masks_visible/val` contiennent les masques apres application des absences
humaines, sans raccord artificiel. Le manifeste donne les empreintes des deux,
les declarations de visibilite par image, les identifiants des objets et les
differences de conversion. L'IoU de conversion est compare au masque apres
declaration d'absence ; `human_suppressed_pixels` distingue la suppression
humaine des changements de contour. Les images incertaines ne sont pas
exportees, pas meme leurs autres objets. Ne pas utiliser les traits de
raccord des polygones YOLO comme preuve de contact cable/clip.
Seules les images approuvees sont exportees. Le train doit contenir au moins
un masque approuve de chacune des trois classes. Chaque ligne de label contient
la classe et les sommets normalises d'un polygone, et non une boite. Le rendu
du label exporte est celui de la vue YOLO a la resolution source. Le
redimensionnement, le reechantillonnage et les predictions du futur modele
devront aussi etre controles a leur propre resolution.

Le premier essai reste PARTIEL (un connecteur, un clip et un cable reperes
par video). Completer toutes les instances visibles avant de constituer un
dataset d'entrainement final ; une conversion possible n'est pas une annotation
complete, ni une validation industrielle.

L'export separe les videos, pas des images consecutives tirees au hasard. Ces
deux prises restent proches : elles servent a un premier essai, pas a demontrer
une fiabilite industrielle. Ajouter ensuite d'autres prises et montages OK/NOK.

Dans le notebook d'entrainement, utiliser **`yolov8n-seg.pt`** comme modele de
depart, le nouveau ZIP et son ordre de classes. Ne pas reutiliser les labels
rectangulaires du dataset demo. `OK/NOK` reste une decision applicative.

### Export d'essai avec les masques actuels

Pour tester le pipeline sans refaire les annotations, `export --draft` accepte
les polygones candidats imparfaits et les images non approuvees, **y compris
les images marquees rejetees**. Il ne change ni les decisions du controle
visuel ni les masques SAM. Le ZIP est explicitement marque **NON VALIDE** ;
le manifeste conserve les avertissements, les decisions et les images exclues.
Sans `--draft`, les exigences de validation habituelles restent actives.

Les images marquees incertaines (`I`) restent exclues. Les absences (`H`)
restent appliquees. Une image est egalement exclue si un objet non vide n'a
aucun polygone representable, ou si tous ses masques sont vides sans
confirmation. Les empreintes des images, des reperes et des masques restent
obligatoires : cette option ne contourne pas un calcul devenu obsolete.

Pour l'essai avec `resultats_sam2_v2.zip`, les masques locaux sont identiques
au dernier retour Colab. Dans `test2`, les reperes du cable ont ete effaces
apres le calcul : on restaure donc ce retour **dans un nouveau dossier**, sans
remplacer la session en cours. `test1` utilise la session actuelle et conserve
sa decision de rejet dans le manifeste de l'essai.

```bash
python scripts/annotate_sam2.py restore data/resultats_sam2_v2.zip --output data/annotations_sam2_export_essai_v2
python scripts/annotate_sam2.py export --draft --train data/annotations_sam2_clip_bas_retour/test1 --val data/annotations_sam2_export_essai_v2/test2 --output data/datasets/faisceau_seg_essai_actuel
```

Ces commandes refusent d'ecraser un dossier ou un ZIP existant. Apres le
premier export, utiliser directement
`data/datasets/faisceau_seg_essai_actuel.zip` dans Colab, ou choisir un nouveau
nom de sortie pour refaire l'essai. Ce ZIP est un dataset, pas `best.pt` :
il faut encore entrainer le modele de segmentation.

La regle metier demandee est **NOK uniquement si le clip reste attache au
connecteur**, pas lorsque ces deux classes sont simplement presentes.
L'export ne calcule pas cette relation et ne modifie pas Node-RED. Les masques
imparfaits ou les raccords de polygones ne prouvent pas une liaison physique.
Avec `test1` seul en train et `test2` seul en validation, les scores servent
uniquement a tester l'execution, pas a evaluer la discrimination OK/NOK.

## Calcul sur une machine GPU locale

Dans une venv distincte compatible avec PyTorch recent, installer
`requirements-sam2.txt`. SAM recommande Linux ou WSL pour Windows. Les poids
Tiny officiels et leur configuration sont ceux utilises par le notebook.

```bash
python scripts/annotate_sam2.py propagate data/annotations_sam2/test1 --checkpoint /chemin/sam2.1_hiera_tiny.pt --device cuda
python scripts/annotate_sam2.py review data/annotations_sam2/test1
```

## Verifications et limites

Les tests locaux couvrent la conservation des coordonnees, la conversion des
masques en polygones, la separation train/val, les transferts ZIP et
l'invalidation des anciennes annotations. Ils utilisent des masques artificiels
et un predicteur simule ; ils ne prouvent pas la qualite de SAM sur les videos.

```bash
python -m unittest discover -s test -p 'test_sam2_annotation.py' -v
```

La propagation reelle doit etre verifiee avec les reperes choisis et le GPU.
La selection graphique doit etre testee dans une session de bureau active.

References : [SAM 2 officiel](https://github.com/facebookresearch/sam2),
[format YOLO-Seg](https://docs.ultralytics.com/datasets/segment/),
[raccords multi-contours](https://docs.ultralytics.com/reference/data/converter/#ultralytics.data.converter.merge_multi_segment).
