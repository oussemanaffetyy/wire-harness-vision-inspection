# Dashboard d'inspection

Le flow `wire_harness_dashboard_flow.json` utilise **FlowFuse Dashboard**
(`@flowfuse/node-red-dashboard`, déjà inclus dans les dépendances du projet).
La détection Python, MQTT et l'écriture de `C:\test\IACom.txt` ne sont pas modifiées.

## Ouvrir le dashboard

Après mise à jour du projet, arrêter puis relancer le Node-RED du projet :

```bat
node-red
```

Dans le second terminal CMD, à la racine du projet :

```bat
.venv\Scripts\activate
python run.py
```

Ouvrir <http://127.0.0.1:1880/inspection/> et actualiser la page.
Sur Mac, utiliser `npm --prefix nodered start` pour le dashboard.

Le lanceur du projet charge automatiquement le JSON. Pour une installation
Node-RED personnelle, remplacer l'ancien flow d'inspection par cet export ;
ne pas ajouter une deuxième copie. Garder **un seul `ui-base`** et ne pas
supprimer les autres flows personnels.

## Ce qui apparaît

- La vidéo annotée et le verdict envoyé par Python, sans FPS ni compteurs.
- Une explication du résultat et l'heure de réception du résultat côté Python.
- Les 300 derniers changements de verdict, de source vidéo ou de boucle.
- Le filtre **Défauts** et la pagination, sans ascenseur interne.

Un résultat de plus de 15 secondes ou une connexion navigateur interrompue
est affiché en **pause**, avec le dernier verdict daté. La règle Python reste
inchangée : NOK si contact/proximité clip-connecteur, sinon OK par défaut.
OK n'est pas une preuve de conformité, notamment si un objet manque à la détection.

## Retrouver le fichier TXT

**Ouvrir un TXT** permet de sélectionner `C:\test\IACom.txt`. Le fichier est
lu uniquement dans le navigateur, jamais envoyé au serveur ni modifié. Les
lignes OK/NOK identiques consécutives sont regroupées ; les 300 derniers
changements sont conservés à l'écran. Taille maximale : 20 Mo.

La vidéo et le verdict continuent de fonctionner pendant la consultation.
**Revenir à la session** réaffiche l'historique MQTT courant. Le fichier importé
est une copie au moment de l'ouverture : le rouvrir pour voir ses nouvelles lignes.

**Exporter TXT** télécharge l'historique sélectionné (avec le filtre courant)
dans un nouveau fichier. Il ne remplace pas `IACom.txt` et n'est pas une archive
de chaque image analysée. Les horaires de l'export sont locaux au navigateur ;
ceux du journal Python proviennent du PC qui exécute Python.

L'historique MQTT est partagé entre navigateurs et survit à une actualisation.
Avec la configuration fournie, il est en mémoire et repart à zéro au redémarrage
de Node-RED. Le journal Python reste l'archive persistante. Un fichier TXT importé
reste local à l'onglet et disparaît de l'écran après actualisation.

## Maintenance

`dashboard.vue` contient le rendu, `inspection_history.js` le traitement MQTT.
Après modification de ces sources :

```bash
node nodered/build_dashboard.cjs
node --test test/test_dashboard.cjs
```

Le JSON généré contient tout : aucune dépendance vers ces sources n'est nécessaire
lors de l'import dans Node-RED. Aucun accès Internet n'est utilisé par le dashboard.
