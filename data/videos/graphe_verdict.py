import re
from pathlib import Path
import matplotlib.pyplot as plt


# ============================================================
# CONFIGURATION
# ============================================================

# Remplacez ce nom si votre fichier de log porte un autre nom.
LOG_FILE = "/Users/bigsur/Documents/wire-harness-vision-inspection/github_ready/data/logs/IACom.txt"

# Images produites
OUTPUT_FILES = {
    "test1.MOV": "graphe_test1.png",
    "test2.MOV": "graphe_test2.png",
}


# ============================================================
# LECTURE ET PARSING DU LOG
# ============================================================

def parse_log(log_file):
    """
    Extrait les frames et statuts pour test1.MOV et test2.MOV.

    Retour :
        {
            "test1.MOV": [(frame, verdict), ...],
            "test2.MOV": [(frame, verdict), ...]
        }

    verdict :
        OK  -> 1
        NOK -> 0
    """

    data = {
        "test1.MOV": [],
        "test2.MOV": []
    }

    current_video = None

    # Exemple :
    # Video source: /.../test1.MOV | global frame=0
    video_pattern = re.compile(
        r"Video source:.*?(test1\.MOV|test2\.MOV)"
    )

    # Exemple :
    # Inspection event: status=NOK frame=19 ...
    event_pattern = re.compile(
        r"Inspection event:\s*status=(OK|NOK)\s+frame=(\d+)"
    )

    with open(log_file, "r", encoding="utf-8", errors="ignore") as f:

        for line in f:

            # ------------------------------------------------
            # Détection de la vidéo courante
            # ------------------------------------------------
            video_match = video_pattern.search(line)

            if video_match:
                current_video = video_match.group(1)
                continue

            # ------------------------------------------------
            # Extraction status + frame
            # ------------------------------------------------
            event_match = event_pattern.search(line)

            if event_match and current_video is not None:

                status = event_match.group(1)
                frame = int(event_match.group(2))

                # Conversion numérique
                verdict = 1 if status == "OK" else 0

                data[current_video].append((frame, verdict))

    return data


# ============================================================
# NETTOYAGE DES DONNÉES
# ============================================================

def clean_data(values):
    """
    Trie les données par numéro de frame.

    Le log peut contenir plusieurs exécutions du même test.
    Si une frame apparaît plusieurs fois, on garde le dernier
    verdict enregistré pour cette frame.
    """

    frames_dict = {}

    for frame, verdict in values:
        frames_dict[frame] = verdict

    return sorted(frames_dict.items())


# ============================================================
# CRÉATION DU GRAPHIQUE
# ============================================================

def create_graph(data, video_name, output_file, line_color):

    values = clean_data(data)

    if not values:
        print(f"Aucune donnée trouvée pour {video_name}")
        return

    frames = [frame for frame, verdict in values]
    verdicts = [verdict for frame, verdict in values]

    # Figure adaptée à Word / rapport
    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    ax.set_facecolor("white")

    # --------------------------------------------------------
    # Courbe principale
    # --------------------------------------------------------
    ax.step(
        frames,
        verdicts,
        where="post",
        linewidth=2.2,
        color=line_color,
        label="Verdict"
    )

    # Points pour améliorer la lisibilité
    ax.scatter(
        frames,
        verdicts,
        s=18,
        color=line_color,
        zorder=3
    )

    # --------------------------------------------------------
    # Titre et axes
    # --------------------------------------------------------
    ax.set_title(
        f"Évolution du verdict - {video_name}",
        fontsize=16,
        fontweight="bold",
        pad=18
    )

    ax.set_xlabel(
        "Numéro de frame",
        fontsize=12,
        fontweight="bold"
    )

    ax.set_ylabel(
        "Verdict  (0 = NOK / 1 = OK)",
        fontsize=12,
        fontweight="bold"
    )

    # Axe Y binaire
    ax.set_ylim(-0.15, 1.15)

    ax.set_yticks([0, 1])
    ax.set_yticklabels(
        ["0 = NOK", "1 = OK"],
        fontsize=11
    )

    # --------------------------------------------------------
    # Grille
    # --------------------------------------------------------
    ax.grid(
        True,
        linestyle="--",
        linewidth=0.7,
        alpha=0.35
    )

    ax.set_axisbelow(True)

    # --------------------------------------------------------
    # Nettoyage esthétique
    # --------------------------------------------------------
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(axis="x", labelsize=10)
    ax.tick_params(axis="y", labelsize=10)

    # Marge horizontale
    ax.margins(x=0.02)

    plt.tight_layout()

    # --------------------------------------------------------
    # Sauvegarde haute résolution
    # --------------------------------------------------------
    plt.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close()

    print(f"Graphique créé : {output_file}")
    print(f"  Nombre de frames détectées : {len(frames)}")


# ============================================================
# PROGRAMME PRINCIPAL
# ============================================================

def main():

    log_path = Path(LOG_FILE)

    if not log_path.exists():
        raise FileNotFoundError(
            f"Le fichier '{LOG_FILE}' est introuvable.\n"
            "Placez le script dans le même dossier que le log "
            "ou modifiez LOG_FILE."
        )

    print("Lecture du fichier :", log_path)
    print()

    data = parse_log(log_path)

    # Test 1 : bleu
    create_graph(
        data["test1.MOV"],
        "test1.MOV",
        OUTPUT_FILES["test1.MOV"],
        line_color="#1565C0"
    )

    # Test 2 : vert
    create_graph(
        data["test2.MOV"],
        "test2.MOV",
        OUTPUT_FILES["test2.MOV"],
        line_color="#00897B"
    )

    print()
    print("Terminé.")
    print("Images générées :")
    print(" - graphe_test1.png")
    print(" - graphe_test2.png")


if __name__ == "__main__":
    main()