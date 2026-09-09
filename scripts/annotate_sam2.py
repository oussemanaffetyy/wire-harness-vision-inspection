"""Annotation video assistee : reperes, SAM 2.1, controle et export YOLO-Seg.

Les commandes prepare/points/review/export/pack ne chargent pas PyTorch.
Seule propagate demande l'environnement SAM 2.1 (voir ANNOTATION_SAM2.md).
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import stat
import tempfile
import textwrap
import uuid
import zipfile

import cv2
import numpy as np


CLASSES = {0: "connector", 1: "clip", 2: "cable"}
COLORS = {0: (60, 60, 220), 1: (220, 120, 40), 2: (60, 190, 70)}
ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
CONVERSION_POLICY = {
    "version": 5, "min_iou": 0.98, "max_noise_area": 3,
    "max_noise_fraction": 0.001, "business_rules": "none",
    "encoding": "pixel_centers_float32", "merge": "nearest_return_bridges_v2_preserve_holes",
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_hash(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def image_path(job, index):
    return Path(job) / "frames" / f"{index:05d}.jpg"


def load_job(job):
    job = Path(job)
    meta = read_json(job / "sequence.json")
    prompts = read_json(job / "prompts.json")
    expected = {str(k): v for k, v in CLASSES.items()}
    if meta.get("schema") != SCHEMA or meta.get("classes") != expected:
        raise ValueError("Schema ou classes incompatibles dans sequence.json.")
    if not meta.get("frames"):
        raise ValueError("La sequence est vide.")
    for index, info in enumerate(meta["frames"]):
        path = image_path(job, index)
        if not path.is_file() or file_hash(path) != info["sha256"]:
            raise ValueError(f"Image absente ou modifiee : {path}")
    return meta, prompts


def fingerprint(meta, prompts):
    return hashlib.sha256(json.dumps([meta, prompts], sort_keys=True).encode()).hexdigest()


def load_visibility(job, meta, prompts):
    """Corrections humaines separees des points et des masques bruts SAM."""
    path = Path(job) / "visibility.json"
    value = read_json(path) if path.exists() else {
        "schema": 1, "sequence_fingerprint": fingerprint(meta, {}), "objects": {}}
    if (not isinstance(value, dict) or set(value) != {"schema", "sequence_fingerprint", "objects"}
            or type(value["schema"]) is not int or value["schema"] != 1
            or value["sequence_fingerprint"] != fingerprint(meta, {})
            or not isinstance(value["objects"], dict)):
        raise ValueError("visibility.json incompatible avec les images de cette sequence.")
    classes = {str(obj["id"]): obj["class_id"] for obj in prompts["objects"]}
    for key, record in value["objects"].items():
        if (key not in classes or not isinstance(record, dict)
                or set(record) != {"class_id", "frames"} or type(record["class_id"]) is not int
                or record["class_id"] != classes[key] or not isinstance(record["frames"], dict)):
            raise ValueError("visibility.json : objet inconnu ou classe modifiee.")
        for index, state in record["frames"].items():
            if (not index.isascii() or not index.isdigit() or str(int(index)) != index
                    or not 0 <= int(index) < len(meta["frames"]) or state not in ("absent", "ignore")):
                raise ValueError("visibility.json : indice ou etat invalide.")
    return value


def visibility_at(visibility, index):
    if visibility and visibility["objects"] and index is None:
        raise ValueError("Un indice d'image est requis pour appliquer la visibilite.")
    return {key: record["frames"][str(index)] for key, record in (visibility or {}).get("objects", {}).items()
            if str(index) in record["frames"]}


def set_visibility(visibility, obj, start, end, state, frame_count):
    if (type(start) is not int or type(end) is not int or not 0 <= min(start, end)
            or max(start, end) >= frame_count or state not in ("absent", "ignore", "clear")):
        raise ValueError("Plage de visibilite invalide.")
    record = visibility["objects"].setdefault(str(obj["id"]), {"class_id": obj["class_id"], "frames": {}})
    for index in range(min(start, end), max(start, end) + 1):
        if state == "clear":
            record["frames"].pop(str(index), None)
        else:
            record["frames"][str(index)] = state
    if not record["frames"]:
        visibility["objects"].pop(str(obj["id"]))


def visible_masks(masks, states):
    return {key: np.zeros_like(mask, dtype=bool) if states.get(key) == "absent" else mask.copy()
            for key, mask in masks.items()}


def validate_prompts(meta, prompts):
    objects = prompts.get("objects", [])
    ids = [obj.get("id") for obj in objects]
    if not objects or len(set(ids)) != len(ids):
        raise ValueError("Ajouter des objets avec la commande points (identifiants uniques).")
    for obj in objects:
        if type(obj["id"]) is not int or obj["id"] < 1 or obj["class_id"] not in CLASSES:
            raise ValueError("Identifiant ou classe d'objet invalide.")
        if not obj.get("prompts"):
            raise ValueError(f"Objet {obj['id']} : aucun repere. Ajouter des clics ou supprimer cet objet.")
        for key, prompt in obj["prompts"].items():
            if str(int(key)) != key or not 0 <= int(key) < len(meta["frames"]):
                raise ValueError(f"Indice d'image invalide : {key}")
            points, labels, box = prompt.get("points", []), prompt.get("labels", []), prompt.get("box")
            if len(points) != len(labels) or any(label not in (0, 1) for label in labels):
                raise ValueError("Un label 0/1 est requis pour chaque clic.")
            if not points and box is None:
                raise ValueError("Repere vide.")
            coords = list(points)
            if box is not None:
                if len(box) != 4 or not (box[0] < box[2] and box[1] < box[3]):
                    raise ValueError("Rectangle invalide.")
                coords += [box[:2], box[2:]]
            for point in coords:
                if len(point) != 2 or not np.isfinite(point).all():
                    raise ValueError("Coordonnees invalides.")
                if not (0 <= point[0] < meta["width"] and 0 <= point[1] < meta["height"]):
                    raise ValueError("Un repere sort de l'image.")
            if box is None and 1 not in labels:
                raise ValueError("Ajouter au moins un clic positif a chaque image corrigee.")


def prepare(video, output, sample_fps=6.0, rotation=0, max_frames=600):
    video, output = Path(video).resolve(), Path(output).resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if sample_fps <= 0 or max_frames < 1:
        raise ValueError("La cadence et le nombre maximal d'images doivent etre positifs.")
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem)
    job = output / name
    if job.exists():
        raise FileExistsError(f"Dossier deja present : {job}. Utiliser points ou un autre --output.")
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"Video illisible : {video.name}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cadence video inconnue.")
        stride = max(1, round(fps / sample_fps))
        (job / "frames").mkdir(parents=True)
        frames, source_index, truncated = [], 0, False
        size = None
        rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if source_index % stride == 0:
                if len(frames) >= max_frames:
                    truncated = True
                    break
                if rotation:
                    frame = cv2.rotate(frame, rotations[rotation])
                height, width = frame.shape[:2]
                if size is not None and size != (width, height):
                    raise ValueError("La resolution change dans la sequence.")
                size = width, height
                path = image_path(job, len(frames))
                if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                    raise OSError(f"Ecriture impossible : {path}")
                frames.append({"source_index": source_index, "time_s": source_index / fps, "sha256": file_hash(path)})
            source_index += 1
        if not frames:
            raise ValueError("Aucune image extraite.")
        write_json(job / "sequence.json", {
            "schema": SCHEMA, "classes": CLASSES, "source_name": video.name,
            "source_sha256": file_hash(video), "width": size[0], "height": size[1],
            "fps": fps, "stride": stride, "rotation": rotation, "truncated": truncated, "frames": frames,
        })
        write_json(job / "prompts.json", {"objects": []})
        print(f"{job} : {len(frames)} images, {fps / stride:g} images/s" + (" (sequence tronquee)" if truncated else ""))
    finally:
        capture.release()
    return job


def merge_multi_segment(segments):
    """Raccords aller-retour, selon le principe merge_multi_segment d'Ultralytics.

    Implementation NumPy autonome : aucune importation de PyTorch pour review.
    Un raccord est un artifice du format, jamais une preuve de cable visible.
    """
    polygon = np.asarray(segments[0], dtype=np.int32).reshape(-1, 2).copy()
    for segment in segments[1:]:
        segment = np.asarray(segment, dtype=np.int32).reshape(-1, 2)
        # Bounded temporary arrays, including for long, detailed cable contours.
        best = None
        for start in range(0, len(polygon), 256):
            delta = polygon[start:start + 256, None].astype(float) - segment[None]
            distances = np.sum(delta * delta, axis=2)
            i, j = np.unravel_index(np.argmin(distances), distances.shape)
            candidate = (float(distances[i, j]), start + int(i), int(j))
            if best is None or candidate < best:
                best = candidate
        _, i, j = best
        loop = np.roll(segment, -j, axis=0)
        polygon = np.concatenate((polygon[:i + 1], loop, loop[:1], polygon[i:]))
    return polygon


def normalized_polygon(polygon, shape):
    height, width = shape
    # Pixel centers survive text -> float32 -> int32 without shifting an edge
    # left/up because of floating-point rounding in the YOLO mask loader.
    return (polygon.astype(float) + 0.5) / [width, height]


def polygon_text(polygon, shape):
    return " ".join(f"{value:.8f}" for value in normalized_polygon(polygon, shape).ravel())


def polygon_raster(polygon, shape):
    raster = np.zeros(shape, dtype=np.uint8)
    if polygon is not None:
        height, width = shape
        values = np.fromstring(polygon_text(polygon, shape), sep=" ", dtype=np.float32).reshape(-1, 2)
        vertices = (values * np.array([width, height], dtype=np.float32)).astype(np.int32)
        cv2.fillPoly(raster, [vertices], 1)
    return raster.astype(bool)


def polygon_from_mask(mask):
    """Convertir sans dilatation ; controler la fidelite au masque original."""
    binary = np.asarray(mask, dtype=bool).astype(np.uint8)
    if binary.ndim != 2:
        raise ValueError("Un masque doit avoir deux dimensions.")
    if not binary.any():
        return None, ["masque vide : verifier l'absence"], False
    original = binary.copy()
    messages = []
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    small = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] <= CONVERSION_POLICY["max_noise_area"]]
    removed = sum(int(stats[i, cv2.CC_STAT_AREA]) for i in small)
    if small and removed <= binary.sum() * CONVERSION_POLICY["max_noise_fraction"]:
        binary[np.isin(labels, small)] = 0
        messages.append(f"{removed} pixel(s) parasite(s) retire(s) de la copie YOLO")
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    exterior = [c for i, c in enumerate(contours) if hierarchy[0][i][3] == -1]
    filled = np.zeros_like(binary)
    for contour in exterior:
        cv2.fillPoly(filled, [contour], 1)
    holes = filled.astype(bool) & ~binary.astype(bool)
    # Inner rings are traversed too. An out-and-back cut encodes a hole without
    # filling its opening; the raster check below rejects even one added pixel.
    polygon = merge_multi_segment(contours)
    if len(polygon) < 3 or cv2.contourArea(polygon) <= 0:
        return None, messages + ["contour trop fin ou degenere"], True
    if len(exterior) > 1:
        messages.append(f"{len(exterior)} fragments ; raccords de format a controler")
    if holes.any():
        messages.append("Ouvertures internes : aucun remplissage autorise")
    raster = polygon_raster(polygon, binary.shape)
    iou = np.logical_and(original, raster).sum() / np.logical_or(original, raster).sum()
    added = int(np.count_nonzero(raster & ~original.astype(bool)))
    lost = int(np.count_nonzero(original & ~raster))
    if added or lost:
        messages.append(f"YOLO : +{added} / -{lost} pixels, IoU={iou:.4f}")
    blocked = bool(iou < CONVERSION_POLICY["min_iou"])
    if blocked:
        messages.append("conversion du contour trop imprecise")
    if np.any(raster & holes):
        blocked = True
        messages.append("remplissage d'une ouverture interdit")
    # Keep blocked candidates for review and explicit, unvalidated draft exports.
    return polygon, messages, blocked


def review_settings(job, objects, frame_count=None):
    """Compatibilite avec les anciens appels : aucune regle de clip vide.

    Ne pas lire les anciens fichiers review_settings.json : ils reposent sur
    une hypothese metier abandonnee. Ils restent sur disque pour l'historique.
    """
    return {}


def prompt_issues(masks, objects, frame_index):
    """Verifier la cible sur les seules images reperees, sans modifier les masques."""
    issues = []
    for obj in objects:
        prompt = obj.get("prompts", {}).get(str(frame_index), {})
        mask = masks[str(obj["id"])]
        for number, (point, label) in enumerate(zip(prompt.get("points", []), prompt.get("labels", [])), 1):
            x = min(mask.shape[1] - 1, max(0, round(point[0])))
            y = min(mask.shape[0] - 1, max(0, round(point[1])))
            if bool(mask[y, x]) != bool(label):
                kind = "positif hors du masque" if label else "negatif inclus dans le masque"
                issues.append(f"Repere #{obj['id']} {CLASSES[obj['class_id']]} : clic {number} {kind} ; verifier la cible")
    return issues


def convert_frame(masks, objects, settings=None, frame_index=None, visibility=None):
    """Qualite d'annotation seulement ; l'ancien argument settings est ignore."""
    states = visibility_at(visibility, frame_index)
    effective = visible_masks(masks, states)
    result = {"polygons": {}, "rasters": {}, "added": {}, "removed": {},
              "statistics": {}, "messages": [], "blocked": False,
              "visibility": states, "effective_masks": effective,
              "excluded": "ignore" in states.values()}
    for obj in objects:
        key = str(obj["id"])
        original = effective[key].astype(bool)
        if states.get(key) == "absent":
            polygon, issues, blocked = None, ["absence declaree par l'utilisateur : masque export vide"], False
        else:
            polygon, issues, blocked = polygon_from_mask(original)
        raster = polygon_raster(polygon, original.shape)
        result["polygons"][key] = polygon
        result["rasters"][key] = raster
        result["added"][key] = raster & ~original
        result["removed"][key] = masks[key].astype(bool) & ~raster
        union = np.count_nonzero(original | raster)
        result["statistics"][key] = {
            "added_pixels": int(result["added"][key].sum()),
            "removed_pixels": int(result["removed"][key].sum()),
            "iou": float(np.count_nonzero(original & raster) / union) if union else 1.0,
            "human_suppressed_pixels": int(np.count_nonzero(masks[key])) if states.get(key) == "absent" else 0,
        }
        result["messages"].extend(f"#{key} {CLASSES[obj['class_id']]} : {issue}" for issue in issues)
        result["blocked"] |= blocked

    def block(message):
        result["blocked"] = True
        result["messages"].insert(0, message)

    if frame_index is not None:
        checked_objects = [obj for obj in objects if states.get(str(obj["id"])) not in ("absent", "ignore")]
        for issue in prompt_issues(effective, checked_objects, frame_index):
            block(issue)
    if result["excluded"]:
        block("Objet incertain : IMAGE ENTIERE EXCLUE de l'entrainement (C dans points pour annuler)")

    return result


def mask_report(masks, objects, settings=None, frame_index=None, visibility=None):
    result = convert_frame(masks, objects, settings, frame_index, visibility)
    return result["polygons"], result["messages"], result["blocked"]


def conversion_token(meta, prompts, info, index, result, settings):
    values = {"source": fingerprint(meta, prompts), "frame_index": index,
              "mask": info["mask_hashes"][str(index)],
              "policy": CONVERSION_POLICY, "visibility": result["visibility"],
              "labels": {key: polygon_text(polygon, (meta["height"], meta["width"]))
                         if polygon is not None else None for key, polygon in result["polygons"].items()}}
    return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def track_object(predictor, state, obj, frame_count):
    predictor.reset_state(state)
    for key, prompt in sorted(obj["prompts"].items(), key=lambda item: int(item[0])):
        points = prompt.get("points", [])
        predictor.add_new_points_or_box(
            inference_state=state, frame_idx=int(key), obj_id=obj["id"],
            points=np.asarray(points, dtype=np.float32) if points else None,
            labels=np.asarray(prompt.get("labels", []), dtype=np.int32) if points else None,
            box=np.asarray(prompt["box"], dtype=np.float32) if prompt.get("box") is not None else None,
            normalize_coords=True,
        )
    first = min(int(key) for key in obj["prompts"])
    visited = set()
    for reverse in (False, True):
        if reverse and first == 0:
            continue
        for index, object_ids, logits in predictor.propagate_in_video(state, start_frame_idx=first, reverse=reverse):
            if index in visited:
                continue
            if not 0 <= index < frame_count or obj["id"] not in object_ids:
                raise ValueError("Sortie SAM incoherente avec la sequence.")
            values = logits[object_ids.index(obj["id"])]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values)
            if values.ndim != 3 or values.shape[0] != 1 or not np.isfinite(values).all():
                raise ValueError("Masque SAM invalide.")
            visited.add(index)
            yield index, values[0] > 0
    if visited != set(range(frame_count)):
        raise ValueError("SAM n'a pas retourne toutes les images.")


def read_masks(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key].astype(bool) for key in archive.files}


def propagate(job, checkpoint, sam_config, device):
    job, checkpoint = Path(job).resolve(), Path(checkpoint).resolve()
    meta, prompts = load_job(job)
    validate_prompts(meta, prompts)
    settings = review_settings(job, prompts["objects"], len(meta["frames"]))
    visibility = load_visibility(job, meta, prompts)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Poids SAM 2.1 absents : {checkpoint}")
    try:
        import torch
        if tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:3]) < (2, 5, 1):
            raise RuntimeError("SAM 2.1 demande PyTorch >= 2.5.1. Utiliser le notebook Colab ou une venv separee.")
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError("SAM 2.1 absent. Voir ANNOTATION_SAM2.md ; ne pas modifier la venv de l'application.") from exc
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    run = job / "predictions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8])
    (run / "masks").mkdir(parents=True)
    # A run is activated only after every object and frame has been computed.
    with torch.inference_mode():
        predictor = build_sam2_video_predictor(sam_config, str(checkpoint), device=device, apply_postprocessing=False)
        state = predictor.init_state(str(job / "frames"), offload_video_to_cpu=True, offload_state_to_cpu=True)
        for obj in prompts["objects"]:
            print(f"Propagation objet {obj['id']} ({CLASSES[obj['class_id']]})", flush=True)
            for index, mask in track_object(predictor, state, obj, len(meta["frames"])):
                if mask.shape != (meta["height"], meta["width"]):
                    raise ValueError("La taille du masque ne correspond pas a l'image source.")
                path = run / "masks" / f"{index:05d}.npz"
                masks = read_masks(path) if path.exists() else {}
                masks[str(obj["id"])] = mask
                np.savez_compressed(path, **masks)
    quality = {}
    hashes = {}
    for index in range(len(meta["frames"])):
        path = run / "masks" / f"{index:05d}.npz"
        _, messages, blocked = mask_report(read_masks(path), prompts["objects"], settings, index, visibility)
        quality[str(index)] = {"messages": messages, "blocked": blocked}
        hashes[str(index)] = file_hash(path)
    write_json(run / "run.json", {
        "fingerprint": fingerprint(meta, prompts), "checkpoint": checkpoint.name,
        "checkpoint_sha256": file_hash(checkpoint), "sam_config": sam_config,
        "torch": torch.__version__, "device": device, "quality": quality, "mask_hashes": hashes,
        "visibility": visibility,
    })
    write_json(run / "review.json", {})
    write_json(job / "active_run.json", {"path": run.relative_to(job).as_posix()})
    print(f"Masques calcules : {run}\nEtape suivante : review (aucune image encore approuvee).")
    return run


def active_run(job, meta, prompts):
    job = Path(job).resolve()
    pointer = job / "active_run.json"
    if not pointer.is_file():
        raise ValueError("Pas de masques disponibles. Lancer propagate dans l'environnement SAM.")
    run = (job / read_json(pointer)["path"]).resolve()
    if not run.is_relative_to(job / "predictions"):
        raise ValueError("Chemin de predictions invalide.")
    info = read_json(run / "run.json")
    if info["fingerprint"] != fingerprint(meta, prompts):
        raise ValueError("Les reperes ont change. Recalculer les masques avec propagate avant validation/export.")
    return run, info


def checked_masks(run, info, index):
    path = run / "masks" / f"{index:05d}.npz"
    if file_hash(path) != info["mask_hashes"][str(index)]:
        raise ValueError("Masque modifie apres calcul : recalculer puis revalider.")
    return read_masks(path)


def overlay(frame, masks, objects):
    canvas = frame.copy()
    for obj in objects:
        mask = masks.get(str(obj["id"]))
        if mask is None or not mask.any():
            continue
        color = np.array(COLORS[obj["class_id"]], dtype=np.float32)
        canvas[mask] = (0.65 * canvas[mask] + 0.35 * color).astype(np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, tuple(int(c) for c in color), 1)
        y, x = np.argwhere(mask)[0]
        cv2.putText(canvas, f"#{obj['id']} {CLASSES[obj['class_id']]}", (int(x), max(15, int(y))), cv2.FONT_HERSHEY_SIMPLEX, 0.45, tuple(int(c) for c in color), 1, cv2.LINE_AA)
    return canvas


def review_canvas(frame, masks, objects, converted, preview_mode):
    selected = converted["rasters"] if preview_mode == 1 else masks
    canvas = overlay(frame, selected, objects)
    if preview_mode == 2:
        for obj in objects:
            key = str(obj["id"])
            canvas[converted["added"][key]] = (220, 40, 220)
            canvas[converted["removed"][key]] = (0, 220, 255)
    return canvas


def viewer(job, review=False, max_height=780):
    job = Path(job).resolve()
    meta, prompts = load_job(job)
    visibility = load_visibility(job, meta, prompts)
    visibility_dirty, range_start = False, None
    run, run_info, decisions = None, None, {}
    if review:
        run, run_info = active_run(job, meta, prompts)
        decisions = read_json(run / "review.json")
    elif (job / "active_run.json").exists():
        try:
            run, run_info = active_run(job, meta, prompts)
        except ValueError:
            pass
    title = "Controle des masques" if review else "Reperes SAM 2.1"
    cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE)
    cv2.createTrackbar("Image", title, 0, max(1, len(meta["frames"]) - 1), lambda value: None)
    view = {"index": 0, "selected": 0, "sx": 1.0, "sy": 1.0, "w": 0, "h": 0}
    message, mask_visible = "", True
    settings = review_settings(job, prompts["objects"], len(meta["frames"])) if review else None
    preview_mode, detail_offset = 1, 0
    cached_index, converted, token = None, None, None

    def current_object():
        if not prompts["objects"]:
            return None
        view["selected"] %= len(prompts["objects"])
        return prompts["objects"][view["selected"]]

    def mouse(event, x, y, flags, param):
        nonlocal message
        obj = current_object()
        if review or obj is None or not (0 <= x < view["w"] and 0 <= y < view["h"]):
            return
        if event not in (cv2.EVENT_LBUTTONDOWN, cv2.EVENT_RBUTTONDOWN):
            return
        if str(obj["id"]) in visibility_at(visibility, view["index"]):
            message = "C : annuler la declaration de visibilite avant de placer des clics."
            return
        prompt = obj["prompts"].setdefault(str(view["index"]), {"points": [], "labels": []})
        prompt["points"].append([min(meta["width"] - 1, x / view["sx"]), min(meta["height"] - 1, y / view["sy"])])
        prompt["labels"].append(1 if event == cv2.EVENT_LBUTTONDOWN else 0)

    cv2.setMouseCallback(title, mouse)
    try:
        while True:
            index = min(cv2.getTrackbarPos("Image", title), len(meta["frames"]) - 1)
            view["index"] = index
            if index != cached_index or not review:
                frame = cv2.imread(str(image_path(job, index)))
                if frame is None:
                    raise ValueError("Image illisible.")
                masks = checked_masks(run, run_info, index) if run else {}
                if review:
                    converted = convert_frame(masks, prompts["objects"], settings, index, visibility)
                    token = conversion_token(meta, prompts, run_info, index, converted, settings)
                    detail_offset, message = 0, ""
                cached_index = index
            states = visibility_at(visibility, index)
            if not mask_visible:
                canvas = frame.copy()
            elif review:
                canvas = review_canvas(frame, masks, prompts["objects"], converted, preview_mode)
            else:
                canvas = overlay(frame, visible_masks(masks, states), prompts["objects"])
            obj = current_object()
            if not review:
                for item in prompts["objects"]:
                    prompt = item["prompts"].get(str(index), {})
                    color = COLORS[item["class_id"]]
                    for point, label in zip(prompt.get("points", []), prompt.get("labels", [])):
                        xy = tuple(round(v) for v in point)
                        cv2.circle(canvas, xy, 5, color, 2)
                        cv2.putText(canvas, "+" if label else "-", (xy[0] + 6, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
                    if prompt.get("box"):
                        x1, y1, x2, y2 = map(round, prompt["box"])
                        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            scale = min(max_height / meta["height"], 1100 / meta["width"], 1.5)
            w, h = round(meta["width"] * scale), round(meta["height"] * scale)
            view.update(w=w, h=h, sx=w / meta["width"], sy=h / meta["height"])
            display = np.full((max(h, 650), w + 570, 3), 245, dtype=np.uint8)
            display[:h, :w] = cv2.resize(canvas, (w, h))
            lines = [job.name, f"Image {index + 1}/{len(meta['frames'])} - {meta['frames'][index]['time_s']:.2f}s", ""]
            if review:
                decision = decisions.get(str(index), {})
                status = decision.get("status", "a verifier")
                if status == "approved" and decision.get("conversion_token") != token:
                    status = "ancienne approbation : revalider"
                mode_name = ("SAM brut (avant corrections humaines)", "YOLO export", "Differences YOLO / SAM brut")[preview_mode]
                lines += [f"Vue : {mode_name}", f"Decision : {status}",
                          "A : annotation correcte (NE VEUT PAS DIRE OK)",
                          "B : image sans objet | R : rejeter | U : attente",
                          "V : SAM / YOLO / differences | M : image seule",
                          "N/P : image suivante/precedente | Q : quitter",
                          "J/K : defiler les details ci-dessous",
                          "Magenta : pixels ajoutes | Jaune : retires",
                          "Verifier TOUTES les pieces visibles, pas 3 seules.",
                          "Clip mobile : aucun emplacement impose.",
                          f"Visibilite declaree : {states or 'aucune'}",
                          "Conversion BLOQUEE" if converted["blocked"] else "Conversion possible, validation humaine requise", ""]
                details = [line for issue in converted["messages"] for line in textwrap.wrap(issue, 75)]
                room = max(1, (display.shape[0] - 50) // 22 - len(lines) - 1)
                detail_offset = min(detail_offset, max(0, len(details) - room))
                lines += details[detail_offset:detail_offset + room]
                if len(details) > room:
                    lines.append(f"Details {detail_offset + 1}-{min(len(details), detail_offset + room)}/{len(details)} (J/K)")
            else:
                lines += ["TAB : selectionner un objet existant",
                          "0/1/2 : NOUVEL objet connector/clip/cable",
                          "Clic gauche : inclure | droit : exclure",
                          "B : rectangle | U : annuler le dernier clic",
                          "X : effacer les reperes | D : supprimer l'objet",
                          "H : objet entierement cache / hors champ",
                          "I : incertain -> image exclue de l'entrainement",
                          "G : debut de plage (G encore pour annuler)",
                          "C : retirer la declaration H/I (image ou plage)",
                          "N/P : suivante/precedente | M : image seule",
                          "S : enregistrer | Q : enregistrer/quitter", "",
                          "H/I/C agit sur l'objet selectionne seulement.",
                          "H vide son masque ; I exclut toute l'image.",
                          "Un objet visible mais immobile n'est PAS absent.",
                          "Pour corriger les contours : recalcul SAM.",
                          f"Plage : indice {range_start[1]} -> {index}, puis H/I/C" if range_start
                          else f"Indice {index} : H/I/C agit sur cette seule image.", ""]
                for item in prompts["objects"]:
                    state = states.get(str(item["id"]), "suivi SAM")
                    lines.append(("> " if item is obj else "  ") + f"#{item['id']} {CLASSES[item['class_id']]} : {state}")
                if run:
                    lines.append("Masques precedents : H/I immediat ; clics -> recalcul.")
            if not review:
                lines += ["", message]
            for row, line in enumerate(lines):
                cv2.putText(display, line[:77], (w + 15, 27 + 22 * row), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (35, 35, 35), 1, cv2.LINE_AA)
            if review:
                cv2.putText(display, message[:77], (w + 15, display.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (30, 30, 180), 1, cv2.LINE_AA)
            cv2.imshow(title, display)
            key = cv2.waitKey(30) & 0xFF
            if ord("A") <= key <= ord("Z"):
                key += ord("a") - ord("A")
            if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                break
            if key in (ord("q"), 27):
                break
            if key in (ord("n"), ord("p")):
                cv2.setTrackbarPos("Image", title, max(0, min(len(meta["frames"]) - 1, index + (1 if key == ord("n") else -1))))
            elif key == ord("m"):
                mask_visible = not mask_visible
            elif review:
                if key == ord("v"):
                    preview_mode = (preview_mode + 1) % 3
                elif key in (ord("j"), ord("k")):
                    detail_offset = max(0, detail_offset + (1 if key == ord("j") else -1))
                elif key in (ord("a"), ord("b")):
                    empty = not any(mask.any() for mask in converted["effective_masks"].values())
                    if converted["blocked"] or (empty and key != ord("b")) or (not empty and key == ord("b")):
                        message = "Refuse : corriger le masque, ou B si TOUS sont vides."
                    elif not mask_visible or preview_mode == 0:
                        message = "Afficher la vue YOLO ou differences (V/M) avant approbation."
                    else:
                        decisions[str(index)] = {"status": "approved", "empty_confirmed": empty,
                                                 "conversion_token": token}
                        message = "Annotation approuvee. Aucun statut OK/NOK attribue."
                elif key in (ord("r"), ord("u")):
                    decisions[str(index)] = {"status": "rejected" if key == ord("r") else "pending"}
                if key in (ord("a"), ord("b"), ord("r"), ord("u")):
                    write_json(run / "review.json", decisions)
            elif key in (ord("0"), ord("1"), ord("2")):
                range_start = None
                new_id = max([item["id"] for item in prompts["objects"]], default=0) + 1
                prompts["objects"].append({"id": new_id, "class_id": key - ord("0"), "prompts": {}})
                view["selected"] = len(prompts["objects"]) - 1
            elif key == 9:
                view["selected"] += 1
                range_start = None
            elif key == ord("g") and obj:
                range_start = None if range_start else (obj["id"], index)
                message = "Plage annulee." if range_start is None else "Choisir la fin avec le curseur, puis H/I/C."
            elif key in (ord("h"), ord("i"), ord("c")) and obj:
                start = range_start[1] if range_start and range_start[0] == obj["id"] else index
                state = {ord("h"): "absent", ord("i"): "ignore", ord("c"): "clear"}[key]
                set_visibility(visibility, obj, start, index, state, len(meta["frames"]))
                visibility_dirty, range_start = True, None
                message = f"#{obj['id']} : {state}, indices {min(start, index)}-{max(start, index)}. Q pour sauvegarder."
            elif key == ord("d") and obj:
                if visibility["objects"].pop(str(obj["id"]), None) is not None:
                    visibility_dirty = True
                range_start = None
                prompts["objects"].remove(obj)
                view["selected"] = 0
            elif key == ord("x") and obj:
                obj["prompts"].pop(str(index), None)
            elif key == ord("u") and obj:
                prompt = obj["prompts"].get(str(index), {})
                if prompt.get("points"):
                    prompt["points"].pop()
                    prompt["labels"].pop()
                elif prompt.get("box"):
                    prompt.pop("box")
                if not prompt.get("points") and not prompt.get("box"):
                    obj["prompts"].pop(str(index), None)
            elif key == ord("b") and obj:
                if str(obj["id"]) in states:
                    message = "C : annuler la declaration de visibilite avant de placer un rectangle."
                    continue
                roi_title = "Rectangle : ENTREE pour valider, C pour annuler"
                x, y, rw, rh = cv2.selectROI(roi_title, cv2.resize(frame, (w, h)), fromCenter=False, showCrosshair=True)
                cv2.destroyWindow(roi_title)
                if rw > 1 and rh > 1:
                    prompt = obj["prompts"].setdefault(str(index), {"points": [], "labels": []})
                    prompt["box"] = [x / view["sx"], y / view["sy"], min(meta["width"] - 1, (x + rw) / view["sx"]), min(meta["height"] - 1, (y + rh) / view["sy"])]
            elif key == ord("s"):
                write_json(job / "prompts.json", prompts)
                if visibility_dirty:
                    write_json(job / "visibility.json", visibility)
                    visibility_dirty = False
                message = "Reperes enregistres."
    finally:
        write_json(run / "review.json", decisions) if review else write_json(job / "prompts.json", prompts)
        if not review and visibility_dirty:
            write_json(job / "visibility.json", visibility)
        cv2.destroyAllWindows()


def approved_frames(job):
    job = Path(job).resolve()
    meta, prompts = load_job(job)
    validate_prompts(meta, prompts)
    run, info = active_run(job, meta, prompts)
    decisions = read_json(run / "review.json")
    settings = review_settings(job, prompts["objects"], len(meta["frames"]))
    visibility = load_visibility(job, meta, prompts)
    frames = []
    for index in range(len(meta["frames"])):
        if "ignore" in visibility_at(visibility, index).values():
            continue
        decision = decisions.get(str(index), {})
        if decision.get("status") != "approved":
            continue
        masks = checked_masks(run, info, index)
        converted = convert_frame(masks, prompts["objects"], settings, index, visibility)
        if converted["blocked"]:
            raise ValueError(f"{job.name} image {index} : geometrie non exportable. Relancer review.")
        if not any(mask.any() for mask in converted["effective_masks"].values()) and not decision.get("empty_confirmed"):
            raise ValueError("Image vide non confirmee.")
        token = conversion_token(meta, prompts, info, index, converted, settings)
        if decision.get("conversion_token") != token:
            raise ValueError(f"{job.name} image {index} : approbation ancienne ou conversion modifiee. Relancer review.")
        frames.append((index, converted["polygons"]))
    if not frames:
        raise ValueError(f"Aucune image approuvee dans {job.name}. Lancer review.")
    return meta, prompts, run, frames


def draft_exclusion(converted, decision):
    """Un essai accepte les imperfections, pas la disparition d'un objet visible."""
    if converted["excluded"]:
        return "Image declaree incertaine par l'utilisateur."
    for key, mask in converted["effective_masks"].items():
        polygon = converted["polygons"][key]
        if mask.any() and polygon is None:
            return f"Objet #{key} non vide sans polygone representable."
        if polygon is not None:
            values = normalized_polygon(polygon, mask.shape)
            if (len(values) < 3 or not np.isfinite(values).all()
                    or not np.all((values >= 0) & (values < 1))):
                return f"Objet #{key} : coordonnees YOLO invalides."
    if not any(mask.any() for mask in converted["effective_masks"].values()) and not decision.get("empty_confirmed"):
        return "Image entierement vide non confirmee."
    return None


def draft_frames(job):
    """Selection explicite sans approbation ; conserver les refus et avertissements."""
    meta, prompts = load_job(job)
    validate_prompts(meta, prompts)
    run, info = active_run(job, meta, prompts)
    visibility = load_visibility(job, meta, prompts)
    decisions = read_json(run / "review.json")
    frames, skipped = [], []
    for index in range(len(meta["frames"])):
        masks = checked_masks(run, info, index)
        converted = convert_frame(masks, prompts["objects"], frame_index=index, visibility=visibility)
        reason = draft_exclusion(converted, decisions.get(str(index), {}))
        if reason:
            skipped.append({"source_name": meta["source_name"], "source_sha256": meta["source_sha256"],
                            "frame_index": index, "reason": reason})
            continue
        frames.append((index, converted["polygons"], conversion_token(meta, prompts, info, index, converted, {})))
    if not frames:
        raise ValueError(f"Aucune image exportable dans {job.name}, meme en mode essai : {skipped}")
    return meta, prompts, run, frames, skipped


def export_dataset(train_jobs, val_jobs, output, draft=False):
    output = Path(output).resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("L'export existe deja. Choisir un autre --output.")
    sources, datasets, skipped = {}, [], []
    for split, jobs in (("train", train_jobs), ("val", val_jobs)):
        for job in jobs:
            job = Path(job).resolve()
            if draft:
                meta, prompts, run, frames, excluded = draft_frames(job)
                skipped.extend(dict(record, split=split) for record in excluded)
            else:
                meta, prompts, run, frames = approved_frames(job)
                decisions = read_json(run / "review.json")
                frames = [(index, polygons, decisions[str(index)]["conversion_token"]) for index, polygons in frames]
            source = meta["source_sha256"]
            if source in sources:
                raise ValueError("Meme video reutilisee : train et val doivent provenir de videos distinctes.")
            sources[source] = split
            datasets.append((split, job, meta, prompts, run, frames))
    present = {obj["class_id"] for split, _, _, prompts, _, frames in datasets if split == "train"
               for _, polygons, _ in frames for obj in prompts["objects"] if polygons[str(obj["id"])] is not None}
    if present != set(CLASSES):
        raise ValueError("Le train doit contenir des masques exportables pour les trois classes.")
    manifest = {"classes": CLASSES, "conversion_policy": CONVERSION_POLICY,
                "export_mode": "draft" if draft else "approved", "skipped": skipped,
                "usage": "Essai non valide, y compris images rejetees et conversions imparfaites." if draft else "Annotations approuvees dans review.",
                "mask_warning": "masks_sam conserve les sorties brutes, y compris les hallucinations. masks_visible applique les absences humaines avant raccords YOLO. Les raccords ne prouvent pas un contact physique.",
                "samples": []}
    for split, job, meta, prompts, run, frames in datasets:
        settings = review_settings(job, prompts["objects"], len(meta["frames"]))
        info = read_json(run / "run.json")
        (output / "images" / split).mkdir(parents=True, exist_ok=True)
        (output / "labels" / split).mkdir(parents=True, exist_ok=True)
        (output / "masks_sam" / split).mkdir(parents=True, exist_ok=True)
        (output / "masks_visible" / split).mkdir(parents=True, exist_ok=True)
        for index, polygons, expected_token in frames:
            originals = checked_masks(run, info, index)
            visibility = load_visibility(job, meta, prompts)
            converted = convert_frame(originals, prompts["objects"], settings, index, visibility)
            token = conversion_token(meta, prompts, info, index, converted, settings)
            decision = read_json(run / "review.json").get(str(index), {})
            if token != expected_token:
                raise ValueError(f"{job.name} image {index} : conversion modifiee pendant l'export.")
            if draft:
                reason = draft_exclusion(converted, decision)
                if reason:
                    raise ValueError(f"{job.name} image {index} : {reason}")
            elif converted["blocked"] or decision.get("status") != "approved" or decision.get("conversion_token") != token:
                raise ValueError(f"{job.name} image {index} : validation modifiee pendant l'export. Relancer review.")
            polygons = converted["polygons"]
            name = f"{meta['source_sha256'][:12]}_{index:05d}"
            shutil.copy2(image_path(job, index), output / "images" / split / f"{name}.jpg")
            mask_name = f"masks_sam/{split}/{name}.npz"
            shutil.copy2(run / "masks" / f"{index:05d}.npz", output / mask_name)
            visible_name = f"masks_visible/{split}/{name}.npz"
            np.savez_compressed(output / visible_name, **converted["effective_masks"])
            lines = []
            for obj in prompts["objects"]:
                polygon = polygons[str(obj["id"])]
                if polygon is None:
                    continue
                lines.append(str(obj["class_id"]) + " " + polygon_text(polygon, (meta["height"], meta["width"])))
            (output / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            manifest["samples"].append({"image": f"images/{split}/{name}.jpg", "source_name": meta["source_name"],
                                        "source_sha256": meta["source_sha256"], "source_index": meta["frames"][index]["source_index"],
                                        "run": run.name, "reviewed": not draft,
                                        "review_decision": decision.get("status", "pending"),
                                        "quality_blocked": converted["blocked"], "conversion_messages": converted["messages"],
                                        "original_masks": mask_name, "mask_sha256": info["mask_hashes"][str(index)],
                                        "visible_masks": visible_name, "visible_mask_sha256": file_hash(output / visible_name),
                                        "visibility": converted["visibility"],
                                        "objects": [{"id": obj["id"], "class_id": obj["class_id"]} for obj in prompts["objects"]],
                                        "review_settings": settings, "conversion": converted["statistics"],
                                        "conversion_token": token})
    # No 'path: .' : Ultralytics should use the location of this YAML file.
    (output / "data.yaml").write_text("train: images/train\nval: images/val\nnames:\n  0: connector\n  1: clip\n  2: cable\n", encoding="utf-8")
    write_json(output / "manifest.json", manifest)
    if draft:
        (output / "LIRE_AVANT_ENTRAINEMENT.txt").write_text(
            "DATASET D'ESSAI NON VALIDE\n\n"
            "Masques SAM actuels convertis sans correction manuelle supplementaire.\n"
            "Les candidats imparfaits et les images precedemment rejetees sont inclus.\n"
            "Les decisions review et les masques sources n'ont pas ete modifies.\n"
            "Consulter manifest.json : decisions, ecarts de conversion, exclusions.\n"
            "Classes : 0 connector, 1 clip, 2 cable. Modele : yolov8n-seg.pt.\n"
            "L'annotation peut etre partielle ou viser le mauvais objet.\n"
            "Ce dataset teste le pipeline ; il ne demontre pas sa fiabilite.\n"
            "Avec test1 seul en train et test2 seul en val, aucun score ne prouve\n"
            "la distinction OK/NOK. Ajouter des videos variees pour un vrai modele.\n\n"
            "Regle metier demandee : NOK uniquement si le clip reste attache\n"
            "au connecteur. Leur simple presence simultanee ne suffit pas.\n"
            "L'export ne calcule pas cette relation et ne modifie pas Node-RED.\n"
            "Les raccords de format YOLO ne sont pas une preuve de contact.\n"
            "Ce ZIP est un dataset, pas le modele best.pt : entrainer avant usage.\n",
            encoding="utf-8")
    with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED) as handle:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                handle.write(path, f"{output.name}/{path.relative_to(output).as_posix()}")
    label = "images d'essai NON VALIDEES" if draft else "images approuvees"
    print(f"Export : {archive} ({len(manifest['samples'])} {label}, {len(skipped)} exclues)")
    return archive


def pack_jobs(jobs, output):
    output = Path(output).resolve()
    checked = []
    for job in jobs:
        job = Path(job).resolve()
        meta, prompts = load_job(job)
        validate_prompts(meta, prompts)
        review_settings(job, prompts["objects"], len(meta["frames"]))
        load_visibility(job, meta, prompts)
        checked.append((job, meta))
    if len({job.name for job, _ in checked}) != len(checked):
        raise ValueError("Les sequences doivent avoir des noms distincts.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", zipfile.ZIP_DEFLATED) as handle:
        handle.write(Path(__file__).resolve(), "scripts/annotate_sam2.py")
        handle.write(ROOT / "requirements-sam2.txt", "requirements-sam2.txt")
        for job, meta in checked:
            paths = [job / "sequence.json", job / "prompts.json"]
            if (job / "visibility.json").exists():
                paths.append(job / "visibility.json")
            for path in paths + [image_path(job, i) for i in range(len(meta["frames"]))]:
                handle.write(path, f"jobs/{job.name}/{path.relative_to(job).as_posix()}")
    print(f"Archive pour Colab : {output}")


def restore_results(archive, output):
    """Importer un retour Colab dans un dossier neuf, sans ecraser les reperes locaux."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Le dossier de retour existe deja. Choisir un autre --output.")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".sam2_import_", dir=output.parent) as temporary:
        staging = Path(temporary)
        with zipfile.ZipFile(archive) as handle:
            members = handle.infolist()
            if len(members) > 20000 or sum(item.file_size for item in members) > 2_000_000_000:
                raise ValueError("Archive trop volumineuse pour cet outil d'essai.")
            seen = set()
            for item in members:
                parts = Path(item.filename).parts
                if ("\\" in item.filename or ":" in item.filename or ".." in parts
                        or not parts or parts[0] != "jobs" or len(parts) < 2
                        or stat.S_ISLNK(item.external_attr >> 16)):
                    raise ValueError("Chemin interdit dans l'archive.")
                target = staging.joinpath(*parts).resolve()
                if not target.is_relative_to(staging / "jobs") or target in seen:
                    raise ValueError("Chemin duplique ou hors archive.")
                seen.add(target)
            handle.extractall(staging)
        jobs_dir = staging / "jobs"
        jobs = sorted(jobs_dir.iterdir()) if jobs_dir.exists() else []
        if not jobs or any(not job.is_dir() for job in jobs):
            raise ValueError("Aucune sequence valide dans l'archive.")
        for job in jobs:
            meta, prompts = load_job(job)
            validate_prompts(meta, prompts)
            settings = review_settings(job, prompts["objects"], len(meta["frames"]))
            run, info = active_run(job, meta, prompts)
            # Older notebook download cells may omit the sidecar. The run
            # snapshot preserves declarations supplied to the GPU calculation.
            if not (job / "visibility.json").exists() and "visibility" in info:
                write_json(job / "visibility.json", info["visibility"])
            visibility = load_visibility(job, meta, prompts)
            for index in range(len(meta["frames"])):
                mask_report(checked_masks(run, info, index), prompts["objects"], settings, index, visibility)
            # Les apercus Colab ne constituent pas une validation humaine.
            write_json(run / "review.json", {})
        jobs_dir.rename(output)
    print(f"Retour Colab importe : {output}\nEtape suivante : review pour chaque sequence.")
    return output


def main():
    parser = argparse.ArgumentParser(description="Annotation assistee SAM 2.1 : connector=0, clip=1, cable=2.")
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("prepare", help="Extraire les images dans de nouvelles sequences.")
    command.add_argument("videos", nargs="+", type=Path)
    command.add_argument("--output", type=Path, default=ROOT / "data/annotations_sam2")
    command.add_argument("--sample-fps", type=float, default=6.0)
    command.add_argument("--rotation", type=int, choices=[0, 90, 180, 270], default=0)
    command.add_argument("--max-frames", type=int, default=600)
    for name in ("points", "review"):
        command = commands.add_parser(name, help="Placer les reperes." if name == "points" else "Valider les masques calcules.")
        command.add_argument("job", type=Path)
        command.add_argument("--max-height", type=int, default=780)
    command = commands.add_parser("propagate", help="Calculer les masques (environnement SAM requis).")
    command.add_argument("job", type=Path)
    command.add_argument("--checkpoint", required=True, type=Path)
    command.add_argument("--sam-config", default="configs/sam2.1/sam2.1_hiera_t.yaml")
    command.add_argument("--device", choices=["auto", "cuda", "cpu", "mps"], default="auto")
    command = commands.add_parser("pack", help="Creer le ZIP des reperes pour Colab.")
    command.add_argument("jobs", type=Path, nargs="+")
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("restore", help="Importer le ZIP de masques retourne par Colab.")
    command.add_argument("archive", type=Path)
    command.add_argument("--output", required=True, type=Path)
    command = commands.add_parser("export", help="Exporter les images approuvees, ou un essai explicite avec --draft.")
    command.add_argument("--train", type=Path, nargs="+", required=True)
    command.add_argument("--val", type=Path, nargs="+", required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--draft", action="store_true", help="Essai NON VALIDE : inclure aussi les images rejetees et les candidats imparfaits. Ne modifie pas review.")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            for video in args.videos:
                prepare(video, args.output, args.sample_fps, args.rotation, args.max_frames)
        elif args.command in ("points", "review"):
            if args.max_height < 200:
                raise ValueError("--max-height doit etre >= 200.")
            viewer(args.job, args.command == "review", args.max_height)
        elif args.command == "propagate":
            propagate(args.job, args.checkpoint, args.sam_config, args.device)
        elif args.command == "pack":
            pack_jobs(args.jobs, args.output)
        elif args.command == "restore":
            restore_results(args.archive, args.output)
        else:
            export_dataset(args.train, args.val, args.output, draft=args.draft)
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError, OSError) as exc:
        parser.exit(1, f"Erreur : {exc}\n")


if __name__ == "__main__":
    main()
