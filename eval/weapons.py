"""Choosing the weapon detector: candidates on the Sevilla frames of Cam1 + Cam5 (decision 29.09).

Run from the repository root:
    uv run python -m eval.weapons [--imgsz 480 640]

For every candidate and input size the frames are predicted once with a low confidence (predictions cached in
results/weapons/*.jsonl), then:
  (a) AP50 per kind (`gun`, `knife`; VOC all-point interpolation, IoU >= 0.5);
  (b) per confidence threshold: recall of frames with a weapon (any gun/knife box at or above the
      threshold) and the share of weapon-free frames with such a box ("false frames");
  (c) ms per frame of the model call (letterbox + inference + NMS), mean / p95.
Selection rule: the highest frame recall with false frames <= 2 %; if close, the faster model.
"""

import argparse
import json
import os
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from sentinel.agents.pose_features import iou_matrix

GT_CLASS_MAP = {"handgun": "gun", "short_rifle": "gun", "knife": "knife"}
CANDIDATE_CLASS_MAP = {"pistol": "gun", "rifle": "gun", "gun": "gun", "knife": "knife"}
CANDIDATES = {
    "hf-cosgun99-gun-knife-yolo11n": Path("weights/candidates/hf-cosgun99-gun-knife-yolo11n/best.pt"),
    "gh-halok600-weaponshield-yolov8s": Path("weights/candidates/gh-halok600-weaponshield-yolov8s/best.pt"),
}
THRESHOLDS = [round(0.25 + 0.05 * i, 2) for i in range(10)]  # 0.25 … 0.70
MAX_FALSE_FRAMES = 0.02
PRED_CONF = 0.01  # low, so that AP sees the whole precision/recall curve
PERSON_WEIGHTS = Path("weights/yolo26n.pt")  # the pipeline's person detector (decision 29.09)
PERSON_IMGSZ = 480
CROP_PAD = 0.25


@dataclass(frozen=True)
class Detection:
    frame: str
    kind: str
    conf: float
    box: tuple[float, float, float, float]


def read_voc(xml_path: Path) -> list[tuple[str, tuple[float, float, float, float]]]:
    """Weapon boxes of a Pascal VOC file, mapped to gun / knife; other classes are dropped."""
    objects = []
    for obj in ET.parse(xml_path).getroot().iter("object"):
        kind = GT_CLASS_MAP.get((obj.findtext("name") or "").strip().lower())
        if kind is None:
            continue
        b = obj.find("bndbox")
        objects.append((kind, tuple(float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))))
    return objects


def average_precision(
    detections: Iterable[Detection], truth: dict[str, list], kind: str, iou_threshold: float = 0.5
) -> float:
    """VOC-style AP (all-point interpolation) for one kind; each true box can be matched once."""
    true_boxes = {f: np.array([b for k, b in objs if k == kind]).reshape(-1, 4) for f, objs in truth.items()}
    n_true = sum(len(b) for b in true_boxes.values())
    if n_true == 0:
        return float("nan")
    used = {f: np.zeros(len(b), dtype=bool) for f, b in true_boxes.items()}
    hits = []
    for d in sorted((d for d in detections if d.kind == kind), key=lambda d: -d.conf):
        boxes = true_boxes.get(d.frame, np.zeros((0, 4)))
        hit = False
        if len(boxes):
            ious = iou_matrix(np.array([d.box]), boxes)[0]
            ious[used[d.frame]] = -1.0
            j = int(np.argmax(ious))
            if ious[j] >= iou_threshold:
                used[d.frame][j] = True
                hit = True
        hits.append(hit)
    if not hits:
        return 0.0
    tp = np.cumsum(hits)
    fp = np.cumsum(~np.array(hits))
    recall = np.concatenate([[0.0], tp / n_true, [1.0]])
    precision = np.concatenate([[0.0], tp / (tp + fp), [0.0]])
    for i in range(len(precision) - 2, -1, -1):  # precision envelope, right to left
        precision[i] = max(precision[i], precision[i + 1])
    steps = np.where(recall[1:] != recall[:-1])[0]
    return float(np.sum((recall[steps + 1] - recall[steps]) * precision[steps + 1]))


def frame_rates(
    detections: Iterable[Detection], truth: dict[str, list], thresholds: Sequence[float]
) -> list[tuple[float, float, float]]:
    """(threshold, recall of weapon frames, share of weapon-free frames with a detection)."""
    best: dict[str, float] = {}
    for d in detections:
        best[d.frame] = max(best.get(d.frame, 0.0), d.conf)
    armed = [f for f, objs in truth.items() if objs]
    empty = [f for f, objs in truth.items() if not objs]
    rates = []
    for t in thresholds:
        recall = sum(best.get(f, 0.0) >= t for f in armed) / len(armed)
        false = sum(best.get(f, 0.0) >= t for f in empty) / len(empty)
        rates.append((t, recall, false))
    return rates


def person_crop(
    box: tuple[float, float, float, float], frame_shape: tuple[int, int], pad: float
) -> tuple[int, int, int, int]:
    """Square around a person box, side = longer box side x (1 + pad), shifted (not cut) to stay inside."""
    h_img, w_img = frame_shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = min(int(round(max(box[2] - box[0], box[3] - box[1]) * (1 + pad))), w_img, h_img)
    x1 = min(max(int(round(cx - side / 2)), 0), w_img - side)
    y1 = min(max(int(round(cy - side / 2)), 0), h_img - side)
    return x1, y1, x1 + side, y1 + side


def merge_detections(detections: Iterable[Detection], iou_threshold: float) -> list[Detection]:
    """Greedy NMS per frame and kind: a weapon seen in two overlapping crops is kept once (highest score)."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: -d.conf):
        same = [k.box for k in kept if k.frame == d.frame and k.kind == d.kind]
        if not same or iou_matrix(np.array([d.box]), np.array(same))[0].max() < iou_threshold:
            kept.append(d)
    return kept


def pick_threshold(rates: list[tuple[float, float, float]]) -> tuple[float, float, float] | None:
    """Highest recall with false frames within the limit (the lowest such threshold)."""
    allowed = [r for r in rates if r[2] <= MAX_FALSE_FRAMES]
    return max(allowed, key=lambda r: (r[1], -r[0])) if allowed else None


def predict_frames(
    weights: Path, frames: list[Path], imgsz_list: Sequence[int], out_dir: Path, name: str
) -> None:
    """Predict every frame once per input size; detections and per-frame model time go to JSONL files."""
    import cv2
    from ultralytics import YOLO

    from sentinel.agents.weapons import map_class

    model = YOLO(str(weights), task="detect")
    warm = cv2.imread(str(frames[0]))
    for imgsz in imgsz_list:  # warm-up, not measured
        for _ in range(3):
            model.predict(warm, imgsz=imgsz, conf=PRED_CONF, verbose=False)
    files = {imgsz: (out_dir / f"{name}_{imgsz}.jsonl").open("w", encoding="utf-8") for imgsz in imgsz_list}
    try:
        for path in frames:
            image = cv2.imread(str(path))
            for imgsz in imgsz_list:
                started = time.perf_counter()
                result = model.predict(image, imgsz=imgsz, conf=PRED_CONF, verbose=False)[0]
                ms = (time.perf_counter() - started) * 1000
                dets = []
                for xyxy, score, cls_id in zip(
                    result.boxes.xyxy.numpy(),
                    result.boxes.conf.numpy(),
                    result.boxes.cls.numpy(),
                    strict=True,
                ):
                    kind = map_class(result.names[int(cls_id)], CANDIDATE_CLASS_MAP)
                    if kind is not None:
                        dets.append({"kind": kind, "conf": float(score), "box": [float(v) for v in xyxy]})
                files[imgsz].write(json.dumps({"frame": path.stem, "ms": ms, "detections": dets}) + "\n")
    finally:
        for f in files.values():
            f.close()


def detect_people(frames: list[Path], out_path: Path) -> dict[str, list[list[float]]]:
    """Person boxes per frame from the pipeline's detector (cached, the pipeline pays for it anyway)."""
    if out_path.exists():
        return json.loads(out_path.read_text())
    import cv2
    from ultralytics import YOLO

    model = YOLO(str(PERSON_WEIGHTS), task="detect")
    people = {}
    for path in frames:
        result = model.predict(
            cv2.imread(str(path)), imgsz=PERSON_IMGSZ, classes=[0], conf=0.25, verbose=False
        )[0]
        people[path.stem] = result.boxes.xyxy.numpy().tolist()
    out_path.write_text(json.dumps(people))
    return people


def predict_crops(
    weights: Path, frames: list[Path], people: dict[str, list], crop_imgsz: int, out_path: Path
) -> None:
    """Weapon model on a padded square around every person; boxes mapped back to the frame and merged."""
    import cv2
    from ultralytics import YOLO

    from sentinel.agents.weapons import map_class

    model = YOLO(str(weights), task="detect")
    warm = cv2.imread(str(frames[0]))[:320, :320]
    for _ in range(3):
        model.predict(warm, imgsz=crop_imgsz, conf=PRED_CONF, verbose=False)
    with out_path.open("w", encoding="utf-8") as out:
        for path in frames:
            image = cv2.imread(str(path))
            ms, found = 0.0, []
            for box in people.get(path.stem, []):
                x1, y1, x2, y2 = person_crop(tuple(box), image.shape, CROP_PAD)
                started = time.perf_counter()
                result = model.predict(image[y1:y2, x1:x2], imgsz=crop_imgsz, conf=PRED_CONF, verbose=False)[
                    0
                ]
                ms += (time.perf_counter() - started) * 1000
                for xyxy, score, cls_id in zip(
                    result.boxes.xyxy.numpy(),
                    result.boxes.conf.numpy(),
                    result.boxes.cls.numpy(),
                    strict=True,
                ):
                    kind = map_class(result.names[int(cls_id)], CANDIDATE_CLASS_MAP)
                    if kind is not None:
                        b = (xyxy[0] + x1, xyxy[1] + y1, xyxy[2] + x1, xyxy[3] + y1)
                        found.append(Detection(path.stem, kind, float(score), tuple(float(v) for v in b)))
            dets = [
                {"kind": d.kind, "conf": d.conf, "box": list(d.box)} for d in merge_detections(found, 0.5)
            ]
            row = {"frame": path.stem, "ms": ms, "crops": len(people.get(path.stem, [])), "detections": dets}
            out.write(json.dumps(row) + "\n")


def load_predictions(path: Path) -> tuple[list[Detection], list[float]]:
    detections, times = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        times.append(row["ms"])
        detections += [
            Detection(row["frame"], d["kind"], d["conf"], tuple(d["box"])) for d in row["detections"]
        ]
    return detections, times


def evaluate(detections: list[Detection], times: list[float], truth: dict[str, list]) -> dict:
    rates = frame_rates(detections, truth, THRESHOLDS)
    return {
        "ap50": {k: average_precision(detections, truth, k) for k in ("gun", "knife")},
        "rates": rates,
        "pick": pick_threshold(rates),
        "ms_mean": float(np.mean(times)),
        "ms_p95": float(np.percentile(times, 95)),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", type=Path, default=Path("data/raw/us-gun/Images"))
    parser.add_argument("--split", type=Path, default=Path("data/splits/us_gun.json"))
    parser.add_argument("--imgsz", type=int, nargs="+", default=[480, 640])
    parser.add_argument("--out-dir", type=Path, default=Path("results/weapons"))
    parser.add_argument("--crop-imgsz", type=int, nargs="*", default=[], help="also run on person crops")
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cameras = json.loads(args.split.read_text())["parts"]["select"]
    frames = sorted(p for p in args.images.glob("*.jpg") if p.name.split("-", 1)[0] in cameras)
    truth = {p.stem: read_voc(p.with_suffix(".xml")) for p in frames}

    from sentinel.bench import read_power_state

    summary = {
        "frames": len(frames),
        "armed_frames": sum(bool(v) for v in truth.values()),
        "boxes": {
            k: sum(1 for objs in truth.values() for kind, _ in objs if kind == k) for k in ("gun", "knife")
        },
        "power": asdict(read_power_state()),
        "load_before": os.getloadavg()[0],
        "results": {},
    }
    for name, weights in CANDIDATES.items():
        cached = all((args.out_dir / f"{name}_{s}.jsonl").exists() for s in args.imgsz)
        if not cached:
            predict_frames(weights, frames, args.imgsz, args.out_dir, name)
        for imgsz in args.imgsz:
            dets, times = load_predictions(args.out_dir / f"{name}_{imgsz}.jsonl")
            summary["results"][f"{name}@{imgsz}"] = evaluate(dets, times, truth)
            print(name, imgsz, json.dumps(summary["results"][f"{name}@{imgsz}"]["pick"]), flush=True)
        if args.crop_imgsz:
            people = detect_people(frames, args.out_dir / f"people_{PERSON_IMGSZ}.json")
            summary["people_per_frame"] = float(np.mean([len(v) for v in people.values()]))
        for crop_imgsz in args.crop_imgsz:
            path = args.out_dir / f"{name}_crops{crop_imgsz}.jsonl"
            if not path.exists():
                predict_crops(weights, frames, people, crop_imgsz, path)
            dets, times = load_predictions(path)
            summary["results"][f"{name}@crops{crop_imgsz}"] = evaluate(dets, times, truth)
            print(
                name,
                f"crops{crop_imgsz}",
                json.dumps(summary["results"][f"{name}@crops{crop_imgsz}"]["pick"]),
            )
    summary["load_after"] = os.getloadavg()[0]
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
