"""Key frame of an incident: the frame with person and weapon boxes drawn and faces blurred (SPEC §7).

The face area comes from the face keypoints (nose, eyes, ears); without them, the top fifth of the person box
is blurred instead. Frames with people are research data: key frames stay under results/ (outside git).
"""

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from sentinel.agents.detector import Track
from sentinel.agents.weapons import WeaponBox

FACE_KEYPOINTS = (0, 1, 2, 3, 4)  # COCO: nose, eyes, ears
FACE_POINT_CONF = 0.3
PERSON_COLOUR = (0, 200, 0)  # BGR
WEAPON_COLOUR = (0, 0, 255)
BYSTANDER_COLOUR = (128, 128, 128)  # people in view who are not part of the incident


@dataclass(frozen=True, eq=False)
class Snapshot:
    """What is needed to draw the key frame of one analysed frame."""

    image: np.ndarray
    tracks: list[Track]
    skeletons: dict[int, np.ndarray]
    weapons: list[WeaponBox]
    # people the pose model saw without a confirmed track (box, keypoints): their faces are blurred too
    others: list[tuple[tuple[float, float, float, float], np.ndarray | None]] = field(default_factory=list)


def _face_region(
    box: tuple[float, float, float, float], skeleton: np.ndarray | None, shape: tuple[int, ...]
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    box_h = y2 - y1
    points = (
        [skeleton[i, :2] for i in FACE_KEYPOINTS if skeleton[i, 2] >= FACE_POINT_CONF]
        if skeleton is not None
        else []
    )
    if points:
        pts = np.array(points)
        cx, cy = pts.mean(axis=0)
        half = max(0.12 * box_h, float(np.ptp(pts, axis=0).max()))
        region = (cx - half, cy - half, cx + half, cy + half)
    else:
        region = (x1, y1, x2, y1 + 0.2 * box_h)
    h, w = shape[:2]
    return (
        int(max(0, region[0])),
        int(max(0, region[1])),
        int(min(w, region[2])),
        int(min(h, region[3])),
    )


def render_keyframe(
    image: np.ndarray,
    tracks: list[Track],
    skeletons: dict[int, np.ndarray],
    weapons: list[WeaponBox],
    blur_faces: bool,
    others: list[tuple[tuple[float, float, float, float], np.ndarray | None]] = (),
    highlight: set[int] | frozenset[int] | None = None,  # the incident's participants; None: everyone alike
) -> np.ndarray:
    out = image.copy()
    if blur_faces:
        people = [(t.box, skeletons.get(t.track_id)) for t in tracks] + list(others)
        for box, skeleton in people:
            x1, y1, x2, y2 = _face_region(box, skeleton, out.shape)
            if x2 - x1 > 1 and y2 - y1 > 1:
                size = max(x2 - x1, y2 - y1) // 2 * 2 + 1
                out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (size, size), 0)
    for track in sorted(tracks, key=lambda t: highlight is None or t.track_id in highlight):
        x1, y1, x2, y2 = (int(round(v)) for v in track.box)
        if highlight is not None and track.track_id not in highlight:
            cv2.rectangle(out, (x1, y1), (x2, y2), BYSTANDER_COLOUR, 1)  # faint, unlabelled
            continue
        cv2.rectangle(out, (x1, y1), (x2, y2), PERSON_COLOUR, 2)
        cv2.putText(
            out, f"#{track.track_id}", (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, PERSON_COLOUR, 1
        )
    for weapon in weapons:
        x1, y1, x2, y2 = (int(round(v)) for v in weapon.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), WEAPON_COLOUR, 2)
        label = f"{weapon.kind} {weapon.conf:.2f}"
        cv2.putText(out, label, (x1, max(12, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, WEAPON_COLOUR, 1)
    return out


class KeyframeWriter:
    """Renders a snapshot and writes <output_dir>/<incident_id>/<name>.jpg (the incident machine's saver)."""

    def __init__(self, output_dir: Path, blur_faces: bool) -> None:
        self.output_dir = Path(output_dir)
        self.blur_faces = blur_faces

    def __call__(
        self, incident_id: str, name: str, snapshot: Snapshot | None, highlight: set[int] | None = None
    ) -> Path | None:
        if snapshot is None:
            return None
        image = render_keyframe(
            snapshot.image,
            snapshot.tracks,
            snapshot.skeletons,
            snapshot.weapons,
            self.blur_faces,
            snapshot.others,
            highlight,
        )
        path = self.output_dir / incident_id / f"{name}.jpg"
        path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return path
