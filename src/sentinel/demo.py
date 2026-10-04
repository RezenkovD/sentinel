"""Demo video for the defence: the analysed clip with the incident state, the risk and the Telegram card.

    uv run sentinel demo --source results/sanity_videos/lifecycle_long_gaps.mp4 --out results/demo/fight.mp4
    uv run sentinel demo --sevilla Cam5-From09-47-49To09-48-30_x264 --out results/demo/weapon.mp4 --label Хол

The clip runs through the real pipeline (the configured mode, real notifications when the bot is set up);
every analysed frame is written with blurred faces, the incident's participants in green and bystanders faint,
weapons in red, a banner with the automaton's state and the window risk, and on the right the text of the last
Telegram message. The output holds people from research datasets: it is not committed and not published.
"""

import argparse
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PANEL_W = 420
HEIGHT = 480
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
STATES = {  # banner colour (BGR) per automaton state
    "норма": (60, 140, 60),
    "увага": (0, 160, 230),
    "тривога": (40, 40, 210),
    "затихання": (0, 120, 220),
}


def _font(size: int):
    try:
        return ImageFont.truetype(FONT, size)
    except OSError:
        return ImageFont.load_default()


def _wrap(text: str, width: int) -> list[str]:
    import textwrap

    lines = []
    for line in text.split("\n"):
        lines += textwrap.wrap(line, width) or [""]
    return lines


def compose(
    frame: np.ndarray,
    state: str,
    risk: float | None,
    decided_by: str,
    t: float,
    card: str | None,
    origin: float | None = None,
) -> np.ndarray:
    """The frame scaled to HEIGHT with a state banner, and a PANEL_W panel with the last message."""
    h, w = frame.shape[:2]
    video = cv2.resize(frame, (round(w * HEIGHT / h), HEIGHT))
    banner = 40
    cv2.rectangle(video, (0, 0), (video.shape[1], banner), STATES.get(state, (90, 90, 90)), -1)
    panel = np.full((HEIGHT, PANEL_W, 3), (54, 43, 33), dtype=np.uint8)  # Telegram dark blue (BGR)
    image = Image.fromarray(cv2.cvtColor(np.hstack([video, panel]), cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    if origin is not None:  # the same clock as the messages
        clock = time.strftime("%H:%M:%S", time.localtime(origin + t))
    else:
        minutes, seconds = divmod(int(t), 60)
        clock = f"{minutes:02d}:{seconds:02d}"
    risk_text = "—" if risk is None else f"{risk:.2f}".replace(".", ",")
    draw.text(
        (12, 9),
        f"{state.upper()}  ·  ризик {risk_text}  ·  рішення {decided_by}  ·  {clock}",
        font=_font(19),
        fill=(255, 255, 255),
    )
    if risk is not None:  # a thin risk bar under the banner
        draw.rectangle((0, banner, int(video.shape[1] * risk), banner + 5), fill=(255, 255, 255))
    x0 = video.shape[1] + 16
    draw.text((x0, 14), "Telegram · Sentinel", font=_font(17), fill=(150, 170, 190))
    if card:
        y = 52
        for n, line in enumerate(
            _wrap(card.replace("🔴 ", "● ").replace("✅ ", "✔ ").replace("⚠️ ", "▲ "), 34)
        ):
            colour = (
                (235, 95, 95)
                if n == 0 and line[:1] in "●▲"
                else (120, 210, 130)
                if n == 0
                else (230, 235, 240)
            )
            draw.text((x0, y), line, font=_font(18), fill=colour)
            y += 26
    else:
        draw.text((x0, 52), "повідомлень немає", font=_font(17), fill=(120, 135, 150))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def sevilla_clip(segment: str, images: Path = Path("data/raw/us-gun/Images")):
    """Frames of a Sevilla segment (stills at 2 frames/s) as a stream."""
    from sentinel.eval.extract import sevilla_frames
    from sentinel.io.source import Frame

    for i, (t, path, _) in enumerate(sevilla_frames(images, segment)):
        yield Frame(image=cv2.imread(str(path)), t=t, index=i)


def main(argv: list[str] | None = None) -> None:
    from dotenv import find_dotenv, load_dotenv

    from sentinel.config import DEFAULT_CONFIG, load_config
    from sentinel.eval.extract import dataset_config
    from sentinel.io.source import LatestFrameReader
    from sentinel.keyframe import render_keyframe
    from sentinel.notify import caption, make_notifier
    from sentinel.pipeline import build_pipeline

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="video file")
    source.add_argument("--sevilla", help="Sevilla segment name (stills at 2 frames/s)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", help="camera name in the messages (default: source.label)")
    parser.add_argument("--no-telegram", action="store_true", help="console only, even with the bot set up")
    args = parser.parse_args(argv)
    load_dotenv(find_dotenv(usecwd=True))
    config = load_config(args.config)
    if args.sevilla:
        config = dataset_config(config, "us_gun")
    label = args.label or config.source.label
    env = {} if args.no_telegram else os.environ
    origin = time.time()
    notifier = make_notifier(env, config.notify.incidents_jsonl, config.notify.edit_interval_s, label, origin)
    run_id = time.strftime("%Y%m%d-%H%M%S")
    pipeline = build_pipeline(
        config, config.source.camera_id, run_id, Path(f"results/demo_{run_id}.jsonl"), notifier
    )
    if args.sevilla:
        frames = sevilla_clip(args.sevilla)
    else:
        frames = iter(LatestFrameReader(args.source, "offline", target_fps=config.analysis.fps))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer, card, incidents = None, None, 0
    try:
        for frame in frames:
            events = pipeline.process(frame)
            for event in events:
                card = caption(event, label, origin)
                incidents += event.kind == "open"
            found = pipeline.last_found
            incident = pipeline.incidents.current
            image = render_keyframe(
                frame.image,
                found.tracks,
                found.skeletons,
                found.weapons,
                blur_faces=True,
                others=found.others,
                highlight=incident.participants if incident is not None else set(),  # calm: all faint
            )
            risk, decision = pipeline.last_risk, pipeline.last_decision
            out = compose(
                image,
                pipeline.incidents.state.value,
                None if risk is None else risk.risk,
                decision.get("source", "A"),
                frame.t,
                card,
                origin,
            )
            if writer is None:
                fps = config.analysis.fps
                writer = cv2.VideoWriter(
                    str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), fps, out.shape[1::-1]
                )
            writer.write(out)
        for event in pipeline.finish():
            card = caption(event, label, origin)
        if writer is not None and card:  # hold the final message for two seconds
            for _ in range(int(2 * config.analysis.fps)):
                writer.write(compose(image, "норма", None, "A", frame.t, card, origin))
    finally:
        if writer is not None:
            writer.release()
        notifier.close()
    print(f"{args.out}: {incidents} incident(s)")


if __name__ == "__main__":
    main()
