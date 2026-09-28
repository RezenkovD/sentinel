"""Command-line interface: `sentinel run | bench | eval`."""

import argparse
from datetime import date
from pathlib import Path

from sentinel.config import DEFAULT_CONFIG, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the pipeline on a camera, RTSP stream or video file")
    sub.add_parser("eval", help="run the experiment on dataset episodes")

    bench = sub.add_parser("bench", help="measure processing speed (ms/frame, RTF)")
    bench.add_argument("--source", type=Path, nargs="+", required=True, help="video file(s)")
    bench.add_argument(
        "--weights", type=Path, nargs=2, metavar=("DETECT", "POSE"), help="default: from config"
    )
    bench.add_argument("--format", choices=["torch", "openvino"], default="torch")
    bench.add_argument("--device", help="cpu for torch; intel:cpu or intel:gpu for openvino")
    bench.add_argument("--imgsz", type=int, nargs="+", help="default: bench.imgsz from config")
    bench.add_argument("--fps", type=float, nargs="+", help="analysis rates; default: bench.fps from config")
    bench.add_argument(
        "--threads", type=int, help="torch intra-op threads; default: models.threads from config"
    )
    bench.add_argument("--warmup", type=int, help="default: bench.warmup_frames from config")
    bench.add_argument("--out", type=Path, help="markdown report; default: results/bench_<date>.md")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "bench":
        from sentinel.bench import run_bench

        det, pose = args.weights or (config.models.person.weights, config.models.pose.weights)
        out = run_bench(
            sources=args.source,
            det_weights=det,
            pose_weights=pose,
            backend=args.format,
            device=args.device or ("intel:cpu" if args.format == "openvino" else config.models.device),
            imgsz_list=args.imgsz or config.bench.imgsz,
            fps_list=args.fps or config.bench.fps,
            warmup=config.bench.warmup_frames if args.warmup is None else args.warmup,
            threads=args.threads or config.models.threads,
            person_conf=config.models.person.conf,
            tracker_cfg=config.models.person.tracker,
            pose_conf=config.models.pose.conf,
            min_iou=config.models.pose.min_iou,
            out=args.out or Path("results") / f"bench_{date.today().isoformat()}.md",
        )
        print(f"report: {out}")
        return 0
    print(f"`sentinel {args.command}` is not implemented yet")
    return 2
