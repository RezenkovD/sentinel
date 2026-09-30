"""Command-line interface: `sentinel run | bench | eval`."""

import argparse
import importlib
from datetime import date
from pathlib import Path

from sentinel.config import DEFAULT_CONFIG, load_config

# `sentinel eval <name> [args]` -> sentinel.eval.<module>.main(args); True = the script takes --config
EVAL_COMMANDS = {
    "build-episodes": ("build_episodes", False),
    "weapons": ("weapons", False),
    "sanity": ("sanity", True),
    "people": ("people", True),
}


def eval_command(name: str, rest: list[str], config: Path) -> tuple[str, list[str]]:
    """Module and arguments of an eval script; the top-level --config goes to scripts that take one."""
    if name not in EVAL_COMMANDS:
        raise SystemExit(f"unknown eval command {name!r}; choose from: {', '.join(EVAL_COMMANDS)}")
    module, takes_config = EVAL_COMMANDS[name]
    argv = list(rest)
    if takes_config and "--config" not in argv:
        argv = ["--config", str(config), *argv]
    return f"sentinel.eval.{module}", argv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="YAML config (default: %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the pipeline on a camera, RTSP stream or video file")
    run.add_argument("--source", required=True, help="video file, camera index (e.g. 0) or rtsp:// URL")
    run.add_argument("--paced", action="store_true", help="play a file at its own speed, like a live camera")
    ev = sub.add_parser("eval", help="dataset scripts: splits, episodes, training, experiment")
    ev.add_argument("name", help=", ".join(EVAL_COMMANDS))
    ev.add_argument("rest", nargs=argparse.REMAINDER, help="arguments of the eval script")

    sub.add_parser("export", help="export the config's models to OpenVINO (skipped while up to date)")

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
    bench.add_argument("--paced", action="store_true", help="real-time run: clips play at their own speed")
    bench.add_argument(
        "--pipeline",
        nargs="+",
        metavar="WEAPONS",
        help="full pipeline for weapon-detector variants, e.g. 480:1 640:1 640:2 crops320:2",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "eval":
        module, script_argv = eval_command(args.name, args.rest, args.config)
        result = importlib.import_module(module).main(script_argv)
        return result if isinstance(result, int) else 0
    config = load_config(args.config)
    if args.command == "run":
        import json

        from sentinel.pipeline import run_source

        summary = run_source(config, args.source, args.paced)
        print(json.dumps(summary, ensure_ascii=False))
        return 1 if summary["source_error"] or summary["source_lost"] else 0
    if args.command == "export":
        from sentinel.models import export_openvino, weights_to_export

        for pt in weights_to_export(config):
            print(f"{pt} -> {export_openvino(pt)}")
        return 0
    if args.command == "bench" and args.pipeline:
        from sentinel.bench import WeaponVariant, run_pipeline_bench

        out = args.out or Path("results") / f"bench_pipeline_{date.today().isoformat()}.md"
        variants = [WeaponVariant.parse(v) for v in args.pipeline]
        models = config.models.model_copy(update={"backend": args.format})
        config = config.model_copy(update={"models": models})
        warmup = config.bench.warmup_frames if args.warmup is None else args.warmup
        print(f"report: {run_pipeline_bench(config, args.source, variants, warmup, out)}")
        return 0
    if args.command == "bench":
        from sentinel.agents.pose_features import MatchRules
        from sentinel.bench import run_bench

        det, pose = args.weights or (config.models.people.weights, config.models.pose.weights)
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
            person_conf=config.models.people.conf,
            tracker_cfg=config.models.people.tracker,
            pose_conf=config.models.pose.conf,
            pose_rules=MatchRules(
                min_iou=config.models.pose.min_iou,
                min_inside=config.models.pose.min_inside,
                keypoint_conf=config.models.pose.keypoint_conf,
                margin=config.models.pose.box_margin,
            ),
            out=args.out or Path("results") / f"bench_{date.today().isoformat()}.md",
            paced=args.paced,
        )
        print(f"report: {out}")
        return 0
    print(f"`sentinel {args.command}` is not implemented yet")
    return 2
