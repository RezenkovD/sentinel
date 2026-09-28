"""Command-line interface: `sentinel run | bench | eval`."""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="sentinel", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("run", help="run the pipeline on a camera, RTSP stream or video file")
    sub.add_parser("bench", help="measure processing speed (ms/frame, RTF)")
    sub.add_parser("eval", help="run the experiment on dataset episodes")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print(f"`sentinel {args.command}` is not implemented yet")
    return 2
