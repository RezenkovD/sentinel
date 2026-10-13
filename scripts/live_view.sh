#!/usr/bin/env bash
# Live view of `sentinel run` / `sentinel demo` for the show (session 08): neither prints anything per window, so
# this follows the journal (results/run_<time>.jsonl or results/demo_<time>.jsonl) and prints one line per analysis window — video time, people,
# A's / B's risk and level, automaton state, who decided, whether B was asked, incident events, weapon in hand.
# Run it in a second terminal next to the run; Ctrl+C stops only the view. Needs jq.
#   scripts/live_view.sh            # waits for the next run to write its first window
#   scripts/live_view.sh --last     # follows the newest existing journal (a run that has already started)
set -euo pipefail
cd "$(dirname "$0")/.."
command -v jq >/dev/null || { echo "потрібен jq (sudo apt install jq)"; exit 1; }
if [[ "${1:-}" == --last ]]; then
  journal=$(ls -t results/run_*.jsonl results/demo_*.jsonl 2>/dev/null | head -1)
  [[ -n "$journal" ]] || { echo "у results/ ще немає журналів run_*.jsonl / demo_*.jsonl"; exit 1; }
else
  marker=$(mktemp)
  trap 'rm -f "$marker"' EXIT
  echo "чекаю на перше вікно нового sentinel run / demo…"
  journal=""
  while [[ -z "$journal" ]]; do
    sleep 0.5
    journal=$(find results -maxdepth 1 \( -name 'run_*.jsonl' -o -name 'demo_*.jsonl' \) -newer "$marker" | sort | tail -1)
  done
fi
echo "журнал: $journal"
tail -n +1 -F "$journal" 2>/dev/null | jq --unbuffered -r '
  def fixed($d): if . == null then "—" else (pow(10; $d)) as $m | (. * $m | round) as $c
    | "\($c / $m | floor),\($c % $m | tostring | ("0" * ($d - length)) + .)" end;
  [ "\(.t | fixed(1)) с",
    "людей \(.scene.summary.n_persons // 0)",
    "ризик \(.risk.risk | fixed(2))",
    (.state // "—"),
    "рішення \(.decision.source // "A")" + (if .decision.asked_b then " · запит до B" else "" end),
    ([.scene.persons[]?.holds | select(. != null)] | if length > 0 then "ЗБРОЯ В РУЦІ: " + join(", ") else empty end),
    ([.events[]? | {open: "▶ ІНЦИДЕНТ ВІДКРИТО", close: "■ ІНЦИДЕНТ ЗАКРИТО"}[.kind] // .kind]
      | if length > 0 then join(", ") else empty end)
  ] | join("  │  ")'
