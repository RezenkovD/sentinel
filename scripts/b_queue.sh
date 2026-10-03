#!/usr/bin/env bash
# Queue of aggregator B on the free Groq tier (session 05): the pilot, then B on 120 stratified val clips.
# Every step is resumable through B's cache and waits out the provider limits by itself; the user timer
# sentinel-b.timer starts this again (after a reboot or a crash) whenever it is not running.
set -eu
cd "$(dirname "$0")/.."
UV="$HOME/.local/bin/uv"
log() { echo "$(date '+%F %T') $*" >> results/b_queue.log; }
[ -f results/b_queue.done ] && exit 0  # everything answered: the timer has nothing to do
log "queue started"
"$UV" run --no-sync sentinel eval b-pilot --prompt b_v1 --per-stratum 15 --repeat-windows 40 --repeats 2 --out results/b_pilot_b_v1 >> results/b_pilot_b_v1.log 2>&1
log "pilot done"
"$UV" run --no-sync sentinel eval b-run --clips 120 >> results/b_run_val120.log 2>&1
log "B on 120 val clips done"
touch results/b_queue.done
