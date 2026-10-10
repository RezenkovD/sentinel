#!/usr/bin/env bash
# Pre-show check (session 07, A4): one OK / FAIL line per item; secrets are never printed (only whether they
# are set, and the services' yes / no answers). Exit code = number of FAILs.
#   scripts/demo_check.sh
set -uo pipefail
cd "$(dirname "$0")/.."
fails=0
ok() { printf 'OK    %s\n' "$1"; }
fail() { printf 'FAIL  %s\n' "$1"; fails=$((fails + 1)); }
check() { if eval "$2" >/dev/null 2>&1; then ok "$1"; else fail "$1${3:+ — $3}"; fi; }
py() {  # a Python check from stdin: print its OK / FAIL lines and count the FAILs
  local out
  out=$(uv run --no-sync python - 2>&1)
  if printf '%s\n' "$out" | grep -qE '^(OK|FAIL) '; then
    printf '%s\n' "$out" | grep -E '^(OK|FAIL) '
    fails=$((fails + $(printf '%s\n' "$out" | grep -c '^FAIL ')))
  else
    fail "перевірка на Python не виконалась"
  fi
}

# power and load
ac=$(cat /sys/class/power_supply/A*/online 2>/dev/null | head -1)
[[ "$ac" == 1 ]] && ok "живлення від мережі" || fail "живлення від мережі — підключити зарядку"
profile=$(powerprofilesctl get 2>/dev/null || echo "?")
[[ "$profile" == performance ]] && ok "профіль живлення: performance" \
  || fail "профіль живлення: $profile — powerprofilesctl set performance"
load=$(cut -d' ' -f1 /proc/loadavg)
awk -v l="$load" 'BEGIN { exit !(l < 2.0) }' && ok "load average 1 хв: $load (< 2)" \
  || fail "load average 1 хв: $load (≥ 2) — закрити Docker-тести й зайві програми"

# camera
check "вебкамера /dev/video0 доступна для читання" "test -r /dev/video0" "немає пристрою або прав"

# secrets: presence only
for key in TELEGRAM_TOKEN TELEGRAM_CHAT_IDS GROQ_API_KEY; do
  check "$key у .env заповнено (значення не показується)" "grep -Eq '^${key}=.+' .env" "немає або порожнє"
done

# services, through Python so that no secret appears on a command line or in the output
py <<'PYEOF'
import requests
from dotenv import dotenv_values

env = dotenv_values(".env")


def line(good: bool, text: str) -> None:
    print(("OK    " if good else "FAIL  ") + text)


token = (env.get("TELEGRAM_TOKEN") or "").strip()
try:
    r = requests.get(f"https://api.telegram.org/bot{token}/getMe", timeout=10)
    good = r.ok and bool(r.json().get("ok"))
    line(good, "бот Telegram відповідає (getMe)" + ("" if good else f" — HTTP {r.status_code}"))
except requests.RequestException as e:
    line(False, f"бот Telegram відповідає (getMe) — {type(e).__name__}")
key = (env.get("GROQ_API_KEY") or "").strip()
try:
    r = requests.get("https://api.groq.com/openai/v1/models", headers={"Authorization": f"Bearer {key}"}, timeout=10)
    ids = {m.get("id") for m in r.json().get("data", [])} if r.ok else set()
    good = "openai/gpt-oss-120b" in ids
    line(good, "Groq досяжний, модель openai/gpt-oss-120b є (список моделей, токени не витрачаються)"
         + ("" if r.ok else f" — HTTP {r.status_code}"))
except requests.RequestException as e:
    line(False, f"Groq досяжний — {type(e).__name__}")
PYEOF

# demo files
for f in results/demo/1_fight_h264.mp4 results/demo/2_weapon_h264.mp4 results/demo/3_calm_h264.mp4 \
  results/demo/1_fight_src.mkv; do
  d=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "$f" 2>/dev/null)
  [[ -n "$d" ]] && ok "$f (${d%.*} с)" || fail "$f — немає або не читається (див. docs/demo.md)"
done

# Docker image and weights
check "Docker-образ sentinel:dev є" "docker image inspect sentinel:dev" "scripts/protocol.sh --build"
py <<'PYEOF'
import hashlib
import json
from pathlib import Path

from sentinel.config import DEFAULT_CONFIG, load_config

frozen = json.loads(Path("docs/experiment_freeze.json").read_text())["state"]["weights_sha256"]
cfg = load_config(DEFAULT_CONFIG)
paths = {"people": cfg.models.people.weights, "pose": cfg.models.pose.weights, "weapons": cfg.weapons.weights}
for name, path in paths.items():
    p = Path(path)
    good = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == frozen[name]
    text = "збігається із заморожуванням 04.10 (= env.md)" if good else "НЕ збігається або файлу немає"
    print(("OK    " if good else "FAIL  ") + f"ваги {name}: SHA-256 {text}")
PYEOF

# aggregator B queue: it must not spend the free daily limit during the show
state=$(systemctl --user is-active sentinel-b.service 2>/dev/null)
done_b=$([[ -f results/b_queue.done ]] && echo так || echo ні)
if [[ "$state" == active ]]; then
  fail "sentinel-b працює — на час показу: systemctl --user stop sentinel-b.timer sentinel-b.service"
else
  ok "sentinel-b не працює (стан: ${state:-невідомий}; прогін B на val завершено: $done_b)"
fi
echo "FAIL: $fails"
exit "$fails"
