#!/usr/bin/env bash
# Trials protocol (session 06, step 2): the Docker image from a clean environment — versions and hashes, pytest,
# a paced run of test clips per scenario — then the protocol goes to docs/reports/protocol_<date>.md.
#   scripts/protocol.sh [--build] [--telegram]
set -euo pipefail
cd "$(dirname "$0")/.."
image=sentinel:dev
args=()
for a in "$@"; do
  case "$a" in
    --build) docker build -t "$image" . ;;
    --telegram) args+=(--telegram) ;;
    *) echo "unknown option $a" >&2; exit 2 ;;
  esac
done
day=$(date +%F)
git_rev="$(git rev-parse --short HEAD)$(git diff --quiet HEAD -- src configs tests Dockerfile || echo '-dirty')"
env_file=()
[[ " ${args[*]-} " == *" --telegram "* ]] && env_file=(--env-file .env)
docker run --rm --user "$(id -u):$(id -g)" "${env_file[@]}" \
  -e SENTINEL_IMAGE="$(docker image inspect -f '{{.Id}}' "$image")" \
  -e SENTINEL_IMAGE_CREATED="$(docker image inspect -f '{{.Created}}' "$image")" \
  -e SENTINEL_GIT="$git_rev" \
  -v "$PWD/weights:/app/weights:ro" -v "$PWD/data:/app/data:ro" -v "$PWD/results:/app/results" \
  "$image" eval protocol --out "results/protocol_$day.md" "${args[@]}"
cp "results/protocol_$day.md" "docs/reports/protocol_$day.md"
echo "docs/reports/protocol_$day.md"
