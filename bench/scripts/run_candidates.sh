#!/usr/bin/env bash
# Evaluate every change in a candidates file in parallel, one query process each.
#   DB=store/fresh.db HALF=B N=1 [CRN=1] bash scripts/run_candidates.sh gowhy/mined_qwen3.txt logs/mined2
# Each change gets <outdir>/qI.txt (the query) and <outdir>/cI.log (its output).
# Runs are cached in the store, so rerunning only fills what is missing.
set -euo pipefail
cands=$1; out=$2
ROOT=${ROOT:-/data2/yujia/Yingli/gowhy}
PY=${PY:-/data2/yujia/Yingli/gowhy_setup/envs/alfw/bin/python}
DB=${DB:-store/fresh.db}; HALF=${HALF:-}; N=${N:-1}; WORKERS=${WORKERS:-48}
export TMPDIR=$ROOT/tmp
export GOWHY_SERVERS=${GOWHY_SERVERS:-'{"qwen3-8b":[8041,8042,8043,8044]}'}
cd "$ROOT"; mkdir -p "$out" "$TMPDIR"
half_arg=(); [ -n "$HALF" ] && half_arg=(--half "$HALF")
[ -n "${CRN:-}" ] && half_arg+=(--crn)   # seeded replays (common random numbers)
i=0
grep '^UPDATE' "$cands" | while IFS= read -r ch; do
  i=$((i + 1))
  printf 'WHATIF %s WITH n = %s\n' "$ch" "$N" > "$out/q$i.txt"
  setsid nohup "$PY" -u gowhy/query.py --db "$DB" "${half_arg[@]}" --workers "$WORKERS" \
    "@$out/q$i.txt" > "$out/c$i.log" 2>&1 < /dev/null &
  echo "c$i: $ch"
done
