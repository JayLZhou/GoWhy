#!/usr/bin/env bash
# WebShop end to end, after harness/gen_fresh.py --env webshop has written data/fresh/Qwen3-8B-ws.jsonl:
#   1. wait for generation to finish   2. build store/ws_q3.db
#   3. LLM-proposed candidates from half A, plus generic ones
#   4. evaluate every candidate on half B with seeded replays (n = 4)
# Each step logs to logs/ws_*.log. Safe to rerun: runs are cached.
set -uo pipefail
ROOT=/data2/yujia/Yingli/gowhy
PY=/data2/yujia/Yingli/gowhy_setup/envs/webshop/bin/python
export TMPDIR=$ROOT/tmp GOWHY_SERVERS='{"qwen3-8b":[8041,8042,8043,8044]}'
cd "$ROOT"

until grep -q '^total' logs/ws_gen.log; do sleep 120; done
echo "[$(date +%T)] generation done: $(grep '^total' logs/ws_gen.log)"

$PY gowhy/store.py ws_q3 > logs/ws_store.log 2>&1
echo "[$(date +%T)] store: $(grep -E 'traces|success|invalid' logs/ws_store.log | tr '\n' ' ')"

$PY gowhy/propose.py --db store/ws_q3.db --per-trace 2 --keep 20 --out gowhy/proposed_ws.txt > logs/ws_propose.log 2>&1
echo "[$(date +%T)] proposals: $(grep -c '^UPDATE' gowhy/proposed_ws.txt)"

{
  grep '^UPDATE' gowhy/proposed_ws.txt
  echo "UPDATE events SET action = reject:3 WHERE valid = 0 SCOPE ALL"
  echo "UPDATE events SET action = hint:2 WHERE valid = 0 SCOPE ALL"
  echo "UPDATE events SET action = hint:2 WHERE repeat = 1 SCOPE ALL"
} | sort -u > gowhy/pool_ws.txt
echo "[$(date +%T)] pool: $(wc -l < gowhy/pool_ws.txt) candidates"

PY=$PY DB=store/ws_q3.db HALF=B N=4 CRN=1 WORKERS=10 bash scripts/run_candidates.sh gowhy/pool_ws.txt logs/ws_pool
echo "[$(date +%T)] pool launched"
