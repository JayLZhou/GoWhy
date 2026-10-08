#!/usr/bin/env bash
# Third model: Llama-3.1-8B-Instruct on ALFWorld. Pre-specified fixes (from Qwen3), all 300 games.
set -uo pipefail
ROOT=/data2/yujia/Yingli/gowhy; PY=/data2/yujia/Yingli/gowhy_setup/envs/alfw/bin/python
cd $ROOT; export TMPDIR=$ROOT/tmp
until grep -q DL_DONE logs/dl_llama.log; do sleep 60; done
echo "[$(date +%T)] download: $(grep have logs/dl_llama.log | tr '\n' ' ')"
# free GPU 3: its Qwen3 replica's in-flight requests retry on the other replicas
kill $(cat logs/vllm_qwen3-8b_8044.pid) 2>/dev/null; sleep 30
MODEL=/data2/yujia/models/Llama-3.1-8B-Instruct NAME=llama-3.1-8b GPUS=3 PORT0=8051 \
  EXTRA_ARGS="--max-model-len 32768" bash scripts/serve.sh
until curl -s -m 5 localhost:8051/v1/models | grep -q llama; do sleep 20; done
echo "[$(date +%T)] llama served"
export GOWHY_SERVERS='{"llama-3.1-8b":[8051]}'
$PY -u harness/gen_fresh.py --n 300 --model Llama-3.1-8B-fresh --served llama-3.1-8b --workers 96 > logs/llama_gen.log 2>&1
echo "[$(date +%T)] gen: $(grep '^total' logs/llama_gen.log)"
$PY gowhy/store.py fresh_llama > logs/llama_store.log 2>&1
echo "[$(date +%T)] store: $(grep -E 'success|invalid' logs/llama_store.log | tr '\n' ' ')"
mkdir -p logs/llama
printf 'WHATIF UPDATE events SET action = reject:1 WHERE k < 0 SCOPE ALL WITH n = 1 DEPLOY m = 2\n' > logs/llama/q0.txt
printf "WHATIF UPDATE events SET action = hint:2 WHERE verb = 'take' AND obj_match = 0 SCOPE ALL WITH n = 2 DEPLOY m = 2\n" > logs/llama/q1.txt
printf "WHATIF UPDATE events SET action = replan:2 WHERE (verb = 'take' AND obj_match = 0) OR (verb = 'move' AND recep_match = 0) SCOPE ALL WITH n = 2 DEPLOY m = 2\n" > logs/llama/q2.txt
printf 'WHATIF UPDATE events SET action = reject:3 WHERE valid = 0 SCOPE ALL WITH n = 2 DEPLOY m = 2\n' > logs/llama/q3.txt
for i in 0 1 2 3; do
  $PY -u gowhy/query.py --db store/fresh_llama.db --workers 48 @logs/llama/q$i.txt > logs/llama/c$i.log 2>&1 &
done
wait
echo "[$(date +%T)] llama queries done"
