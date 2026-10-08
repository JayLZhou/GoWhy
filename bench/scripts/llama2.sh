#!/usr/bin/env bash
# Llama-3.1-8B, part 2: harness-fix query after the store is built and the Qwen3 pools are done.
set -uo pipefail
ROOT=/data2/yujia/Yingli/gowhy; PY=/data2/yujia/Yingli/gowhy_setup/envs/alfw/bin/python
cd $ROOT; export TMPDIR=$ROOT/tmp
until grep -q '^total' logs/llama_gen.log; do sleep 60; done
echo "[$(date +%T)] gen: $(grep '^total' logs/llama_gen.log)"
$PY gowhy/store.py fresh_llama > logs/llama_store.log 2>&1
echo "[$(date +%T)] store: $(grep -E 'success|invalid' logs/llama_store.log | tr '\n' ' ')"
pools_done() { for d in ws_pool replan_alf replan_ws; do for f in logs/$d/c*.log; do grep -qE '^  cost:|Traceback' $f || return 1; done; done; }
until pools_done; do sleep 120; done
echo "[$(date +%T)] qwen3 pools done; switching GPUs 0-2 to llama"
for p in 8041 8042 8043; do kill $(cat logs/vllm_qwen3-8b_$p.pid) 2>/dev/null; done; sleep 40
MODEL=/data2/yujia/models/Llama-3.1-8B-Instruct NAME=llama-3.1-8b GPUS="0 1 2" PORT0=8052 \
  EXTRA_ARGS="--max-model-len 32768" bash scripts/serve.sh
for p in 8052 8053 8054; do until curl -s -m 5 localhost:$p/v1/models | grep -q llama; do sleep 20; done; done
echo "[$(date +%T)] 4 llama replicas up"
export GOWHY_SERVERS='{"llama-3.1-8b":[8051,8052,8053,8054]}'
mkdir -p logs/llama
printf 'WHATIF UPDATE events SET action = reject:1 WHERE k < 0 SCOPE ALL WITH n = 1 DEPLOY m = 2\n' > logs/llama/q0.txt
printf 'WHATIF UPDATE events SET action = reparse WHERE valid = 0 SCOPE ALL WITH n = 2 DEPLOY m = 2\n' > logs/llama/q1.txt
for i in 0 1; do $PY -u gowhy/query.py --db store/fresh_llama.db --workers 96 @logs/llama/q$i.txt > logs/llama/c$i.log 2>&1 & done
wait
echo "[$(date +%T)] llama queries done"
