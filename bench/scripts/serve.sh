#!/usr/bin/env bash
# One vLLM replica per GPU (no tensor parallel), consecutive ports.
#   MODEL=/path/to/models/Qwen3-8B NAME=qwen3-8b GPUS="1 2 3" PORT0=8041 bash scripts/serve.sh
# --generation-config vllm is required: otherwise vLLM applies the model's own
# top_p/top_k and the sampling policy no longer matches the logs.
set -euo pipefail
: "${MODEL:?}" "${NAME:?}" "${GPUS:?}" "${PORT0:?}"
SETUP=${SETUP:-/data2/yujia/Yingli/gowhy_setup}
ROOT=${ROOT:-/data2/yujia/Yingli/gowhy}
PY=$SETUP/envs/vllm011/bin/python
mkdir -p "$ROOT/logs"
port=$PORT0
for g in $GPUS; do
  log=$ROOT/logs/vllm_${NAME}_${port}.log
  CUDA_VISIBLE_DEVICES=$g nohup "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --served-model-name "$NAME" --port "$port" \
    --generation-config vllm --gpu-memory-utilization 0.9 \
    --enable-prefix-caching ${EXTRA_ARGS:-} > "$log" 2>&1 &
  echo $! > "$ROOT/logs/vllm_${NAME}_${port}.pid"
  echo "$NAME gpu=$g port=$port pid=$! log=$log"
  port=$((port + 1))
done
