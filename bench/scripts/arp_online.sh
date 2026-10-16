#!/usr/bin/env bash
# ARP online validation: new GPU replays, 10 runs per method at budget 0.3, two stores.
ROOT=/data2/yujia/Yingli/gowhy; PY=/data2/yujia/Yingli/gowhy_setup/envs/alfw/bin/python
cd $ROOT; export TMPDIR=$ROOT/tmp; mkdir -p logs/arp
CH='UPDATE events SET action = reject:3 WHERE valid = 0 SCOPE ALL'
run() {  # db servers extra method rep_base run
  GOWHY_SERVERS="$2" $PY -u gowhy/arp.py online --db store/$1.db $3 --change "$CH" --budget 0.3 \
    --method $4 --rep-base $5 --run $6 --workers 16 --out store/arp_online_$1.jsonl \
    > logs/arp/$1_$4_$6.log 2>&1
}
for r in $(seq 0 9); do
  run fresh '{"qwen3-8b":[8041,8042]}' "--extra-runs data/fresh/Qwen3-8B-fresh.rep2.jsonl" arpdr 200 $r &
  run fresh '{"qwen3-8b":[8041,8042]}' "--extra-runs data/fresh/Qwen3-8B-fresh.rep2.jsonl" uniform 300 $r &
  run fresh_q25 '{"qwen2.5-7b":[8032,8033]}' "" arpdr 200 $r &
  run fresh_q25 '{"qwen2.5-7b":[8032,8033]}' "" uniform 300 $r &
done
wait
echo ARP_ONLINE_DONE
