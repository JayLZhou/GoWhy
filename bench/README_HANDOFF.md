# GoWhy 交接 README（2026-10-16）

题目：**GoWhy: Certified Why and How-To Queries over Replayable Agent Executions**

一句话：在可重放的 agent 执行上，用"干预某一步 + 重放 + 对照"回答两类带统计保证的查询：
why（这次失败的根因在哪个 agent 的哪一轮）和 how-to（改哪里能提高成功率）。
单 agent 部分（ALFWorld / WebShop / ScienceWorld）已完成；多 agent 部分用 AgentsNet
（MAFBench 的协作任务），harness 已写好并通过自测，GPU 实验还没开始。

计划与验收标准：`bench/PLAN_MERGE.md`。全部历史记录与踩过的坑：`bench/PLAN.md`。
理论：`paper_v3/theory_note.tex`（完整证明）。论文草稿：`paper_v4/main.tex`（how-to 单主线版）。

---

## 1. 代码在哪

本地仓库 `~/Project/GoWhy/`（GitHub: JayLZhou/GoWhy，远端落后于本地，未推送最近改动）。

```
bench/
  harness/
    aeb.py          ALFWorld harness（AgentDebug 逐字复现），start_episode 分发 webshop:/sciworld:
    webshop.py, sciworld.py
    agentsnet.py    多 agent harness（AgentsNet 逐字复现）：执行、日志、重放、干预、自测  ← 新
    llm.py          vLLM 调用；Policy(name)(prompt, seed) 单轮；Policy.chat(messages, seed) 多轮  ← 新增 chat
    gen_fresh.py    采集单 agent 轨迹库
    ab.py           新任务上的 A/B（从头部署）
  gowhy/
    store.py        SQLite 库（traces / events / runs）
    changes.py      改动语言 UPDATE events SET action=<effect> WHERE <pred> SCOPE ALL
    engine.py       分叉点重放 replay_one；共享重放树（donor / log_steps）；补丁重放 replay_patch
    lattice.py      候选格（原子 × 效果），日志上界
    lattice_gai.py  how-to 搜索 + 认证（sh / random / ub-order / lin），--prefix，--extra
    offline_bench.py 离线无偏基准（缓存重放上跑搜索算法）
    pair_eval.py    耦合实验（分歧质量）
    csi.py          反事实库迭代（物化 + 多轮）
    gepa_lite.py    GEPA 式提示优化基线
    propose.py      大模型提议规则
  scripts/serve.sh  起 vLLM 单卡副本
  PLAN.md, PLAN_MERGE.md, THEORY.md, DESIGN_V2.md, PAPER.md
paper_v3/           旧论文 + theory_note.tex（理论文档，编译 tectonic theory_note.tex）
paper_v4/           当前论文（编译 tectonic main.tex；acmart.cls、bst、ref.bib 在目录内）
```

## 2. 服务器上在哪（14 号机 yujia-server，10.249.185.22，共享账号 yujia）

```
/data2/yujia/Yingli/gowhy/              代码同步目标（rsync bench/ 过去，排除 store logs tmp）
  store/                                SQLite 库与结果（300M）：fresh_q25.db fresh.db fresh_llama.db ws_q3.db sw_q25.db，
                                        gai_*.jsonl, universe500.json, offline_u500.jsonl, pair_eval.jsonl, csi*_summary.jsonl, ab_*.jsonl
  data/fresh/*.jsonl                    单 agent 轨迹库原始数据（220M）
  data/agentsnet/train.parquet          AgentsNet 的 117 张图（ws/ba/dt × 4..100 节点）
  logs/                                 运行日志；*_run.sh 是各实验的启动脚本
  tmp/                                  TMPDIR（根分区曾被写满，必须设）
/data2/yujia/Yingli/gowhy_setup/
  envs/alfw      Python 3.10：alfworld 0.4.2、textworld、openai、pandas、pyarrow、networkx、regex、scipy
  envs/vllm011   vllm 0.11.0、transformers 4.57.1
  envs/webshop   WebShop（含 JVM：lib/jvm）；envs/sciworld  scienceworld 1.2.3
  alfworld_data/ ALFWorld 数据（2.1G）；install.sh / ws_install.sh / sw_install.sh 建环境脚本
/data2/yujia/models/Qwen2.5-7B-Instruct（软链到 fengjia 的目录）、Llama-3.1-8B-Instruct；Qwen3-8B 在 /data2/jfeng/project/models/Qwen3-8B
```

换机器要搬的：`gowhy/`（代码 + store + data）、`gowhy_setup/`（环境可用 install.sh 重建；alfworld_data 2.1G 要拷）、模型权重。
AgentsNet 图数据也可重新下：`curl -L -o train.parquet https://hf-mirror.com/datasets/disco-eth/AgentsNet/resolve/main/data/train-00000-of-00001.parquet`
（14 号机连不上 GitHub，hf-mirror / tuna / gitee 可用）。

## 3. 起模型

```bash
cd /data2/yujia/Yingli/gowhy
MODEL=/data2/yujia/models/Qwen2.5-7B-Instruct NAME=qwen2.5-7b GPUS="0 1" PORT0=8034 bash scripts/serve.sh
# Llama: MODEL=/data2/yujia/models/Llama-3.1-8B-Instruct NAME=llama-3.1-8b GPUS="0 1" PORT0=8034 EXTRA_ARGS="--max-model-len 32768" bash scripts/serve.sh
# 等待就绪: until curl -s localhost:8034/v1/models | grep -q qwen; do sleep 20; done
# PID 在 logs/vllm_<name>_<port>.pid；停: kill $(cat logs/vllm_qwen2.5-7b_8034.pid)
```
每个任务都要设：
```bash
export TMPDIR=/data2/yujia/Yingli/gowhy/tmp GOWHY_SERVERS='{"qwen2.5-7b":[8034,8035]}'
P=../gowhy_setup/envs/alfw/bin/python
```
长任务用 `setsid nohup bash logs/xxx_run.sh > logs/xxx_driver.log 2>&1 < /dev/null &`（直接 nohup 会随 ssh 会话被杀）。

## 4. 多 agent（AgentsNet）怎么跑

```bash
# 自测（不需要 GPU）：重放不干预 0 次调用；干预后重新生成的 agent-轮次 == 因果锥
$P harness/agentsnet.py selftest --task coloring

# 采集执行（E2）：任务 × 拓扑 × 规模 × 图编号 × 重复；结果 store/agentsnet/<task>/<gen>_<n>_<idx>_rep<r>.json + summary.jsonl
$P harness/agentsnet.py gen --task coloring matching vertex_cover leader_election consensus \
   --generators ws ba dt --sizes 4 8 16 --indices 0 1 2 3 --reps 2 --served qwen2.5-7b --procs 8

# 对一条日志做干预并重放（只重新生成因果锥内的 agent-轮次）
$P harness/agentsnet.py replay --trace store/agentsnet/coloring/ws_8_0_rep0.json --agent 3 --turn 2 --op resample --rep 100
#   --op hint --text "..."   在该 agent 该轮的提示后追加一段话再生成
#   --op replace --text "..."  直接把该轮回复换成给定文本
```
日志格式：`agents[v][t] = {hist_hash, user, reply, fallback_reply?, parsed, seed, reused, intervened}`；
`answers`、`score`、`success`（score == 1）、`calls`、`reused`。turn 0 是首轮消息，turn T 是最终答案。
一次执行成本 ≈ n × (T+1) 次调用（n=8、T=5 约 48 次）。温度 0.7（AgentsNet 原文用 API 默认，论文里要说明）。

## 5. 单 agent 部分怎么跑（已完成，复现用）

```bash
$P harness/gen_fresh.py --env alfworld --served qwen2.5-7b ...        # 采轨迹 -> data/fresh/*.jsonl
$P gowhy/store.py fresh_q25                                          # 建库 -> store/fresh_q25.db
$P gowhy/lattice_gai.py --db store/fresh_q25.db --half B --method sh --search-budget 800 --seed 1   # how-to 搜索+认证
$P harness/ab.py --change "UPDATE events SET action = replan:2 WHERE valid = 0 SCOPE ALL" --served qwen2.5-7b --m 2   # 新游戏 A/B
$P gowhy/offline_bench.py --db store/fresh_q25.db --half B --universe store/universe500.json --budget 200 400 800 --runs 200  # 离线基准（无 GPU）
$P gowhy/pair_eval.py --db store/fresh_q25.db --half B --universe store/universe500.json --pairs 111 --rep 2000  # 耦合实验
```

## 6. 现状与下一步

已有结果（都在 PLAN.md）：重放估计与真部署 6 组吻合；how-to 认证的修复在新游戏上成立（Qwen2.5 41.2%→48.4%，Llama 0.4%→16.8%）；
端到端一次 how-to 约 2 万次调用（比逐候选重放少 50–100 倍）；同预算 GEPA 零提升（7B 反思）；耦合省 1.8 倍（相近对 5.2 倍）；
无偏基准上 SH 好于随机与按上界顺序；CSI 第二轮四次都没认证出东西；理论（剪枝恰好免费、穷尽、上下界、耦合分离、CSE）证明齐全。

下一步（按 PLAN_MERGE.md 的 E1–E9）：
1. 卡空出来后跑 `gen`（E2），看 Qwen2.5-7B 在 n=4/8/16 的成功率；若几乎全失败，考虑 n=4 全连接或换 14B。
2. 写 `gowhy/why.py`：why 查询 = 因果过去剪枝（只保留能传到见证节点的 (v,t)）→ Prune-and-Race / CSE → 新种子认证；
   真值用注入故障（在成功执行里 replace 某条消息）和小规模穷举。
3. 基线：大模型判官（AgentLocate / Who&When 式）、随机位置、末轮选点、穷举。
4. 指标：agent 级 / 轮级定位准确率、修复成功率、调用数；预算曲线；耦合节省。

## 7. 坑

- `/data2` 会被别人临时写满（已发生三次）：写库报 "database or disk is full"，进程会死。脚本要可续跑（pair_eval 已改成每对独立种子、自动补跑）。
- 用 `setsid nohup`；`pkill -f` 用方括号模式（`[l]attice_gai`）避免杀到自己的 shell。
- 共享账号 `yujia` 上会有别人的 vLLM（4 卡并行 TP 进程）；我们的服务是单卡 `api_server`，按 PID 文件停，不要按名字杀。
- vLLM 不可逐字复现：同种子 ≠ 耦合，必须用重放树（engine.replay_one 的 donor / agentsnet 的 hist_hash 复用）。
- 认证规则是严格的 LCB > τ 且估计 > τ（曾有效果 0、方差 0 被判通过的 bug）。
- 多种子大规模重复不要做（用户要求）；本地电脑不跑任何后台任务。
