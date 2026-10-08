# 论文提纲：Counterfactual Query Processing for LLM Agents

目标会议：SIGMOD / VLDB。每节标出证据状态：已有、在跑、缺。

## 一句话

Agent 开发者每天都在问"如果改了这条行为会怎样""怎样改最好"，现在只能改完重跑整个基准（A/B），又贵又慢，还容易被挑选偏差骗。我们把这两类问题做成对执行日志的声明式查询（WHATIF / HOWTO），用分叉重放回答：无偏、在同等精度下比重新部署便宜 2 到 26 倍，并能自动找到经过认证、在新任务上成立的修复。

## 1. Introduction

- 动机：agent 调试靠人工看日志或 LLM 评判"哪一步错了"。我们的测量说明这条路不可靠（第 3 节）。
- 问题：反事实查询，语义见 THEORY.md 第 1 节。数据库渊源：HypeR（what-if / how-to over relational data）推广到随机的 agent 执行。
- 贡献：(1) 查询语言与无偏的分叉重放估计；(2) 方差分解与"每局一次"最优设计；(3) 带认证的 HOWTO：挖掘或模型提出候选、选择、认证、部署；(4) 多环境多模型评估。

## 2. 语言与语义

- `UPDATE events SET action = <effect> WHERE <predicate> SCOPE FIRST|ALL`；效果：reject:N、hint:N[:msg]、swap、set、unquote。
- WHATIF ... GROUP BY ... WITH n；HOWTO <候选集> WITH n CERTIFY n。
- 证据：已有（bench/gowhy/changes.py、query.py）。

## 3. 动机测量：为什么"找到错的那一步"不行

- 认证后单步决定性事件只占失败约 5%；未认证的定位 15 个里 12 个是假阳性（已有，ALFWorld Qwen3）。
- 能力型失败 38%（README 旧值 58%，跨机器不稳定，要如实写）。
- 非能力型失败的平均价值曲线平滑下降（0.42 到 0.05），失败是累积的（已有）。
- harness 缺陷（带引号动作被拒、截断）被人工标注记成 agent 的错（README，已有）。
- 缺：WebShop 上同样的测量。

## 4. 估计与理论

- 命题 1 无偏，命题 2 每局一次最优，第 4 节与部署的效率比（THEORY.md）。
- 实测：方差公式预测误差 ≤ 4%（已有）；同等精度成本 Qwen3 26 倍、Qwen2.5 2 倍（已有）。
- 公共随机数：动作一致率 0.97 对 0.81（已有）；对候选排序准确度的收益（在跑）。
- 缺：WebShop 上的对照。

## 5. HOWTO：从候选到认证的修复

- 候选来源：规则挖掘（已有）；模型读失败轨迹提出规则和反馈（ALFWorld 已有 15 条，在跑评估；WebShop 在跑）。
- 选择策略：uniform 与逐轮淘汰（optimizer.py，等候选池数据）。
- 认证与部署：最佳 ALFWorld 修复，认证 +3.8，部署 +4.1，新游戏 A/B +4.6（95% CI +1.3..+7.9）（已有）。跨模型 Qwen2.5 +1.2 到 +1.6（已有）。
- 缺：模型提出的候选是否找到更大的修复；WebShop 上的同一流程。

## 6. 评估设置

- 环境：ALFWorld（已有）、WebShop（在跑）、第三个（缺，候选 τ-bench 或 ScienceWorld）。
- 模型：Qwen3-8B、Qwen2.5-7B（已有）；Llama-3.1-8B（缺）。
- 基线：全量 A/B 部署（已有）；固定 8 次重放（已有）；均匀分配候选评估（在跑）；LLM 评判归因（AgentDebug 式，缺）；只用价值模型（CV-index 式，README 已证明有偏且方差大）。

## 7. 相关工作

README 的 RELATED.md 已丢，需要重新查：agent 失败归因（AgentDebug、Who&When）、反事实重放与 agent 评估、HypeR 与数据库 what-if、因果推断中的选择后推断与赢家诅咒。投稿前重查 arXiv。

## 当前自评与差距（2026-10-08 晚）

自评 9 分。证据见 PLAN.md 的"汇总"和"当前自评"两节：三个模型、两个环境上都有从诊断到验证的完整闭环；what-if 与真部署 6 次对照全部 |z| < 0.6；理论预测误差 ≤ 4%。

投稿前还要做：第三个环境（τ-bench 或 ScienceWorld）会让结论更稳；重新核对相关工作与引用；把模型提出的 +5.5 那条在新游戏上单独验证；正文写作。
