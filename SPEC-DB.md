# GoWhy 作为数据库：存什么、查什么、怎么改、怎么插、怎么删（v1，2026-10-01）

一句话：**基础事实只追加不覆盖，派生结果带依赖集可失效重算，每个答案带 explain 可回放。**

## 1. 存什么

### 1.1 基础对象（事实，带 provenance 与版本）
| 表 | 字段 | 说明 |
|---|---|---|
| Variable | id, name, type, domain, granularity, view_id, encoder? | 生态里的一个可干预 / 可测量的量 |
| Edge | src, dst, mechanism_id, provenance_ids[], version_range, status ∈ {supported, contested, retracted} | 一条因果边；同一对节点允许多条互相矛盾的声明并存，标 contested |
| Mechanism | id, child, parents[], class ∈ {linear, tree, pwl-nn, table, llm-summary}, params_ref, fitted_on(view_id, data_ref), version | 子变量如何依赖父变量 |
| Evidence | id, kind ∈ {experiment, replay, observational, expert, discovery}, source_ref, trust, timestamp, seed? | 边 / 机制 / 效应的来源；trust 由 kind 决定的偏序：experiment ≈ replay > observational > expert > discovery |
| View | id, kind ∈ {local, global}, variables[], mapping ∈ {LAV, GAV}, owner, refresh_policy | 数据源注册；global view 由 mediator 从 local views 合并得到 |
| Episode | id, view_id, x, t̄, m̄, y, timestamp, seed, agent_version | 原始观测单位（agent 轨迹、A/B 记录）；可选，支持 L0 直接入库 |
| Version | id, parent, timestamp, diff(edges±, mechanisms±, evidence±) | 全库版本链；任何"当时的答案"可按版本回放 |

### 1.2 派生对象（物化，带依赖集）
| 表 | 字段 | 说明 |
|---|---|---|
| Effect | estimand(X→Y, population, conditioning), value, CI, estimator, assumptions[], deps{edges, mechanisms, evidence, episodes}, version, status ∈ {fresh, stale, unknown} | 效应表的一格，例如 (skill, 任务类) → τ |
| Certificate | query, status ∈ {identifiable, partial[lo,hi], not}, adjustment_set / front-door set, missing{variables, evidence, logs}, deps, version | identify 的结果，也是 Effect 的前置 |
| IndexArtifact | kind ∈ {FCM groups & counts, FDCut cut structure, TESSERA provenance closure}, scope, deps, version | 三个索引的物化结构 |

### 1.3 不变量
- I1 每个版本内 Edge 集合（status=supported）无环。
- I2 每条 Edge、每个 Mechanism 至少一条 Evidence；无来源的声明拒绝入库。
- I3 每个 Effect 必有 Certificate 与依赖集；依赖集里任何对象变化 → Effect.status ← stale。
- I4 低 trust 的写入不能静默覆盖高 trust 的事实；只能并存为 contested。
- I5 任何历史答案可由 (version, seed, estimator) 复现。

## 2. 查什么

执行流：parse → identify（出 Certificate）→ plan（选估计量与索引）→ execute → 附 explain。

| 查询 | 语法示意 | 语义 | 走哪条路 |
|---|---|---|---|
| 观测 | `MATCH (c:Campaign)-[:TARGETS]->(u:User) RETURN ...` | 普通图模式匹配 | 图引擎 |
| 干预 | `WHAT IF SET c.discount = 0.10 RETURN effect(u.churn), interval(0.95)` | E[Y \| do(X=x′), population] | identify → back-door / front-door（FCM、FDCut）/ 重放融合 |
| 反事实 | `COUNTERFACTUAL episode e SET T = t′ RETURN Y` | abduction–action–prediction，需 Mechanism 与该 episode 的噪声可推断 | TESSERA 溯源免重放，否则重放 |
| 事故归因 | `WHY episode e OUTCOME harm CANDIDATES tool_outputs` | 每个候选的 PN / PS 区间与责任分配 | 界 + 少量重放 |
| 组件归因 | `ATTRIBUTE outcome TO components OVER traces WHERE ...` | 每个组件的干预边际；稀疏时 group testing | P7 / N3 |
| 处方 | `PRESCRIBE minimize u.churn SUBJECT TO budget <= B OVER interventions(...)` | 预算内最优干预集 | P6 |
| 可识别性 | `IDENTIFY effect(Y \| do(X)) FROM views` | identifiable / partial [lo,hi] / not + 缺什么 | Certificate |
| 解释 | `EXPLAIN <query id>` | 估计目标、识别路径、假设、证据 id、版本 | 只读派生对象 |
| 时间旅行 | `... AS OF version v` / `AS OF time t` | 在旧版本上执行任意查询 | 版本链 |

约定：任何 WHAT IF / COUNTERFACTUAL / ATTRIBUTE 的返回体固定为 {value, interval, certificate, explain}，不可识别时 value 为空、certificate 给区间与缺口，不允许返回一个裸数。

## 3. 如何改

- **改机制 / 改边 = 追加新版本**，旧版本保留可 AS OF 查询；没有原地覆盖。
- **失效传播**：改动对象 o → 所有 deps ∋ o 的 Effect、Certificate、IndexArtifact 置 stale；重算按策略：eager（小图）或 lazy（下次查询时）；增量重算只动受影响部分（FCM 的局部组、FDCut 的割结构、TESSERA 的受影响版本），这是论文 P3。
- **结构改动**（加 / 删边、改方向）先过 I1 无环检查；改方向等于删旧边加新边，两边各自带 Evidence。
- **冲突写入**：新声明与现有声明矛盾（反向边、同边不同机制）时按 trust 偏序处理：高压低 → 旧声明降为 contested 并保留；低碰高 → 新声明入库为 contested，不改当前答案；同级 → 两者并存 contested，Certificate 标注"依赖争议边"。争议边只能由更高 trust 的 Evidence 或显式仲裁（记录仲裁者与理由）解决。
- **数据更新**（新 Episode 流入某 View）：该 View 上 fitted 的 Mechanism 与依赖它的 Effect 置 stale；重估后与旧值比较，超阈值即打 drift 标记（效应变了，不只是数据多了）。
- **改 View 映射**（schema 变化）：等于改所有经该 View 落地的 Variable 的语义，强制新版本 + 全量重算 Certificate。

## 4. 如何插入

- **INSERT Variable**：挂到一个 View；同名变量跨 View 需显式对齐（mediator 的 alignment），否则视为不同变量。
- **INSERT Edge**：必带 ≥1 Evidence（I2）；无环检查；若存在反向 supported 边 → 走冲突规则进 contested；插入后重算涉及 (src, dst) 祖先 / 后代的 Certificate（一条新边可能开一条新的 back-door 路径，让原本可识别的效应变不可识别，也可能提供新的 front-door 中介，让不可识别的变可识别，两个方向都要重判）。
- **INSERT Mechanism**：绑定 child 与 parents，parents 必须是当前版本里的 supported 父集；带 fitted_on。
- **INSERT Evidence**：附着到 Edge / Mechanism / Effect；可能把 Edge 从 contested 提升为 supported（当 trust 高于对方）；触发依赖它的 Effect 重估。
- **INSERT Episode（批量日志入库）**：按 View 的映射落成 (x, t̄, m̄, y)；检查是否记录了中介 m̄，没有则该 View 上一切配置对比查询的 Certificate 直接标 not（Replay-Free 推论）；增量更新 FCM 计数；为所有 standing query 重跑 identify。
- **REGISTER View**：mediator 把 local view 合并进 global view：变量对齐、边缘一致性检查（重叠变量上的独立性结构不矛盾）、来源加权仲裁（论文 P1、P2）；合并失败则 View 以"未集成"状态存在，只能回答 view 内查询。
- **INSERT Effect**：只允许两种来源，estimator 产出（deps 自动填）或外部声明（kind=expert/literature，必带引用）；不允许无来源的数。

## 5. 如何删除

- **删除 = 撤回（tombstone）**，不是物理删除：对象在新版本里 status=retracted，AS OF 旧版本仍可见。理由：C5 事后审计，授权过某次 agent 行动的那个反事实必须能重建。
- **删 Edge**：依赖它的 Effect 不是 stale 而是 unknown（识别路径可能被拆掉），重跑 identify 后再决定是重算还是降为 partial；反过来删边也可能关掉一条 back-door 路径让别的效应变得可识别，同样重判。
- **删 Evidence**：仅由它支撑的 Edge / Mechanism 降为 unsupported（不满足 I2）→ 自动 retracted，并按上面的删边规则传播；有其他证据的只重算 trust。
- **删 Episode（被遗忘权 / 数据下线）**：从 View 数据中移除；所有 fitted_on 含该数据的 Mechanism 与依赖 Effect 重估；派生的摘要、索引计数同步扣减（删记录不等于遗忘，派生物必须重派生）；版本链记录"因删除而变化的效应"供审计。
- **删 View**：其独占的 Variable、Edge、Mechanism 全部 retracted；global view 重新合并。
- **物理清除（purge）**：只为合规；清除本身写入 Version.diff 作为审计记录，清除后相关 AS OF 查询返回"该版本含已清除数据，不可复现"。

## 6. 和普通图数据库的三点不同（README 里要说）
1. 查询的语义是 do / counterfactual，不是模式匹配；执行前必须过 identify，答案必带假设。
2. 派生对象（效应、证书、索引）带依赖集，事实一变派生物按因果结构失效，不是按表失效。
3. 删除是撤回，历史答案永远可回放，因为答案曾经授权过行动。
