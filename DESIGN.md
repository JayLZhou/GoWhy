# GoWhy 系统设计 v0.1（2026-10-01）

> 状态：设计稿，未开工，等你过一遍 §11 的未决问题再动代码。
> 一句话：**给 agent 用的因果数据库，行动前先问 what-if，事后能答 why。** 它是把因果图当数据、把因果问题当查询、把 agent 当用户的那一层。

## 0. 这份文档管什么

| 文档 | 回答的问题 |
|---|---|
| `README.md` | 对外：卖点、quickstart、不是什么 |
| `SPEC-DB.md` | 存什么、查什么、怎么改 / 插 / 删（语义与不变量 I1–I5） |
| `SPEC-TRACES.md` | agent 轨迹怎么入库（模板、最小日志契约） |
| `../CWS-PROJECT-MODEL.md` | 模块、论文挂接、90 天里程碑 |
| **`DESIGN.md`（本文）** | 系统怎么搭：参考了谁、架构、三个入口、内核判定流程、v0.1 切片、验收 |

本文不重复两份 SPEC 的内容，只在需要落到实现时引用。

## 1. 目标与非目标

三个能力，每个对应一个能演示的入口：

| # | 能力 | 入口 | 来自 | 别人为什么没有 |
|---|---|---|---|---|
| 1 | 从自己的日志算出工具到底有没有用，不重放 | `gowhy analyze traces/` | N1 定理 1、2 | CausalFlow、CAR 都要重放 |
| 2 | 先判能不能答：可识别 / 部分识别 / 不可识别，缺什么 | `identify` | CausalGQL 的可识别性检查 + M1 记录设计 | DoWhy 会算但不拦你；LLM 会编 |
| 3 | agent 行动前先问一句 | MCP 五个 tool | L2 的 demo | 规则和提示词不知道后果 |

非目标（README 已写）：不是给统计学家的估计库（DoWhy、EconML 是后端）；不是因果发现包（causal-learn 的产物是我们的输入）；不是记忆层（Mem0、MemOS 是邻居）；不是 Neo4j 加因果插件。

v0.1 另外三条不做：不做因果发现；不做跨视图联合识别（论文 P1 / P2，mediator 的事）；不做十亿边规模（FCM / FDCut / TESSERA 索引留插件接口，v0.1 不接）。

## 2. 参考系统：借什么、不借什么

### 2.1 LOTUS 逐项对位（你点名的）

LOTUS 现在的自我介绍是"用 agent 和 LLM 批处理数据集，精度更高、成本更低"；核心是语义算子（`sem_filter / sem_map / sem_join / sem_agg / sem_extract`，新版加了在 `Corpus` 上跑的 agentic map / filter / reduce）加一个对用户透明的优化器（批处理、模型级联与代理模型、整条管线惰性规划），论文里的保证是"相对一个 gold algorithm 的精度保证"。

| LOTUS 的设计决策 | GoWhy 对应的决策 |
|---|---|
| 少量声明式算子，用户说要什么，不说怎么算 | 五个因果算子：`what_if / identify / counterfactual / attribute / explain`。用户不选估计量、不写调整集 |
| 算子直接长在 DataFrame 上（`df.sem_filter(...)`），不要求先建库 | 两条路都给：`db.what_if(...)`（有库），`df.causal.what_if(graph=...)`（一个 DataFrame 加一张图就能问） |
| 优化器把贵路径（大模型全量跑）换成便宜路径（代理模型、级联），保证相对 gold 的精度 | **gold = 干预（重放或随机实验），便宜路径 = 日志上的识别公式。** planner 能用日志就不重放；证书说明等价成立的假设；可抽少量重放检验便宜路径（N1 命题 6），这一步对应 LOTUS 抽样跑 gold 来定级联阈值 |
| 保证带失败概率 δ | 每个数带 CI；重放条数按 (ε, δ) 给：R₀ = ⌈ln(2/δ) / (2ε²)⌉ |
| 惰性规划整条管线再执行 | 查询先出计划和证书（`EXPLAIN`），再执行；`identify` 就是只规划不执行 |
| `lotus.settings.configure(lm=...)` 一处配置 | `gowhy.connect(path)` 一处入口；LLM 不是内核依赖，只在边缘（v0.2 的 NL→查询、抽边） |
| 仓库：`lotus/ examples/ benchmarks/ docs/ tests/`，`examples/op_examples/` 一算子一例；README 一屏跑通 | 同构：`gowhy/ examples/ bench/ docs/ tests/`，一算子一例；README 三分钟 quickstart |
| 新版把 agent 放到第一屏 | 我们的第一用户就是 agent：MCP 是主入口，不是附属 |

不借的：LOTUS 的执行内核是 LLM 调用，结果是近似的；我们的执行内核是识别加估计，答不出时必须明说答不出。这是两个系统最根本的区别，也是 README 里"LLM 会编"那句话的依据。

### 2.2 其它数据库

| 系统 | 借的设计 | 落到 GoWhy 哪里 |
|---|---|---|
| DuckDB | 进程内嵌入、单文件、零配置；`connect()` + `sql()`；直接扫 DataFrame / Parquet / 目录；`EXPLAIN` | `gowhy.connect("x.gowhy")`、`db.sql("WHAT IF ...")`；`analyze traces/` 直接扫目录不用先入库 |
| SQLite | 标准库自带的可靠单文件存储 | v0.1 的存储后端，零额外依赖 |
| Datomic、Dolt | 事实不可变，只追加；任意历史时刻可查 | 行带 `(v_from, v_to)`；`AS OF VERSION v`；删除 = 撤回（SPEC-DB §5） |
| Materialize、DBSP、Noria | 派生状态带依赖，源变则增量维护 | Effect / Certificate 带依赖集，写入时按因果结构失效（SPEC-DB I3；论文 P3） |
| Neo4j、Kùzu、GQL 标准 | 属性图模型；`MATCH` 模式语法 | 查询语言的外形；`MATCH` 在我们这里是**单位表的定义**（§4），不是另一套引擎 |
| Mem0、MemOS、Chroma | MCP 一行接入；三行 quickstart 的传播方式 | `{"command": "gowhy", "args": ["serve", "--mcp"]}`；不借记忆语义 |
| OpenTelemetry GenAI、LangSmith、Langfuse | 轨迹的事实标准格式 | `ingest` 的输入格式（SPEC-TRACES §6） |

### 2.3 先例与差异

| 先例 | 它做了什么 | 我们多出来的 |
|---|---|---|
| DoWhy / PyWhy | model → identify → estimate → refute 四步，库 | identify 是强制闸门不是可选步骤；结果可存、可版本化、可失效；refute 变成 planner 的自动 cross-check；可作估计后端 |
| HypeR（SIGMOD'22）、CaRL（SIGMOD'20） | 关系库上的 what-if / how-to 查询语法与因果语义 | 证书、部分识别与"缺什么"、provenance 与版本、轨迹模板、agent 入口。写 related work 时要逐条核对原文 |
| CausalGQL（自家） | BMIF 片段、可识别性感知优化器的设计 | 设计思路全部继承；**代码没有可继承的**（见 §11 Q1） |

### 2.4 由此定下的六条原则

1. **identify 是闸门。** 任何因果查询先出证书再执行；不可识别时不返回点估计。
2. **答案是四元组。** `{value, interval, certificate, explain}`，不允许裸数（SPEC-DB §2）。
3. **事实只追加，派生可失效。** 边、证据、数据行不可变带版本；效应和证书带依赖集。
4. **算子少，内核通用。** 五个算子共用一条 identify → plan → estimate 管线；轨迹分析不是特例代码，是通用引擎加一张模板图（§7.4）。
5. **嵌入式优先。** 一个文件、一个进程、两个依赖（numpy、pandas）；服务化是以后的事。
6. **日志优先，干预兜底。** 能从日志算的不重放；必须干预的，说清楚重放哪一臂、多少条。

## 3. 架构

```text
入口      Python 算子 API        GoWhy QL (db.sql)        CLI            MCP server (stdio)
            │                        │                    │                 │
            └────────────┬───────────┴──────────┬─────────┴─────────────────┘
                         ▼                      ▼
内核              ┌────────────┐   ┌──────────────────────────────────────────┐
                  │  parser    │──▶│ identify：图手术 → back-door → front-door  │
                  └────────────┘   │           → ID/IDC（完备）→ 支撑检查 → 界  │──▶ Certificate
                                   ├──────────────────────────────────────────┤
                                   │ planner：枚举（视图 × 策略），按证据信任与  │
                                   │          精度排序，cross-check             │
                                   ├──────────────────────────────────────────┤
                                   │ estimators：分层 / front-door / 通用插入 / │
                                   │             界；bootstrap CI              │──▶ Effect
                                   └──────────────────────────────────────────┘
                         ▲                      ▲
资产              ┌────────────────────────────────────────────────────────────┐
                  │ store（SQLite 单文件）：Variable · Edge · Evidence · View ·  │
                  │ Episode · Version ｜ 派生：Effect · Certificate · Query log │
                  └────────────────────────────────────────────────────────────┘
                         ▲
写入              register_view(DataFrame / 文件) · ingest traces/（模板 + 行） · assert / retract edge
```

一次查询的生命周期：

1. 解析成 `CausalQuery{kind, X=x, baseline, Y, W=w, view?, as_of?}`。
2. 查缓存：规范化查询 + 当前版本，Effect 为 fresh 则直接返回。
3. 对每个记录了 X、Y、W 的视图跑 identify，得到候选计划与证书。
4. planner 选主计划；没有可识别计划则走界。
5. 估计，bootstrap 出 CI，做数据层支撑检查。
6. 写 Query log（id、文本、版本、种子、证书、explain），写 Effect（带依赖集）。
7. 返回四元组。

## 4. 数据模型：三层

| 层 | 是什么 | 例子 | v0.1 |
|---|---|---|---|
| 模式层因果图 | 变量与因果边，边带证据与状态 | `discount → retained`，证据 expert；`frustration` 未记录 | 做 |
| 单位表（视图） | 一行一个单位，列是变量的取值；视图声明哪些列是随机化的 | `orders_log`（观测）、`discount_ab`（随机化 discount）、一个 agent 版本的轨迹 | 做 |
| 实例属性图 | 节点、边、属性；`MATCH` 模式把它拉平成单位表 | `(c:Campaign)-[:TARGETS]->(u:User)` | v0.2 |

两点说明：

- **变量"是否隐"是相对视图的。** 变量三态：视图里有列（已记录）、图里有但视图里没列（未记录，可建议去记）、标 `latent`（原则上测不到，如隐难度 U）。`identify` 的"缺什么"只在第二类里找。
- **`MATCH` 是视图定义。** 属性图上的模式定义"单位是什么"并把属性拉成列，之后的 identify / plan / estimate 完全相同。所以 v0.1 不做 `MATCH` 不影响内核，v0.2 加上也不改内核。

表结构、冲突写入、撤回规则见 SPEC-DB §1、§3–§5。v0.1 实现其中：Variable、Edge、Evidence、View、Episode、Version、Effect、Certificate；不实现 Mechanism 表与 IndexArtifact。

## 5. 结果契约

所有因果算子返回同一个结构：

```json
{
  "query_id": "q-000042",
  "version": 17,
  "estimand": "E[retained | do(discount=1), segment=B] − E[retained | do(discount=0), segment=B]",
  "value": 0.031,
  "interval": [0.012, 0.049],
  "certificate": {
    "status": "identifiable",
    "strategy": "backdoor",
    "formula": "Σ_z P(y | x, z, w) P(z | w)",
    "adjustment_set": ["loyalty"],
    "view": "orders_log",
    "evidence": "observational",
    "assumptions": ["图在版本 17 下成立", "loyalty 之外没有 discount 与 retained 的未记录共因", "各层内两种取值都出现过"],
    "missing": [],
    "complete": true
  },
  "explain": {"estimator": "stratified", "n": 60000, "seed": 0, "edges_used": ["..."], "alternatives": ["..."], "warnings": []}
}
```

状态的定义（这是一个要你确认的口径，§11 Q5）：

| status | 含义 | value | interval |
|---|---|---|---|
| `identifiable` | 图上可识别且数据有支撑 | 点估计 | 置信区间 |
| `partial` | 识别集是一个比无假设界更窄的区间（部分人群无支撑、PN 界、调用效应的 [τ−1, τ]） | 空 | 识别区间（端点带抽样误差） |
| `not` | 没有比无假设界更好的结论，或没有视图同时记录 X 与 Y，或依赖未仲裁的争议边 | 空 | 无假设界（若结果有界） |

`missing` 的四种动作：`log`（该多记哪些变量）、`experiment`（随机化谁、每臂多少条）、`replay`（重放哪一臂、多少条）、`arbitrate`（哪条争议边要裁决）。

## 6. 三个入口，一个内核

### 6.1 Python 算子 API

```python
import gowhy

db = gowhy.connect("retail.gowhy")                      # 或 ":memory:"

db.edge("loyalty", "discount", evidence=gowhy.Evidence("expert", "定价组口述"))
db.edge("discount", "retained", evidence=gowhy.Evidence("experiment", "ab-2026-07"))
db.register_view("orders_log", df, kind="observational")
db.register_view("discount_ab", df_ab, kind="experiment", randomized=["discount"])

r = db.what_if({"discount": 1}, outcomes=["retained", "refund"], where={"segment": "B"})
r["retained"].value, r["retained"].interval, r["retained"].certificate.status
print(r.explain())

db.identify("support_call", "retained")                 # 只判不算
db.counterfactual(unit={...}, set={"discount": 0}, outcome="retained")
db.attribute("correct", to=["skill:cite-first", "skill:long-context"])
db.as_of(version=12).what_if(...)                       # 时间旅行
```

LOTUS 式免建库用法：

```python
import gowhy.pandas
df.causal.what_if(graph="loyalty->discount; loyalty->retained; discount->retained",
                  set={"discount": 1}, outcome="retained")
```

### 6.2 GoWhy QL（v0.1 子集）

```text
WHAT IF SET discount = 1 [VS SET discount = 0 | VS STATUS QUO]
  RETURN effect(retained), effect(refund) [WHERE segment = 'B'] [USING view] [AS OF VERSION 12]

IDENTIFY effect(retained | do(support_call)) [WHERE ...] [USING view]

COUNTERFACTUAL EPISODE 'ep-000123' SET depth = 10 RETURN correct
COUNTERFACTUAL GIVEN discount = 1, retained = 0, segment = 'B' SET discount = 0 RETURN retained

WHY retained = 0 BECAUSE support_call = 1 [WHERE ...]          -- PN 区间

ATTRIBUTE correct TO `skill:cite-first`, `skill:long-context` [USING view]

EXPLAIN q-000042
SHOW VARIABLES | EDGES | VIEWS | EFFECTS | VERSIONS
ASSERT EDGE a -> b EVIDENCE expert 'reason'
RETRACT EDGE a -> b 'reason'
```

与 SPEC-DB §2 的差：v0.1 没有 `MATCH`、`PRESCRIBE`、`SWITCH_RISK`；`WHERE` 只能写处理变量的非后代（处理前协变量）。不写 `VS` 时的默认基线：处理变量只有两个取值就取另一个，否则取现状（观测策略下的 E[Y]）。

### 6.3 CLI

```bash
gowhy init --example retail            # 生成 gowhy.db：零售图 + 观测日志 + 一个小 A/B
gowhy sql "WHAT IF SET discount = 1 RETURN effect(retained), effect(refund) WHERE segment = 'B'"
gowhy analyze traces/ --tool retriever --treatment config.depth --mediator output.gold_hits --outcome outcome.correct
gowhy ingest traces/ --agent rag-react@1.4.2
gowhy serve --mcp
gowhy demo                             # 先问再动的脚本化 agent
```

`analyze` 无状态（内存库，扫完即走）；`ingest` 才落库。README 里现在写的 `--treatment retrieval_depth --mediator retrieved_docs` 是变量名，实现里需要的是 JSON 路径，README 要跟着改，或者做自动探测（单工具单配置键时不用给路径）。

### 6.4 MCP：五个 tool

| tool | 输入 | 输出 |
|---|---|---|
| `what_if` | `set{var: value}`、`outcomes[]`、`baseline?`、`where?` | 每个结果一行：效应、区间、状态、依据；`query_id` |
| `identify` | `treatment`、`outcome`、`where?` | 状态、策略或原因、`missing[]` |
| `counterfactual` | `unit{...}` 或 `episode`、`set{...}`、`outcome` | 反事实结果的概率区间与假设 |
| `attribute` | `outcome`、`components[]`、`view?` | 每个组件的干预边际、区间、状态 |
| `explain` | `query_id?` | 估计目标、识别路径、假设、证据、版本；不带参数时返回库概览（有哪些变量、视图、能问什么） |

设计决定：

- **server 用标准库手写 stdio JSON-RPC**（`initialize / tools/list / tools/call`），不依赖 mcp SDK。`pip install gowhy` 只带 numpy 和 pandas，一行配置就能接。
- **返回体两份**：一段给模型读的短文本，加 `structuredContent` 里的完整四元组。短文本的样子：

  ```text
  WHAT IF discount=1 (vs discount=0) WHERE segment=B
    retained  +0.031  [95% CI +0.012, +0.049]  identifiable · back-door {loyalty} · orders_log n=24k
    refund    +0.052  [95% CI +0.041, +0.063]  identifiable · back-door {} · orders_log n=24k
  query_id q-000042 · 用 explain 看识别路径与假设
  ```
- **agent 只读。** v0.1 的 MCP 不暴露写入（断言边、入库）。谁能写是 vision paper 的 C5，先不让 agent 改自己的世界模型。
- **不可识别时 tool 不报错**，正常返回 `status=not` 和 `missing`。agent 要学会的正是"这个问题现在答不了"。

## 7. 内核设计

### 7.1 identify：判定流程

输入：查询 (X=x, Y, W=w)、版本 v、一个视图。输出：Certificate。

| 步 | 做什么 | 结果 |
|---|---|---|
| 0 | 合法性：变量存在；W 都是 X 的非后代；相关祖先集里没有未仲裁的争议边 | 否则 `not` + 原因 |
| 1 | 取图：版本 v 的 supported 边；视图未记录的变量当隐变量；视图里随机化的变量删掉入边 | 工作图 G |
| 2 | Y 不是 X 的后代 | `identifiable`，效应恒为 0（no-causal-path） |
| 3 | back-door：候选 = 已记录的 X 非后代。构造式判定：若存在合法调整集，则 An(X∪Y∪W) ∩ 候选 就是一个（van der Zander 等）；再贪心删到极小 | `strategy=backdoor`（X 全被随机化时调整集为空，标 `randomized`） |
| 4 | front-door（单处理）：在 X→Y 有向路径上的已记录变量里枚举中介集 M（规模 ≤ 3），协变量 C ⊇ W。三个条件：M 截断全部有向路径；G 去掉 X 出边后 X ⊥ M \| C；G 去掉 M 出边后 M ⊥ Y \| X, C | `strategy=frontdoor`，公式 Σ_c P(c\|w) Σ_m P(m\|x,c) Σ_x′ P(x′\|c) E[Y\|m,x′,c] |
| 5 | ID / IDC（Shpitser–Pearl）：把 G 做隐投影得到 ADMG，跑完备算法 | 可识别 → 符号公式；不可识别 → hedge |
| 6 | 数据层支撑检查：公式里每个条件项在数据里有没有支撑 | 缺口质量 g > 0 → 降为 `partial`，区间宽 g·range(Y) |
| 7 | 不可识别：结果有界则给无假设界（Manski）；算 `missing` | `partial` 或 `not` |

三点设计理由：

- **为什么要第 5 步。** back-door 加 front-door 不完备（napkin 图两者都不适用但可识别）。没有完备算法兜底，"不可识别"这个判词就可能是错的，而这个判词正是卖点 2。`certificate.complete` 标明判词是否由完备算法给出。完备性的范围是"单个视图内"；多个视图合起来能否识别是 gID，属于论文 P1，v0.1 不做，证书里写明。
- **为什么要第 6 步。** 图判据看不见结构零。例：没调工具时工具输出恒为空，图上 front-door 成立，数据上内层求和有一整块没有支撑。N1 定理 2 的现象一半来自图（见 §7.4），一半来自支撑，所以支撑检查必须在内核里，不能留给估计器报错。
- **`missing` 就是 M1 的单查询版。** `log`：在"图里有、视图没记、非 latent"的变量上按规模 1→3 枚举最小集合，使其视为已记录后第 3–5 步通过。`experiment`：随机化 X，二值结果每臂 n = ⌈2z²σ̂²/ε²⌉。多查询工作负载的最小记录集（hitting set 贪心）是 v0.2，也就是 M1 论文本体。

### 7.2 planner

- 候选：每个记录了 X、Y、W 的视图，各出一个（策略，证书）。
- 排序：状态（identifiable > partial > not）→ 证据信任（experiment ≈ replay > observational）→ 预期精度（最小支撑格的样本数）。
- cross-check：其余可识别计划也估一遍，写进 `explain.alternatives`。两个计划的区间不相交就告警：图或"视图人群等于目标人群"的假设被数据反驳。这是 DoWhy 的 refute 变成默认行为。
- v0.1 没有代价模型。v0.2 把重放成本 c 放进来：N1 命题 3 的交叉点（有效日志数 N₊ρ_min 对重放预算 R）决定"日志算还是重放"；FCM / FDCut / TESSERA 作为物理算子进入计划空间，接口沿用 CWS-PROJECT-MODEL §4 的 `build / supports / answer / maintain`。

### 7.3 estimators

| 估计器 | 何时用 | 备注 |
|---|---|---|
| 分层 back-door | 调整集离散 | 无支撑的层自动转成区间 |
| 条件 front-door 插入 | 第 4 步通过，M、C 离散 | 无支撑的格自动转成区间 |
| ID 通用插入 | 只有第 5 步通过 | 在经验联合分布上对符号公式求值；求和规模超阈值则拒绝并说明 |
| 线性回归调整 | 处理变量连续 | 证书里加"线性结果模型"假设 |
| 无假设界 | 不可识别且结果有界 | E[Y·1(X=x)] + P(X≠x)·[y_lo, y_hi] |
| PN / 反事实界 | `why`、`counterfactual` | 见 §7.5 |

统一做法：连续协变量按分位数分箱并在证书里注明；CI 用非参数 bootstrap（默认 200 次，种子入 Query log，满足 I5 可复现）；区间型结果的 CI 取下端点的下分位与上端点的上分位。v0.1 不做 DML / EIF（KDD 框架要求的那套），留给 v0.2 或直接接 EconML 后端（§11 Q6）。

### 7.4 traces：把 N1 做成通用引擎的一个模板

轨迹模板把"调不调"和"用什么配置"拆成两个变量（SPEC-TRACES §2 里是一个 `T_k`，这里细化）：

```text
X（任务上下文，含查询串） → called, config, M, Y
U（隐难度，latent）       → called, config, Y
called → config, M        config → M        M → Y
```

在这张图上，N1 的两个定理不需要特判代码，直接从通用 identify 掉出来（我手推核对过 d-分离条件，实现时用模拟数据再核一遍）：

- **定理 1（配置效应可识别）**：查询 `WHAT IF SET config = t WHERE called = 1`。条件 front-door 以 M 为中介、C = {X, called} 通过；公式恰为 Σ_x P(x|called) Σ_m P(m|t,x,called) Σ_t′ P(t′|x,called) E[Y|m,t′,x,called]，即定理 1。
- **定理 2（调用效应不可识别）**：查询 `IDENTIFY effect(Y | do(called))`。back-door 被 U 挡住；front-door 第二个条件被 called ← U → config → M 打破；ID 算法在 {called, config, Y} 这个 c-component 上返回 hedge。若模板里没有 config（单配置工具），图上 front-door 成立，但第 6 步支撑检查发现 M 在 called=0 时恒为空，同样降为 `partial`。

`analyze` 在通用引擎之上只加四样轨迹专属的东西：

1. 最小日志契约检查（SPEC-TRACES §6）。没记工具输出 → 配置对比 `not`，`missing: log 工具输出`。
2. naive 日志对比并排列出，标注"混杂，不是因果量"。
3. 调用效应的证书：Δ ∈ [τ₊(t) − 1, τ₊(t)]，`missing: replay 不调工具那一臂 R₀ 条`，按 ε ∈ {0.1, 0.05, 0.02} 给 R₀。
4. 若日志里已有重放行（`replay.of` 指向原 episode）：不调工具臂的重放给 β̂，Δ̂ = τ̂₊ − β̂（混合估计）；同配置重放与日志估计做双样本检验（命题 6，证伪 A1）。

另外报一个"等效重放数" R_eq = τ̂(1−τ̂)/se²：日志给出的精度相当于多少次重放，这是"不用重跑"省了多少的直接度量。

多步轨迹（定理 4，A3′）、组件变量的 group testing（N3）不进 v0.1。

### 7.5 counterfactual、why、attribute 的语义

- **`attribute`** = 对每个组件 S_j 发一条 `WHAT IF SET S_j = 1 VS S_j = 0`，各带各的证书。组件是随机化注入的 → `identifiable`；是关键词触发的 → 被隐难度混杂，`not` + `missing: experiment`。这就是"归因即查询"（桥 A）的最小实现，也直接演示 N4 的"日志效应可以反号"。不做 Shapley、不做定价。
- **`why`**（Q1 的最小实现）：二值结果，问"若当时不做 x，y 还会发生吗"。返回必要性概率 PN 的 Tian–Pearl 区间：下界 max(0, [P(y) − P(y_x′)] / P(x, y))，上界 min(1, [P(y′_x′) − P(x′, y′)] / P(x, y))，其中 P(y_x′) 走 identify。P(y_x′) 自己只有区间时，把区间带进去。可选假设 `monotone` 时取下界为点值，并写进证书。
- **`counterfactual`**：单位级反事实。二值结果返回 P(Y_x′ = 1 | 该单位的事实) 的区间（与 PN 同一套界，按单位的处理前协变量取子人群）；若该 episode 有匹配的重放行，直接返回重放结果，证据类型 replay。v0.1 没有 Mechanism 表，不做 abduction–action–prediction；连续结果的反事实推到 v0.2。

这三个算子的共同点：单位级的量一般只有界。系统的做法是给界，并说清什么额外假设或多少次重放能把界收成点。

### 7.6 派生状态与失效

- Effect 的键 = 规范化查询 + 视图；依赖集 = An(X ∪ Y ∪ W) 的变量集 + 所用视图。
- 写入时：新增或撤回的边，终点落在某 Effect 的依赖变量集里 → 该 Effect 置 `stale`（撤回边时置 `unknown`，重判）；视图数据变 → 依赖该视图的 Effect 置 `stale`。
- v0.1 是 lazy：下次查询时重算。增量重算是论文 P3。

### 7.7 存储

- 单文件 SQLite，扩展名 `.gowhy`。表：`variables, edges, evidence, edge_evidence, views, rows, versions, queries, effects`。
- 所有事实行不可变，带 `(v_from, v_to)`；状态变化 = 关旧行开新行；`AS OF VERSION v` = 过滤 `v_from ≤ v < v_to`。
- 数据行 v0.1 存成 JSON 负载。量级约束：十万行级没问题；百万行以上换成"视图指向外部 Parquet / 目录"（DuckDB 的做法），那时再把 DuckDB 作为可选扫描后端。
- 冲突写入按 SPEC-DB §3 的 trust 偏序。实现上的一个细化：同级互斥的两条反向边都 contested 时，凡相关祖先集含这对变量的查询一律返回 `not` + `missing: arbitrate`，不猜方向。

## 8. 三个验收演示

每个演示都用真值已知的模拟数据，所以"对不对"可以自动检验。

| # | 演示 | 数据 | 通过标准 |
|---|---|---|---|
| D1 | `gowhy analyze traces/` | N1 的 SCM 生成的 RAG 轨迹：depth ∈ {3, 10}、不调工具一臂、隐难度影响调用与深度 | front-door 的区间盖住真值而 naive 对比不盖；调用效应返回 `partial` 加 R₀；删掉工具输出字段重跑返回 `not` 加"记工具输出" |
| D2 | `identify` | 零售图：`support_call → retained` 被未记录的 `frustration` 混杂 | 返回不可识别、无假设界、`missing` 里同时有"记 frustration"和"随机化 support_call 每臂 n 条"；把 frustration 加进视图后同一查询变 `identifiable` |
| D3 | MCP 先问再动 | 零售图：discount 对 segment B 的真效应为留存 +3 点、退款 +5 点；日志里折扣多发给低忠诚客户，naive 留存效应为负 | 脚本化 agent 经真实 stdio MCP 调 `what_if`，拿到"留存 +3、退款 +5"，按净值规则改用 free_shipping；同一 server 在 Claude Code 里配一行能调通 |

D3 的 agent 是脚本，不需要 LLM key，CI 里能跑；真 LLM 的版本是 L2 论文的实验，不是系统验收。

## 9. v0.1 范围与实现

| 做 | 不做（下一版） |
|---|---|
| store：变量、边、证据、视图、行、版本、AS OF、撤回、冲突写入 | Mechanism 表、IndexArtifact、物理清除 |
| identify：back-door、条件 front-door、ID / IDC、支撑检查、界、`missing` | 跨视图联合识别（gID）、工具变量、可搬运性 |
| 估计：分层、front-door、通用插入、线性回退、bootstrap | DML / EIF、连续中介、高维上下文 |
| 算子：what_if、identify、counterfactual（界）、why（PN）、attribute、explain | prescribe、switch_risk、多步轨迹、group testing |
| 入口：Python、QL 子集、CLI、MCP（只读） | `MATCH`、NL→查询、HTTP 服务 |
| 轨迹：JSONL 适配、模板、analyze、ingest、重放融合 | OTel / LangGraph / Claude Code 适配器 |
| 例子：retail、rag-traces；三个演示 | CWS-Bench 接入、FCM / FDCut / TESSERA 插件 |

实现：Python ≥ 3.9，依赖 numpy、pandas，估计 3000 行左右加测试。仓库结构：

```text
gowhy/
  README.md  DESIGN.md  SPEC-DB.md  SPEC-TRACES.md  pyproject.toml
  gowhy/
    graph.py        DAG、祖先 / 后代、图手术、d-分离
    idalg.py        隐投影、ADMG、ID / IDC、符号公式与求值
    identify.py     判定流程、Certificate、missing
    estimate.py     估计器、bootstrap、界
    planner.py      候选枚举、排序、cross-check
    store.py        SQLite、版本、不变量、失效
    db.py           Database：五个算子 + 写入 API
    ql.py           GoWhy QL 解析与执行
    traces.py       适配、模板、analyze、ingest
    mcp_server.py   stdio JSON-RPC
    cli.py          init / sql / analyze / ingest / serve / demo
    pandas.py       df.causal 访问器
    examples/       retail、rag_traces 的模拟器（带真值）
  examples/         一算子一例 + consult_before_act
  tests/  bench/  docs/
```

开工顺序（每一步有可跑的测试才进下一步）：

1. `graph` + `idalg` + `identify`：纯图逻辑，单元用例定死。
2. `estimate`：对模拟 SCM 的真值做数值检验。
3. `store` + `db` + `planner`：D2 通过。
4. `ql` + `cli`。
5. `traces`：D1 通过。
6. `mcp_server` + demo：D3 通过。
7. README 按实际能跑的命令重写。

## 10. 测试

- **图判定用例**：bow 图（不可识别）、front-door 图、napkin 图（可识别且 back-door、front-door 都不适用）、IV 图（不可识别）、M-bias（不得调整对撞点）、随机化视图（调整集为空）。
- **数值真值**：每个估计器对模拟 SCM 检验，估计落在真值的 3 个标准误内；CI 覆盖率抽查。
- **N1 复核**：复用 `campaigns/replay-free/verify/verify_frontdoor.py` 的 SCM，通用引擎的输出要与那份脚本的定理 1 数值一致；定理 2 的两个 SCM 在引擎里得到相同的证书。
- **store 不变量**：I1 无环、I2 无证据拒收、I4 低信任不覆盖高信任、I5 同版本同种子复现、AS OF 回放、失效传播。
- **入口**：QL 解析往返；MCP 以子进程方式做 initialize → tools/list → tools/call 全流程。

## 11. 未决问题（要你定）

| # | 问题 | 我的建议 | 理由 |
|---|---|---|---|
| Q1 | 实现语言 | v0.1 纯 Python | 我查了 `~/Project/CausalGQL`：8 个 crate 里 Rust 源码合计 38 行，全是占位。CWS-PROJECT-MODEL 里"store 来自 CausalGQL 的 Rust 核心"目前没有可搬的东西。Python 先把语义和接口定死，Rust 留给规模化的索引插件 |
| Q2 | 名字 | 包名 `gowhy`，学术名 CWS | README 已是 GoWhy；你贴的对比表最后一列写的是 cws，需要统一 |
| Q3 | `MATCH` 进不进 v0.1 | 不进 | 按 §4，它只是单位表的定义，不影响内核；但 Go-Y 的身份是属性图，demo paper 之前必须补上 |
| Q4 | 存储后端 | SQLite | 零依赖；DuckDB 等数据量上来再作可选扫描后端 |
| Q5 | `partial` 与 `not` 的口径 | 按 §5 的表 | README 说调用效应"not identifiable"，SPEC-TRACES 说返回 `partial [τ−1, τ]`，两处要统一成一个定义 |
| Q6 | 估计后端 | v0.1 自带插入估计 + bootstrap；DML 走 v0.2 的 EconML 可选后端 | N-系列论文要 EIF / DML，但系统 v0.1 先保证判定正确 |
| Q7 | 高维工具输出怎么变成 M | 用户给摘要函数（命中数、长度、分位），系统只做分箱 | 摘要选得不好会破坏 A2（T 对 Y 的作用全经 M）；自动选摘要是一个研究问题，不放进 v0.1 |
| Q8 | MCP 是否只读 | 只读 | C5：先不让 agent 改自己的世界模型 |
| Q10 | 世界模型先支持哪一类（§15） | 先做语言世界模型（工具响应）和自家仿真器；感知型不排期 | 前两类和 N1、MicroWorld 直接接上；感知型要先解决变量抽取 |
| Q9 | 仓库位置 | 现在的 `AutonomousMath/gowhy/` 只适合放设计稿；开工时挪到 `~/Project/gowhy` 并 `git init` | AutonomousMath 是数学流水线的仓库，且不是 git 仓库 |

## 12. 与论文的挂接

| 模块 | 支撑的论文 | 论文反过来给系统什么 |
|---|---|---|
| `traces` + `analyze` | N1 Replay-Free | 定理 1、2、命题 3、6 |
| `identify` 的 `missing` | M1 Causal Logging Design | 多查询最小记录集算法 |
| MCP + demo | L2 Consult Before You Act | 何时该查的 VoI 策略 |
| `attribute` | P7 Attribution as a Causal Query、N4 Causal UpSkill | 组件效应表、认证流程 |
| `why` | Q1 Probability of Causation for Incidents | 重放收紧 PN 界的样本量 |
| planner + 索引插件 | cws 系统论文（demo 先行）、P3 | 代价模型、增量维护 |

## 13. 怎么接到 agent 上：三档

agent 通过 MCP 调 GoWhy。接法分三档，约束力依次变强：

| 档 | 做法 | agent 不配合时 | 需要改谁 |
|---|---|---|---|
| 1 工具 | MCP 一行配置，agent 多出 `what_if / identify / explain` | 可以不调 | 谁都不用改 |
| 2 指令 | server 的 `instructions` 与工具描述写明"改变世界的动作之前先问"；或写进 agent 的系统提示 / skill | 可能忘 | agent 的提示 |
| 3 闸门 | 动作工具多一个必填参数 `query_id`。执行前检查：这个查询问的是同一个动作和同一个人群，且每个结果都可识别；否则拒绝 | 做不了 | 动作工具的包装层 |

第 3 档的两个副产品：

- **审计。** 每个动作都挂着授权它的那条查询，`explain(query_id)` 能重建当时的估计目标、假设、证据和版本。这就是 vision paper C5 里"授权过行动的反事实事后可审计"。
- **事后对账。** 动作执行后把真实结果写回，和当时的预测区间比。长期落在区间外，说明图或数据漂了（论文 P3、M8 的触发条件）。

闸门不在 GoWhy 内核里，它是动作工具一侧的一个薄包装：GoWhy 只需要提供"按 `query_id` 查证书"的接口。v0.1 给一个 Python 装饰器和一个 MCP 代理两种包装方式。

L2（Consult Before You Act）研究的是第 2 档里"哪些步值得问"（信息价值对查询代价）；第 3 档是工程上的兜底，两者不冲突：闸门只装在后果重大的动作上。

## 14. 第一步：`demo/`（已完成，2026-10-01）

在动 v0.1 之前先做了一个第 3 档的最小 demo，目的是验证三件事：手写的 MCP server 能通；identify 闸门加"缺什么"的交互读起来成立；估计对真值是对的。

- 位置：`demo/`，说明见 `demo/README.md`。
- 内容：真值已知的零售世界；最小引擎（back-door 估计，ID 算法判不可识别，无假设界，`missing`）；MCP server（`what_if / identify / explain` 加带闸门的 `apply_offer`）；脚本化 agent。
- 结果：13 个测试通过。打折在 B 段的估计为留存 +0.030、退款 +0.050（真值 +0.03、+0.05），日志直接对比的留存效应为 −0.069；回访动作被判不可识别并给出两条补救；记录 `frustration` 后同一查询变可识别。
- 没做的：§8 的 D1（轨迹分析）、存储与版本、查询语言、front-door 与通用估计、`counterfactual` 与 `attribute`。未在 Claude Code 里实测。

demo 的 `engine.py` 是 v0.1 `graph / idalg / identify / estimate` 四个模块的雏形，接口按本文 §5 的结果契约写的，开工时拆开即可。

## 15. 世界模型（2026-10-01 加，设计，未实现）

### 15.1 说的是哪种世界模型

| 类 | 例子 | 它给 agent 什么 | 和 GoWhy 的关系 |
|---|---|---|---|
| 语言世界模型（agent 用的） | Qwen-AgentWorld、MCP-Cosmos、WorldEvolver、ADWM | 行动前预测工具返回 / 环境响应，或离线评估新策略 | **主战场**：和 `what_if` 是同一个卖点，差一张证书 |
| 仿真器 | 自家的 MicroWorld、MiroFish（"上帝视角注入变量"） | 一个可以随便干预的平行世界 | 仿真里的 do() 是精确的；缺的是"仿真等于现实吗" |
| 感知型世界模型 | OpenWorldLib 收的那些：交互视频生成、3D、VLA | 像素 / 具身层面的下一状态 | 暂不直接支持；要先有"从感知状态抽出因果变量"这一步（vision paper 的 C3） |

### 15.2 一个判断，三个定位

世界模型从日志轨迹学到的是 p(o | s, a)。agent 规划时需要的是 p(o | s, do(a))。两者相等的条件是：行为策略选 a 时看到的东西都记在 s 里，且这个 (s, a) 在训练数据里出现过。前一条是可识别性，后一条是支撑，正好是 `identify` 的第 3–6 步。集成模型之间的分歧只能量出估计方差，量不出混杂偏差：所有成员会一致地给出同一个错的答案。

由此 GoWhy 对世界模型有三个定位：

1. **GoWhy 自己就是一个带证书的世界模型。** 结构化、小而可审计；别的世界模型告诉 agent 会发生什么，GoWhy 多告诉它这句话凭什么成立、什么时候不成立。
2. **GoWhy 是学出来的世界模型外面那层信任层。** 逐条预测打标签：可识别且有支撑 / 无支撑（外推）/ 被混杂。
3. **世界模型是 GoWhy 的一档证据和一种估计器。** 便宜但有偏的"想象中的重放"，用少量真实重放校正。

### 15.3 接口

**A. 作为世界模型被调用**（定位 1）。在五个算子之上加一层薄封装，不加新内核：

```python
db.predict(state={"segment": "B", "loyalty": 0}, action={"discount": 1}, outcomes=["retained", "refund"])
db.best_action(state=..., candidates=[{"discount": 1}, {"free_shipping": 1}], objective=...)   # = 处方查询
db.rollout(state=..., plan=[...], horizon=3)                                                  # 多步，依赖时序图，v0.3
```

`predict` 返回的是水平 E[Y | do(a), s]（规划要的是水平，不是对比），仍是四元组。不可识别的转移返回区间，规划器按区间做悲观规划，这一点接上离线 RL 的 pessimism，但不确定性来自识别而不是来自集成。

**B. 接一个外部世界模型**（定位 2、3）。GoWhy 不托管模型，只登记：

```python
db.register_world_model("agentworld", predict=fn, trained_on="tool_traces@1.4.2", inputs=["X", "T"], outputs=["M"])
```

登记后发生三件事：

- **打标签。** 对它的每类预测跑 identify（以 `trained_on` 视图的记录变量为准），证书写进 Mechanism。视图数据变了，这个 Mechanism 置 stale。
- **进计划空间。** planner 的证据从两档变三档：日志（免费，要过识别）、世界模型推演（便宜，有模型偏差）、真实重放或实验（贵，gold）。
- **用真实样本校正。** 推演当预测、真实重放当标注，走 prediction-powered 估计：θ̂ = (1/N) Σ_j f(s_j, a) + (1/n) Σ_i (Y_i − f(s_i, a))。只要那 n 条是目标人群的随机样本，不管 f 多差都无偏；方差约为 Var(f)/N + Var(Y − f)/n，f 预测得越准，需要的真实重放越少。证书里的证据类型写 `simulated, rectified with n real`。

**C. 反过来服务世界模型**。评测：拿可识别的效应或重放真值去量世界模型的"干预保真度"，和"观测保真度"分开报。采集：`missing` 给出的"在哪个状态区域随机化哪个动作"就是世界模型该补的数据。

对 SPEC-DB 的两处增量：Evidence.kind 加 `simulated`（信任度低于真实实验与重放，是否高于观测日志取决于是否校正过）；Mechanism.class 加 `world-model`（外部可调用，带 `trained_on`）。

### 15.4 和 N1 接在一起的两个推论（待写成命题并核对）

在 trace SCM（§7.4）里：

- **工具响应的世界模型可以从日志学，结果的世界模型不行。** A1 说 M 只依赖 (X, T, ε_M)，所以 p(m | x, t) 的观测条件分布就是干预分布，语言世界模型学"工具会返回什么"是可识别的。而 p(y | x, t) 被隐难度 U 混杂，直接学"这样做任务会不会成功"的世界模型是有偏的。这给现在一批"用世界模型做 agent 离线评估"的工作划了一条线。
- **世界模型正好补上 front-door 的第一因子。** 定理 1 的公式里 p(m | t, x) 在 M 是长文本时没法用计数估，这是 §11 Q7 的难处。语言世界模型就是这个因子的神经估计器：从它采样 m，再用结果模型算内层期望。所以 Q7 的答案可以从"用户给摘要函数"升级为"接一个工具响应世界模型"。
- 定理 2 说调用效应必须重放。有了 15.3-B 的校正，结论变成：必须重放，但世界模型能把重放条数按 Var(Y − f)/Var(Y) 的比例降下来。

### 15.5 落地顺序

| 步 | 内容 | 放哪版 |
|---|---|---|
| 1 | demo 加一幕：一个在日志上训的世界模型说"回访有害"，GoWhy 标出这条预测被混杂；500 条真实随机回访加世界模型推演，校正后得到 +0.04 | demo（下一步，可立刻做） |
| 2 | `predict / best_action` 封装；`register_world_model` 与三档证据；prediction-powered 校正 | v0.2 |
| 3 | 仿真器适配：MicroWorld / MiroFish 的一次运行入库为 `simulated` 视图，注入变量记为随机化；仿真效应与真实日志上可识别的效应做对账 | v0.2 |
| 4 | 工具响应世界模型作为 front-door 第一因子；`rollout` 多步 | v0.3 |
| 5 | 感知型世界模型 | 不排期，等 C3 |

### 15.6 邻居（2026-10-01 搜到的，未做完整查新）

- ADWM（arXiv 2606.05558）：用扩散世界模型做 LLM agent 的离线策略评估。摘要里没有处理日志混杂，是 15.4 第一条的直接对象。
- Learning Implicit Causal World Models from Multi-Agent Demonstrations（2607.26336）、BECAUSE（2407.10967）：在世界模型内部学因果结构。我们不学结构，做的是模型外面的识别证书与校正，方向不同，要引。
- MCP-Cosmos（2605.09131）、Qwen-AgentWorld（2606.24597）、WorldEvolver（2606.30639）：agent 行动前用世界模型预演。是 `what_if` 的对照组，也是 15.3-B 的接入对象。
- 理论锚点：Richens & Everitt（ICLR'24，稳健的 agent 必然学到近似因果模型）、Ortega 等（2021，序列模型把自己的动作当观测会产生 delusion）、prediction-powered inference（Angelopoulos 等，2023）。这三条是凭记忆写的，引用前核对。

若要单独成文，题目大致是"学出来的世界模型，哪些预测是因果的"：识别证书 + 少量重放的校正 + N1 的两个推论。开题前先查新。
