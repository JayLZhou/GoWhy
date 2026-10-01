# agent 轨迹怎么进 GoWhy（v1，2026-10-01）

核心做法：**每种 agent 一张"轨迹因果模板"（trace template），轨迹是模板上的行。** 模板是一个小因果图，节点是这类 agent 每步的决策、工具输出、结果、成本，外加一个隐难度；一条轨迹就是这些节点的一次取值。模板负责识别（哪些查询能答），轨迹负责估计（答多少）。

## 1. 一条轨迹长什么样（入库前）

以 OpenTelemetry 式 span 树或任意 JSONL 为输入，最少需要：

```json
{
  "episode_id": "ep-000123",
  "agent": {"name": "rag-react", "version": "1.4.2", "model": "qwen2.5-7b", "seed": 17},
  "task": {"id": "hotpot-8841", "prompt": "...", "class": "bridge-2hop", "meta": {...}},
  "steps": [
    {"k": 1, "kind": "tool", "tool": "retriever", "config": {"depth": 5},
     "input": {"query": "who founded ..."}, "output": {"docs": ["d17","d42","d9","d88","d3"], "gold_hits": 1},
     "latency_ms": 210, "cost": 0.0},
    {"k": 2, "kind": "llm", "input_ctx_tokens": 1830, "output": "The founder is ...", "tokens": 96}
  ],
  "outcome": {"correct": 1, "judge": null},
  "cost": {"tokens": 1926, "usd": 0.0031},
  "components": ["skill:cite-first", "tool:retriever@2.1"]
}
```

必须有：任务上下文、每步的决策（调了哪个工具、什么配置、或没调）、**每步工具的返回内容或其摘要**、结果。没有工具返回内容，配置对比查询会被证书直接判 not（Replay-Free 推论），入库时就报警。

## 2. 映射到数据模型

| 轨迹里的东西 | 落到哪张表 | 备注 |
|---|---|---|
| agent 名 + 版本 | View（kind=local，owner=该 agent） | 一个 agent 版本一个 View；模型换了就是新 View |
| 任务上下文（prompt、查询串、任务类、元数据） | Variable `X` 及其子字段 | 查询串必须在 X 里，否则 A1 不成立 |
| 第 k 步决策（调不调、哪个工具、配置） | Variable `T_k`，取值域 {t₀} ∪ 𝒯₊ | 多个工具各自一个 T |
| 第 k 步工具返回（文档集、返回值） | Variable `M_k`（原文存 Episode，摘要存 Variable 值） | 摘要例：命中金标数、相关度分位、返回长度 |
| 最终结果 / 判官分 | Variable `Y`（及 `Y_judge`） | judge 是有偏代理，另存 |
| token / 时延 / 美元 | Variable `C` | 成本也是结果，处方查询要用 |
| 隐难度 | Variable `U`，标 latent | 模板里显式存在，永不赋值；identify 靠它判 |
| 模板边 X→T_k、U→T_k、X→M_k、T_k→M_k、M_k→Y、U→Y、X→Y、M_k→C | Edge（Evidence kind=template，trust=structural） | 来自 agent 代码结构，不是数据发现 |
| 一条轨迹 | Episode 一行（x, t̄, m̄, y, c, seed, version） | 原始 JSON 整体保留，可回放 |
| 重放同一轨迹（do(T_k=t′)） | 新 Episode 行，parent=原 id，Evidence kind=replay | 与原行配对，是干预证据 |
| A/B 随机分配的版本旗标、检索器种子 | Episode 的 design 字段 | 是天然随机化，identify 会用作 IV / 直接随机化证据 |
| components（skills、子 agent、工具版本） | Variable `S_j ∈ {0,1}`（在场 / 消融） | N3 / N4 / P7 的治疗变量 |

多步：`T_1, M_1, …, T_K, M_K`，模板里再加 `M_{<k} → T_k` 的边当且仅当 agent 代码里后一步读了前一步的返回（自适应）。这条边有没有决定序贯 front-door 能不能用（定理 4 的 A3′）。模板从 agent 代码或 span 依赖关系推，推不出来就默认有（保守）。

多 agent 团队：团队是一组组件变量 `S_1..S_n`，轨迹里记谁在场；结果 Y 是团队结果；每个 agent 的边际 = do(S_j=0) 的效应；联盟语境固定为当前团队（Team Pricing 的教训：不做跨联盟平均）。

## 3. 入库流程（`gowhy ingest traces/ --agent rag-react@1.4.2`）

1. **模板推断**：读 agent 代码里的 tool 声明或 span 依赖，生成 / 校验该 View 的轨迹模板；用户可手改（加删边、标自适应）。
2. **字段映射**：JSON 路径 → 模板变量；工具返回内容 → 摘要函数（可插拔）。
3. **完整性检查**：缺中介 → 证书预判 not 并报警；缺查询串 → 警告 A1 可能失效；缺种子 → 重放不可精确复现。
4. **写 Episode**：追加行，带版本。
5. **增量更新索引**：FCM 的局部组计数按新行加；效应表里依赖该 View 的格子置 stale。
6. **standing queries**：对该 agent 注册过的查询（哪个 depth 好、skill 有没有用）重跑 identify + 估计，写回效应表。

## 4. 入库后立刻能问的

| 问题 | 查询 | 走哪个定理 |
|---|---|---|
| depth 5 还是 10 好 | `WHAT IF SET T_1 = 10 RETURN effect(Y) POPULATION called` | Replay-Free 定理 1，零重放 |
| 要不要调工具 | 同上 SET T_1 = t₀ | 定理 2：证书返回 partial [τ−1, τ]，并给"重放无工具臂 R₀ 条" |
| 这个 skill 有没有用 | `ATTRIBUTE Y TO S_j` | N4 认证：要求 Evidence 含随机化，否则只给日志估计并标 confounded |
| 谁弄坏了这次 | `WHY episode ep-000123 OUTCOME Y=0 CANDIDATES M_1, S_*` | Q1：PN 区间，贪心解码一次重放判定 |
| 新版本比旧版本坏多少题 | `SWITCH_RISK view@1.4.2 -> view@1.5.0` | Q2：Fréchet 界 + 配对重放 |
| 该多记什么 | `IDENTIFY effect(Y | do(T_1)) FROM view` | M1：缺的变量列表 |

## 5. 轨迹图和世界图怎么接（桥 A）

agent 的动作变量（比如"给客户打折"）同时是轨迹模板里的 `T_k` 和业务世界图里的一个可干预变量。入库时把两者对齐（同一 Variable id，两个 View 引用）。于是：
- 事前：agent 在轨迹里要做 T_k 前，向世界图问 `WHAT IF SET discount`（L2）。
- 事后：世界图里的结果变化，归因回轨迹里的哪一步（P7）。
两张图共用 store、共用版本，这就是"归因即查询"能成立的存储基础。

## 6. 别的框架、别的模型产生的轨迹能不能接

能，前提是守一个五字段的最小日志契约；模板机制本身不依赖框架或模型。

**最小日志契约（Minimum Log Contract）**：任务上下文（含发给工具的查询串）、每步决策（工具 + 配置，或"没调"）、每步工具返回（原文或摘要）、结果、种子 / 版本。缺第三项配置对比不可识别；缺第一项 A1 存疑；缺第五项重放只能分布式重放。

**标准输入格式**：以 OpenTelemetry GenAI 语义约定的 span 树为规范输入（`gen_ai.*` 属性、tool call span、父子依赖）。适配器：`gowhy ingest --format otel | langgraph | openai-agents | claude-code | autogen | jsonl`。LangSmith、MLflow、Braintrust、Langfuse 都能导出 OTel 或近似格式，只写一次转换。

**模板与模型解耦**：模板由 agent 的控制流决定（哪几步、读不读前一步返回），不由背后的 LLM 决定。同一控制流换 backbone = 同一模板下的两个 View；这正是 Q2（升级坏掉率）需要的配对结构。不同控制流 = 不同模板，各自识别，效应表里按模板分区。

**跨模型比较的边界**：把"模型版本"当治疗时，路由日志是混杂的（难题送大模型），从日志只能给 Fréchet 界；要点估计得有 A/B 旗标或配对重放。系统会把这条写进证书，不会假装能算。

**工具返回的可比性**：M 是工具的输出不是模型的输出，跨模型可比；摘要函数按工具定义（检索器：命中金标数 / 相关度分位；代码执行器：退出码 / 测试通过数；搜索：结果重叠率），与 agent 无关。

**隐难度共享**：U 定义在任务上，同一任务集跨模型共享；这使"模型 A 在难题上被高估"这类比较可以在同一 U 下讨论。

**多 agent 框架**（AutoGen、CrewAI、LangGraph 多节点）：每个节点 / 子 agent 是一个组件变量 S_j，谁在场从 span 树读；团队结果是 Y。

**API 不暴露种子时**：重放不能精确复现，改为分布式重放（同 episode 跑 R 次取均值），证书里标"distributional replay, R=…"。

**接不进来的情况**：只有最终答案没有步骤（黑盒 API 调用）；工具返回被截断到无法算摘要；多轮对话里用户输入本身是后续步的"工具返回"但没记录。三种情况入库时直接报缺口，不做猜测。
