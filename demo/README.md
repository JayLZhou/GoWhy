# GoWhy demo：先问再动

一个 agent 想提升 B 段客户留存。它每做一个动作之前先向 GoWhy 问 `what_if`，动作工具只认问过且可识别的查询。

这是 `../DESIGN.md` 的第一步，不是 v0.1：只有一个处理变量、二值结果、一张观测日志、back-door 估计（背后有完备的 ID 算法判"不可识别"）。没有存储、版本、查询语言、轨迹分析。

## 跑

```bash
python3 agent.py
```

```bash
python3 -m pytest test_demo.py -q
```

依赖只有 numpy 和 pandas。`agent.py` 是脚本，不调 LLM，它通过真实的 MCP stdio 协议和 `server.py` 说话。

## 会看到什么

| 幕 | agent 做什么 | GoWhy 返回 |
|---|---|---|
| 0 | 不问直接打折 | 动作工具拒绝：没有对应的 `what_if` |
| 1 | 问打折 | 留存 +3 点、退款 +5 点，可识别；日志直接对比是 −7 点（符号反的）。agent 算出净值为负，放弃 |
| 2 | 问包邮，然后执行 | 留存 +2 点、退款无影响。执行后模拟器给出真实效应，落在预测区间内 |
| 3 | 问主动回访 | 不可识别，不给数：缺 `frustration`，或者做一个每臂约 4400 人的随机实验。硬要执行被拒 |
| 4 | 审计第 2 幕 | 估计目标、调整集、假设、每条边的来源、这次查询授权了哪个动作 |
| 5 | 把 `frustration` 记进日志后再问回访 | 变成可识别，留存 +4 点左右 |

## 接到真 agent 上

Claude Code 里加一行（换成你机器上的绝对路径，`python3` 要能 import numpy 和 pandas）：

```bash
claude mcp add gowhy-demo -- python3 /absolute/path/to/GoWhy/demo/server.py
```

开一个新会话，说：

> 你是运营 agent，目标是提升 B 段客户的留存。可用的动作是 apply_offer。

server 的 instructions 和工具描述会告诉它先问 `what_if`；就算它不听，`apply_offer` 没有合格的 `query_id` 也执行不了。这一步我只用自己写的 MCP 客户端验证过协议，还没有在 Claude Code 里实测。

## 文件

| 文件 | 内容 |
|---|---|
| `world.py` | 零售世界的模拟器，真值已知；引擎只能看到日志列 |
| `engine.py` | 图、d-分离、back-door 搜索、ID 算法（只判定）、分层估计加 bootstrap、无假设界、"缺什么" |
| `server.py` | MCP server，标准库手写；`what_if / identify / explain` 加一个带闸门的动作工具 `apply_offer` |
| `agent.py` | 脚本化 agent，同时是一个最小 MCP 客户端 |
| `test_demo.py` | 图判定用例、估计对真值的覆盖、MCP 往返与闸门 |

## 世界里的真值

| 动作 | 留存 | 退款 | 日志里的混杂 |
|---|---|---|---|
| discount | +0.03 | +0.05 | 折扣多发给低忠诚客户，日志对比为负 |
| free_shipping | +0.02 | 0 | 包邮多发给高忠诚客户，日志对比偏高 |
| support_call | +0.04 | 0 | 回访多打给不满的客户，而 `frustration` 没记 |

日志 20 万行，默认种子固定为 11（这个种子下输出读起来正好是"+3、+5"）。换种子估计会在真值附近波动，`test_demo.py` 检查 20 个种子下 95% 区间至少盖住真值 16 次。
