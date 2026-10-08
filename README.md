<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/banner-dark.svg">
  <img src="assets/banner-light.svg" alt="GoWhy" width="720">
</picture>

### What-if and how-to queries over LLM agent trajectories

Ask *"what if the agent had done X?"* and *"what change would make it succeed more?"*,<br>
and get answers that are unbiased, certified, and far cheaper than re-running the benchmark.

[![status](https://img.shields.io/badge/status-research%20prototype-orange)](#status-and-roadmap)
[![python](https://img.shields.io/badge/python-3.10-3776ab)](#quickstart)
[![envs](https://img.shields.io/badge/envs-ALFWorld%20%C2%B7%20WebShop-informational)](#results)
[![models](https://img.shields.io/badge/models-Qwen3--8B%20%C2%B7%20Qwen2.5--7B%20%C2%B7%20Llama--3.1--8B-7c3aed)](#results)

[Idea](#the-idea) ·
[Results](#results) ·
[How it works](#how-it-works) ·
[Query language](#the-query-language) ·
[Quickstart](#quickstart) ·
[Layout](#repository-layout) ·
[Lessons](#lessons-and-pitfalls) ·
[Roadmap](#status-and-roadmap)

</div>

---

## The idea

Agent developers keep asking two questions:

- **What-if**: *if the agent had been stopped from picking up objects the task does not need, how much would its success rate change?*
- **How-to**: *which change to the agent would raise its success rate the most?*

Today the answer is to change the agent and re-run the whole benchmark (an A/B test), which is slow, expensive, and easy to fool yourself with when you try many changes and keep the best-looking one.

GoWhy stores agent runs as a trajectory database and answers these questions as **queries**. A change is written as an update to the agent's future steps:

```sql
WHATIF UPDATE events SET action = hint:2
       WHERE verb = 'take' AND obj_match = 0     -- about to pick up something the task does not need
       SCOPE ALL
       GROUP BY task_type WITH n = 1
```

GoWhy finds, in every stored game, the first step where the change would have fired, **forks the run there**, reuses the logged prefix, and replays only the suffix under the changed policy. Games where the change never fires need no replay at all: their outcome provably does not change.

## Results

All numbers come from this repository's code (`bench/`), run on ALFWorld and WebShop with the [AgentDebug](https://github.com/ulab-uiuc/AgentDebug) harness reproduced verbatim. Full log, including every negative result: [`bench/PLAN.md`](bench/PLAN.md).

**The answers are right.** In 6 head-to-head checks, the what-if estimate agrees with actually deploying the change and re-running from scratch (all |z| < 0.6):

| Model, environment, change | What-if | True deployment | z |
| --- | --- | --- | --- |
| Qwen2.5-7B, ALFWorld, reject invalid actions (smoke, 40 games) | +5.0 | +5.0 | 0.00 |
| Qwen3-8B, ALFWorld, reject invalid actions | +0.5 | +0.2 | 0.13 |
| Qwen2.5-7B, ALFWorld, reject invalid actions | +6.4 | +5.3 | 0.39 |
| Qwen3-8B, ALFWorld, feedback on picking up a wrong object | +2.5 | +4.1 | −0.58 |
| Qwen2.5-7B, ALFWorld, same feedback | +1.2 | +1.8 | −0.28 |
| Llama-3.1-8B, ALFWorld, lenient action parsing | +16.3 | +17.4 | −0.40 |

**They are cheap.** At equal precision (±1 point), what-if needs far fewer tokens than redeploying:

| Model (share of games the change touches) | What-if, 1 replay per affected game | What-if, 8 replays | Redeploy |
| --- | --- | --- | --- |
| Qwen3-8B (22%) | 4M tokens | 21M | 103M |
| Qwen2.5-7B (84%) | 48M | 188M | 103M |

Picking the best of 27 candidate changes with one replay per affected game chose the held-out best every time (6/6 splits) for about 12k model calls, roughly 13x cheaper than deploying each candidate.

**They find fixes that hold on unseen tasks.** Each fix below was found from stored trajectories, screened by what-if, certified on fresh replays, and then confirmed by an A/B test on games or goals never used before:

| Model, environment | Fix | Result on unseen tasks |
| --- | --- | --- |
| Qwen3-8B, ALFWorld | feedback when about to pick up an object the task does not need | success +4.6 points (95% CI +1.3..+7.9), 274 games |
| Qwen3-8B, ALFWorld | "replan" (restate task, actions so far, places visited) on wrong object or wrong receptacle | certified +5.6; unseen +4.0 (95% CI +0.2..+7.8) |
| Qwen3-8B, WebShop | the store showed 65% of episodes never buy anything; replan when the agent keeps paging after step 8 | score 0.211 → 0.314, **+49%** (95% CI +0.068..+0.137), 200 goals |
| Llama-3.1-8B, ALFWorld | 95.8% of actions were rejected by the harness (no `</action>`, quotes); lenient parsing | success 0.2% → **17.5%** (deployed) |

**And they tell you what does not work.** After certification, only about 5% of failures have a single decisive step, and 12 of 15 "decisive steps" flagged with 8 replays per cell were false positives. One LLM-proposed rule lowered success by 5 points; checking attributes right before "buy now" lowered WebShop score; broad replan triggers (every 10 steps, every revisit) did not help.

## How it works

The theory is written up in [`bench/THEORY.md`](bench/THEORY.md) (in Chinese).

- **Model.** With a stateless harness, each step is a sample from the policy given the full prompt, so an episode is a Markov chain over prompts.
- **Fork-point replay is unbiased.** Before the first step where the change fires, the changed and original policies behave identically, so the logged prefix is a valid sample under both. At the first firing step, the logged reply is the changed policy's first draw. Unaffected games contribute exactly zero. Per game, `D_i = a_i (V̂_i − Y_i)`; the estimate is the mean of `D_i` and its SE is `sd(D)/√N` (the deployment-basis SE).
- **One replay per affected game is optimal.** `Var(D) = A + B/n`, where `A` is between-game variance and `B` is replay noise. Precision times cost is `A·n + B`, minimized at `n = 1`; sub-sampling games with 1/q weights is worse. The formula predicts observed SEs within 4%, and it explains why an earlier "materialize a fraction of the cells" index failed.
- **Common random numbers.** Each sample gets a seed fixed by (game, replay, step, draw) and independent of the change, so different candidates are compared on coupled randomness. Same-seed action agreement is 0.97 versus 0.81 without seeds.
- **Selection is guarded.** Candidates are mined on half A of the store and tested on half B; the winner is certified on fresh replays not used for selection, then A/B-tested on unseen tasks.

## The query language

```
UPDATE events SET action = <effect> WHERE <predicate> SCOPE FIRST | ALL
WHATIF <change> [GROUP BY <column>] WITH n = <replays> [DEPLOY m = <runs>]
HOWTO  <file of changes> WITH n = <replays> CERTIFY n = <replays>
```

**Effects**

| Effect | Meaning |
| --- | --- |
| `reject:N` | a verifier: resample up to N times while the reply still matches |
| `hint:N[:ID]` | append a checker's feedback (generic, or message ID from `messages.json`) and resample |
| `replan:N[:ID]` | stronger feedback: restate the task, actions so far and places visited, ask for the unfinished part, then resample |
| `reparse` | harness fix: read the reply leniently and map it to the closest admissible action |
| `swap:<model>` | from the firing step on, use another model |
| `set:<action>` | force an action |
| `unquote` | strip quotes from the action |

**Predicate columns** (computed before the step runs, identically on logged events and during replay): `k`, `verb`, `valid`, `fmt_valid`, `quoted`, `repeat`, `revisit`, `obj_match`, `recep_match`, `n_invalid`, `task_type`, and the text columns `action`, `obs`, `task` (use `LIKE`). Predicates are evaluated by SQLite itself.

## Quickstart

The code expects one GPU machine with vLLM and conda. Paths default to the original setup; override with `GOWHY_ROOT`, `ALFWORLD_DATA`, `WEBSHOP_DIR`, `GOWHY_SERVERS`.

```bash
# 1. environments (conda, Python 3.10)
conda create -p envs/alfw python=3.10 && envs/alfw/bin/pip install "alfworld==0.4.2" "textworld==1.7.0" openai scikit-learn pandas pyarrow
conda create -p envs/vllm011 python=3.10 && envs/vllm011/bin/pip install "vllm==0.11.0" "transformers==4.57.1"
# WebShop (optional): Python 3.10 + openjdk 21 (conda-forge), pyserini 0.22.1, spacy 3.7.2 with
# en_core_web_sm/lg 3.7.1, gym 0.24.0, selenium 4.2.0; data from the HF dataset YWZBrandon/webshop-data
# (items_shuffle_1000.json, items_ins_v2_1000.json, items_human_ins.json), then build the Lucene index.

# 2. serve the agent model, one replica per GPU (--generation-config vllm is required)
MODEL=/path/to/Qwen3-8B NAME=qwen3-8b GPUS="0 1 2 3" PORT0=8041 bash bench/scripts/serve.sh

# 3. check the harness reproduces AgentErrorBench prompts exactly (expects 100/100)
cd bench && python harness/validate.py

# 4. collect trajectories and build a store
python harness/gen_fresh.py --n 300 --model Qwen3-8B-fresh --served qwen3-8b
python gowhy/store.py fresh

# 5. ask
python gowhy/query.py --db store/fresh.db \
  "WHATIF UPDATE events SET action = hint:2 WHERE verb = 'take' AND obj_match = 0 SCOPE ALL WITH n = 1 DEPLOY m = 2"

# 6. find a fix: propose on half A, screen on half B, certify, then A/B on unseen games
python gowhy/propose.py --db store/fresh.db --out gowhy/proposed.txt
DB=store/fresh.db HALF=B N=4 CRN=1 bash scripts/run_candidates.sh gowhy/proposed.txt logs/pool
python gowhy/optimizer.py --db store/fresh.db --cands gowhy/proposed.txt --half B
python harness/ab.py --served qwen3-8b --m 2 --change "UPDATE events SET action = ... SCOPE ALL"
```

For WebShop, add `--env webshop` to `gen_fresh.py`, build `store.py ws_q3`, use `--metric score`, and `--splits webshop_test` for the A/B test. `scripts/ws_pipeline.sh` and `scripts/llama2.sh` run whole pipelines end to end.

## Repository layout

```
bench/                      current GoWhy (what-if / how-to over trajectories)
  PLAN.md                   running log: every result, number and pitfall (Chinese)
  THEORY.md                 estimator, unbiasedness, variance and the one-replay theorem (Chinese)
  PAPER.md                  paper outline and evidence status (Chinese)
  harness/
    aeb.py                  AgentDebug ALFWorld harness, verbatim templates and parsing; Episode restore
    webshop.py              AgentDebug WebShop harness
    llm.py                  vLLM client: replicas, retries, per-request seeds
    validate.py             exact prompt reproduction check on AgentErrorBench
    gen_fresh.py            play new games and record full trajectories
    replay.py, gate.py      fidelity check; decisive-step gate with certification
    ab.py                   A/B test of a change on unseen games or goals
  gowhy/
    store.py                SQLite store: traces / events / runs, step attributes
    changes.py              change language, predicates, effects, feedback messages
    engine.py               fork-point replay engine
    query.py                WHATIF / HOWTO, deployment-basis estimates, DEPLOY check
    mine.py, propose.py     rule mining and LLM-proposed candidates (half A only)
    optimizer.py            selection strategies compared on recorded replays
    *.txt, messages.json    candidate sets and feedback messages used in the results
  scripts/                  serve.sh, run_candidates.sh, ws_pipeline.sh, llama2.sh
demo/, DESIGN.md, SPEC-*.md earlier prototype (see below)
```

Trajectory stores, replays and model weights live on the experiment machine and are not in the repository.

## Lessons and pitfalls

These cost real time; [`bench/PLAN.md`](bench/PLAN.md) has the details.

- **Selection bias is everywhere.** Mining and testing on the same games inflates effects; an A/B split that reused the hash that picked between duplicate runs inflated one effect from about +2.5 to +5.7. Keep mining, selection, certification and final testing on disjoint data, and salt every hash.
- **Single logged outcomes are noisy baselines.** Deployment comparisons should average several base-policy runs per game.
- **Uncertified "decisive steps" are mostly noise**: the steepest of many noisy drops is usually a winner's-curse artifact.
- **Harness bugs look like agent failures.** Llama's 0% on ALFWorld was almost entirely a parsing problem.
- **Engineering**: vLLM needs `--generation-config vllm`; a JVM (WebShop's Lucene) does not survive `fork`, so use `spawn`; give each process a random starting replica or the first one overloads; use WAL and retries when many processes write one SQLite file.

## Status and roadmap

Research prototype. Done: harness reproduction (100/100 exact prompts), what-if with deployment checks, the one-replay theory, LLM-proposed candidates, certified fixes on two environments and three models.

Next:

- a third environment (τ-bench or ScienceWorld)
- validate the best LLM-proposed ALFWorld fix (+5.5 on half B) on unseen games
- related-work check and paper writing

## Earlier prototype

Before this line of work, GoWhy was a "causal database for agents": an MCP server that estimated `what_if` effects from observational data with identification checks. That design lives on in [`DESIGN.md`](DESIGN.md), [`SPEC-DB.md`](SPEC-DB.md), [`SPEC-TRACES.md`](SPEC-TRACES.md) and the runnable [`demo/`](demo/) (`python3 demo/agent.py`). The previous README is in the git history.

## Acknowledgements

The agent harnesses reproduce [AgentDebug](https://github.com/ulab-uiuc/AgentDebug); trajectories for validation come from AgentErrorBench (HF `davide221/agenterrorbench`); environments are [ALFWorld](https://github.com/alfworld/alfworld) and [WebShop](https://github.com/princeton-nlp/WebShop).
