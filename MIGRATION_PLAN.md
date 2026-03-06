# MIGRATION_PLAN.md
# CARLA_TEMPO — GitHub Repository Integration Analysis

_Last updated: 2026-03-06_

---

## Repository Under Review

| Field | Value |
|---|---|
| Repository URL | https://github.com/ruvnet/ruflo |
| Repository name | RuFlo v3.5 |
| Type | Enterprise AI Agent Orchestration Platform |
| Language | TypeScript / JavaScript (Node.js 20+), WebAssembly (Rust) |
| Stars | ~19.4k |
| License | Not confirmed |
| Relevance to CARLA_TEMPO | **None — different domain entirely** |

---

## Section 1 — What RuFlo Does

RuFlo is a **multi-agent AI orchestration framework** built on top of Claude Code.
Its purpose is to coordinate 60+ specialized software engineering agents (coder, tester,
reviewer, security auditor, documenter, etc.) in "swarms" to automate software development
workflows.

**Key capabilities:**
- Q-Learning router that assigns tasks to the right agent type
- Swarm topologies (hierarchical, mesh, ring, star)
- 42+ Claude Code skills and 17 custom hooks
- HNSW vector database for agent memory
- Background workers for security audits, GitHub integration, session persistence
- Token optimization (30-50% cost reduction via smart LLM routing)
- Multi-provider LLM support (Claude, GPT, Gemini, Ollama)
- Plugin SDK for extending the platform

**Tech stack**: Node.js 20+, TypeScript, PostgreSQL, SQLite, ONNX Runtime, WebAssembly.

**Intended users**: Software engineering teams who want to automate code review, refactoring,
testing, and DevOps workflows using fleets of Claude Code agents.

---

## Section 2 — Relevance Assessment

| Component | Relevant to CARLA_TEMPO? | Why |
|---|---|---|
| Multi-agent swarm orchestration | ❌ No | CARLA_TEMPO needs one training loop, not an agent swarm |
| TypeScript/Node.js infrastructure | ❌ No | Project is Python/PyTorch end-to-end |
| HNSW vector memory / knowledge graph | ❌ No | No vector retrieval needed in ML training |
| Smart LLM cost routing (Haiku/Sonnet/Opus) | ❌ No | Not an LLM API workflow |
| SLURM job orchestration | ❌ No | RuFlo has no HPC/SLURM integration |
| PyTorch training loop management | ❌ No | Not a Python ML framework |
| CARLA simulator integration | ❌ No | No autonomous driving knowledge |
| InterFuser/timm dataset utilities | ❌ No | Completely unrelated |
| Token cost optimization for API calls | ❌ No | CARLA_TEMPO makes no LLM API calls at training time |
| GitHub workflow automation | ⚠️ Marginal | Could theoretically help with PR automation, but overkill |

**Overall relevance score: 0.5 / 10** — the framework solves a fundamentally different problem.

---

## Section 3 — Direct Comparison

### What CARLA_TEMPO needs:

| Need | Current solution | Does RuFlo help? |
|---|---|---|
| Train InterFuserTemporal on GPU | `temporal/train.py` + SLURM | ❌ No |
| Process CARLA sensor data | `CarlaMVDetDataset` + timm | ❌ No |
| Batch frames into temporal windows | `TemporalWindowDataset` | ❌ No |
| Submit/monitor HPC jobs | `scripts/slurm/*.sbatch` | ❌ No |
| Evaluate waypoint prediction quality | ❌ Missing (need to build) | ❌ No |
| Compare baseline vs temporal metrics | ❌ Missing | ❌ No |
| Run ablation studies (T=2,4,8) | ❌ Missing (need sbatch variants) | ❌ No |
| CARLA closed-loop evaluation | ❌ Missing | ❌ No |

### What RuFlo provides that we already have or don't need:

| RuFlo feature | Our equivalent | Status |
|---|---|---|
| Agent orchestration framework | Claude Code directly (already in use) | Redundant |
| Code review agent | Claude Code directly | Redundant |
| Documentation agent | Manual docs (RUNBOOK, DECISIONS, etc.) | Redundant |
| GitHub PR automation | `git push` + manual PR | Not needed for thesis |
| Cost optimization (30-50% API savings) | No LLM API costs in our pipeline | Not applicable |

---

## Section 4 — Should We Use RuFlo?

### Decision: **IGNORE ENTIRELY**

**Rationale:**

1. **Wrong domain**: RuFlo is a software engineering automation tool. CARLA_TEMPO is a
   machine learning research project. They solve completely different problems.

2. **Wrong language**: RuFlo is TypeScript/Node.js. Every component of CARLA_TEMPO is Python/PyTorch.
   Integrating RuFlo would require maintaining two separate tech stacks with no shared benefit.

3. **Wrong compute model**: CARLA_TEMPO runs GPU training jobs on HPC via SLURM. RuFlo orchestrates
   Claude Code agents via MCP. These execution environments are incompatible.

4. **Wrong abstraction level**: RuFlo coordinates AI assistants working on code. CARLA_TEMPO
   coordinates sensor data flowing through a neural network. These are unrelated problems.

5. **No missing gap to fill**: Every gap in CARLA_TEMPO (eval script, ablation scripts,
   Approach 2 model) is a Python/PyTorch engineering problem. RuFlo cannot fill any of them.

6. **Overhead exceeds benefit**: Setting up RuFlo (Node.js, PostgreSQL, ONNX Runtime, WASM)
   would add significant infrastructure complexity that contributes nothing to the thesis
   research goals.

**What to use instead for workflow automation**: Claude Code directly (already in use and working).

---

## Section 5 — Important Confirmations

✅ **RuFlo must NOT replace the InterFuser codebase** — it has no ML model, no sensor fusion,
   no CARLA integration. It is categorically the wrong tool for replacing any part of InterFuser.

✅ **RuFlo must NOT be installed on Amarel** — it would consume disk/inode quota with no benefit.

✅ **RuFlo must NOT be added to requirements.txt** — Node.js dependencies are incompatible
   with the `interfuser` conda env (Python 3.8).

✅ **All existing temporal/ code must be preserved as-is** — no migration or refactoring needed.

---

## Section 6 — What To Do Instead

The actual gaps in CARLA_TEMPO have nothing to do with workflow orchestration. They are:

| Gap | Correct solution |
|---|---|
| No unified eval script | Write `temporal/eval.py` in Python |
| No ablation scripts | Write `scripts/slurm/temporal_T2.sbatch` etc. |
| Metrics not comparable | Add `eval_l1_error` to `temporal/train.py` |
| No Approach 2 model | Write `temporal/models/interfuser_temporal_attn.py` |
| No CARLA closed-loop eval | Set up CARLA leaderboard harness (leaderboard/) |

These are all Python engineering tasks that can be done directly in Claude Code.

---

## Final Recommendation

| Option | Decision |
|---|---|
| Full adoption of RuFlo | ❌ Reject |
| Partial adoption (specific components) | ❌ Reject — no components are applicable |
| Use as reference for architecture | ❌ Reject — wrong domain, wrong language |
| **Ignore entirely** | ✅ **This is the correct choice** |

**The CARLA_TEMPO project should proceed with its current Python/PyTorch/SLURM stack.
RuFlo is a sophisticated tool for a completely different problem space.**
