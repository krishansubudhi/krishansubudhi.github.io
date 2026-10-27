---
comments: true
author: krishan
layout: post
categories: agents
date: 2026-10-26
pin: true
title: AI agents delete half as much existing code
description: "Analyzing 642 merged pull requests across 16 major open-source repositories: why coding agents bloat codebases by adding logic without pruning dead code, and how code-size ceilings prevent it."
---

AI agents delete half as much existing code as human developers do.

We analyzed 642 merged pull requests across 16 open-source repositories (Electron, LiteLLM, OpenHands, Ruff, Stagehand, LangChain, and others), matching 321 merged AI-agent PRs against an exact 1:1 nearest-in-time cohort of 321 human PRs from the same repos and time windows.

The headline gap in code growth is modest: **88.2%** of agent PRs net-grow codebases compared to **81.0%** for human developers (+7.2%). Most code changes add lines regardless of who writes them.

The real divergence is how agents handle deletions:

| Metric | AI Agent PRs (n=321) | Matched Human PRs (n=321) | Difference |
| :--- | :--- | :--- | :--- |
| **Median deletions per PR** | **5 lines** | **10 lines** | **-50% (half as much code deleted)** |
| **PRs with zero deletions** | **19.0%** (61/321) | **12.8%** (41/321) | +48% more zero-deletion PRs |
| **p90 deletions per PR** | **127 lines** | **283 lines** | -55% less tail refactoring |
| **PRs that net-grow code** | **88.2%** (283/321) | **81.0%** (260/321) | Modest difference (+7.2%) |
| **PRs that net-shrink code** | **6.9%** (22/321) | **10.9%** (35/321) | -37% fewer net-reductive PRs |
| **Median additions per PR** | 67 lines | 92 lines | Agents submit slightly smaller additions |

## Why agents avoid deleting code

Large language models are trained to satisfy prompts while avoiding regressions. When an agent fixes a bug or adds a feature, deleting or restructuring existing code carries high risk: subtle dependencies might break, tests might fail, and shared state might get corrupted. 

Adding code is much safer. If an edge case fails, an agent appends an extra `if` branch, defines a duplicate helper function, or writes a fallback handler. The prompt is solved, the test suite passes, and the old path remains untouched.

Over 10 out of the 16 repositories studied, median deletions for agents were strictly lower than for humans (and tied in two others). In repositories like `opensanctions/opensanctions`, the median agent PR deleted 0 lines versus 5 lines for humans. In `browserbase/stagehand`, median deletions were 3 lines for agents versus 25 for humans.

When you pair an 88% net-growth rate and halved deletions with the high merge velocity of autonomous agents, the codebase inflates rapidly without any deliberate architectural decision.

## Methodology

To measure typical developer and agent workflows rather than synthetic benchmarks:

1. **Explicit agent identification**: We gathered merged PRs with clear agent signatures—bot accounts (`devin-ai-integration[bot]`, `claude[bot]`), agent branch prefixes (`head:codex/*`), or body declarations (`Generated with Claude Code`).
2. **1:1 nearest-in-time matching**: For each repo and time window, each agent PR was paired with the human PR merged nearest in time without replacement. Automated dependency bots (Dependabot, Renovate) and CI release bots were excluded.
3. **Change metrics**: Additions, deletions, and net lines were pulled directly from GitHub's GraphQL API.

## Caveats

- **Attribution bias**: Only PRs explicitly tagged with agent signatures were included in the agent cohort. Human-assisted agent PRs submitted without disclosure count as human, which would dilute rather than exaggerate the measured gap.
- **Task scoping**: In some repos, agents are assigned targeted tasks (e.g. adding provider pricing models or isolated unit test fixes) while human core maintainers handle broader architectural refactors.
- **Outliers**: A handful of package lockfiles and generated asset PRs skew extreme maximums, which is why median and 90th-percentile metrics are used.

## Guarding against agent bloat

If agents default to purely additive changes, codebases need a mechanical counterbalance that prevents code bloat before it lands.

I built [ratchet](https://github.com/krishansubudhi/ratchet) for this reason. Ratchet enforces an automated ceiling on total lines of code directly in CI (`ratchet check`). If an agent adds 40 lines of code, it either has to find dead code or obsolete helpers to prune, or explicitly ask a human maintainer for a logged ceiling grant (`ratchet grant`). When code is cleaned up, the ceiling ratchets down, locking in simplicity.

In our own agent harness, ratchet has blocked over 50 bloat attempts, forcing agents to clean up after themselves instead of endlessly expanding the tree.

---

> 📊 **Explore the Interactive Report Card**: Want to see how your own repo or these 16 open-source projects compare? Try the [AI Agent Bloat & Deletion Report Card](/bloat/) to benchmark your pull requests and get an instant bloat grade.
