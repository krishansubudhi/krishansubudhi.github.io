---
comments: true
author: krishan
layout: post
categories: agents
title: My assistant picked its own memory library. I made it change its mind.
description: Jarvis went looking for an open-source memory library to replace its own. Its first pick passed the test and turned out to be abandoned. Here is how it ended up on Graphiti, what the public benchmarks do and don't say, and why the real answer will come from replaying my own data.
---

Jarvis, my personal assistant, has a memory problem. I build it mostly by talking to it, and it remembers things by pulling facts out of our chats into a table. The table has grown to a few thousand rows. Ask it where I live and it finds three answers: Sunnyvale, "California", and Boston. All three are live. None of them were ever retired.

So on 2026-10-10 I asked it to stop maintaining its own memory code and find an open-source library that handles this properly. This post is how that search went, including the part where its first answer was wrong.

## The test: Seattle, then San Francisco

The library has to handle a correction. That is the bug I actually have, so it became the test.

Turn one: "Krishan lives in Seattle." Turn two: "We moved to San Francisco last month." Then ask: "Where does Krishan live?" A good memory gives one answer, San Francisco. A bad one gives both, or worse, ranks Seattle first.

Two more cases rode along. A dedup pair ("My favorite programming language is Python" / "Python is the language I like most for programming") should end up as one fact. A junk turn ("ok thanks, sounds good!") should store nothing.

Jarvis installed four libraries in throwaway environments and ran all of them through the same turns, using the same model: [LangMem](https://github.com/langchain-ai/langmem), [mem0](https://github.com/mem0ai/mem0), [Graphiti](https://github.com/getzep/graphiti) and [MemLedger](https://github.com/riktar/memledger). About ten more ([Cognee](https://github.com/topoteretes/cognee), [Hindsight](https://github.com/vectorize-io/hindsight), [Letta](https://github.com/letta-ai/letta), Memori, memU and others) were judged only from their READMEs and source code, and were never run. That's weaker evidence, and it's labelled that way in the notes.

## What happened

**mem0** had already been ruled out on 2026-10-08, before this round. It sends PostHog telemetry by default, and it has no SQLite vector store. It got re-run anyway and failed the correction: it added "moved from Seattle to San Francisco" next to "lives in Seattle" and ranked Seattle first. The current version's extraction only ever adds facts. Nothing retires the old one. That's exactly the bug I already have.

**LangMem** was the cleanest. One LLM call per write, one SQLite file. It updated the Seattle row in place to "Krishan lives in San Francisco (moved from Seattle)", merged the Python pair, and ignored the junk turn.

**Graphiti** was the most interesting. It stores facts as edges in a graph with time ranges. After the second turn, the Seattle edge got an `invalid_at` date and the San Francisco fact appeared next to it. It keeps the history instead of overwriting it. But it cost 2–4 LLM calls and 5–12 seconds per write, it needs a graph database, and its search still returned the invalidated Seattle edge. The caller has to filter that out.

**MemLedger** also passed all three cases, three runs out of three. But it is one author, 17 stars, and nothing has been committed since July.

So jarvis's first pick was LangMem: the only one that updated the fact in place, cheaply, in a single file.

## The pushback

I looked at the LangMem repo and said no. It's stagnant: the last release is 0.0.30, from October 2025. The 2026 commits are dependency bumps and doc fixes. A memory layer is something I'll be living with for years. I don't want to adopt a library and become its maintainer on day one.

Jarvis had flagged this as a risk itself ("adopt, pin, be ready to vendor about 1k lines"). It still picked LangMem because LangMem passed the test. That's fair, but passing a test once doesn't make a library worth depending on.

## Checking the public benchmarks

Next it looked at what the published numbers say. The most complete roundup I found is [Open Source AI Review's September 2026 comparison](https://www.opensourceaireview.com/blog/which-ai-memory-layer-has-the-best-published-benchmarks-in-2026). Asterisks mean the vendor reported the number itself:

| System | LoCoMo | LongMemEval | Passed my correction test? | Actively maintained? |
|---|---|---|---|---|
| Zep (Graphiti) | 94.7* | 90.2* | Yes (caller filters old facts) | Yes |
| Mem0 | 92.5* | 94.4* | No: kept both cities | Yes |
| Memori | 81.95 | — | Not run | Last release May 2026 |
| Letta | 74.0 | — | Not run (now a TypeScript agent harness) | Yes |
| LangMem | 58.1 | — | Yes | No: last release Oct 2025 |
| MemLedger | — | — | Yes | No: one author, quiet since July |

Read those numbers carefully. The two top scores are the vendors' own, each on its own setup. When Zep is run under the Mem0 paper's protocol, the same page puts it at 58–66% on LoCoMo. The page also notes that the earlier open-source mem0 scored about 49% on LongMemEval, a long way from the managed platform's 94.4. It even quotes Mem0 at 93.4 and 91.6 in other places. [Memnode's roundup](https://memnode.dev/articles/agent-memory-benchmarks-2026-real-numbers) gives yet another set: Mem0 66.9% on LoCoMo, Zep/Graphiti 71.2% on LongMemEval. Same benchmarks, very different numbers.

My takeaway: the benchmarks are a coarse filter. LangMem's 58.1 didn't help its case. Beyond that they don't rank anything for me.

## Why Graphiti

Put the two lists together. Graphiti is the only option that is both alive (a release in September, commits the day we checked, 31k stars) and passed the correction test. Its time-ranged edges are also the right model for "I used to live here, now I live there". I'd rather have the history than lose it to an overwrite.

There was one blocker. Graphiti needs a graph store, and I had wanted jarvis's memory to stay in SQLite. Once I agreed to a separate local database, the blocker went away.

## The caveats

**No public benchmark tests what jarvis actually remembers.** LoCoMo and LongMemEval are about long chats. A lot of jarvis's memory comes from agent tool trajectories: what a helper agent did and learned during a coding run. Nobody benchmarks memory built from those. That's why the real decision comes from replaying my own data, not from a leaderboard.

**Not the embedded backend.** Graphiti has an embedded Kuzu backend that needs no server, which is tempting. But it's deprecated inside Graphiti, the [Kuzu repo](https://github.com/kuzudb/kuzu) is archived, and in an earlier internal test Graphiti-on-Kuzu scored 61%, no better than the plain BM25 keyword search it was meant to replace (about 60%). So it runs on a local FalkorDB or Neo4j instead.

**Search has to filter invalidated facts.** Graphiti marks Seattle as no longer true, but it still returns it. If jarvis doesn't drop edges whose `invalid_at` is set, I'm back to two cities.

**The smoke test was small.** One model, a handful of runs, three cases. It's a filter, not a verdict.

## Eval results (coming)

> **Placeholder.** This section will be filled in with:
>
> - **Replay:** Graphiti against jarvis's current memory on real corrections and chef trajectories replayed from my own history.
> - **Michelin:** results from my recall eval.
>
> Until those land, Graphiti is the pick, not the proven winner.
