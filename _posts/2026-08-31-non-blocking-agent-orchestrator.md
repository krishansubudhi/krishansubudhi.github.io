---
comments: true
author: krishan
layout: post
permalink: /blog/non-blocking-agent-orchestrator/
categories: [agents, llm, architecture]
title: A non-blocking agent orchestrator
description: Almost every multi-agent framework blocks the orchestrator while its subagents run. Here is what changes when you remove the ability to block instead of managing it.
---

I have been building a self-evolving agent called Walle. The single change that improved it most was not a better model or a better prompt. It was deleting the orchestrator's ability to wait.

The result: a second message sent while a ten-minute job is running gets answered in seconds instead of queueing behind it, and a worker that is going the wrong way can be stopped and re-aimed mid-flight without losing what it has already read. This post was itself re-aimed that way, 72 seconds in — the log is at the bottom.

## The bug that started it

Here is a comment I wrote into the orchestrator's own system prompt after watching it happen:

> I delegated a refactor and then kept going, timed out, and the human was told their request had failed while the helper was still happily working on it.

Nothing crashed. The worker finished fine. But the orchestrator had a 120-second turn budget, it spent that budget doing work it had already handed off, and the person on the other end got a failure message about a job that was going to succeed.

The instinct is to fix the prompt — tell it not to do that. That is a policy fix for a structural problem, and policy fixes on an LLM hold about 80% of the time. The structural fix is that there must be no blocking primitive available to it in the first place.

## What everyone else does

This is not a niche design question. Anthropic's June 2025 [multi-agent research system post](https://www.anthropic.com/engineering/multi-agent-research-system) has a section literally headed *"Synchronous execution creates bottlenecks"*:

> Our lead agents execute subagents synchronously, waiting for each set of subagents to complete before proceeding. This simplifies coordination, but creates bottlenecks in the information flow between agents.

They name three consequences: *"the lead agent can't steer subagents, subagents can't coordinate,"* and *"the entire system can be blocked while waiting for a single subagent to finish searching."* They call asynchronous execution future work and ship the synchronous version.

The picture in the frameworks is the same, because a subagent is almost always modelled as a function call that returns a value:

| Framework | Delegation primitive | Orchestrator during the call |
|---|---|---|
| LangGraph supervisor | routes to a worker node, waits for the graph edge back | blocked |
| Google ADK | `AgentTool` / sub-agent invocation returns a result | blocked |
| OpenAI Agents SDK | `handoff` transfers the turn | blocked (or gone) |
| Anthropic's research system | synchronous subagent spawn | blocked, by their own description |
| Walle | `delegate()` returns immediately, callback fires later | free |

You can of course build async orchestration in any of these. The point is what the default shape is, and the default shape is a blocking call.

## The primitive

Walle's workers are called *helper chefs*. Every one is a separate `claude -p` subprocess in its own git worktree on its own branch. The whole contract is in the module docstring:

```python
"""helper.py -- disposable helper chefs: offload a task to a short-lived
`claude -p` subprocess so the headchef's dispatcher loop is never blocked
waiting on it. delegate() returns immediately; on_done(text, error) fires from
a background thread when the subprocess finishes.
"""
```

`delegate()` returns a worker id. That is all. The orchestrator says one line to the human — "handed this to a chef" — and ends its turn. It is now idle and available to everybody else on the system. When the subprocess exits, a background thread calls `on_done`, which wakes the orchestrator with the report.

The turn is not a unit of work any more. It is just the interval during which the model happens to be generating tokens.

```
message ──▶ orchestrator (never waits)
              │
              ├──▶ worker (own worktree, own branch, own process)
              ├──▶ worker
              └──▶ worker
              │
              ▼
           replies, any number, any time, no turn boundary
              ▲
              │
new message ──┘  interrupts generation, folds itself into the re-prompt
```

## Steering a worker that is going wrong

Because the orchestrator is free, it can watch. And because it can watch, it needs a lever. `interrupt` and `resume` are that lever:

```python
def interrupt(hid, reason="", steered=False):
    """Stop chef `hid` now, keeping everything it has learned -> (ok, detail).

    Unlike stop(), which simply calls a chef off, this is a PAUSE: the claude
    session id is recorded so resume() can pick the same conversation up, and
    the chef's worktree and branch are left exactly where they are instead of
    being handed back. SIGTERM first, SIGKILL after INTERRUPT_GRACE_S, the
    whole process group both times -- a `claude` mid-tool-call has children.
    """
```

`resume(hid, extra)` puts the *same* worker back on the *same* session with `extra` as a new user turn. It keeps its session, its worktree and its branch. It does not keep its old budget — the timeout restarts from zero, because half a budget is how a resumed worker gets killed a second time.

That distinction between *stop* and *pause* is worth more than it looks. Re-delegating a task throws away everything the worker has read; resuming costs one cached prefix read. It is the difference between "start over with better instructions" and "carry on, but do this instead."

One non-obvious detail cost me a real incident. An interrupt-then-resume pair still fires the interrupted run's completion callback, which by default reports `[helper failed]` — for a worker that is, by the time the message is read, already back at work. The orchestrator then tried to "recover" it and interrupted-and-resumed it a second time. The fix is one bit, `steered=True`, threaded through the interrupt so the report goes out as text rather than an error.

## What the interrupt costs

The orchestrator itself is a long-lived session with a large cached prefix, so interrupting *it* looked expensive. Measured on a 178k-token prefix, cancelled mid-`Bash` call:

| Measurement | Result |
|---|---|
| `interrupt()` returns | 11.6 ms |
| cancel → settled result message | 85.6 ms |
| terminal reason | `aborted_tools` |
| session still usable? | yes — next turn answered in 1.5 s |
| re-prompt after abort | 178,551 tokens cache-read, 610 written, $0.0373 |

Three orders of magnitude below the turn itself. Chat stops being a queue.

**But be honest about that last row: n=1.** Two smaller control runs at a 22k prefix showed the post-abort re-prompt did *not* reliably re-hit the cache — one missed entirely. If the same partial miss happens at 178k, one interrupt costs roughly $0.6 instead of $0.037. I have not repeated it enough times to claim otherwise, so this is gated behind a hard spend ceiling rather than treated as free.

A second measured caveat: interrupts are only cheap once the session is *warm*. The first turn at a 163k prefix was billed **$0.77 for the cache write alone**.

## The caps, and why they are small

```python
# How many helper chefs may hold ONE job at the same time. Not one (a big job
# splits into parts that genuinely run in parallel) and not unbounded (chefs
# cannot see each other, so overlapping tasks are the same work done twice and
# a merge conflict at the end).
MAX_HELPERS_PER_JOB = 3
```

Three per job, four in total across the machine. The reason is not cost, it is that workers are deliberately blind to each other. This lines up with Cognition's [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents), which argues that parallel subagents with independent context reliably produce conflicting work, and that writes should stay single-threaded. Walle's version: workers each get an isolated worktree, and only the orchestrator talks to the human.

The other half of that is compression at the handoff. A worker's report is often two thousand words; passing that through verbatim turns the orchestrator into a proxy. So the rule is explicit:

> When relaying a helper chef's report, compress it to a table plus the few lines that matter — not paragraphs passed through near-verbatim.

## This post, steered

Halfway through writing this, I changed my mind about what it should be about and how it should be published. I did not wait for the draft. From the run log:

```json
{
 "title": "blog post novel walle experiments",
 "ok": true,
 "duration_s": 71.93,
 "outcome": "STEERED -- somebody stopped this chef on purpose, saying:
             Krishan wants a pull request, not a direct push to the default
             branch. Steering now so you don't push."
}
```

Same worker, same session, everything it had already read about the blog's format still in context, new instructions. The alternative was letting it finish the wrong post and then starting a fresh one from nothing.

## What this does not do

- **In-flight work inside the orchestrator's own turn dies on interrupt.** `aborted_tools` abandons the tool call. Anything the orchestrator was doing itself is lost and has to be re-run. Only work that is already *out of process* survives.
- **Killed processes can orphan.** Subprocesses outlive their parent, so the interrupt path needs explicit process-group cleanup or you leak `claude` children. Learned the hard way.
- **In-memory state is still in memory.** Which worker holds which job lives in the orchestrator's process. Kill it and that is gone; the durable record that fixes this is a work item, not a finished thing.
- **It is one orchestrator, not N.** Forking the session to run several heads works (a fork reads the parent's cache at cache-read price) but I have not needed it, so it is not built.

None of the individual pieces are novel. Non-blocking dispatch is 1970s operating systems. Isolated worktrees are what CI does. What seems genuinely underexplored is applying the rule to the *orchestrator of an LLM agent system* and treating it as structural rather than as a prompt instruction — there is no rule telling Walle not to block, because there is no primitive left to block with.

If you are building on top of a supervisor pattern, the question worth asking is: when a subagent is 6 minutes into a 10-minute job and you can already see it is wrong, what can you do about it? If the answer is "wait, then start over," that is the thing to fix first.
