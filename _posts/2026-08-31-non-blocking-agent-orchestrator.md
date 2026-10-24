---
comments: true
author: krishan
layout: post
permalink: /blog/non-blocking-agent-orchestrator/
categories: [agents, llm, architecture]
title: A non-blocking agent orchestrator
description: Almost every multi-agent framework blocks the orchestrator while its subagents run. Here is what changes when you remove the ability to block instead of managing it.
---

I have been building a private, self-evolving personal agent. The single change that improved it most was not a better model or a better prompt. It was deleting the orchestrator's ability to wait.

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
| This architecture | `delegate()` returns immediately, callback fires later | free |

You can of course build async orchestration in any of these. The point is what the default shape is, and the default shape is a blocking call.

## Three kinds of helper

In practice, background work takes different shapes. The orchestrator ("head") delegates work across three distinct tiers:

1. **In-turn subagents**: Lightweight, ephemeral helpers used strictly for grunt reads (like broad file searches, grep sweeps, or reading dozens of files). They run synchronously within the head's turn, but their scope is tightly constrained: read-only, fast, returning directly into context before the head formulates its reply.
2. **Read-only scouts**: When looking up code, web references, or past context spans across the repository and might take tens of seconds, waiting in-turn would freeze chat. The head dispatches a scout with `scout(question)`. The tool returns a tag *immediately* ("scout dispatched"), the turn ends, and up to 5 scouts can run concurrently across the system. Scouts have no worktree, cannot write, survive host restarts, and post their answers back as fresh incoming messages when they finish.
3. **Workers**: Autonomous helpers doing heavy implementation work. Each worker runs in its own OS subprocess, with its own git worktree, branch, and budget. Up to 3 workers can run per job (capped at 6 across the system). They outlive the turn, survive restarts, and can be paused or re-aimed mid-flight.

## The delegation primitive

Every worker is a separate subprocess in its own git worktree on its own branch. The contract is captured in the worker module docstring:

```python
"""helper.py -- disposable workers: offload a task to a short-lived
subprocess so the orchestrator's dispatcher loop is never blocked
waiting on it. delegate() returns immediately; on_done(text, error) fires from
a background thread when the subprocess finishes.
"""
```

`delegate()` returns a worker id. That is all. The orchestrator says one line to the human — "handed this to a worker" — and ends its turn. It is now idle and available to everybody else on the system. When the subprocess exits, a background thread records the report and wakes the orchestrator with the outcome.

The turn is not a unit of work any more. It is just the interval during which the model happens to be generating tokens.

```
message ──▶ orchestrator (never waits)
              │
              ├──▶ in-turn subagent (fast read helper, within turn)
              ├──▶ scout (read-only, instant tag, up to 5 concurrent)
              └──▶ worker (own worktree, own branch, outlives turn)
              │
              ▼
           replies, any number, any time, no turn boundary
              ▲
              │
new message ──┘  interrupts generation, folds itself into the re-prompt
```

## Deciding when work is done: predicates, not promises

Most multi-agent systems rely on the worker's own summary to decide whether a task is finished. The worker says "I fixed the issue," the orchestrator reads that claim, and trusts it. If the worker hallucinated or tests broke silently, the job gets marked done anyway.

Here, every delegated job requires a work item filed *before* the worker starts. Crucially, the work item requires a machine-checkable predicate check:

```
check: cmd: pytest tests/test_orders.py -k test_idempotent
```

Other check forms include `file_contains: <path> :: <pattern>` and `http: <url> :: <status>`.

Two rules enforce honesty:
1. **Red first**: The check must fail *right now* before work begins. Opening an item dry-runs the check against the repository. If it already passes, the orchestrator refuses to open the item: a test that already passes cannot measure whether new work succeeded.
2. **Environmental evaluation**: The worker never marks its own work item complete. A periodic background tick evaluates the check against the codebase independently. Only when the predicate actually turns green does the item advance to shipped. The worker's self-report is treated as a narrative summary, not proof of completion.

## Steering a worker that is going wrong

Because the orchestrator is free, it can watch. And because it can watch, it needs a lever. `interrupt` and `resume` are that lever:

```python
def interrupt(wid, reason="", steered=False):
    """Stop worker `wid` now, keeping everything it has learned -> (ok, detail).

    Unlike stop(), which simply kills a worker, this is a PAUSE: the model
    session id is recorded so resume() can pick the same conversation up, and
    the worker's worktree and branch are left exactly where they are instead of
    being handed back. SIGTERM first, SIGKILL after INTERRUPT_GRACE_S, the
    whole process group both times -- a worker mid-tool-call has children.
    """
```

`resume(wid, extra)` puts the *same* worker back on the *same* session with `extra` as a new user turn. It keeps its session, its worktree and its branch. It does not keep its old budget — the timeout restarts from zero, because half a budget is how a resumed worker gets killed a second time.

That distinction between *stop* and *pause* is worth more than it looks. Re-delegating a task throws away everything the worker has read; resuming costs one cached prefix read. It is the difference between "start over with better instructions" and "carry on, but do this instead."

One non-obvious detail cost me a real incident. An interrupt-then-resume pair still fires the interrupted run's completion callback, which by default reports `[worker failed]` — for a worker that is, by the time the message is read, already back at work. The orchestrator then tried to "recover" it and interrupted-and-resumed it a second time. The fix is one bit, `steered=True`, threaded through the interrupt so the report goes out as text rather than an error.

## What the interrupt costs

The orchestrator itself is a long-lived session with a large cached prefix, so interrupting *it* looked expensive. Measured on a 178k-token prefix, cancelled mid-tool call:

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

## Guardrails: tool ceilings and the Oct 24 lesson

Non-blocking delegation keeps long tasks off the orchestrator's thread. But what stops the orchestrator itself from stalling?

Two initial guardrails govern the orchestrator's turns:
- **Tool call ceiling**: A 10-second ceiling per tool call (`TOOL_TIMEOUT_S = 10`). If a command takes longer, it does not belong in the orchestrator's turn; it belongs in a worker or scout.
- **Budget and reply exemption**: The orchestrator is limited to 25 tool calls per turn (warning injected when 5 remain) to prevent runaway wandering. Crucially, `reply` is exempt: the orchestrator must *never* be refused permission to speak to the user.

### The 4m37s freeze

On October 24, a subtle failure proved that non-blocking delegation is still not enough if the orchestrator's own tools can hang.

When the orchestrator opened a work item for a delegated task, opening the item triggered a synchronous dry run of the predicate check to verify that it failed as expected. The check ran a command that hung. That single synchronous tool call blocked the orchestrator for **4 minutes and 37 seconds**. During those four minutes, the orchestrator went completely silent: incoming messages from the user queued up unacknowledged.

Two problems surfaced:
1. **The head's own tool calls need a hard deadline**: Relying on individual tools to return promptly is a trap. In-process MCP tool executions called by the orchestrator require an enforced deadline.
2. **Subprocess timeouts leak grandchild processes**: In Python, running `subprocess.run(shell=True, timeout=...)` sends `SIGKILL` only to `/bin/bash` when the timeout triggers. Any processes spawned by the shell keep running. Even worse, `subprocess.run` waits for stdout/stderr pipes to close, so as long as grandchildren hold open the pipe descriptors, the orchestrator thread stays blocked indefinitely despite the timeout!

The fix in progress: enforce a strict MCP tool call deadline on the orchestrator, and use process-group termination (`start_new_session=True` combined with `os.killpg(os.getpgid(p.pid), signal.SIGKILL)`) so that timeouts actually tear down the entire process tree cleanly.

## The concurrency caps, and why they are small

```python
# How many workers may hold ONE job at the same time. Not one (a big job
# splits into parts that genuinely run in parallel) and not unbounded (workers
# cannot see each other, so overlapping tasks are the same work done twice and
# a merge conflict at the end).
MAX_WORKERS_PER_JOB = 3
```

Three workers per job, six in total across the machine, and five concurrent scouts. The reason is not cost, it is that workers are deliberately blind to each other. This lines up with Cognition's [Don't Build Multi-Agents](https://cognition.ai/blog/dont-build-multi-agents), which argues that parallel subagents with independent context reliably produce conflicting work, and that writes should stay single-threaded. In this setup: workers each get an isolated git worktree, and only the orchestrator talks to the human.

The other half of that is compression at the handoff. A worker's report is often two thousand words; passing that through verbatim turns the orchestrator into a proxy. So the rule is explicit:

> When relaying a worker's report, compress it to a table plus the few lines that matter — not paragraphs passed through near-verbatim.

## This post, steered

Halfway through writing this, I changed my mind about what it should be about and how it should be published. I did not wait for the draft. From the run log:

```json
{
  "title": "blog post agent experiments",
  "ok": true,
  "duration_s": 71.93,
  "outcome": "STEERED -- somebody stopped this worker on purpose, saying:
              Krishan wants a pull request, not a direct push to the default
              branch. Steering now so you don't push."
}
```

Same worker, same session, everything it had already read about the blog's format still in context, new instructions. The alternative was letting it finish the wrong post and then starting a fresh one from nothing.

## What this does not do

- **In-flight work inside the orchestrator's own turn dies on interrupt.** `aborted_tools` abandons the tool call. Anything the orchestrator was doing itself is lost and has to be re-run. Only work that is already *out of process* survives.
- **Killed processes can orphan without process groups.** Subprocesses outlive their parent, so the interrupt and timeout paths need explicit process-group cleanup (`killpg`) or you leak running children holding file descriptors.
- **In-memory state is still in memory.** Which worker holds which job lives in the orchestrator's process. Kill it and that is gone; the durable record that fixes this is a work item, not a finished thing.
- **It is one orchestrator, not N.** Forking the session to run several heads works (a fork reads the parent's cache at cache-read price) but I have not needed it, so it is not built.

None of the individual pieces are novel. Non-blocking dispatch is 1970s operating systems. Isolated worktrees are what CI does. What seems genuinely underexplored is applying the rule to the *orchestrator of an LLM agent system* and treating it as structural rather than as a prompt instruction — there is no rule telling the agent not to block, because there is no primitive left to block with.

If you are building on top of a supervisor pattern, the question worth asking is: when a subagent is 6 minutes into a 10-minute job and you can already see it is wrong, what can you do about it? If the answer is "wait, then start over," that is the thing to fix first.
