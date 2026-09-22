---
comments: true
author: krishan
layout: post
categories: agents
title: The hard part was never the scheduling
description: An orchestrator that talks to a human and dispatches long-running workers has an IO problem that is harder than its scheduling problem. Every design mistake I can point at turned out to be an IO mistake.
---

Handing a job to a background worker and getting an answer back later is, on paper, a scheduling problem. Queue, worker pool, callback, done — the operating systems people finished this in the 1970s.

I have spent almost none of my time on who runs when. I have spent all of it on what crosses the boundary between the orchestrator and the worker, and on what crosses the boundary between the orchestrator and me. Every mistake I can point at in the last few months lives on one of those two wires.

## Dispatch must return immediately — and that forces a second rule

The first one I got right, mostly by accident: [dispatch never blocks]({{ '/blog/non-blocking-agent-orchestrator/' | relative_url }}). If handing work to a worker blocks, the orchestrator goes silent, and a turn where the human sees nothing on the screen is the one outcome I am not willing to accept. Not a wrong answer — those I can correct. Silence I can only wait out.

But non-blocking dispatch only buys you that if the orchestrator can *say something* before the turn is over. So there is a rule, and the rule is deliberately mechanical: before you start anything that might run long, post a message. Not "if you think this will take a while, say so" — that version delegates the judgement to the component I have watched get it wrong. Agents are bad at predicting which of their turns will run long, and the turns they misjudge are, with depressing reliability, exactly the slow ones. The rule that works is the one with no estimate in it.

![Two timelines of the same turn. With blocking dispatch the whole turn is silent until the report arrives; with non-blocking dispatch a message is posted first and the orchestrator keeps talking while the worker runs](/assets/orchestrator-io/dispatch-and-silence.svg)

## A standalone brief is an assertion under oath

A worker has its own context. It cannot see the conversation that produced its job. So every fact it needs has to be restated in the brief — which means the brief is not a pointer to the truth, it is a *copy* of it, and copies are where errors breed.

Mine shipped false premises repeatedly. One brief asserted a quantity of headroom that simply did not exist. Another sent a worker to go optimise "383 subprocess spawns"; the worker audited the tree and found 407 spawns in total, of which 26 were test subprocesses and 369 were invocations of `git`. The number was wrong, and more importantly the *category* was wrong — the thing named in the brief was a rounding error next to the thing actually costing time.

In both cases the worker's first useful act was to correct its own brief. That is now the contract: workers are told to challenge the brief and report corrections before anything else. The reason is not that the orchestrator is careless. It is that the orchestrator has no way to check its own memory — nothing inside it distinguishes a number it read ten minutes ago from a number it is currently inventing. The worker, sitting in front of the actual tree, is the only component in the system that can tell those apart.

## Reports lose their middles

Worker reports are long, and long messages get folded and truncated on the way through. Several of mine arrived with the middle cut out — in one case the survivor list from an audit, which was the entire point of the audit. One arrived as a single sentence from somewhere in the middle of a thought, with no outcome in it at all.

![A report drawn as a stack of lines with its middle replaced by a fold marker: the opening and the closing lines survive, the middle does not](/assets/orchestrator-io/report-lost-middle.svg)

Two rules fell out of that. The first is the one I [already believed about human readers]({{ '/blog/the-reply-was-correct-nobody-could-tell/' | relative_url }}) and had not applied to machines: the report leads with the outcome rather than building to it. A report that ends with its verdict is a report that arrives with no verdict.

The second is the one that matters more. When a fold marker says something is missing, the orchestrator must say **"I do not know what was here."** It must not reconstruct it. Reconstructing a cut-out middle is how a hallucination enters a system through a channel nobody is auditing — every other input has a check somewhere, and the gap in a truncated report has none. The filler is fluent, plausible, consistent with the surrounding text, and produced by the one component whose entire job is to sound coherent.

## The channel has a size, and you will find it

One worker failed twice in a row with a JSON message that exceeded a 1 MiB buffer. The work was finished. The result was unrecoverable. Nothing was wrong with the compute, the scheduling, the prompt or the model.

Budget the channel, not just the compute. The cheap version of this is a hard cap on report length with the long thing written to a file and the path passed instead — which also happens to fix the truncation problem above, because a path does not have a middle to lose.

## Asking back is the missing primitive

A worker that hits a genuine fork — an ambiguous requirement, two defensible designs, a brief that turns out to be factually wrong — has two bad options. Guess, and maybe do an hour of work in the wrong direction. Or stall, and raise the question in its final report, by which point the hour is gone anyway.

The fix is a question channel: the worker parks the job and ends its turn with a question, the orchestrator sees it within minutes instead of at the end, and the answer comes back into the *same* session with everything the worker has already read still in context. That last part is what makes it cheap. Re-dispatching a corrected brief throws away all the reading; answering a question costs one cached prefix.

The discipline that goes with it is a small fixed budget of questions per worker. Without a cap, "ask the orchestrator" becomes cheaper than "read the code", and you have built a very expensive way of not doing research.

## Give the orchestrator a small tool budget on purpose

The orchestrator gets a hard cap: a couple of dozen tool calls per turn and a ten-second ceiling on any one of them. This is not a cost control. It is a routing mechanism. Anything bigger than that ceiling is structurally forced into a worker, where it belongs, and I do not have to rely on the orchestrator's judgement about what counts as "big".

The corollary is the useful bit: **a call that overruns is not a retry signal, it is a delegate signal.** The instinct when a command times out is to run it again with a longer timeout. The right move is to hand the whole question to something that has an hour.

And when the budget does run out, the orchestrator has to say it was cut off. A confident answer built on half the evidence is strictly worse than an honest partial one, because the honest one gets a follow-up question and the confident one gets believed.

## The shape underneath

Read these back and they are one bug at six different layers: a boundary got treated as transparent when it was lossy.

| The boundary | What does not cross | What that cost |
| --- | --- | --- |
| Orchestrator → human | a sign of life | an empty screen while the work runs |
| Orchestrator → worker | context | briefs asserting numbers the tree did not have |
| Worker → orchestrator | text, intact | an audit's survivor list folded away |
| Worker → orchestrator | bytes, past a limit | a finished result lost to a 1 MiB buffer |
| Either direction | time | an overrun read as retry, not as delegate |
| Worker → orchestrator | a question, at all | an hour of guessing, or an hour of stalling |

Context does not cross — so the brief has to carry it, and can be wrong. Text does not cross intact — so the report has to front-load, and must not paper over the hole. Bytes do not cross past a limit — so the channel needs a budget. Time does not cross — so an overrun means delegate, not retry. And questions do not cross *at all* unless you build a wire for them, which is why "guess or stall" looked like a worker-quality problem for months when it was a missing channel.

None of this is scheduling. The scheduler has been fine the whole time.
