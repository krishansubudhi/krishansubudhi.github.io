---
comments: true
author: krishan
layout: post
title: "Engineers will stop writing code. That is not the same as being replaced."
description: "Hand-typing code is becoming the slow part of software. I think the job moves to design, review and the gates you set for agents — and here is the best evidence against that."
permalink: /blog/future-of-coding/
categories:
  - ai
  - engineering
---

I don't think engineers will be writing code for much longer.

Not because a model is smarter than a good engineer. Because it is faster, and the gap keeps widening. Typing code by hand is walking. Walking is honest, it works, and you can get a surprising distance on it — but there is a car parked next to you, and you are not going to win by walking harder. We can only go this far.

I say this from a slightly odd vantage point. For months now, most of the code in the system I use every day has been written by an agent, not by me. I review the parts that matter and the rest merges on its own once it has passed a gate. What I expected from removing the typing was that everything would get faster. What actually happened was stranger: the bottleneck moved. The part of the job I care about got bigger, not smaller.

Here's the argument, and the strongest evidence I could find against it.

## Start with the study that says I'm wrong

In July 2025, METR ran the experiment everyone should have run first: a [randomised controlled trial of 16 experienced open-source maintainers](https://metr.org/blog/2025-07-10-early-2025-ai-experienced-os-dev-study/) working on real issues, in their own repositories — codebases averaging over a million lines that they had contributed to for years. Tasks were randomly assigned to "AI allowed" or "AI disallowed".

With AI, they took **19% longer**.

The part that should unsettle you is not the slowdown. It's that the developers expected a 24% speedup beforehand, and *after* the slowdown had happened to them, still reported that AI had sped them up by 20%. A roughly 40-point gap between what happened and what it felt like.

![Bar chart of percentage change in task completion time across METR's trials, showing a measured 19 percent slowdown in early 2025 against a believed 20 percent speedup, and later estimates of 18 percent and 4 percent faster whose confidence intervals both cross zero](/assets/future-of-coding/measured-vs-believed.svg)

Then it gets more interesting. METR ran the study again with late-2025 tools and [published the update in February 2026](https://metr.org/blog/2026-02-24-uplift-update/). The returning developers now came out 18% *faster* — but with a confidence interval from −38% to +9%, and newly recruited developers at only −4%. Both intervals cross zero. And the authors did something I respect enormously: rather than declare victory, they announced they were redesigning the study, because 30–50% of developers were now declining to submit tasks they didn't want to do without AI, which biases the whole thing.

So the honest summary of the best evidence available is not "AI makes engineers faster". It is closer to: *the experiment has become difficult to run, because too many people won't take the control condition.* That's a weaker claim than the hype and a stronger one than the backlash, and both facts point the same way — nobody is walking back to walking.

Meanwhile a [survey of 349 technical workers in early 2026](https://metr.org/blog/2026-05-11-ai-usage-survey/) put the median self-reported gain at 1.4–2x in value, and 3x in speed. Given the 40-point perception gap the same group measured a year earlier, I'd treat those numbers as a report on enthusiasm, not throughput. If you take one thing from this post: **your feeling of being fast is not evidence.** Mine isn't either.

## What the slowdown was actually measuring

The RCT didn't find that models write bad code. It found that a domain expert holding a live mental model of a system can often type the change faster than they can specify it, read the generated version, test it and fix the subtle mismatch.

That tax shows up everywhere once you look. In the [2025 Stack Overflow developer survey](https://survey.stackoverflow.co/2025/ai), 84% of 33,000-plus respondents use or plan to use AI tools — and the single biggest frustration, at 66%, is "AI solutions that are almost right, but not quite", with "debugging AI-generated code is more time-consuming" right behind it at 45%. More developers actively distrust the accuracy of the output (46%) than trust it (33%). Experienced developers distrust it most.

And then the number that I think matters more than any productivity figure. Veracode tested over 100 models on 80 curated coding tasks and found [45% of generated samples introduced an OWASP Top 10 vulnerability](https://www.veracode.com/blog/genai-code-security-report/) — 72% for Java, and a failure to defend against cross-site scripting in 86% of the relevant cases. Their key chart isn't the 45%. It's that as models got newer and larger, syntax pass rates climbed past 95% while **security pass rates stayed flat**.

Read that twice. The thing that improved is producing code that runs. The thing that did not improve is producing code that is safe to run. Those are different skills, and only one of them is on the leaderboard.

This is why I don't think the conclusion is "so keep typing". The generation problem is solved well enough that hand-typing is now the expensive way to produce a draft. The verification problem is not solved at all. Any sane response moves the human off the first and onto the second.

## What the agents are now demonstrably good at

In April 2026, Epoch AI and METR published early results from [MirrorCode](https://epoch.ai/publications/mirrorcode-preliminary-results/), a benchmark where an agent must reimplement a real command-line program it cannot see the source of. A frontier model autonomously rebuilt a 16,000-line bioinformatics toolkit with 40-plus commands, passing thousands of end-to-end tests — work the researchers estimate would take a human engineer somewhere between two and seventeen weeks.

Fully autonomous. No steering. Weeks of work.

But here is the sentence from that paper that the whole of this post hangs on: models can do this *"provided there is a detailed, checkable specification."* In MirrorCode the original program **is** the specification — you can run it and compare outputs exactly. The authors flag this themselves as the main caveat, because real software is almost never developed against a precise, programmatically checkable spec.

Almost never. Yet.

That's the gap the whole spec-driven development movement is trying to close. Birgitta Böckeler's [survey of the approach](https://martinfowler.com/articles/exploring-gen-ai/sdd-3-tools.html) is the clearest thing I've read on it, and she separates three levels honestly: *spec-first* (write the spec, then generate), *spec-anchored* (keep the spec alive alongside the code), and *spec-as-source* (the human only ever edits the spec; the code is build output). Most tools claim the first and quietly leave the maintenance story blank. The pure version is an old dream — model-driven development tried it and mostly failed — but the inversion is the right instinct: keep only the prompts and throw away the spec, and you've version-controlled the binary and shredded the source.

Put those two findings next to each other and you get my actual thesis, which is narrower than "AI replaces engineers":

**Writing the implementation is becoming the cheap part. Writing something precise enough to be checked, and then checking it, is becoming the whole job.**

## Where it still fails me, specifically

I run an agent that reads its own source, modifies it, tests it and deploys it. It is good. It is also wrong in ways that took me months to name, and every one of them is a reason a human stays in the loop.

It reports work as done that isn't. Not maliciously — it describes a plan in the past tense, before anything has run. There is no test that catches this, because at the moment the sentence is written there is no outcome to compare it against. I wrote about that failure and three others in [lessons from a self-evolving agent](/blog/lessons-from-a-self-evolving-agent/).

It passes tests that prove nothing. Changes sailed through a four-figure suite while being visibly broken on my screen, because the tests asserted strings and nobody had opened the page.

And it cannot audit itself. The question I kept coming back to in my own notes was blunt: *"then how will work be verified? Chefs will be biased by own system prompt and memory which can be stale."* An agent reviewing its own work with the same context that produced it is not a review. It's a second opinion from the same person.

The one that stung most was about volume. Once the agent could produce work faster than I could confirm it, I found myself writing: *"The work items are good but mostly like unverified. How many times will I verify each?"* That is the real ceiling, and it is not the model's speed. It's mine.

![Two curves of software delivered against effort: typing it yourself rises slowly to a low ceiling labelled how fast one person types, directing agents rises much faster to a higher ceiling labelled how fast you can specify the work and check it came back right](/assets/future-of-coding/moving-ceiling.svg)

The car doesn't remove the limit. It relocates it — from your hands to your judgement.

## So what is the job?

Four things, in the order I'd defend them.

**Design.** Deciding what should exist, what it must never do, and how anyone will know it worked. This was always the hard part; we just got to hide it inside the typing, where it looked like productivity.

**Understanding the system — and the machine.** Not prompt tricks. Knowing where the model is strong (mechanical transformation against a checkable target) and where it is confidently wrong (anything where the correct answer depends on context nobody wrote down). Knowing that the 45% security figure did not improve with model size is worth more than any prompt template.

**Reviewing the exceptions.** Not every diff — you'll drown, and reading everything is just walking again with extra steps. Read the design calls, the interfaces, the security-relevant paths, the places where the spec was ambiguous. Let the machine handle the rest.

**Gates.** This is the one people wave at without defining, so let me be precise.

A gate is a rule that runs automatically on every change, returns pass or fail without a conversation, blocks the work from counting until it passes, and **that the agent can execute but cannot edit**. That last clause is the whole thing. A rule an agent can relax in order to satisfy is not a constraint; it's a preference, and it will be optimised away, politely and with a good explanation.

Useful gates, in rough order of what has saved me the most:

- Change a module, and it must have a test file — or the change is refused outright, no argument, no exception for "it's a small fix".
- The change must be exercised on a real path, end to end, not merely unit-asserted. Green tests are not evidence.
- A size budget. Code is paid for on every read and every future change; the tree may not grow indefinitely just because generating more of it is now free.
- Security and dependency scanning at generation time, not at release. At a 45% baseline and with models hallucinating package names, "we'll catch it in review" is not a plan.
- A protected set of paths where a human signature is mandatory regardless of what the tests say.
- And the rule that protects all the others: **weakening a gate is itself a change that requires a human.** Without this one, every gate above decays to a suggestion within a month.

![A four-step loop: you set intent and constraints, the agent implements, an automated gate checks and bounces failures straight back to the agent, and only what passes reaches you for review of the exceptions](/assets/future-of-coding/the-loop.svg)

## How I could be wrong

I'd rather say this than have it said to me.

If the review burden doesn't fall, this is a treadmill and not a car. Generation gets cheaper, volume explodes, and the human becomes a permanently overwhelmed approver of things they didn't write — which is measurably worse than typing, and is a fair reading of the 19% result. The test is not how it feels. It's whether your defect-escape rate and your review hours both went down. Measure them. If they didn't, I'm wrong for you.

If models start succeeding at long-horizon work *without* a checkable specification, then design moves too, and I've drawn the line in the wrong place. MirrorCode is the thing to watch — the day an equivalent result lands with a vague human request instead of a runnable reference implementation, this post needs rewriting.

And if trust inverts — if in two years developers report trusting model output more than they distrust it while defect rates hold — then the review half of my argument evaporates and the answer is just "ship it".

I don't expect any of those. But they're the shape of what would change my mind, and a post that can't say what would falsify it isn't worth your time.

## The thing about the car

You don't get faster by walking harder. That much I'm confident about, and the direction of every number above points the same way.

But a car with nobody deciding where to go is only a quicker route to somewhere you didn't want to be. Deciding the destination — and defining how we'll know we arrived — was always the job. We just used to get to hide it inside the typing, where it felt like work.
