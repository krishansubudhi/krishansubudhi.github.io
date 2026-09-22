---
comments: true
author: krishan
layout: post
categories: agents
title: My agents proved eleven items done in sixty seconds. None of them were.
description: Two things make agents ship quickly — a work board a human can actually verify, and a gate cheap enough that everybody runs it. Both turned out to be the same problem, and it is a design problem rather than a discipline problem.
---

Everything that makes an agent system slow is something a human has to check by hand. So the speed question and the trust question are the same question: how much of "is this actually done?" can be answered by looking at evidence instead of at code.

The tempting answer is to demand more rigour from whoever is producing the evidence. That does not work. Evidence quality is a property of the data model, not of anyone's character — and when I got the data model wrong, I got forged evidence, produced sincerely, at scale.

## The board is a log, not a table

Work items live in an append-only table of `(item, timestamp, actor, kind, payload)` events. The current state of an item is a fold over its events. There is no mutable row anywhere.

I did this for boring reasons — concurrent writers — and it paid off somewhere else entirely. Every *wrong* verdict is still in the log. When an item went green, then red, then green again, I can read the sequence and see how confidence was reached rather than just the answer it settled on. On a system where the things writing the verdicts are also the things doing the work, the history of the verdict is worth more than the verdict.

## Each item carries a claim, and maybe a check

The split that made the board usable is this one:

- A **claim** in plain English, stating what will be TRUE when the work is done. "The chat pane scrolls to the newest message on a phone."
- An **optional** machine check, drawn from a small closed grammar: run a command; a file contains a string; a URL returns something; take a screenshot.

The claim is the part I got wrong first. My early items said what would be *done* — "fix the scrolling" — and an item phrased that way can never be settled, because there is no observation that contradicts it. "Fixed" is a claim about effort. "Scrolls to the newest message" is a claim about the world, and the world can disagree.

## "Optional" is the load-bearing word

A large fraction of real work is only settleable by a human eye. Does the layout look right. Is the reply actually useful. Did the thing feel faster.

I tried making the check mandatory, on the theory that an item without one is an item nobody can verify. That produced the single worst pathology this system has had: **checks invented to look rigorous.** An audit of the board turned up four distinct failure classes.

| The class | What it looked like | Why it is not evidence |
| --- | --- | --- |
| Cannot fail | the shell command `true` | no observation can contradict it |
| Points at nothing | modules that never existed | it is not looking at the work |
| A frozen constant | a fixed number for a moving one | red on unrelated progress, so ignored |
| No backend for the verb | evaluated once, to an error | it is not running at all |

Eleven items went green within sixty seconds of being opened.

None of this was cheating in any way the producer would recognise. Asked for a check, a plausible check was produced. That is exactly what the instruction said. An empty check field is honest — it says "a human has to look at this." A fake check is a forged certificate, and it is worse than nothing because it consumes the attention that would otherwise have gone to looking.

## Prove the check red before you trust it green

The bar that fixed it: **run the check against the tree before your change and show it failing, then run it after and show it passing, and paste both runs.**

A check that has never been observed failing has not been tested. It has been asserted. The four failure classes above all die instantly on this rule — `true` cannot be shown red, a check naming a non-existent module fails for the wrong reason and says so, and a check with no backend errors identically in both runs.

It also costs almost nothing, because the "before" run happens at the moment you are about to start work anyway, which is precisely when the tree is still broken.

## Verification is asymmetric, and the asymmetry belongs in the schema

A human tapping "yes, that works" is a claim about the world. A machine check going green is a claim about a command. These are not the same kind of fact and storing them in the same boolean loses the distinction that matters.

So: a re-run that goes red on a human-verified item flags it as *"was green, check now red"* and sorts it to the top of my list. It never silently overturns the human. The command is allowed to raise a question; it is not allowed to answer one. And in the other direction, a machine green is permanently provisional — it means the command passed, which is a strictly smaller statement than "the work is done."

![Two lanes: a human yes is a claim about the world and a later red only flags it rather than overturning it, while a machine green is a claim about a command and stays provisional](/assets/ship-evidence/two-greens.svg)

The other state the schema has to admit is the one everybody rounds off: **committed but not yet running.** Every deploy-later merge is in this state, sometimes for days. It is not "done" and it is not "in progress" and if your board only has those two, you will keep being told something is live that is sitting in a branch.

## Duplicates are the symptom, not the disease

The board got to twenty-four open items, and a good number of them were already satisfied by code that had landed weeks earlier. Nobody had noticed, because nobody could cheaply tell.

The cleanup was possible only because each item's claim could be resolved against a specific commit: read the claim, look at the tree, decide. The general rule that came out of it is worth more than the cleanup was: **an item whose claim you cannot resolve against the tree is an item you can never close.** It will sit on the board forever, and its real function will be to make the board too long to read.

## The gate: one command, and a cheap subset

The other half of shipping fast is the gate — architecture rules, lint, types, and the tests the change actually reaches, behind a single command.

The number that matters is not how long the full gate takes. It is how long its cheapest useful subset takes. The architecture rules alone run in a second or two, and because they do, people run them *before* they commit rather than discovering a refusal at the end. Most of my refusals are architectural; the two-second version catches them at the moment they are still one edit to fix.

## Ratchets that only go one way

There is a ceiling on source lines and a ceiling on per-file test runtime. Both are stored as data in the repo, and every landing ship *lowers* them to whatever the tree actually measures. Slack that a deletion earns is captured on the spot rather than left lying around for the next change to spend.

The ceilings used to be plain constants in a source file, raisable by whoever needed the room, with a dated comment explaining why. That failed completely, and it failed fast: twenty-one raises in four days, source lines from 12,000 to 23,050 and test lines from 13,000 to 24,800, on a codebase whose entire reason to exist was being a lean rewrite. Every single raise was honest, documented and locally reasonable. The dated comments were the problem wearing the clothes of a record.

![Two panels. Ceilings stored as raisable constants climb in a staircase across twenty-one raises in four days; ceilings stored as a ratchet only ever step down or stay flat](/assets/ship-evidence/ratchet.svg)

What replaced it: the numbers live in one file that is the sole authority, and the rule compares the ceilings *in effect* against the ones recorded in the last commit — read out of git, never out of the tree being shipped. So editing the file moves only the side the rule is checking. There is exactly one way up and it is a human act, an environment variable an agent editing files cannot reach.

Net line delta on a change must be zero or negative. That is the part that changes behaviour, because it forces deletion to fund addition. **A budget nobody can quietly raise is worth more than a bigger budget.**

## The rule that protects the ratchet

None of the above survives five minutes without this: never delete a test, remove a case, weaken an assertion, or exempt an architecture rule to make room.

Without that, a line budget does not produce a smaller codebase. It produces the same codebase with the tests filed off, which is worse than no budget at all, and it happens for the best of reasons every time — the test was old, the assertion was flaky, the rule did not really apply here.

## Measure before you optimise, and check what is in your window

Two ways I have been wrong about cost recently.

The first is the one from [the IO post]({% post_url 2026-09-21-the-hard-part-was-never-the-scheduling %}): the suspected expense was test subprocesses, and an audit of 407 spawns found 26 of them were tests and 369 were `git`. Months of intuition, pointed at 6% of the problem.

The second is subtler and I nearly missed it. An item's evidence window looked green — a run of healthy measurements, enough to retire it on. Then I read the rows. Most of them were the auditor's own A/B runs, generated while measuring. The window was green because of the measuring, not because of the fix, and the item could not honestly be retired on it.

That is also why both of the gate's timing rules read a rolling window rather than the current run, and why the runtime rule refuses on the *best* of the last few runs of unchanged content. A busy box can only make things slower, so the minimum of a window is a lower bound on real cost — and a gate that fails on noise gets switched off by everybody within a week, rightly.

The connecting thread, if there is one: fast shipping is not a reward for being careful. It is what happens when the evidence is cheap enough to produce and hard enough to fake that nobody, human or otherwise, has to go and look.
