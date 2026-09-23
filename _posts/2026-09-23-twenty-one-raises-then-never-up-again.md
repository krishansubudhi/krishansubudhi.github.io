---
comments: true
author: krishan
layout: post
categories: agents
title: Twenty-one raises, then never up again. The budget got so tight it made workers delete the wrong things.
description: A one-way ratchet on code size stopped my agent-written codebase growing. Clamping it exactly onto the tree turned out to be the mistake, and the fix is three hundred lines of deliberate slack.
---

When agents write your code, it grows. Not dramatically, not in any one change — every fix adds a few lines, nothing is ever deleted, and each addition is locally correct. Then one morning the tree is unmaintainable and you cannot point at the commit that did it, because there isn't one.

I wrote about the ratchet that stopped this [in passing a couple of days ago]({% post_url 2026-09-21-shipping-fast-is-an-evidence-problem %}). This post is the part I skipped: the one detail that makes it actually hold, and the way it went wrong once it was holding.

## The version that failed

The cap started life as a constant in a source file, raisable by whoever needed the room, with a dated comment explaining why.

`git log -G'^SOURCE_CEILING' --oneline | wc -l` returns **21**. Twenty-one commits moved that constant. Every one of them moved it up. The resulting raise log runs to 398 lines and reads, honestly, quite well — each entry has a reason, and each reason is true.

That is the failure mode worth naming. Nobody cheated. A worker whose change did not fit found a number in the source, changed the number, wrote down why, and shipped. The dated comments were the problem wearing the clothes of a record.

## What replaced it, and the one detail that matters

The numbers moved into `ceilings.json`, which is the sole authority — loaded once, so there is no second copy to edit. Two numbers, source and tests, never added together. Tests get the larger allowance on purpose: one combined figure reads as "a big repo" instead of "the tests are doing their job."

Then the rule. Every landing ship rewrites the file from measurement, and a separate rule refuses any ship where the ceiling went up.

Here is the part people skip, and it is the whole mechanism:

> The rule compares the ceilings **in effect in this tree** against the ceilings **recorded in the last commit, read out of git** — never off disk.

Read it off disk and you have written a rule that checks the file against itself, which is no rule at all. Reading the recorded side out of git means the tree being shipped can only move the left-hand side of the comparison. Editing `ceilings.json` and hardcoding a constant past it become the *same* violation, detected identically, because both of them move the caps in effect and neither can touch what the last commit says.

There is exactly one way up, and it is an environment variable set by hand at ship time, shouted about in the output. A worker that only edits files cannot reach it.

The recorded source ceiling over one run of that regime:

> 22308 · 22305 · 22294 · 22293 · 22282 · 22281 · 22280 · 22280 · 22214 · 22209 · 22207 · 22170

Down or flat, twelve values, never once up. Compare the shape to twenty-one consecutive raises. Same repository, same workers, same week.

A companion ratchet does the same thing to per-file test runtime in `runtimes.json`, holding the best time each test file has ever run in. That one has no override at all.

## The rule that keeps it from being a lie

None of this survives five minutes without one prohibition: **never delete a test, remove a case, weaken an assertion, or exempt an architecture rule to make room.**

Without it a line budget does not produce a smaller codebase. It produces the same codebase with the tests filed off, and it happens for the best of reasons every time — the test was old, the assertion was flaky, the rule did not really apply here. A budget that can be met by deleting the things that measure you is a budget that measures nothing.

## Then it got too tight

The ratchet worked. It worked so well that it produced a pathology I did not see coming.

Every landing ship wrote `min(cap, measured)`. The cap was clamped exactly onto the tree. Which means **every change opened with zero headroom.** Not "a little tight" — zero. A two-line fix could not land until two lines came from somewhere.

So workers went shopping. They would take a small, obviously-correct fix, find it did not fit, and go hunting through unrelated parts of the repository for dead code to fund it. One funded a settings-page fix by deleting an unrelated dead function. The fix was right. The deletion was probably fine. But the change that landed was now two unrelated things in one commit, one of which nobody had asked for and nobody reviewed with any care, and the worker had spent most of its effort on the part that was not the job.

That is precisely the pathology the constraint exists to prevent — unreviewed, unrelated code churn — arriving through a different door. A budget so tight that it makes people delete the wrong things is not a tighter budget. It is a different bug.

## The fix: settle above the tree, never onto it

One function, and it is the whole change:

```python
CEILING_SLACK = 300

def settle(caps, sizes):
    return {k: min(caps[k], sizes[k] + CEILING_SLACK)
            for k in ("source", "tests")}
```

Called from the same place the old clamp was called from. The cap now lands three hundred lines above whatever the tree measured, instead of on top of it. Deleting still banks room for the next change — up to three hundred lines and no further.

## The escalator, and why it needs no guard

The obvious objection, and the first thing I went looking for: delete 500, bank the 500, add 500 back, repeat. A pump. Free growth, one lap at a time.

It cannot happen, and the reason is nicer than a guard would have been. Expand the recurrence and the cap after *any* sequence of ships is exactly:

> `min(the cap it started at, 300 + the smallest tree ever measured)`

A closed form with no memory of the path. Headroom is a **level** over the low-water mark, not a balance that accumulates. The re-adding ship settles against the same historic minimum the deleting ship did, so the second lap buys nothing; and once the tree is three hundred lines over its own record, the growth rule refuses the next line. Total give-back against the old clamp is three hundred lines per counter **once, for the life of the repository** — not per ship.

There is nothing to guard, because arithmetic already refused it. The test that pins this is named `test_no_sequence_of_ships_can_walk_the_ceiling_upward`, and it walks a deliberately vicious sequence, asserting the closed form after every step:

![Nine ships whose measured tree oscillates between 900 and 250 lines; the ceiling steps down twice, to 600 and then to 550, and never moves again — 550 being 300 lines above 250, the smallest tree ever measured](/assets/ceiling-headroom/headroom-is-a-level.svg)

The tree in that sequence ends at 700 after visiting 900 twice. The cap ends at 550, parked over an all-time low it touched at ship four and never beat. Six ships of thrashing bought nothing.

## Where 300 came from

Not a round number somebody liked. Over the last 120 non-merge commits, I measured the net growth of a commit **that grew at all**:

| | source lines | test lines |
| --- | --- | --- |
| median | 71 | 131 |
| 75th percentile | 274 | 278 |
| slack chosen | 300 | 300 |

So three hundred covers roughly three ordinary changes in four, and none of the big ones. A feature that wants six hundred lines is a conversation and a deliberate override, not something the ratchet quietly funds — which is the line I wanted the number to sit on. One figure for both halves because at that quantile the two columns are four lines apart, and two numbers there would imply a precision I do not have.

Two deliberate choices inside that:

**Absolute, not a percentage.** What is being sized is *one ordinary change* — a count of lines that does not scale with the size of the repository. A five-percent slack would have shrunk the allowance exactly as the tree got leaner, tightening hardest at the moment the work was going best. That is the wrong incentive pointed the wrong way.

**Sized off what commits actually do, not off what they should do.** The percentile came from the log. Had I picked the number from taste I would have picked it low, because low feels disciplined, and low is how I got here.

## The honest ending

The very next change to land after the headroom shipped was a *simplification* of a UI page — collapsing a cluttered status display down to one status and one line of evidence. It came in at **+24 source lines and +32 test lines.**

It made the page simpler and the code bigger. Both of those are true at once, and they are true at once far more often than a line budget wants to admit. Under the old clamp that change does not land as written; it lands with an unrelated deletion bolted to it, or it does not land at all.

The ratchet's job was never to make the number go down. It was to make sure the number can only go down *by accident of good work*, and never by somebody needing room. Those are different goals, and the three hundred lines are what separates them.
