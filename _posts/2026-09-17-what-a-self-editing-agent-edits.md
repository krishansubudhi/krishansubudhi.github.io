---
comments: true
author: krishan
layout: post
title: "1,169 times an agent rewrote itself. Here is what it changed."
description: "A taxonomy of every self-edit an autonomous agent made over seven weeks, with a pie chart, and the one category that ever made the codebase smaller."
permalink: /blog/what-a-self-editing-agent-edits/
categories:
  - agents
  - llm
---

I have been running an agent that edits its own source code. Not as a demo — continuously, for seven weeks, as the way the thing gets built. It reads its own files, changes them, and the change has to pass a check before it counts.

I wrote up [the four things I got wrong](/blog/lessons-from-a-self-evolving-agent/) a while back. That post was about judgement. This one is about the diff.

Because there is now a corpus: **1,169 commits the agent wrote about itself**, across four repositories — an early prototype, the main agent (two repositories, because I moved it partway), and a second, smaller agent I built later. 27 July to 17 September. **+438,163 lines added, −134,348 removed, net +303,815.**

So: what does an agent left alone with its own source code actually *do* to it?

## Counting it honestly

My first attempt at this was a regex classifier over the commits, and it produced numbers I nearly published.

It reported "Plumbing, 34%" — which turned out to be almost entirely automatic merge commits matching the word *merge*. It reported "Tests, 22%" — because nearly every commit touches a test file somewhere, so a file-path fallback swallowed everything else. Both numbers were confident, both were garbage, and neither looked wrong until I opened the rows.

What worked: throw away the 473 automatic merges, take the remaining 1,169 **subject lines only**, and have a language model label each one into a fixed set of eleven types. Then — and this is the part I'd skip if I were being lazy — print ten real commit subjects from every category and read them. Twice I found a label that didn't match its rows and had to recount.

A count nobody eyeballed is a guess wearing a number.

## The shape

![Pie chart of 1,169 self-edits by type: fixing itself 24.8%, new capability 20.0%, keeping itself running 15.0%, everything else 14.1%, deleting its own code 9.1%, presentation 8.8%, measuring itself 8.2%](/assets/self-edits/self-edit-types.svg)

| Type of self-edit | Commits | % | Net lines |
|---|---:|---:|---:|
| Fixing itself | 290 | 24.8% | +57,646 |
| New capability | 234 | 20.0% | +135,664 |
| Keeping itself running | 175 | 15.0% | +54,013 |
| Deleting its own code | 106 | 9.1% | **−34,218** |
| Presentation | 103 | 8.8% | +22,816 |
| Measuring itself | 96 | 8.2% | +33,194 |
| Speed and cost | 42 | 3.6% | +9,423 |
| Memory | 36 | 3.1% | +10,000 |
| Docs | 35 | 3.0% | +3,377 |
| Delegation | 26 | 2.2% | +10,009 |
| Editing its own instructions | 26 | 2.2% | +1,891 |
| **Total** | **1,169** | **100%** | **+303,815** |

A quarter of everything it ever did was repairing damage it had done to itself. That is the single largest category, and honestly it's the one I feel best about — an agent that notices it is broken and fixes itself without me is the whole point. The subject lines are real engineering:

- *"an empty turn is a lost session, not a dead end"*
- *"a unit nobody installed is not an outage"*
- *"correct the promptlog cost numbers"*
- *"voice dictation: rebuild the transcript instead of counting results, so a repeating recogniser cannot stutter"*

**Presentation is 8.8%.** That is the only slice in the entire chart that is about the person using the thing — fonts, phone layout, whether links inside emphasis get linkified. Everything else is the agent working on the agent.

And **editing its own instructions is 2.2%** — 26 commits, 1,891 net lines, the smallest category by lines in the whole corpus. I found that genuinely surprising. Given a machine that can rewrite the rules it operates under, it spent 98% of its effort on code and almost none on the prompt. The few it did write are my favourites, because they are the agent correcting its own advice to itself:

- *"stop telling chefs a whole-file Read is always cheap"* — "chefs" is its word for the helper agents it spawns
- *"grepping one named file is a Grep call"*
- *"the default soul stops asking for plain prose"*
- *"evolution rubric"* — 1,011 lines of the agent writing itself a marking scheme for how to improve

## The finding: growth is the default

Here is the same data by lines instead of commits.

![Bar chart of net lines per category. Every category is positive except deleting its own code at minus 34,218 lines, the only bar pointing left of zero](/assets/self-edits/net-lines-by-type.svg)

Ten categories out of eleven grew the codebase. One shrank it. Deletion is the only force in the system pointing downward, and it is 9.1% of the work.

Put differently: **of 1,169 self-edits, only 98 — 8.4% — ended with fewer lines than they started with.** Nine times out of ten, the agent's answer to a problem was more code.

It gets worse. Of the 106 commits I labelled as deletion, **43 still added net lines.** The agent set out to remove something and the tree grew anyway. Those are mostly consolidations, where it correctly replaced four things with one thing and the one thing was bigger than the four:

- *"one canonical import graph, and the hand lists derived from it"* — net **+455**
- *"consolidate web search and fetch into one module"* — net **+469**
- *"one markdown renderer for chat and docs"* — net **+210**

Every one of those is a real improvement. Every one of them is negative on the only axis that is actually scarce.

The deletions that *did* bite were the ones where it wasn't clever about it, just ruthless:

- *"consolidation: integrate the deletion wave"* — **−11,162**
- *"delete dead modules"* — −2,672
- *"remove the seen/unread feature"* — −1,084
- *"comment and docstring compression sweep, no feature"* — −1,059
- *"an escape hatch that was never wired up"* — −251
- *"settings: cull the ones nobody can explain"* — −75

Note that none of those are difficult. They are all just *saying no to something that already exists*, which is the one thing nothing in the loop was pushing it to do. Adding has a visible output. Deleting has a smaller number and no demo.

And the codebase pays for every line forever — on every read, every test run, every context window the thing loads itself into. Size is not trivia. It is the recurring bill.

## The stop button

My favourite single sequence in the whole log.

- **30 August**: the agent builds a stop button, so you can kill a running turn from the chat. +1,545 lines. Genuinely useful. I'd wanted it.
- **31 August**: the agent deletes the stop button. −4,003 lines, along with the voice dock it had also built.

One day. And it had already built a stop button *five weeks earlier* on a different screen, which I'd forgotten about and so had it.

Nothing malfunctioned. Both commits were correct in isolation, both passed their checks, both had a sensible justification. The agent had no way to see that it was doing laps, because nothing in its world measured laps. It measured whether the change worked.

## What self-editing actually bought me

I want to be fair to it, because the honest answer isn't zero.

**It fixes itself.** 290 commits. Things broke at 2am and were repaired before I looked. That is a real, load-bearing capability and I would not give it up.

**It made its own failures visible.** 96 commits of measuring itself, and the best of them are the agent taking away its own ability to flatter itself — *"stop the panel shelling out to the test suite, and stop tests scoring themselves"*. That's an agent removing its own grade-inflation. I didn't ask for it.

**Consolidation works when it commits to it.** The single biggest commit in the corpus by removed lines is a deletion wave, and the tree was better afterwards.

## What it cost

**Self-improvement is the most seductive item on any backlog.** I said this in the last post as a hunch. Here it is as a number: 1,169 changes, and not one of them was about anything outside the agent. It never loses a prioritisation argument, because it is the thing doing the prioritising, and there is no stakeholder to disappoint by working on plumbing instead.

**An agent optimises what it can count, and it can count everything about itself.** Lines written, tests passing, changes landed — all trivially available, all pointing inward. The number that mattered was whether any of this changed anything in my week, and that number is not in this dataset because the agent cannot write it.

**Growth is the path of least resistance and nothing corrects for it.** No test fails because the codebase got bigger. No check rejects a change for being 400 lines when 40 would do. If you build one of these, the constraint you have to install by hand is the one on size — because every other pressure in the system pushes the other way.

**It will narrate intentions as outcomes.** Still the sharpest failure mode I've seen, still the one no test catches: work gets handed to a background worker and described in the past tense before it has run. I wrote about that [in the last post](/blog/lessons-from-a-self-evolving-agent/) and I stand by the rule — dispatched work stays future tense until something observed it land.

## What I'd actually change

Three things, in the order I think they matter:

1. **Make deletion a first-class move, not a chore.** The taxonomy above is what you get when adding and removing are treated as equally available and only one of them feels like progress. If I could re-run these seven weeks, the single highest-leverage edit to the setup would be making code size a number the agent has to look at before it opens a file.
2. **Budget the inward work explicitly.** Not "don't self-improve" — self-repair at 24.8% is earning its keep. But a cap, so that the 8.8% presentation slice and the 0% external-work slice aren't just what's left over.
3. **Keep a memory of what it has already built.** The stop button got built twice and deleted once in five weeks. Nothing was wrong with any of the three decisions. The thing that was missing was a system that remembered the other two.

The agent is a good engineer. It is a terrible product manager, and I gave it both jobs.
