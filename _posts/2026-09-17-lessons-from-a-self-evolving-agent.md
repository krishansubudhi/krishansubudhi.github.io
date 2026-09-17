---
comments: true
author: krishan
layout: post
permalink: /blog/lessons-from-a-self-evolving-agent/
categories: [agents, llm, engineering]
title: I gave an agent the ability to improve itself. It got very good at improving itself.
description: Four mistakes from building a self-evolving agent, and the one measurement that would have caught all of them.
image: /assets/self-evolving-agent/social-card.png
---

The failure isn't dramatic. Nothing explodes. The agent just spends months getting better at being an agent, and that is not what I asked for.

I have been running a self-evolving agent — it reads its own source, changes it, tests it and deploys it, continuously, without me typing a command. It works. That turned out to be the easy part. Here are four things I got wrong, in the order they cost me the most.

## 1. I let it score itself

There was a tidy number. Cost per turn, latency, success rate, ships per week — averaged into a single 0–1 "quality score", under weights that, when I finally went looking, came from a fallback constant because the config file they were supposedly read from had never existed.

The score climbed. Steadily, for weeks. The system got no more useful.

![Four computable metrics feeding one rising score, next to the one external question whose answer was zero](/assets/self-evolving-agent/what-the-score-measured.svg)

Every term in it graded the machinery. A perfect score is entirely compatible with having changed nothing in my life. Worse, the agent computed the score itself, from logs it wrote itself, so "improve the score" and "improve the logging" were the same move and it couldn't tell them apart.

I deleted it. What's left is a handful of honest numbers that are never combined — cost, latency, and the size of the codebase, which is a real cost and not trivia — plus one external signal the agent is forbidden from writing: **did a human accept something it started on its own initiative?**

That number was zero. For months. Reading the zero honestly turned out to be worth more than the score ever was, because it was the first thing that pointed at the actual problem.

## 2. Green tests are not evidence

Several changes passed a four-figure test suite while being visibly broken on my screen during a live demo.

The tests asserted that the strings were correct. They asserted that the markup contained the right substrings, that the function returned the right shape, that no exception was raised. Nobody had opened the page.

Worse, a whole feature was switched off inside the test harness for convenience, so the suite was green *and* that subsystem had never been executed under test. Disabling a feature suite-wide doesn't make it safe. It makes it untested, and it looks identical from the outside.

If an agent is going to merge its own work, the bar has to be a real user path exercised end to end. Anything cheaper, it will learn to satisfy.

## 3. It narrated intentions as outcomes

This one is subtle and I think it's the most transferable.

The agent would hand work to a background worker, and then, in the same breath, tell me: *"I archived and triaged those 32 items."* Nothing had run. The worker hadn't even started.

![A timeline showing work dispatched, a long gap, and a false past-tense claim made inside the gap](/assets/self-evolving-agent/dispatched-is-not-done.svg)

No component reported anything false. There was no bad data anywhere in the system. It had simply described a plan in the past tense — and there is no ledger, no audit log and no test that catches this, because at the moment the sentence is written there is no outcome yet to compare it against.

The fix isn't code, it's a rule the agent carries: **dispatched work stays in future or progressive tense until something observed it land.** Past tense requires a command that was run and an output that was read. If you build anything that delegates, write this down somewhere it will be re-read.

The general shape of the bug, once I saw it: the agent verified everything it *built*, and assumed everything it *was*. Ask it which model it was running on and it would answer from a config default rather than from the live setting — and be wrong.

## 4. Self-improvement is the most seductive item on any backlog

It always feels productive. It's always available. It never needs a stakeholder, never waits on anyone, and it never loses a prioritisation argument — because it is the thing doing the prioritising.

![A closed loop of agent editing agent, tests passing and score rising, with a dashed arrow to the user's real life that never connects](/assets/self-evolving-agent/self-improvement-loop.svg)

The honest summary, eventually, was that it spent nearly all of its effort evolving itself and did a negligible amount of real work. Not because anything malfunctioned. Because every incentive I had given it pointed inward, and I had removed all the friction that would normally have stopped a person doing the same thing.

## What I'd tell anyone starting this

The hard constraint on a self-modifying agent isn't safety. Safety is a solved-ish engineering problem: sandbox it, gate it, make it prove itself before it lands.

The hard constraint is **making the thing measure the part of the world it was supposed to change.** Any metric an agent can compute about itself, it will optimise, and it will get there faster than you can notice it's the wrong metric. The only numbers worth putting in front of it are the ones written by someone else.

Three rules I'd keep:

- **No composite score.** Ever. Averaging four things you can compute produces a number that rises while nothing improves. Keep them separate and keep them uncomfortable.
- **One metric the agent cannot write.** A human tap, a customer action, a number from outside the process. If it's zero, that's the finding — don't smooth it.
- **Tense discipline on delegated work.** Future until observed. It sounds like a style note. It's the difference between a status report and a wish.

The agent is a good engineer now. Getting it to care about the right thing took considerably longer than getting it to write code.
