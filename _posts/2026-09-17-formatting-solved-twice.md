---
comments: true
author: krishan
layout: post
permalink: /blog/formatting-solved-twice/
categories: [agents, llm, architecture]
title: I solved the same problem twice, in opposite directions
description: Version one put a language model in the reply path to format replies. Version two deleted it and wrote a parser. Here is what each one actually cost.
image: /assets/formatting/two-pipelines-card.png
---

I wanted my agent's replies to render as proper cards — status pills, tables that scroll on a phone, tappable options instead of "reply with 1, 2 or 3".

The obvious two approaches both have a well-known failure mode:

1. **Teach the model a card syntax in its system prompt.** It forgets under load, the grammar costs it tokens mid-answer, and the day you change the renderer the prompt silently describes something that no longer exists.
2. **Write regexes that guess which paragraph is "really" a table.** Guessing.

So I did a third thing. Then, six weeks later, I did a fourth thing, and deleted the third.

![Two reply pipelines side by side: a model in the funnel versus a parser at render time](/assets/formatting/two-pipelines.svg)

## Version one: put a second model in the funnel

Every outgoing message passes through exactly one function on its way to a human. I put a small language model there. It receives the finished reply and a grammar, and its only job is to retype the same content with cards around it. The main model never learns the syntax, never spends tokens on it, and can't forget it under pressure. That part genuinely worked.

Everything else about it was a problem.

**It was slow, in the place where slow hurts most.** Measured on real replies: 4.7 seconds for prose, 11.0 for a table, 11.3 for a checklist. And because it sat in the one funnel every message passes through, it was also charged to the short interim messages — the "I've handed that to a worker, back shortly" line whose entire reason for existing is to arrive fast. Latency scaled with length: 765 characters in 42s, 4.6KB in 83s, and a 7KB reply got killed at the 90-second ceiling on every single run.

**It could lose your work, silently.** A model asked to "re-render this" will sometimes summarise it instead. So the output needed a gauntlet: reject if the result is under half the input's length; a token-retention check that at least half the meaningful words survived; a check that the fences actually close; a check that a card has at least one item in it, because an empty one renders as a grey code block. Eight guards, every one of which exists because the thing in the middle is not deterministic.

**The cheap model wasn't good enough, and failed invisibly.** Same seven real replies, same prompt, same guards: the cheap model formatted 2 of 7, the mid-tier one 5 of 7. Every single cheap-model miss was a *silent echo* — it handed back the input unchanged, exit code 0, no guard tripped, nothing in the logs. The failure mode of an LLM in your pipeline is not an exception. It's plausible output.

**And it was switched off in the test suite,** because it was too slow to run 1,000 times. Which meant the suite was green and that subsystem had never once been executed under test. A feature disabled suite-wide isn't safe. It's untested, and from the outside those look identical.

The eventual shape was defensible — move it off the write path onto a background thread, rewrite the stored message afterwards, let the card swap in under a bubble already on screen. It worked. It was still a language model, a subprocess and eight guards to draw a table.

## Version two: delete the model, write the parser

The rebuild took the opposite bet. **The head writes plain markdown and never a fence grammar. A deterministic parser reads the markdown's own shape and draws it** — at render time, inline in the page request, as a pure function of a string.

No prompt grammar to forget. No guessing: it isn't inferring that a paragraph is "really" a table, it's reading a table that is already a table in markdown.

The house style the model gets is four lines, and every one of them is ordinary markdown it already writes:

```
* `## A heading` and what follows becomes a card with a header bar.
* A run of short `key: value` lines becomes a row of fact tiles.
* A markdown table becomes a table that scrolls sideways on a phone.
* A question ending in `?` followed by two to eight short options,
  one per bullet, becomes tappable chips.
```

That instruction lives in the same source file as the renderer that honours it. Deliberately — they're one decision, and splitting them is exactly how the previous version ended up with a prompt describing a grammar nothing rendered any more.

Colour comes from a word list, not a judgement: `shipped`, `live`, `passed`, `green` draw green; `failed`, `broken`, `blocked`, `timeout` draw red; anything unrecognised gets the neutral pill, because a wrong colour on a status is worse than no colour.

## Side by side

| | v1: model in the funnel | v2: parser at render time |
|---|---|---|
| Mechanism | LLM subprocess retypes the reply | markdown parser → HTML |
| Position | write time, background thread | render time, in the page request |
| Latency | 4.7–11.3s typical, 90s ceiling | zero |
| Cost per reply | a model call | none |
| Deterministic | no — same reply can render differently twice | yes |
| Can it lose content | yes; needs a 50% token-retention check | dropping a word would have to be a bug in a loop |
| Failure mode | plausible wrong output, exit 0 | degrades to plainer markup |
| Guards needed | 8 | 0 |
| Tests | stubbed out; untested for weeks | 369 lines of pure assertions, nothing stubbed |

## What I'd actually take from this

**Decide whether your problem is judgement or structure.** I reached for a model because "make this pretty" *sounds* like judgement. It isn't. A markdown table is already structured; a `key: value` line is already structured. Once the input carries the structure, you want a parser, and a parser has no bad days.

**An LLM in a pipeline fails by being plausible.** A parser that can't handle its input throws. A model that can't handle its input returns something reasonable-looking and wrong, at exit code 0. Every guard in version one existed to detect a failure that had already been reported as a success — and the best guard I had still missed the most common failure.

**Degrade to boring, never to broken.** My favourite line in the rewrite: if one option in a tappable row is too long, the *whole row* drops back to a plain bullet list. Not "render three chips and drop the fourth" — a row of choices with one missing is a worse lie than a list of bullets. Partial success is usually the worst available outcome in a UI.

**The cost of the wrong architecture isn't the code, it's the guards.** Version one is roughly 855 lines, and the majority of it is defending against the non-determinism in the middle. Version two is 607 and most of it is the actual parser. I didn't delete a feature. I deleted the reason the feature needed defending.

---

*The measurement that made me care about any of this — a quarter of every long reply being pipes and dashes — is in [the previous post]({{ '/blog/the-layout-tax/' | relative_url }}).*
