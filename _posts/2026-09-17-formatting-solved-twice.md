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

Here's what it actually does to a real reply. The main model writes ordinary prose, because that's all it knows how to do:

> The deployment for the billing service finished. It touched three files across two commits, ran the test suite before merging, and the gate reported no failures. It went out on the fast lane, no restart needed.

The formatter reads that and retypes it, same facts, into the grammar it was given:

```
## Deploy: billing-service
status: shipped
files: 3
commits: 2
lane: fast · no restart
```

That's the trick, and it's a real one: the main model never has to know cards exist. Whatever it can say in prose, the second pass can reshape, because there's a mind on both ends. That is also the entire cost of this approach.

Everything else about it was a problem.

**It was slow, in the place where slow hurts most.** Measured on real replies: 4.7 seconds for prose, 11.0 for a table, 11.3 for a checklist. And because it sat in the one funnel every message passes through, it was also charged to the short interim messages — the "I've handed that to a worker, back shortly" line whose entire reason for existing is to arrive fast. Latency scaled with length: 765 characters in 42s, 4.6KB in 83s, and a 7KB reply got killed at the 90-second ceiling on every single run.

**It could lose your work, silently.** A model asked to "re-render this" will sometimes summarise it instead. So the output needed a gauntlet before it was allowed to replace the original, roughly:

```python
def accept(original, rendered):
    if len(rendered) < 0.5 * len(original):
        return False          # trimmed to fit, not reformatted
    if token_overlap(original, rendered) < 0.5:
        return False          # the words themselves didn't survive
    if not fences_balanced(rendered):
        return False
    if not card_well_formed(rendered):
        return False          # e.g. a card with zero items in it
    if rendered.strip() == original.strip():
        return False          # see below
    return True
```

Eight checks like this, in production, every one of them because the thing in the middle is not deterministic.

**The cheap model wasn't good enough, and failed invisibly.** Same seven real replies, same prompt, same guards: the cheap model formatted 2 of 7, the mid-tier one 5 of 7. Every single cheap-model miss was that last check — `rendered == original`, a *silent echo*. It handed back the input unchanged, exit code 0, no guard tripped, nothing in the logs. The failure mode of an LLM in your pipeline is not an exception. It's plausible output.

It also wasn't worth calling for everything. Below 240 characters the reply is already short enough to read at a glance, so the whole pass is skipped; below 450 it gets a plainer "receipt" style instead of a full card. And because latency scaled with length, the timeout had to scale with it too:

```python
def timeout_for(nchars):
    return min(300, 60 + 25 * nchars / 1000)
```

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

Same deploy message as before, but now the main model writes the structured version directly, because the house style told it to:

```
## Deploy: billing-service
status: shipped
files: 3
commits: 2
lane: fast, no restart
```

The parser sees a heading, then short `key: value` lines, and draws a card with a header bar and a row of fact tiles — no model, no retyping, just a string being read for the shape it's already in. The catch is right there too: if the model had written the prose version instead, this parser renders prose. It has no way to fix that, because fixing it was never its job.

Colour comes from a word list, not a judgement:

```python
GOOD = {"shipped", "live", "passed", "green"}
BAD  = {"failed", "broken", "blocked", "timeout"}

def pill_colour(word):
    if word in GOOD: return "green"
    if word in BAD:  return "red"
    return "neutral"   # unrecognised word — say nothing, don't guess
```

A wrong colour on a status is worse than no colour.

The tappable chips have the same instinct behind them:

```python
MAX_CHIPS, MAX_CHARS = 8, 80

def parse_chips(options):
    if len(options) > MAX_CHIPS or any(len(o) > MAX_CHARS for o in options):
        return None   # whole row drops to a plain bulleted list
    return [chip(o) for o in options]
```

That limit used to be 60. Real questions kept producing options that landed at 65–68 characters, and the whole row would silently vanish back into a bulleted list — a regression nobody noticed for a while because a bulleted list still *works*, it's just not what the chip was for. There's now a test that asserts an option of exactly 78 characters survives. Notice the failure shape: one option two characters too long doesn't get truncated and it doesn't get dropped on its own — the entire row degrades. A row of four choices with the fifth silently missing is a worse lie than a plain list, so the rule is all-or-nothing on purpose.

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

**Degrade to boring, never to broken.** The chip row above is my favourite example of this: not "render three chips and drop the fourth," the whole row, every time. Partial success is usually the worst available outcome in a UI.

**The cost of the wrong architecture isn't the code, it's the guards.** Version one is roughly 855 lines, and the majority of it is defending against the non-determinism in the middle. Version two is 607 and most of it is the actual parser. I didn't delete a feature. I deleted the reason the feature needed defending.

---

*The measurement that made me care about any of this — a quarter of every long reply being pipes and dashes — is in [the previous post]({{ '/blog/the-layout-tax/' | relative_url }}). And neither pipeline matters if the card says the wrong thing first: [why formatting is a correctness problem, not a cosmetic one]({{ '/blog/the-reply-was-correct-nobody-could-tell/' | relative_url }}).*
