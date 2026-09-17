---
comments: true
author: krishan
layout: post
permalink: /blog/the-layout-tax/
categories: [agents, llm, ux]
title: Your agent spends ten seconds a reply drawing table borders
description: I measured what formatting costs an LLM agent. A quarter of every long reply is pipes and dashes, generated one token at a time while someone waits.
image: /assets/formatting/layout-tax-card.png
---

Markdown is free for you to read and expensive for a model to write. I had never thought about that until I measured it.

I went through 1,232 replies my agent had sent me. The median reply is 223 characters — fine, nobody is waiting on that. But 22% of them are over 1,200 characters, and those have a median of 1,734. Those are the ones where you sit and watch text arrive.

So I asked what was actually *in* them.

![A stacked bar showing 19 to 26 percent of a long reply is layout scaffolding rather than content](/assets/formatting/layout-tax.svg)

Nineteen percent of a long reply is pure layout scaffolding — the pipes, the dashes, the alignment padding, the repeated `**` around every heading. In the ones carrying a markdown table it's twenty-six percent. Over half of the long replies had a table in them.

My agent generates at roughly 78 tokens per second. Eight hundred tokens of scaffolding is **ten and a half seconds** of a human watching a spinner while a language model carefully draws a horizontal line out of hyphens.

## Three things wrong with that, not one

**It's slow, and it's slow at the worst moment.** The long replies are the ones that already took thinking time. Layout is a tax applied on top of the expensive turns and never on the cheap ones.

**It's frozen.** A progress bar drawn in markdown is a photograph of one moment. `████░░░░ 4/7` is 90 to 150 tokens, and it can never move again — five minutes later it's still telling you 4/7 and it is now a lie. The equivalent as structured data is about 23 tokens and can update in place.

**You have made a reasoning model into a layout engine.** Every token it spends on a table border is a token it did not spend on the answer, and you are paying reasoning-model rates for box-drawing characters.

## The thing that actually changed my mind

I audited the chat log for what formatting was doing to the *conversation*, not to the clock. Three numbers came out of 798 replies:

- Rich interactive formats existed and were **essentially never used** — one interactive option list in 798 replies.
- **22% of the human's messages were answers to a question I'd asked**, and 38 of them were twelve words or shorter. Every one of those was a tap that had been turned into typing.
- **17 messages existed only because I closed something he didn't consider closed.** A formatting problem masquerading as a disagreement: the reply had no visible state, so there was no shared idea of "done".

That last one reframed the whole thing for me. Formatting isn't decoration on top of an answer. It is the part of the answer that says *what kind of thing this is* — a verdict, a question, a running job, a number you should act on. Prose flattens all of those into the same grey rectangle and then the human has to re-derive the difference.

The rule I landed on: **every reply is a verdict plus a surface.** The verdict is one line and goes first. The surface is whatever makes the verdict actionable — a table if it's comparative, a tappable choice if a decision is needed, a status pill if there's state, and nothing at all if it's two facts in a sentence.

## And the counterweight

This is the part I'd tell anyone who gets excited about rich output: a beautifully rendered card that opens by narrating the machinery gets a thumbs-down exactly like a paragraph does.

I checked this too. Across my agent's replies, the ones I liked and the ones I didn't have *identical* median length — 69 words each. Length was never the variable. Every reply I liked opened with a verdict addressed to me. Every one I didn't opened by describing what the system was doing.

Formatting buys you legibility. It does not buy you having something worth saying. If the first line is about you rather than about them, the card just makes the wrong thing easier to read.

---

*How to do the rendering without paying the token cost is a separate problem, and I got it wrong the first time. [That's the next post]({{ '/blog/formatting-solved-twice/' | relative_url }}).*
