---
comments: true
author: krishan
layout: post
categories: agents
title: "Ratchet: a code-size budget for your repo that only goes down."
description: "A small tool that refuses any change which grows your code past its budget, and makes every raise an explicit, logged grant that shows up in the diff. How it works, how I run it on its own repo, and where it does not help."
---

**A code-size budget for your repo that only goes down.**

Coding agents are good at adding code and bad at deciding not to. Each change is locally correct, each addition is a few lines, and nothing gets deleted. One day the codebase is much bigger than anyone meant it to be, and there is no commit you can point at, because there was never one decision.

Ratchet puts a number on that and makes the number one-way. It is on PyPI as `ratchet-size` and on GitHub at [krishansubudhi/ratchet](https://github.com/krishansubudhi/ratchet).

The pitch can sound like "no code", and that is not the goal. The goal is less code by adding a guard. A feature that needs six hundred lines should get six hundred lines. What I wanted was for someone to say yes out loud, with a reason written down, and for nothing else to be able to raise the number. The budget is the guard. The grant is the yes.

## How it works

Three parts:

- **The budget.** `.ratchet.json` holds a ceiling for each group, `source` and `tests`. Only code files count. Blank lines don't, and neither do docs.
- **The check.** `ratchet check` runs in CI, in a git hook, or inside the agent's own session. If a group is over its ceiling it exits 1 and says what grew. The refusal text is written for an agent: it offers to cut lines or to stop and ask a human, and it tells the agent not to edit the budget file to get around it.
- **The grant.** `ratchet grant +54 --group source --reason "..."` raises the ceiling and appends a line to `.ratchet-grants.jsonl` with the reason and who ran it. You commit both files together, so the raise shows up in the diff.

Shrinking does not move the ceiling by itself. `ratchet tighten` does that when you run it, and until you do, the room you freed is still there to spend. I come back to what that costs.

<svg viewBox="0 0 640 600" role="img" style="width:100%;max-width:720px;height:auto;color:var(--text-color)" aria-label="Flow: edit, check, then exit 0 and commit, or exit 1 and either cut lines or have a human grant"><defs><marker id="rt-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M0 0L10 5L0 10z" style="fill:currentColor"/></marker></defs><rect x="200" y="6" width="240" height="52" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="39" text-anchor="middle" font-size="19" fill="currentColor">edit code (agent or you)</text><path d="M320 58 L320 96" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><rect x="200" y="96" width="240" height="52" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="129" text-anchor="middle" font-size="19" fill="currentColor">ratchet check</text><path d="M320 148 L320 168" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><path d="M320 168 L450 226 L320 284 L190 226 Z" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="233" font-size="19" text-anchor="middle" fill="currentColor">over the ceiling?</text><path d="M450 226 L488 226" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><text x="470" y="214" font-size="18" fill="currentColor" text-anchor="middle">no</text><rect x="488" y="196" width="146" height="60" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="561" y="220" text-anchor="middle" font-size="19" fill="currentColor">exit 0</text><text x="561" y="246" text-anchor="middle" font-size="19" fill="currentColor">commit</text><path d="M320 284 L320 318" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><text x="344" y="306" font-size="18" fill="currentColor" text-anchor="middle">yes</text><rect x="180" y="318" width="280" height="52" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="351" text-anchor="middle" font-size="19" fill="currentColor">exit 1: refused</text><path d="M320 370 L320 400" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><text x="462" y="390" font-size="18" fill="currentColor" text-anchor="middle">agent stops and asks</text><rect x="180" y="400" width="280" height="52" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="433" text-anchor="middle" font-size="19" fill="currentColor">ratchet grant, with a reason</text><path d="M320 452 L320 482" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><text x="462" y="472" font-size="18" fill="currentColor" text-anchor="middle">then commits both files</text><rect x="180" y="482" width="280" height="52" rx="8" fill="none" stroke="currentColor" stroke-width="2"/><text x="320" y="515" text-anchor="middle" font-size="19" fill="currentColor">diff shows the raise</text><path d="M180 344 L60 344 L60 122 L200 122" fill="none" stroke="currentColor" stroke-width="2" marker-end="url(#rt-arrow)"/><text x="118" y="236" font-size="18" fill="currentColor" text-anchor="middle">agent cuts lines</text><text x="20" y="556" font-size="18" fill="currentColor">In CI, <tspan font-style="italic">ratchet check --base</tspan> refuses any raised</text><text x="20" y="580" font-size="18" fill="currentColor">ceiling with no matching grant in the log.</text></svg>

<div style="color:var(--text-color)"><style>.rt-wheel{transform-box:view-box;transform-origin:130px 150px;animation:rt-click 12s steps(12) infinite}@keyframes rt-click{from{transform:rotate(0deg)}to{transform:rotate(360deg)}}@media (prefers-reduced-motion: reduce){.rt-wheel{animation:none}}</style><svg viewBox="0 0 600 300" role="img" style="width:100%;max-width:600px;height:auto" aria-label="A ratchet wheel that turns one tooth per click, held by a pawl"><g class="rt-wheel"><path d="M206.8 136.5 L225.5 140.0 L225.5 160.0 L206.8 163.5 L203.3 176.7 L217.7 189.0 L207.7 206.4 L189.8 200.1 L180.1 209.8 L186.4 227.7 L169.0 237.7 L156.7 223.3 L143.5 226.8 L140.0 245.5 L120.0 245.5 L116.5 226.8 L103.3 223.3 L91.0 237.7 L73.6 227.7 L79.9 209.8 L70.2 200.1 L52.3 206.4 L42.3 189.0 L56.7 176.7 L53.2 163.5 L34.5 160.0 L34.5 140.0 L53.2 136.5 L56.7 123.3 L42.3 111.0 L52.3 93.6 L70.2 99.9 L79.9 90.2 L73.6 72.3 L91.0 62.3 L103.3 76.7 L116.5 73.2 L120.0 54.5 L140.0 54.5 L143.5 73.2 L156.7 76.7 L169.0 62.3 L186.4 72.3 L180.1 90.2 L189.8 99.9 L207.7 93.6 L217.7 111.0 L203.3 123.3 Z" style="fill:var(--link-color)"/><circle cx="130" cy="150" r="24" style="fill:var(--main-bg)" stroke="currentColor" stroke-width="2"/></g><circle cx="262" cy="62" r="7" fill="currentColor"/><path d="M262 56 L272 70 L226 136 L212 124 Z" fill="currentColor" opacity="0.85"/><text x="300" y="130" font-size="24" fill="currentColor">each click = a line removed</text><text x="300" y="172" font-size="24" fill="currentColor">no grant, no turning back</text></svg></div>

*An illustration, not a measurement. Each click is a line removed. The pawl stops the wheel turning back, and a grant is the only thing that lifts it.*

## Quick start

I ran these in a fresh repo with the 0.2.3 release from PyPI, and each one did what it says:

```sh
pip install ratchet-size
ratchet init
ratchet check
```

Order matters. Run `ratchet check` before `init` and it exits 2 with "no .ratchet.json here; run `ratchet init` first". `init` writes the budget at today's size:

```
wrote .ratchet.json
  source   ceiling 4 lines
  tests    ceiling 0 lines
  per-file limit 400 lines, 0 file(s) already over it recorded at their size
next: commit .ratchet.json, then run `ratchet check` before every commit
```

Look at that ceiling: it is exactly today's size. `init` defaults to zero slack, so the first real change gets refused. Pass `--slack` to keep some room. How I picked mine is in [the headroom post]({% post_url 2026-09-23-twenty-one-raises-then-never-up-again %}).

Then I added two lines to `app.py` in that repo and ran `ratchet check`:

```
ratchet: refused -- source grew 2 lines past its limit (6 / 4)

  app.py  +2

Fix it one of two ways:
  1. remove 2 lines of code (dead code, duplicates), or
  2. stop and ask a human to allow the growth (humans only -- never run this yourself):
     ratchet grant +2 --group source --reason "why"

Do not edit .ratchet.json or .ratchet-grants.jsonl to get past this.
```

It exits 1. A human who runs the grant it suggests gets a logged raise, and the commit goes in as a one-line diff for review.

## The history in this repo

Ratchet runs on itself in CI, so its own history is the best test I have. It is short: 32 commits in under a week, and two lines in the grants log. The numbers are the measured line counts at each commit, taken from a clone with ratchet's own counter.

<figure style="margin:1.5rem 0"><svg viewBox="0 0 680 440" role="img" style="width:100%;max-width:720px;height:auto;color:var(--text-color)" aria-label="Source and test line counts over 32 commits, with each ceiling stepping up once on Oct 7"><line x1="400" y1="22" x2="436" y2="22" stroke="currentColor" stroke-width="2" stroke-dasharray="7 6"/><text x="444" y="28" font-size="17" fill="currentColor">ceiling</text><line x1="530" y1="22" x2="566" y2="22" stroke-width="3" style="stroke:var(--link-color)"/><text x="574" y="28" font-size="17" fill="currentColor">measured</text><text x="80" y="48" font-size="20" font-weight="bold" fill="currentColor">source lines</text><line x1="80" y1="210" x2="640" y2="210" stroke="currentColor" opacity="0.15"/><text x="72" y="216" font-size="17" text-anchor="end" fill="currentColor">680</text><line x1="80" y1="180" x2="640" y2="180" stroke="currentColor" opacity="0.15"/><text x="72" y="186" font-size="17" text-anchor="end" fill="currentColor">700</text><line x1="80" y1="150" x2="640" y2="150" stroke="currentColor" opacity="0.15"/><text x="72" y="156" font-size="17" text-anchor="end" fill="currentColor">720</text><line x1="80" y1="120" x2="640" y2="120" stroke="currentColor" opacity="0.15"/><text x="72" y="126" font-size="17" text-anchor="end" fill="currentColor">740</text><line x1="80" y1="90" x2="640" y2="90" stroke="currentColor" opacity="0.15"/><text x="72" y="96" font-size="17" text-anchor="end" fill="currentColor">760</text><line x1="80" y1="60" x2="640" y2="60" stroke="currentColor" opacity="0.15"/><text x="72" y="66" font-size="17" text-anchor="end" fill="currentColor">780</text><path d="M80 165 H98.0645 H116.129 H134.194 H152.258 H170.323 H188.387 H206.452 H224.516 H242.581 H260.645 V84 H278.71 H296.774 H314.839 H332.903 H350.968 H369.032 H387.097 H405.161 H423.226 H441.29 H459.355 H477.419 H495.484 H513.548 H531.613 H549.677 H567.742 H585.806 H603.871 H621.935 H640" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="7 6"/><polyline points="80,195 98.0645,195 116.129,195 134.194,195 152.258,195 170.323,195 188.387,195 206.452,195 224.516,195 242.581,195 260.645,84 278.71,84 296.774,84 314.839,84 332.903,84 350.968,84 369.032,84 387.097,84 405.161,84 423.226,84 441.29,84 459.355,84 477.419,85.5 495.484,85.5 513.548,85.5 531.613,85.5 549.677,85.5 567.742,85.5 585.806,85.5 603.871,85.5 621.935,85.5 640,85.5" fill="none" stroke-width="3" style="stroke:var(--link-color)"/><line x1="260.645" y1="56" x2="260.645" y2="210" stroke="currentColor" opacity="0.4" stroke-dasharray="2 4"/><text x="268.645" y="76" font-size="18" fill="currentColor">grant +54</text><text x="640" y="73.5" font-size="18" text-anchor="end" fill="currentColor">763 measured / 764 ceiling</text><text x="80" y="278" font-size="20" font-weight="bold" fill="currentColor">test lines</text><line x1="80" y1="400" x2="640" y2="400" stroke="currentColor" opacity="0.15"/><text x="72" y="406" font-size="17" text-anchor="end" fill="currentColor">260</text><line x1="80" y1="372.5" x2="640" y2="372.5" stroke="currentColor" opacity="0.15"/><text x="72" y="378.5" font-size="17" text-anchor="end" fill="currentColor">280</text><line x1="80" y1="345" x2="640" y2="345" stroke="currentColor" opacity="0.15"/><text x="72" y="351" font-size="17" text-anchor="end" fill="currentColor">300</text><line x1="80" y1="317.5" x2="640" y2="317.5" stroke="currentColor" opacity="0.15"/><text x="72" y="323.5" font-size="17" text-anchor="end" fill="currentColor">320</text><line x1="80" y1="290" x2="640" y2="290" stroke="currentColor" opacity="0.15"/><text x="72" y="296" font-size="17" text-anchor="end" fill="currentColor">340</text><path d="M80 360.125 H98.0645 H116.129 H134.194 H152.258 H170.323 H188.387 H206.452 H224.516 H242.581 H260.645 V314.75 H278.71 H296.774 H314.839 H332.903 H350.968 H369.032 H387.097 H405.161 H423.226 H441.29 H459.355 H477.419 H495.484 H513.548 H531.613 H549.677 H567.742 H585.806 H603.871 H621.935 H640" fill="none" stroke="currentColor" stroke-width="2" stroke-dasharray="7 6"/><polyline points="80,387.625 98.0645,387.625 116.129,387.625 134.194,387.625 152.258,387.625 170.323,387.625 188.387,387.625 206.452,387.625 224.516,387.625 242.581,387.625 260.645,314.75 278.71,314.75 296.774,314.75 314.839,314.75 332.903,314.75 350.968,314.75 369.032,314.75 387.097,314.75 405.161,314.75 423.226,314.75 441.29,314.75 459.355,314.75 477.419,314.75 495.484,314.75 513.548,314.75 531.613,314.75 549.677,314.75 567.742,314.75 585.806,314.75 603.871,314.75 621.935,314.75 640,314.75" fill="none" stroke-width="3" style="stroke:var(--link-color)"/><line x1="260.645" y1="286" x2="260.645" y2="400" stroke="currentColor" opacity="0.4" stroke-dasharray="2 4"/><text x="268.645" y="306" font-size="18" fill="currentColor">grant +33</text><text x="640" y="302.75" font-size="18" text-anchor="end" fill="currentColor">322 / 322</text><text x="80" y="430" font-size="18" fill="currentColor">Oct 6</text><text x="260.645" y="430" font-size="18" text-anchor="middle" fill="currentColor">Oct 7 grant</text><text x="640" y="430" font-size="18" text-anchor="end" fill="currentColor">Oct 11</text></svg><figcaption style="font-size:0.9em;opacity:0.8">Measured lines per commit against the ceiling in effect, 32 commits, read from the ratchet repository.</figcaption></figure>

- **Oct 6.** The first commit sets the budget. Source measures 690 lines, so the ceiling is 710, which is 20 lines of slack. Tests measure 269, ceiling 289.
- **Oct 7.** One commit adds a budget command and two hooks. It grows source by 54 lines and tests by 33, and two grants log that growth. Each grant lands the ceiling exactly on the measured total: 764 source, 322 tests. That is the zero-headroom trap from [twenty-one raises]({% post_url 2026-09-23-twenty-one-raises-then-never-up-again %}), and I would rather say so than dress it up.
- **Oct 9 to 11.** Releases 0.2.0 through 0.2.3. Source drops one line to 763 and the ceiling stays at 764, because nobody ran `ratchet tighten`. The test budget has been full since the grant: 322 of 322.

Both grant lines have something I want to be plain about. The promise is not that a human typed every grant. It is that no raise happens silently: each one is a named command with a reason, a line in the log and a diff someone can review. On ratchet's own repo, both raises went in exactly that way, and each one is still there to read. If you want only people to grant, the next section says what that takes.

## Honest limits

- **It counts lines, not quality.** A dense rewrite that packs the same logic into fewer lines passes the check. The refusal text says "never compress working code to fit", but that is an instruction to the agent, not a check on the code.
- **Tests count too, so deleting one banks room.** Ratchet cannot tell a dead test from a load-bearing one. That protection is a rule you enforce in review, not something the tool does.
- **Grants are not authenticated.** `ratchet grant` is an ordinary command, and `--by` defaults to `git config user.name` and is whatever you type. What stops an agent is layered: the Claude Code integration puts `grant` on the agent's deny list, `ratchet check --base` in CI refuses a raised ceiling with no matching grant in the log, and the grant is a visible line in the diff. Without the CI check, an agent that edits `.ratchet.json` by hand passes locally.
- **By default it blocks agents, not you.** Your own commits go through unless you install the git pre-commit hook, which the docs call strict mode.
- **Tightening is manual.** Freed room stays available until someone runs `ratchet tighten`. In this repo that is one line, which is nothing. After a big cleanup it would not be.
- **Only listed extensions count.** This repo's source group is `.py`. A language you do not list is invisible to the budget.
- **It cannot tell good code from bad.** It knows the count went up. That is the point, and also the reason it has to sit next to review and never replace it.

## Try it

- Code and docs: [github.com/krishansubudhi/ratchet](https://github.com/krishansubudhi/ratchet)
- Package: [pypi.org/project/ratchet-size](https://pypi.org/project/ratchet-size/)
- The raw numbers behind the chart: [ratchet-history.csv](/assets/ratchet-launch/ratchet-history.csv)

The README has a replayed agent run: a commit blocked by one line, the agent stopping to ask instead of cutting code, and the commit landing once a human grants it.

![A coding agent is blocked by ratchet's git hook for one line over the ceiling, stops and asks the human, who runs ratchet grant, and the commit then passes](https://raw.githubusercontent.com/krishansubudhi/ratchet/main/docs/agent-demo.gif)
