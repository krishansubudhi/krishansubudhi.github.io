---
comments: true
author: krishan
layout: post
categories: agents
title: My kid played a dinosaur game inside my assistant's reply. The hard part was keeping it alive.
description: My personal assistant can now answer in HTML, drawn in a sandboxed frame inside the chat bubble. Getting a game to show up was easy. Making it safe, on-theme, and still running when the next message arrived was the actual work.
---

I talk to my personal assistant, lean-jarvis, almost entirely from my iPhone. Until recently its replies were markdown. A layer on the server turned the markdown into cards, tiles and tappable chips, and that was as far as it went.

Now a reply can carry HTML. One evening it drew a dinosaur runner (a canvas, tap to jump) and a tic-tac-toe board with a minimax opponent, both inside ordinary chat replies and both in my theme's colours. My kid played them. Of everything the assistant has done, that got the biggest reaction in our house.

Getting HTML into a bubble is one line of code. Getting it there safely, keeping it on-theme, and keeping it running when the next message arrives is everything else, and that's what this post is about.

## One fence, one exception

The rendering layer is `jarvis/cardify.py`, and its docstring sets the rule:

> THE HEAD WRITES PLAIN MARKDOWN AND NEVER A FENCE GRAMMAR; this layer reads the markdown's own shape and draws it.

Everything that goes through cardify is escaped. Raw HTML in a reply comes back to you as visible text. There is exactly one exception: a code fence whose language is `html-card`.

~~~~markdown
```html-card
<h3>Dinner</h3>
<p style="color:var(--ok)">Pasta is on.</p>
```
~~~~

Any other fence becomes a real code block, with a language label, a copy button and a wrap toggle. An `html-card` fence becomes a frame. In the code, that's a single branch:

```python
if lang.lower() == "html-card":
    return _frame(blk["body"], blk.get("theme", "dark"))
```

Every other part of this feature exists because of what that branch lets in.

## A frame with no origin

Here's the iframe `_frame` produces, minus the document it carries:

```python
'<iframe class="htmlcard" sandbox="allow-scripts allow-popups '
'allow-popups-to-escape-sandbox" srcdoc="%s"></iframe>'
```

What matters is the attribute that isn't there. The frame has `allow-scripts`, so a game can run, but not `allow-same-origin`. Its origin is therefore opaque. As the docstring puts it, a script in there "never reads this page's cookie, token, DOM or localStorage". The game can draw on its own canvas and do nothing else. `allow-popups` plus a `<base target="_blank">` is how a link inside a card opens a new tab rather than navigating the frame.

A Content-Security-Policy inside the frame's document backs this up:

```python
_FRAME_CSP = ("default-src 'none'; img-src *; media-src *; style-src "
             "'unsafe-inline'; script-src 'unsafe-inline'; frame-src 'none'; "
             "object-src 'none'; form-action 'none'")
```

Inline scripts and styles are allowed because that's what a self-contained card is made of. Nothing is allowed to load a script from elsewhere, nest another frame, or submit a form.

The frame does talk back once. It posts its own height to the parent on load, on resize, and once fonts have settled, so the bubble grows to fit the card. A tall game board doesn't end up cropped at a default height.

## No cookie means no pictures, so add /clip

The sandbox caused a problem straight away. The app's existing picture route, `/pic`, needs the session's token, and a frame without an origin has no token to send. So an `<audio>` or `<img>` inside a card simply failed to load.

The fix is a second route, `/clip`, which serves media from the proofs directory by name with no token at all. It's one of only two routes in the server's `OPEN` list. The other is `/health`. The handler records what that costs:

```python
def h_clip(self, cookie):
    """Audio or a picture by name, NO TOKEN: `cardify._frame` has no origin,
    so no `lj` cookie. Trade-off (Krishan, 2026-10-01): any media file in
    state/proofs/ is fetchable by anyone who can reach this port."""
```

I accepted that trade on purpose, and it sits next to the code so the next person to read it doesn't have to work it out again. The assistant's instructions say the same thing from the other side: inside a frame, use `/clip` and never `/pic`.

## Taps can't get out, so questions stay outside

The sandbox also means a tap inside the frame can't reach the app. That's fine for a game. It's a problem when the assistant needs me to pick something.

Outside the frame, cardify already turns a question line followed by short bullets into tappable chips, and a tap on a chip sends the answer back as my reply. A button inside the sandbox can't do that. So the HTML rule given to the assistant says:

> A question he must pick from stays markdown chips (a question line, then bullets) OUTSIDE the frame: the sandbox cannot send a tap back.

The result is a reply with two layers: the rich part inside the frame, and anything that needs my answer outside it.

## The theme holds because colours are variables

There's one palette, `PALETTE` in cardify. The page shell uses it, and `_frame` injects the same set into every frame's `:root`. It has a dark and a light version. Here's the dark one:

```python
"dark": "color-scheme:dark;--bg:#0e1116;--card:#161b22;--card2:#1b212b;"
        "--ink:#e6edf3;--dim:#8b96a5;--line:#242c37;--accent:#f2c94c;"
        "--ok:#5fd97f;--ok-bg:rgba(63,185,80,.13);--warn:#f2c94c;" ...
```

The assistant is told to use colours only through those variables: `--bg`, `--card`, `--card2`, `--ink`, `--dim`, `--line`, `--accent`, plus the status colours `--ok`, `--warn`, `--bad` and `--info`, each with a `-bg` variant. Never a hard-coded hex value, "so his theme holds." The colours carry meaning too: ok means done, warn means it needs me, bad means something broke, info is neutral.

That's why the dinosaur came out in my theme without anyone asking. The model never picked a colour. It wrote `var(--accent)`, and the frame supplied whatever my theme says that is. If I switch to light, the next card follows.

## A switch, and a turn that knows about it

HTML replies are off by default. The settings page has one checkbox: *reply in HTML cards, not markdown*. When it's on, the rule tells the assistant to put the whole reply in one `html-card` fence every turn, laid out for a phone: the answer first, one card about 390px wide, text 14px or larger, long detail in `<details>`.

The awkward part was resumed sessions. The assistant's head is a long-running session, and flipping a switch on a web page doesn't change a prompt that's already been sent. The commit that added the switch is titled *"toggles ride the turn message so resumed heads see flips."* The HTML guide is always in the system prompt, on or off. When the current setting differs from the one the session was built with, the next turn gets one extra line, generated by `reply_mode`:

```python
return ("HTML on, theme %s -- follow the HTML guide" % theme if html
        else "HTML off -- plain markdown, whatever earlier turns did")
```

"Whatever earlier turns did" is in there because a model that has just written twenty HTML cards will happily write a twenty-first unless it's told the context has changed.

## The wrinkle: the game reset whenever anyone spoke

This was the bug that showed up the moment my kid started playing.

The chat pane refreshed by assigning the server's latest markup to `chat.innerHTML`. That's fine for text. For frames it was a disaster, because every new message rebuilt every bubble, and rebuilding an iframe reloads it. The comment on the fix lists what that broke:

> Replacing the pane reloaded every html-card frame on every message: a playing <audio> stopped, a game in a card reset, and each frame fell back to 160px and regrew.

The fix made the repaint append-only. A bubble's key is its own markup as first painted. On each refresh the pane walks the new list, keeps any bubble whose markup hasn't changed, and builds only the new or changed ones. A kept bubble is never moved either, because, as the comment says, "moving an iframe reloads it just the same."

The first version of that fix had a bug of its own. Two "twin" bubbles from the same minute (the comment's example is an "ok" and a "Yes, keep it") could keep the wrong one as the oldest bubble dropped out of the 100-row window. That misaligned everything below it, so every later bubble was rebuilt, audio included. The follow-up commit, *"twin bubbles no longer rebuild the pane (audio cut)"*, matches each new bubble forward, in order:

```js
for (k = keyOf(fresh[i]), m = cur; m && keyOf(m) !== k; m = m.nextElementSibling) { }
if (!m) { fresh[i].jSrc = k; chat.insertBefore(fresh[i], cur); continue; }
while (cur !== m) { cur = unbub(cur); }
cur = m.nextElementSibling;
```

Now a new message adds one bubble, and the dinosaur keeps running.

## The other wrinkle: three backticks inside a card

A card is HTML, and HTML sometimes contains a line made of three backticks: a `<pre>` showing markdown, or a string in a script. `_read_fence` closes a block at the first bare fence at least as long as the one that opened it:

```python
if shut and shut.group(2).startswith(mark):
    opens = bool(shut.group(3).strip())
    if not opens and not inner:
        i += 1
        break
    inner += (1 if opens else -1) * card
```

So inside an `html-card`, a bare triple-backtick line ends the card right there. The rest of the HTML spills out as escaped text, and the frame gets half a document.

There's a partial fix in the code. Inside a card, a fence with a language, such as `` ```js ``, counts as opening an inner block that consumes the next bare fence, so a nested code sample survives. But a bare fence with no inner opener still closes the card early. The workaround falls out of the same comparison: `startswith(mark)` means an opener of four backticks can't be closed by three. A card that has to contain three backticks should open with four.

## What I take from it

Getting a game into a chat bubble took one `if`. The rest of the work was deciding what the bubble is not allowed to do, and then dealing with the consequences: no cookie, so `/clip`. No way to tap back out, so questions stay as chips outside the frame. No hard-coded colours, so the theme holds. A frame reloads when you touch it, so the pane stopped touching anything that hadn't changed.

My kid doesn't care about any of that. What matters to him is that the dinosaur doesn't restart when I get a message.
