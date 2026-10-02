---
comments: true
author: krishan
layout: post
categories: agents
title: My kid played a dinosaur game inside my assistant's reply. The hard part was keeping it alive.
description: My personal assistant can now answer in HTML, drawn in a sandboxed frame inside the chat bubble. Getting a game to show up was easy. Keeping it safe, on-theme, and running when the next message arrived was the actual work.
---

I talk to my personal assistant, lean-jarvis, almost entirely from my iPhone. Until recently its replies were markdown, which the server turned into cards and tappable chips.

Now a reply can carry HTML. One evening it drew a dinosaur runner inside an ordinary chat reply. My kid played it. Of everything the assistant has done, that got the biggest reaction in our house.

![The Dino Run game drawn inside a lean-jarvis reply: a white pixel dinosaur jumps green cacti on a dark card while the score climbs, then hits one and shows GAME OVER](/assets/dino-chat-bubble/dino-run.gif)
_The real card, rendered headless. A small script presses space when a cactus gets close, then stops so you can see it lose._

## One fence, one frame

Everything in a reply is escaped, so raw HTML shows up as text. There's exactly one exception: a code fence whose language is `html-card`. Its body goes into an iframe inside the bubble:

```html
<iframe sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox"
        srcdoc="..."></iframe>
```

What matters is the attribute that isn't there. The frame has `allow-scripts`, so a game can run, but not `allow-same-origin`. Its origin is opaque, so a script inside can't read the app's cookie, token, page or storage. It can draw on its own canvas and nothing else. A Content-Security-Policy in the frame also stops it from loading outside scripts, nesting frames or submitting forms.

## The theme holds because colours are variables

The app has one palette, dark and light, and every frame gets the same set of CSS variables on `:root`: `--bg`, `--card`, `--ink`, `--accent`, and status colours `--ok`, `--warn`, `--bad`, `--info`. The assistant is told to use only those, never a hex value.

So the model never picks a colour. It writes `var(--accent)` and the frame fills in whatever my theme says. That's why both games came out looking native without anyone asking. The cacti are `--ok`, the best score is `--accent`, and if I switch to light the next card follows.

![A tic-tac-toe card in the same dark theme: X has won on the diagonal, the three winning squares are tinted green, and the status line reads "You beat Jarvis!"](/assets/dino-chat-bubble/tic-tac-toe-won.png)
_The tic-tac-toe card from the same evening. Its opponent is minimax with a 15% chance of a random move, which is the only reason this game was winnable._

## The wrinkle: the game reset whenever anyone spoke

This bug showed up as soon as my kid started playing. The chat pane refreshed by replacing its whole contents with the server's latest markup. That's fine for text. For frames it was a disaster: rebuilding an iframe reloads it, so every new message restarted the dinosaur and cut off any audio that was playing.

The fix made the repaint append-only. Each bubble is keyed by its own markup as first drawn. On refresh the pane keeps every bubble whose markup hasn't changed and builds only the new ones. It never moves a kept bubble either, because moving an iframe reloads it too.

The first version had its own bug. Two look-alike bubbles from the same minute could be matched to the wrong one as the oldest message scrolled out of the window. That threw everything below out of line and rebuilt it all, audio included. Matching each new bubble forward, in order, fixed that.

## What I take from it

Getting a game into a chat bubble took one `if`. The rest of the work was deciding what the bubble isn't allowed to do, and then not touching anything that hadn't changed.

My kid doesn't care about any of that. What matters to him is that the dinosaur doesn't restart when I get a message.
