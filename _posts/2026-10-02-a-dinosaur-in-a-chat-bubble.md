---
comments: true
author: krishan
layout: post
categories: agents
title: My kid played a dinosaur game inside my agent's reply. The hard part was keeping it alive.
description: My personal agent can now answer with a small web page, drawn in a sandboxed frame inside the chat bubble. Getting a game to show up was easy. Keeping it safe, on-theme, and running when the next message arrived was the actual work.
---

I talk to an evolving personal agent built on top of Claude Code, almost entirely from Chrome on my Android phone. Until recently its replies were plain formatted text.

Now a reply can carry a small web page. One evening my agent drew a dinosaur runner inside an ordinary chat reply. My kid played it. Of everything the agent has done, that got the biggest reaction in our house.

![The Dino Run game drawn inside a chat reply: a white pixel dinosaur jumps green cacti on a dark card while the score climbs, then hits one and shows GAME OVER](/assets/dino-chat-bubble/dino-run.gif)
_The real card, rendered headless. A small script presses space when a cactus gets close, then stops so you can see it lose._

## One frame, and what it can't do

Everything in a reply is escaped, so raw HTML shows up as text. There's exactly one exception: a block the agent explicitly marks as a web page. That goes into a sandboxed iframe inside the bubble.

What matters is the permission that isn't there. The frame may run scripts, so a game can work, but it doesn't get the app's origin. A script inside can't read the app's page, storage or anything I'm signed in with. It can draw on its own canvas and nothing else. A content security policy also stops it from loading outside scripts, nesting frames or submitting forms.

## The theme holds because colours are variables

The app has one palette, dark and light, and every frame is handed the same small set of named colours: background, card, text, accent, and a few status colours. The agent is told to use only those, never a raw colour value.

So the model never picks a colour. It asks for "the accent" and the frame fills in whatever my theme says. That's why both games came out looking native without anyone asking, and if I switch to light the next card follows.

![A tic-tac-toe card in the same dark theme: X has won across the middle row, the three winning squares are tinted green, and the status line reads "You beat the agent!"](/assets/dino-chat-bubble/tic-tac-toe-won.png)
_The tic-tac-toe card from the same evening. Its opponent is minimax with a 15% chance of a random move, which is the only reason this game was winnable._

## The wrinkle: the game reset whenever anyone spoke

This bug showed up as soon as my kid started playing. The chat refreshed by replacing everything with the latest version. That's fine for text. For frames it was a disaster: rebuilding an iframe reloads it, so every new message restarted the dinosaur and cut off any audio that was playing.

The fix made the refresh append-only. The chat keeps every bubble that hasn't changed and builds only the new ones. It never moves a kept bubble either, because moving an iframe reloads it too.

The first version had its own bug. Two look-alike bubbles from the same minute could be matched to the wrong one as old messages scrolled away, which rebuilt everything below, audio included. Matching new bubbles forward, in order, fixed that.

## What I take from it

Getting a game into a chat bubble was the easy part. The real work was deciding what the bubble isn't allowed to do, and then not touching anything that hadn't changed.

My kid doesn't care about any of that. What matters to him is that the dinosaur doesn't restart when I get a message.
