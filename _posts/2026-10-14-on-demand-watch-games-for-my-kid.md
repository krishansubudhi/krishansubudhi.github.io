---
comments: true
author: krishan
layout: post
categories: agents
date: 2026-10-14
permalink: /posts/on-demand-watch-games-for-my-kid/
title: I asked my watch for a game for my kid, and had it that evening
description: I talk to a personal agent from my phone and my Wear OS watch. I asked it for a game for my kid, then kept asking, and ended the evening with five games on my wrist. They are small and disposable, and I wouldn't have built them any other way.
---

I have a personal AI agent that I talk to from my phone and from my Wear OS watch. One evening I asked it, out loud, into my wrist, for a game for my kid. It built one and installed it on my watch the same evening. By bedtime there were five.

None of them are good enough to ship, and that isn't the point. They are software made for one kid, on demand, by a parent who never opened an editor. I was the product manager, speaking from my wrist.

## Balloon Pop

The first game was balloons. They float up the round screen, you tap one, and it pops with a sound and a buzz on your wrist. There's no way to lose: a balloon that floats off the top is just gone.

![Balloon Pop: coloured balloons rise on a dark round watch face and one has just burst in a ring of specks, with a gold score at the top](/assets/img/posts/watch-games/balloon-pop.gif)
_A rendering of Balloon Pop from the app's drawing code, with a script doing the tapping. Not a recording of the watch._

## Feedback from my wrist

I tried it and spoke my notes back, the way I would to a designer sitting next to me. The button was too big and too white. Pressing back reset the game. I wanted browns, and I wanted a sound.

The next build had all of it: a small gold button, saving and resuming, twenty colours including three browns, and a pop sound the app synthesises itself. I didn't write a ticket. I said a few sentences and got a new version.

## Then my kid used it

This part is my favourite, because I got it wrong twice.

My kid kept pressing back and leaving the game. So the agent removed the back button and the swipe, and made leaving deliberate: press and hold for three seconds while a gold ring fills. Nobody exits that by accident.

Then getting out became too hard. The next build added a swipe from the left edge as a second way out.

![Balloon Pop mid-hold: a grey ring with a gold arc about two thirds full near the bottom of the screen](/assets/img/posts/watch-games/hold-to-exit.png)
_The exit ring, in a rendering. It only appears once a press is clearly a hold and not a tap._

I couldn't have predicted either problem by thinking about it. I found them by handing over the watch.

## The picker, and four more

With more than one game I needed a picker: big round picture tiles, two to a row, no words.

![The game picker: four round tiles on black, blue with balloons, green with a star and ball, pink with four coloured quarters, purple with a die](/assets/img/posts/watch-games/games-picker.png)
_The picker, as a rendering. Colours and icons follow the app's._

**Tilt Ball.** I asked for a gyroscope game. You tip your wrist to roll a ball toward a gold star, and a new star appears each time you reach one.

![Tilt Ball: a red ball on a dark round face with a faint gold rim, rolling toward a pulsing gold star](/assets/img/posts/watch-games/tilt-ball.gif)
_A rendering of Tilt Ball. A script steers the ball, standing in for a wrist._

**Copy Me.** I asked for something unique that my kid and I could play together, and the agent chose a Simon-style game. Four coloured quarters light up in a sequence, one of us taps it back, then the other takes the next, longer round. A mistake has no fail sound. The sequence just plays again.

![Copy Me: a round face in four quarters, red, blue, green and yellow, with the blue one lit and a gold dot and number in the middle](/assets/img/posts/watch-games/copy-me.gif)
_A rendering of Copy Me. The loop is cut to two steps to keep it small._

**Shake Dice.** I asked for a dice game. Shake your wrist or tap, and a die tumbles and lands. A two-finger tap switches between one die and two.

![Shake Dice: a red die and a blue die tumbling on a round dark face, then landing](/assets/img/posts/watch-games/shake-dice.gif)
_A rendering of Shake Dice: a one-die roll, then a two-dice roll._

## Versions 2.5 to 2.10

The watch app went from version 2.5 to 2.10 in about one evening. Each version was a voice request, and each was signed with the same key, so the watch updated itself and I never reinstalled anything. A request is one sentence. If the sentence was wrong, I say another.

## What I can and can't say

- **The pictures are renderings.** I had no emulator, so I redrew each game with the shapes, colours and layout from the app's drawing code. They are faithful, but they aren't screenshots.
- **The newer games haven't been tested on a real watch by the agent.** The balloon notes above came from my own use. Tilt Ball, Copy Me and Shake Dice still need me to check the feel, and in particular whether Tilt Ball rolls the way you tip it.

## Why bother

I don't think these games are worth keeping. They have no levels and nobody else will install them. That's what interests me. Software this small and personal used to cost more than it was worth, so it never got made. Now one sentence from my wrist is enough, and a wrong version costs a few minutes. It's a game for one kid on one evening, and it may not exist next month. I'm fine with that.
