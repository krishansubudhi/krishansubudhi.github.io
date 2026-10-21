---
comments: true
author: krishan
layout: post
categories: agents
title: The phone goes in the bag. A realistic kit for a personal agent you wear.
description: An earbud with a small camera, a watch as the control hub, a ring, and an agent running on a server you own. What works today and what is still hard.
---

Laptops did not disappear when phones arrived. They went into the bag, and they come out when there is real work to do: a long document, a spreadsheet, a lot of typing. I think the phone is heading the same way. Most of what we pull it out for is small: what's next, reply to this, log that, play something, how far did I run. A personal agent that can hear you, see what you see when you ask, and talk back in your ear can do most of those without a screen.

This is not a "ten years from now" post. Everything below is built from parts that ship today or are one modest step away. The interesting part is what you have to give up to make it work on a real battery.

## The kit

![An open-ear earbud with a small camera at eye level](/assets/wearable-agent/earbud.png)

**An open-ear earbud with a small camera.** This is the main sensor. The ear sits almost exactly at eye line, so a tiny camera there sees roughly what you see. It doesn't change how you look. You don't need a prescription, and it clips to the ear, or to glasses if you already wear them. Open-ear means you still hear the world. It handles voice in and voice out, plus a photo when you ask for one.

**Glasses are optional.** Camera glasses work well, but they change your face, and people who need prescription lenses have to customise them. So they are an add-on for people who want a display, not the default.

**A watch as the control hub.** The watch is where you tap to say yes, feel a haptic nudge, glance at one line of text, and pay. It has LTE, so it works without the phone nearby.

**A ring for health.** Sleep, heart rate and temperature, synced in batches a few times a day. It needs no live connection. It also covers the night: the watch charges while you sleep, and the ring keeps tracking.

**An EMG wristband, optional.** For people who want silent finger gestures instead of voice in a quiet room.

**An agent server you own.** The thinking happens on a home machine or a cloud server running your personal agent, the way people already run things like Muse, Dot or a Grok bot. That server holds your memory, your tools and your accounts. The things you wear are its eyes, ears and hands.

![Present with people, agent in the background](/assets/wearable-agent/woman.png)

## The hard part: the watch can't carry everything

The obvious design routes every sensor through the watch, since it's the one device with a cellular radio. That falls apart on battery. A typical smartwatch battery holds about 1–2 Wh. An LTE radio sending continuously uses a few hundred milliwatts. Stream audio from two earbuds and video from a camera through it, and the watch is dead by lunch. It also gets warm on your wrist, and a watch can't shed much heat.

So the design splits the job in two:

- **The watch is the control plane, not the data plane.** It decides what is allowed, routes the request, and handles consent and haptics. It moves small messages, not media.
- **Heavy sensors talk to the server directly,** over Wi-Fi or their own eSIM when they have it, or in short bursts relayed when they don't. The watch never holds the video.
- **Capture is event-driven, not streamed.** The earbud runs voice activity detection locally and wakes up only when you speak. The camera takes a frame or a short clip when you tap the watch to say yes, not all day.
- **The ring batch-syncs.** Health data is not urgent and should never keep a radio awake.

Do this and each device's radio is idle most of the day. That is the only way a ring, a pair of earbuds and a watch each last a full day on batteries their size.

![A run with no phone: pace in your ear](/assets/wearable-agent/s1-run.png)

## Three privacy rules

1. **Explicit capture consent.** The camera records only after a deliberate tap on the watch. While it is live, the watch pulses and the earbud's outer LED is lit, so people around you can see it.
2. **Coarse location only.** The agent knows "home", "gym" or "office", not a continuous GPS trail.
3. **Context-gated output.** Voice is public. When other people are around, sensitive things like money, health or personal alerts wait until you're alone, or show up as a few words on the watch.

## What a day looks like

- **Morning run.** Pace and distance in your ear, a tap on the watch to log it. No phone strapped to your arm.
- **Lunch.** "Log this." The camera takes one frame, the agent estimates the meal, and the watch buzzes to confirm.
- **With people.** Nothing is recorded by default. If something is worth keeping, a photo of the kids' build or a note from a conversation, one tap says yes.
- **Evening.** The phone comes out of the bag for the things that really need a screen. Then it goes back.

## How realistic is this?

Honestly, most of the parts exist.

- **Voice** is the piece that changed most. Speech recognition is now good enough for everyday commands. Spoken replies sound natural, and round-trip voice latency is down to around a second on a good connection. A year or two of steady progress makes it feel like talking to a person rather than a system.
- **Models** that can see an image and act on it are already capable enough for "what is this, log it, remind me." Small on-device models handle wake words and voice detection today. The big reasoning stays on the server.
- **Power** is the real limit, and batteries are not getting dramatically better. Battery energy density improves by single-digit percentages a year. The wins come from duty cycling and better low-power radios and chips, not from bigger cells. That's why the split above matters more than any single gadget.
- **Hardware.** Open-ear earbuds, LTE watches and health rings all ship now. Camera glasses ship too. The one missing product is the earbud with a camera. It is a modest step, not a breakthrough. The camera modules are already small enough.

What is still hard isn't the hardware. It's **consent and trust**: a camera at eye level near other people has to be obviously off unless you ask, with a visible indicator and a deliberate tap. And it's **integration**: one agent that can read your calendar, your messages and your health data, and act for you, without you living inside one company's ecosystem. That's why I like the agent running on a server you choose.

![The phone stays in the bag, like a laptop](/assets/wearable-agent/s4-night.png)

## The shift

The goal isn't to kill the phone. It's to stop reaching for it fifty times a day for things that take five seconds. An earbud that sees when you ask, a watch you tap to say yes, a ring that quietly keeps score, and an agent you own doing the work in the background. The phone goes in the bag, next to the laptop, and comes out when there's real work to do.
