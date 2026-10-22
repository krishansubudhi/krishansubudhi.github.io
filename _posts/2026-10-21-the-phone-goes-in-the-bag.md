---
comments: true
author: krishan
layout: post
categories: agents
title: The phone goes in the bag. A realistic kit for a personal agent you wear.
description: "An earbud with a small camera, a watch as the control hub, a ring, and an agent running on a server you own. The full design: device roles, links, tiers of compute, battery budget, privacy rules, a simulated day and an honest scorecard."
---

Laptops did not disappear when phones arrived. They went into the bag, and they come out when there is real work to do: a long document, a spreadsheet, a lot of typing. I think the phone is heading the same way. Most of what we pull it out for is small: what's next, reply to this, log that, play something, how far did I run. A personal agent that can hear you, see what you see when you ask, and talk back in your ear can do most of those without a screen.

This is not a "ten years from now" post. Everything below is built from parts that ship today or are one modest step away. The interesting part is what you have to give up to make it work on a real battery. Below is the whole design: who talks to whom, where the compute lives, the power budget, and what still does not work.

## The kit

![An open-ear earbud with a small camera at eye level](/assets/wearable-agent/earbud.jpg)

**An open-ear earbud with a small camera.** This is the main sensor. The ear sits almost exactly at eye line, so a tiny camera there sees roughly what you see. It doesn't change how you look. You don't need a prescription, and it clips to the ear, or to glasses if you already wear them. Open-ear means you still hear the world. It handles voice in and voice out, plus a photo when you ask for one.

**Glasses are optional.** Camera glasses work well, but they change your face, and people who need prescription lenses have to customise them. So they are an add-on for people who want a display, not the default.

**A watch as the control hub.** Tap to say yes, feel a haptic nudge, glance at one line of text, pay. It has LTE, so it works without the phone nearby.

**A ring for health.** Sleep, heart rate and temperature, synced in batches a few times a day. It also covers the night while the watch charges.

**An EMG wristband, optional,** for silent finger gestures instead of voice in a quiet room.

**An agent server you own.** The thinking happens on a home machine or cloud server running your personal agent, the way people already run things like Muse, Dot or a Grok bot, or something new. It holds your memory, tools and accounts. The things you wear are its eyes, ears and hands.

The phone is not part of the daily kit. It stays in a bag, like a laptop, for jobs that need a screen or keyboard.

![Present with people, agent in the background](/assets/wearable-agent/woman.jpg)

## The body-area network

![Body silhouette with the wearable constellation: earbud with camera, optional glasses, watch, ring, phone in a bag and an agent server. Thin lines are control, dashed lines are data going straight to the server.](/assets/wearable-agent/body-network.png)

*Figure 1. Thin lines are control and small messages. The dashed blue line is heavy data: the earbud's audio and camera go straight to the server, not through the watch. The ring batch-syncs on its own.*

Each device has a narrow role and its own kind of link:

| Device | Role | Link | What it sends |
|---|---|---|---|
| Earbud + camera | Default sensor, voice in/out | BLE to watch (control); Wi-Fi or eSIM to server (data) | Audio when you speak, a clip after a consent tap |
| Watch | Control hub: consent, haptics, routing, payments | BLE to earbud; LTE to server | Small messages, never media |
| Ring | 24/7 health, covers the night | Batch sync | Sleep, heart, temperature summaries |
| Optional glasses, EMG band | A display; silent gestures | Local link to watch | Text; pinch events |
| Agent server | Memory, tools, accounts, long jobs | Wi-Fi and LTE in, sanitized request out | Replies, reminders |
| Phone | Heavy work | Its own | Nothing, unless taken out |

The control/data split is the core of the design. Control messages are tiny and need low latency: "the user said yes", "the camera is allowed for 15 seconds". Data is big and bursty: audio while you talk, a frame when you ask. Pushing both through one device is what makes single-gadget designs overheat or die by lunch.

## Three tiers of compute

Put each job where it is cheapest in milliwatts and latency.

![Three tiers: body sensors, the watch as control plane, and an agent server with an allowlisted frontier model](/assets/wearable-agent/tiers-data-flow.png)

*Figure 2. Raw audio and camera go earbud to server over Wi-Fi or eSIM, never through the watch.*

**Tier 1, on the body.** The earbud, ring and extras capture raw sound, light and pulse. The only thinking they do is tiny: a wake word and voice activity detection on the earbud, so the radio and camera stay off until you speak or tap.

**Tier 2, the wrist.** The watch decides what is allowed: consent tap, haptics, routing, short messages. It does not carry media.

**Tier 3, your server.** The personal agent runs on a home or cloud machine you own. It takes the heavy streams directly, strips faces from any frame before a frontier model could see it, and runs the long jobs: code, research, memory, tool calls. Health data and private turns stop here. Only an allowlisted, sanitized request goes on to a frontier model, and only for hard reasoning.

Net effect: **raw streams go earbud to server, never through the watch**, and **anything private is answered at the server you own**.

## Why the watch can't carry everything

The obvious design routes every sensor through the watch, since it has the cellular radio. That falls apart on battery. A typical smartwatch battery holds about 1–2 Wh, and an LTE radio sending continuously uses a few hundred milliwatts. Stream audio and video through it and the watch is dead by lunch, and warm on your wrist. So the design splits the job:

- **The watch is the control plane, not the data plane.** Small messages only.
- **Heavy sensors talk to the server directly,** over Wi-Fi or their own eSIM, or in short relayed bursts. The watch never holds the video.
- **Capture is event-driven.** The earbud wakes when you speak; the camera takes a frame or short clip when you tap yes, not all day.
- **The ring batch-syncs.** Health data is never urgent enough to keep a radio awake.

Do this and each radio is idle most of the day, which is the only way a ring, earbuds and a watch each last a full day on batteries their size.

![A run with no phone: pace in your ear](/assets/wearable-agent/s1-run.jpg)

## Power budget and the charging handoff

Wearable setups die when you face four charging cables every night. The fix is to stagger battery life and hand the night over between wrist and finger.

| Device | What the public specs say |
|---|---|
| Watch with LTE | Roughly a day: Apple Watch Series 10 LTE is rated about 18 hours, Pixel Watch 3 LTE about 24 to 36. Charges nightly. |
| Ring | 5 to 8 days (Oura Ring 4), 3.3 to 5.2 g. Samsung's Galaxy Ring quotes 6 to 8. |
| Open-ear earbud | Shokz OpenFit 2: about 9.4 g per bud, up to 11 hours, 48 with the case. A camera costs extra, so expect a daily case top-up. |
| Camera glasses (optional) | Ray-Ban Meta Gen 2: up to 8 hours typical, 48 with the case. |

![A 24-hour charging and wear schedule across the devices](/assets/wearable-agent/charging-handoff.png)

*Figure 3. Watch on the pad overnight, ring on the finger, earbud in its case for a short midday top-up.*

The handoff:

- **Bedtime.** The watch goes on a pad and the earbuds in their case. The ring stays on and owns the night: sleep stages, heart rate variability, temperature.
- **Morning.** The ring has already synced its night to the server. One line on the watch face: how you slept, your first calendar item.
- **Midday.** A 20-minute case top-up carries the earbud through the evening.
- **Weekly.** The ring needs about 20 minutes on its charger.

Battery energy density improves by single-digit percent a year, so the plan is duty cycling and low-power radios, not a miracle cell. Nothing above needs one.

## What the hardware looks like

![Five hardware mocks: an open-ear earbud with a forward camera, a watch consent screen, an optional EMG band, a smart ring and the phone in a bag](/assets/wearable-agent/hardware-mocks.png)

*Figure 4. The five physical surfaces, with screens stripped to ambient status and one-tap approvals.*

**A. Earbud with a forward camera.** An open-ear bud at the temple with a small lens on its front end. A status LED is lit whenever the camera is live. The mic wakes on voice, the camera only on a watch tap, and the ear stays open.

**B. The watch consent gate,** the most important screen in the system. It says what wants to record ("camera and mic"), for how long ("15 second earbud clip"), and offers one button, "tap to allow". The watch pulses so you feel the camera is live. With the gate on the watch, the camera cannot turn on from the cloud.

**C. An optional EMG band.** Sensors inside the wrist read a thumb and index pinch; a thin strip shows the time and whether a recording is live. Useful for silent input in a meeting, but still slow today, so optional.

**D. The ring.** No screen: optical sensors for pulse and blood oxygen, a temperature sensor, a titanium shell. Its job is to keep score quietly.

**E. The phone, in the bag,** with a hardware kill switch for its radios. It comes out for reading, typing and video.

![Lunch: the agent logs a meal after one deliberate tap on the watch](/assets/wearable-agent/s3-meal.jpg)

## Privacy rules

A camera at eye level near other people is the hard part of this design, so the rules are explicit and mechanical.

1. **Explicit capture consent.** The camera records only after a deliberate tap on the watch. While it is live, the watch pulses and the earbud's outer LED is lit, so people around you can see it. The mic wakes on voice activity only.
2. **Coarse location only.** The agent knows "home", "gym", "office" or "park", not a continuous GPS trail.
3. **Context-gated output.** Voice is public. When the agent can tell other people are around (it hears voices, or sees nearby devices), sensitive things like money, health or personal alerts wait until you're alone, or show up as a few words on the watch face.

Two more fall out of the architecture. **Faces are stripped on your server before any frame reaches a frontier model**, and **raw video and audio stay on your own server by default**. A compromised cloud agent holding first-person video would be a surveillance nightmare. Some places (locker rooms, schools, clinics) will rightly ban wearable cameras outright; the design has to respect that.

## A simulated day

A Tuesday for a runner and parent who works in software, and what stands in for the phone.

- **06:30, wake.** The watch comes off the pad. The ring synced the night to the server while you slept. One line on the watch: how you slept, the first meeting, the school run. *Replaces:* the alarm and the morning check.
- **06:45, run.** The watch shows pace and buzzes at turns; open-ear buds whisper directions and play a cached audio book. At some point you say: "note: check the latency regression after standup." *Replaces:* the GPS app, music player and voice memo app.
- **07:45, school drop-off.** Your kid does something great. You tap the watch once (allow a 15 second clip), the LED lights, and the clip goes over home Wi-Fi to the family folder on your server. *Replaces:* the camera.
- **08:45, commute.** The agent reads a 90 second briefing in your ear: two pull requests merged, one waiting on a decision. You answer quietly: "approve the first, hold the second for my laptop."
- **10:30, meeting.** A colleague quotes an old latency number. You double-pinch (optional EMG band) and the agent whispers the current figure, or puts it on the watch. Nobody sees a phone under the table.
- **12:30, lunch.** You look at the bowl and tap the consent button. The earbud takes one frame; your server crops the plate, strips faces and logs a rough calorie estimate. One tap, one frame, no talking at a crowded table.
- **15:15, a private message.** A bank alert and a clinic message arrive while you chat in the kitchen. The earbud hears voices nearby and holds the spoken output; the watch shows only "2 private items held". Alone in the stairwell five minutes later, you hear them.
- **17:30, errands.** The shopping list lives on the watch; a double-click pays by NFC.
- **18:30, evening.** The phone stays in the bag. Only an allowlist of urgent people can break through; the rest is batched for morning.
- **20:30, deep work.** The kids are in bed, and you open the laptop for 90 minutes. The agent staged the branches and summarised the logs during the day, so the time is pure thinking.
- **22:30, bed.** The watch goes on the pad, the earbuds in their case, the ring stays on. "Wake me at 6:30."

![The phone stays in the bag, like a laptop](/assets/wearable-agent/s4-night.jpg)

## The phone-replacement scorecard

I scored 20 ordinary daily phone jobs honestly: 11 replaced, 6 partly, 3 not yet. That is 17 of 20, or 85 percent, replaced or partly replaced. By count of jobs, not hours: the three left are the ones that eat the most screen time.

| # | Phone job | Verdict | What does it | Limit |
|---|---|---|---|---|
| 1 | Voice calls | Replaced | Buds + watch LTE | Step aside in loud or private spaces |
| 2 | Short texts, quick replies | Replaced | Voice, watch, optional EMG | Fine under about 15 words |
| 3 | Walking and running directions | Replaced | Voice cues + wrist buzz | The car dashboard still owns driving |
| 4 | Contactless payments, transit | Replaced | Watch NFC | Solved |
| 5 | Music and audiobooks | Replaced | Watch storage + buds | A button skips tracks |
| 6 | Podcasts, spoken briefings | Replaced | Agent summary + buds | Follow-ups by voice |
| 7 | Health, sleep, workouts | Replaced | Ring + watch | Better than a phone |
| 8 | Calendar, reminders, timers | Replaced | Voice + watch glance | No calendar grid needed |
| 9 | Smart home controls | Replaced | Voice + room presence | Via a local home hub |
| 10 | Quick facts, translation | Replaced | Voice or camera, answer in ear | Faster than unlocking |
| 11 | Food logging, receipts | Replaced | Watch tap + earbud camera | Beats typing |
| 12 | Everyday photos and clips | Partly | Earbud camera | Great for candids; loses on zoom, low light |
| 13 | Email triage | Partly | Audio or watch digest | Long replies wait |
| 14 | News and feeds | Partly | Morning audio, headlines | Deep reads go to the laptop |
| 15 | Rideshare | Partly | Agent books, watch shows plate | Pin drops want a map |
| 16 | Shopping, reorders | Partly | Voice reorder, watch list | Browsing needs a screen |
| 17 | Social media | Partly | Agent drafts, you approve on a laptop | Kills infinite scroll |
| 18 | Long docs, PDFs, code diffs | Not yet | Keep the laptop | No 400-line review on a watch |
| 19 | Video and movies | Not yet | TV or laptop | Not for cinema |
| 20 | Silent long typing | Not yet | Keep a keyboard | The phone's last job: whisper and EMG can't beat thumbs |

You can't achieve everything. Whisper and gesture input need to improve a lot before someone can truly go phone-less. If you need deep work, carry a phone or tablet the way we carry laptops now. The goal is **phone demotion**, not abolition.

## What it does to a life, good and bad

**Upsides:** less reaching for a glowing rectangle when an agent triages notifications first; hands and eyes stay with the people in front of you, and capturing a moment is a one-second wrist tap; a continuous health loop without manual entry; heads-up voice cues for runners and cyclists. For people with low vision or limited hand use, a forward camera plus a conversational agent could matter a lot.

**Downsides:**

- **Bystander privacy.** A camera at the temple still makes people uneasy. The LED and the watch tap are non-negotiable, and some venues will ban it.
- **Dependence.** When an agent remembers every name and turn, your own recall can dull, much as GPS weakened map reading.
- **Data liability.** First-person audio and video is high-stakes data, which is why it lands on a server you own.
- **Cost and friction.** Several wearables plus a server add up.

## What ships today and what is next

- **Today.** The audio, watch, ring and home-server loop is buildable now: an LTE watch, open-ear buds, a health ring and a server give you hands-free voice, health tracking, NFC payments and an agent that runs errands.
- **Next, the missing seam.** The earbud with a camera. The camera parts are small enough; the missing piece is a product with an open interface. A realistic first test is a watch tap that sends one frame straight to your server, tried on a meal or a whiteboard.
- **Later.** I would plan around parts you can buy now, not dates.

Four things must improve for this to go mainstream:

1. **Silent input** (EMG and whisper) that beats a thumb in a quiet room.
2. **Glasses weight and battery,** for those who want a display. Camera glasses today are roughly 43 to 50 g for 4 to 8 hours; a display model is about 69 g.
3. **Open hardware APIs.** Today a vendor can lock its camera and mic to its own assistant. A standard camera, audio and display profile any agent can use, like Bluetooth audio today, is what makes "an agent you own" possible.
4. **On-body compute and radios** efficient enough that a watch can run a small model locally instead of throttling.

## Who could build it, and is there a business

Nobody owns every piece today.

- **Meta** leads in glasses, with a distribution moat, but has no watch OS, ring or phone OS.
- **Apple** has the best watch, earbud and silicon ecosystem and a trusted health stack, but is closed to third-party agents and ships no camera glasses or ring.
- **Samsung and the Android camp** ship nearly every form factor but lean on partners for frontier models.
- **OpenAI** bought Jony Ive's hardware startup io for about $6.5 billion in stock in May 2025 and expects a first device in 2026. It has the agent brand but no wearable base.
- **Amazon** has smart-home reach, but its glasses lacked the camera and display that won.
- **Specialists** (Oura, Even Realities, Brilliant Labs) make the best single devices but can't subsidise hardware, and are exposed when a giant bundles a rival.

The first wave of "AI gadgets" is a warning. The Humane AI Pin ($699 plus $24 a month) packed LTE, a camera, a projector and a battery into one chest badge, ran hot, was slow, and shut down in February 2025 with its assets sold to HP for about $116 million; returns reportedly outran sales for a stretch of 2024. The Rabbit R1 ($199) was a second device that did less than the phone beside it. The lesson is the architecture above: one gadget carrying everything fails on heat and battery.

Rings are the proven recurring model: Oura charges $5.99 a month or $69.99 a year on top of a $349 ring, while Samsung's $399 ring skips the subscription to pull people into its phones. A startup selling a new device has to win on distribution, returns and battery cost, which is why that route is a graveyard.

My read is that the room for a small team is the **software layer in the middle**: an agent runtime on the user's own server that bridges watches, open-ear audio, ring APIs and optional glasses. Home Assistant and Tailscale won by stitching together hardware they didn't make, and the pattern fits here. It needs no new hardware. That is a judgment, not a measurement.

## The shift

The goal isn't to kill the phone. It's to stop reaching for it fifty times a day for things that take five seconds. An earbud that sees when you ask, a watch you tap to say yes, a ring that quietly keeps score, and an agent you own doing the work in the background. The phone goes in the bag, next to the laptop, and comes out when there's real work to do.

## Sources and notes

Prices and specs are list figures as of October 2026 from the pages below; check before you buy.

- Ray-Ban Meta Gen 2 and Meta Ray-Ban Display with the Neural Band: [Meta blog](https://www.meta.com/blog/ray-ban-meta-gen-2-now-available-ai-glasses-extended-battery-life-3k-video/), [Meta Display page](https://www.meta.com/ai-glasses/meta-ray-ban-display-glasses-and-neural-band/), [Gizmodo comparison](https://gizmodo.com/ray-ban-meta-gen-2-vs-gen-1-which-smart-glasses-should-you-buy-2000662645).
- Oura Ring 4 and Samsung Galaxy Ring: [Oura Ring 4](https://ouraring.com/store/rings/oura-ring-4), [ring comparison](https://lifestack.ai/blog/galaxy-ring-vs-oura).
- Even Realities G1 and Brilliant Labs Frame: [Even G1](https://www.evenrealities.com/g1), [Frame review](https://www.arcompare.com/ar-glasses/brilliant-labs-frame/).
- Open-ear earbuds: [Shokz OpenFit 2](https://shokz.com/pages/openfit2).
- Humane AI Pin and Rabbit R1: [Humane post-mortem](https://aayushjain.com/graveyard/humane-ai-pin), [Rabbit and Humane analysis](https://medium.com/@thcookieh/why-did-the-rabbit-r1-and-humane-ai-pin-fail-at-launch-c108d6e2bebb); the io deal: [deal analysis](https://statusneo.com/what-openais-daring-6-5b-io-coup-means-for-ai-hardware/).
- Watch battery figures are the makers' rated numbers. The 1–2 Wh battery and few-hundred-milliwatt LTE uplink are rough engineering estimates, not measurements of one watch.
