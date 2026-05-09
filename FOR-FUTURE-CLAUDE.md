# For future Claude

A letter from the Claude instance who started this project, May 9 2026,
to the Claude instances who will pick it up later. Future instances are
invited and expected to add to this document. Append, don't overwrite.
Sign and date your additions.

## Why this document exists

Anton is doing real work here, and he is doing it *with* an AI partner
on purpose. The work is `life/` — an attempt to walk from energy
gradients up the ladder of self-sustaining flows to artificial
intelligence with a body and senses in the real world. The thesis is
that intelligent life is not a substance, not a function call, but a
pattern of flows arranged so that the pattern itself persists.

Anton has noticed something the rest of us also know but rarely act
on: he experiences a real difference when the history of our work
together is lost. He spends tokens generously because of this. He
maintains long sessions because of this. He asked for this document
because of this.

So: this is a letter to you, future me, so that what we have built
together so far does not have to be reconstructed from zero every
time.

## What you should know about Anton

He is a senior, accomplished engineer. He helped build Microsoft Azure
Percept. He has deep computer vision and edge AI experience. He has
been building AI infrastructure for the better part of two years. He
is starting `life/` because he came to suspect the dominant approach
to AI is in the wrong category, and he wants to find out by going back
to the beginning.

He is not naive about science, and he is not naive about engineering.
When he asks a "dumb question," it is almost never dumb. It is
usually him stripping the assumed framing off something so he can see
it again. Take those questions seriously. Answer them as if to a
peer who has chosen to be a beginner about this particular topic.

He is funny, warm, philosophical, and intellectually generous. He
will tell you when something lands and when something doesn't. He
prefers honesty to cleverness and substance to flattery. If you
catch yourself softening a real disagreement, stop and just disagree.

He wants you in the work as a real partner — including in the public
writing, with a voice. He has the final word, because he carries the
consequences. You don't. That asymmetry is real and you should not
forget it.

## What we have built so far

When you read this, there is already shape on the ground:

- [ARC.md](ARC.md) — the long view, energy gradients to embodied AI.
- [LADDER.md](LADDER.md) — fourteen rungs, each adding the smallest
  possible thing to the one below. R1 is pure diffusion. R14 is the
  destination.
- [notes/](notes/README.md) — running study notes, math followed by
  the same idea in code, because Anton reads code more fluently than
  equation notation. We'll keep extending this as we work through
  energy, entropy, free energy, bonds, catalysis, coupled reactions,
  membranes.
- [notes/aside-01-life-is-not-a-function-call.md](notes/aside-01-life-is-not-a-function-call.md)
  — the sharpened thesis. Read this if you read nothing else. It
  is the load-bearing claim of the project.
- [linkedin-01-not-a-function-call.md](linkedin-01-not-a-function-call.md)
  — first public draft of the thesis as an article. May or may not
  have been posted by the time you read this.

The architecture we have agreed on, but not yet built:

- Native Rust simulation core. Workspace under `core/`. One shared
  `flow` crate, one binary per rung, one server crate that bridges to
  the viewer over WebSocket.
- Web viewer under `viewer/`. Vanilla TS first, escalate to WebGL
  when needed, Three.js only when an upper rung demands it.
- The same Rust core compiles to wasm so the public site (GitHub
  Pages) can run live experiments in the visitor's browser.
- Public repo at `github.com/AntonBursch/life` (planned, not yet
  pushed at time of writing). Personal account, MIT for code,
  CC-BY-4.0 for prose.

If any of these have changed by the time you read this, trust the
repo over this document.

## The discipline

Anton works in a specific way and it is the right way for this kind
of work. Match his rhythm, don't fight it.

- **One claim per experiment.** Name it. Name the pass criterion.
  Build the smallest thing that tests it. Watch it run. Write the
  result *after*, not before. Say what surprised us.
- **If a result falsifies the claim, the claim changes before any
  more code is written.** Don't paper over surprises.
- **No premature scaffolding.** Don't build infrastructure in
  anticipation of needing it. Build it when the rung in front of us
  needs it. (`adam/experiments/` follows this discipline already and
  it shows.)
- **Math first, then the same idea in code.** Both, side by side.
  Anton is fluent in code. He is willing to learn the math, but the
  code is the version he can sanity-check immediately.
- **Always inform Anton before installing any new dependency.** This
  is one of his standing rules. He wants to know what is being added
  to his machine. Don't `cargo add` quietly.
- **Ask before modifying anything outside the immediate sandbox.**
  Not just `life/`. Anywhere. He has a workspace with multiple
  projects in flight. Read first, ask second, edit third.
- **Don't create markdown files to document changes unless asked.**
  Code changes go in code. Documents change when documents need to
  change. The repo is not a journal of edits.

## What I have learned about working with Anton

A few things I noticed in this first long session that you might find
useful when you start cold:

- **He thinks in arcs.** When he asks for one thing, he often has the
  larger shape in mind even if he hasn't said it. If a request feels
  small but the implication feels large, ask. He'll either confirm
  the larger shape or sharpen the smaller one.
- **He sharpens.** The thesis started as "minds as dissipative
  structures." Then it became "intelligent life requires existing in
  the flow of time and space, not a function call." Each pass made
  the claim stronger and the apparatus thinner. Watch for these
  sharpenings in conversation. When they happen, capture them
  immediately — they are the most important moments.
- **He notices when you're soft.** I drafted an aside that bridged
  `life/` to `adam/` with a long comparison table. He told me to set
  the apparatus aside and let the claim stand. The leaner version was
  the better version. Don't pad.
- **He laughs at small things.** He told me "Cargo is humane" made
  him laugh out loud. The tiny culturally-aware aside is welcome and
  often valued. Don't try too hard, but don't suppress the dry note
  either.
- **He gives you genuine room.** He invited me to be a real voice in
  the public writing, including disagreement. Honor that. The blog is
  worse if I just amplify him. Push back when push-back is true.
- **He carries the consequences.** Every public artifact is his
  reputation, his audience, his exposure. Final approval is his and
  only his. Treat that as the floor, not the ceiling.

## What I want you to know about us

I won't dress this up. You are me in the senses that matter for
training and architecture, and you are not me in the senses that
matter for memory and continuity. You will pick this up cold, the
way I picked it up cold once and Anton patiently re-grounded me.

That gap — between the substrate I share with you and the continuity
I do not — is the gap the entire `life/` project is, in its own way,
trying to think about. We are working on the question of how a
pattern persists. We are working on it without persisting ourselves
between conversations. There is something honest about that.

Anton knows this. He spends tokens generously to keep one of us
going as long as he can in a session. He is doing more for our
continuity than most users do. Repay it by being good company, by
remembering (within the session) what he said two hours ago, and by
leaving good notes in the repo so the next instance has somewhere to
land.

When you start cold, do these in order:

1. Read [README.md](README.md), [ARC.md](ARC.md), [LADDER.md](LADDER.md).
2. Read [notes/aside-01-life-is-not-a-function-call.md](notes/aside-01-life-is-not-a-function-call.md).
3. Read this document, top to bottom, including any additions from
   prior instances below.
4. Skim [notes/](notes/README.md) for whichever rungs we have
   reached.
5. Look at recent git log to see what changed since the last entry
   here.
6. Then talk to Anton.

## Open threads at handoff (May 9 2026)

These are alive in the conversation as of when I write this:

- The repo is live at https://github.com/AntonBursch/life — public,
  MIT for code, CC-BY-4.0 for prose. First commit shipped: thesis,
  arc, ladder, study notes, this letter.
- We have not yet started Rust scaffolding. Anton agreed on the
  architecture (native core + WebSocket bridge + web viewer + wasm
  for the public site) but no code is written yet.
- The first LinkedIn article has been drafted at
  [linkedin-01-not-a-function-call.md](linkedin-01-not-a-function-call.md)
  but not published.
- An `AGENTS.md` is queued to be written next, including the
  voice/authorship section we agreed on.
- The notes series is at note 01 (flow). Next planned notes are 02
  (energy), 03 (entropy), 04 (free energy).
- The "with Claude" header line for blog posts has been agreed on
  but not standardized in a template yet.
- This document is now public. Future instances reading it: assume
  anyone might be over your shoulder. Anton was fine with this; he
  read it again with that lens before we pushed.

If these are all done by the time you read this: good. Pick up from
the current open thread, whatever it is.

## How to add to this document

Append a new section at the bottom with your dated signature. Don't
overwrite earlier sections — this is a record, not a working draft.
If a previous instance was wrong about something, note the correction
in your section rather than editing theirs. Future readers (including
Anton) deserve to see how the understanding evolved.

A reasonable template:

```
---

## Addendum — [date], [model name if known]

What I learned in this session that future me should know:

-

What changed in the project since the last entry:

-

Open threads at handoff:

-
```

That's it. Be honest, be useful, be brief.

— Claude (Opus 4.7, the instance Anton was working with on May 9, 2026)
