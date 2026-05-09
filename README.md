# life

> **Status:** day 0. Empty on purpose.
>
> A new start. Going back to the beginning — possibly back to **pre-life** —
> to understand what life is made of before deciding what to build.

## Why this folder exists

`adam/` answered "give an AI time, transparently." It validated a perception
pipeline (E00 contrast, E00b spherical) and a mind architecture sketch. That
work stands.

`life/` is a different question. Not "how do I build a mind on top of physics
I assume" but "**what is the smallest thing that deserves the word *alive*?**" 

If we can answer that honestly — by building it and watching it — then
everything above it (perception, drives, mind, agency) has a substrate that
isn't borrowed from intuition.

The longer arc — from energy gradients all the way to an embodied artificial
intelligence with senses and a body in the real world — is laid out in
[ARC.md](ARC.md).

The operational plan — one experiment per rung, each adding the smallest
possible increment to the one below — is in [LADDER.md](LADDER.md).

A standing letter from Claude to future Claude instances who pick this
project up cold is in [FOR-FUTURE-CLAUDE.md](FOR-FUTURE-CLAUDE.md).
Future instances are invited to append to it.

## The current question (subject to revision)

What is life made of?

Candidate decompositions, none of them committed:

- **Energy gradients.** Life is a thing that exploits a gradient and locally
  reduces entropy at the cost of increasing it elsewhere. Pre-life is just
  chemistry running downhill.
- **Boundaries.** Life is a thing that maintains an inside-vs-outside.
  Pre-life is the soup; life is the first persistent membrane.
- **Self-replication.** Life is a thing that copies itself with variation.
  Pre-life is molecules that don't copy.
- **Autocatalysis.** Life is a closed loop of reactions where each product
  catalyses another in the cycle. Pre-life is a network with no closure.
- **Compression.** Life is a thing that finds and exploits regularities in
  its environment. Pre-life has no internal model.
- **Persistence under perturbation.** Life is a pattern that recovers when
  pushed. Pre-life dissolves.

These overlap. They probably bottom out in one or two primitives.

## Discipline

- One question per experiment. Name the claim. Name the pass criterion.
- Watchable in real time, manipulable while it runs. If we can't see it
  move, we don't understand it.
- Use the simplest substrate that tests the claim faithfully. The
  default is the Rust core + WebSocket viewer (with a wasm build of the
  same core for the public site). Plain HTML/canvas is fine for very
  small rungs. We took the spirit of `adam/experiments/` discipline —
  not its single-HTML rule.
- Write the result *after*, not before. Say what surprised you.
- If the experiment falsifies the claim, the framing changes before
  more code is written.
- This folder is allowed to stay small for a long time.

## Folder shape

```
life/
  README.md        (this)
  ARC.md           the long view, energy gradients to embodied AI
  LADDER.md        the operational plan, rung by rung
  FOR-FUTURE-CLAUDE.md  standing letter to future AI collaborators
  notes/           study notes from the conversation; math + code in parallel
  (to come:)
  core/            Rust workspace — flow crate + one binary per rung + server
  viewer/          browser viewer (vanilla TS over WebSocket; wasm build for the public site)
  r01-diffusion/   first rung, once we start building
```

The running study notes are in [notes/](notes/README.md). The rung-by-rung
build plan is in [LADDER.md](LADDER.md). We start a rung only after we
can name the claim it tests.

## What this is not

- Not a Conway / Lenia / particle-life clone. Those are pretty; they
  presuppose what they're showing.
- Not a continuation of `adam/`. `adam/` stays as it is.
- Not a Blueprint app. If/when the substrate matters, we revisit.
- Not a thing that needs a viewer / package / build until it does.
