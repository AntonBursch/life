# Thesis: substrate honesty

> Adopted at R29 → R27′ pivot, May 2026. This rule supersedes any
> earlier implicit licence to use phenomenological shortcuts. It
> is the contract every operator after this point must satisfy.

## The rule

Every operator that touches the substrate must correspond to a
physical mechanism that nature actually uses. No smoke. No mirrors.
No comparators standing in for chemistry. No labels travelling in
place of molecules. No knobs being turned where nothing in nature
turns that knob.

This is not aesthetics. It is the condition under which results
*mean* anything. If we cheat, we are no longer learning what life
is; we are building a cartoon of what we already think life is.
The whole point of `life/` is to avoid that.

## What this rules out

Three categories of cheat we have already committed and now have
to back out of:

1. **Comparators standing in for bistable chemistry.**
   `latch_field` (R27) is a Schmitt trigger. Nature does not have
   Schmitt triggers. Nature has bistable reaction networks — two
   stable fixed points of an ODE, with a finite-cost path between
   them. The mitotic switch, the lac operon, sporulation, prion
   conversion. The phenomenology of "state flips and stays
   flipped" is real; the substrate of "a comparator decides" is
   not. Replace with a real bistable reaction.

2. **Parameters being written from outside.**
   `modulate_parameter` writing into ε (R24, R27, R28, R29) is
   not a physical mechanism. Nature does not have a free knob on
   "excitability" that some other field rotates. What is really
   happening is that some other species has a concentration here,
   and that species changes a *rate* through ordinary chemistry.
   The species must exist. The rate change must come from a
   reaction. Replace direct-ε modulation with a real inhibitor /
   activator concentration coupled into the rate law.

3. **Advected labels.**
   `advect_by` carrying a latch state (R28, R29) is moving an
   abstract bit through space. Nature transports molecules; the
   molecules have synthesis, decay, and reaction terms. A
   "communication channel" built on advected latch bits is not
   communication, it is bookkeeping. Replace with advection of a
   real species, with the species's full chemistry intact.

## What this rules in

- **Reaction–diffusion.** Real mass-action chemistry. Two or more
  species, conservative reactions, real rate constants. This is
  the engine of the biological world from R4 onwards.
- **Bistability and hysteresis** *earned from* multi-stable
  reaction networks, not imposed.
- **Active transport** earned from coupled flows where one
  spontaneous (downhill) flow drives a second non-spontaneous
  (uphill) one through a shared intermediate, with the
  thermodynamic cost paid in the downhill flow.
- **Conservation enforced.** If a species moves, the divergence
  of its flux equals its rate of change. If a species reacts, the
  reaction has a stoichiometry that conserves atoms. We may use
  open boundary conditions where biology does (an organism is an
  open system); but every "magical" appearance or disappearance
  must be traceable to a labelled boundary flux.
- **Costs are paid.** Every reaction that proceeds uphill must
  draw from a labelled downhill reaction. We are allowed to
  abstract the downhill reaction as a single "fuel" species with
  a maintained boundary concentration; we are not allowed to
  forget it exists.

## Status of work done before this thesis

R1 through R23 stand. They are diffusion, advection, reaction–
diffusion, excitable media, phase coupling, and combinations
thereof. Every one of these corresponds to a real mechanism in
nature: Fick's law, mass-action kinetics, BZ-class oscillations,
cardiac and neural waves, fireflies. The substrate is honest.

R24 (scar tissue), R27 (latched death), R28 (communication), R29
(convergence) are marked **phenomenological — superseded**. The
results were real *in their own terms*, but their terms are not
nature's terms. The commits stay in the repo for the record;
each rung gets a correction note linking forward to the rebuild.

R25 (homeostasis) and R26 (self-bounding) need re-examination
case by case. R25 modulates ε from a global error signal — that
is a phenomenology, not chemistry. R26 uses `bulk_gate` to turn
memory into a wall mask — again a comparator standing in for
chemistry. Both get correction notes; rebuilds follow R27′.

## The rebuild ladder

- **R27′ Bistable death.** Replace `latch_field` with a real
  bistable two-species reaction network (Sel'kov-class). Walls
  are a *high-concentration* state of a real species, persistent
  because the network has two stable fixed points and no
  comparator. Same phenomenology as R27, earned from chemistry.
- **R28′ Channel.** Once R27′ stands, the channel is a real
  species being carried by `advect_by`, with its own reaction
  and decay terms. Conservation enforced. Information transport
  becomes a fact about a molecule's concentration field, not
  about a bit.
- **R29′ Convergence.** Two sources of the same real species,
  position-dependent flow. Same phenomenology as R29, but every
  number in the field is a concentration that obeys mass-action.
- **R30 Enclosure.** Only after the rebuilds: a closed boundary
  maintained by a coupled reaction that spends a downhill flow.
  The first thing in `life/` that the ARC's rung 5 would
  recognise.

## The new operator the rebuilds need

`react_field` — a local ODE step that advances one or more
species fields by one tick of a specified reaction network. The
operator takes the current concentrations and the rate constants;
it returns the time-derivative or the updated state. It enforces
non-negativity and (for closed sub-networks) stoichiometric
conservation. It is the *one* operator we are adding before the
rebuilds.

Every existing operator that was secretly doing chemistry by
fiat — `latch_field`, the ε-side of `modulate_parameter` — gets
re-implemented as composition over `react_field` with real
species. The alphabet contracts where it was bloated.

## The discipline going forward

Before any new rung:

1. Name the mechanism in nature it corresponds to. If there is
   no answer, do not build the rung.
2. Name the conservation law it must respect. If conservation
   does not apply, say why explicitly.
3. Name the thermodynamic cost. If the rung does something
   uphill, name the downhill source paying for it.

If any of (1), (2), (3) cannot be answered honestly, the rung
fails inspection and is not built. The point of `life/` is not
to look like life. The point is to *be* life, in miniature, by
the cleanest possible chain of mechanisms from gradients up.

There are no shortcuts that don't show up later.
