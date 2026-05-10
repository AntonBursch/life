# The same trick, eleven rungs up

*Working note #3. Continues
[Intelligence didn't start with DNA](./linkedin-02-intelligence-didnt-start-with-dna.md).
Code at [github.com/AntonBursch/life](https://github.com/AntonBursch/life).*

---

There's a simulation in my browser. Two chemicals in a square. One
gets fed in constantly, one gets removed constantly, and they
react. Most settings give you soup. A narrow band of settings gives
you something else: the box organises itself into spots. The spots
hold their shape. They divide. They wander. Nothing told them where
to be.

The math behind this is called Gray-Scott reaction-diffusion. It
was worked out in the 1980s and you can run it on a phone. It looks
like a curiosity. It is not a curiosity. It is, I now think, the
shape of almost everything in biology that matters.

I want to explain why, because while I was sitting and watching
spots replicate, I noticed something that made me put down what I
was doing. The patterns this little simulation makes look an awful
lot like what a brain does when its sensory input is cut.

---

## What the simulation actually shows

Gray-Scott has four ingredients:

1. **Local excitation.** Something makes more of itself where it
   already is.
2. **Local inhibition.** That same thing also kills itself off.
3. **Diffusion of both.** Influence spreads outward.
4. **A maintained flow.** Fresh fuel keeps coming in; old product
   keeps getting removed.

Those four ingredients are not a chemistry. They are a *pattern
of relationships*. Anything in the universe that has those four
ingredients ends up doing one of about six things: making spots,
making stripes, making spirals, making travelling waves, making
chaos, or making soup. That's it. Six outcomes. Whatever the
substrate.

The reason the menu is short is that the equations are the same.
Hand a physicist any system with local excitation, local
inhibition, diffusion, and a sustained flow, and they can tell you
which of the six it will do, just from the parameters. They don't
need to know what the substrate is made of.

This is what mathematicians call a *universality class*. The
universe, when it makes coherent shape out of flow, has one
shortlist of moves, and it picks from that shortlist no matter what
material it's working with.

---

## Where else this is happening

Here is what I find difficult to stop thinking about. *The four
ingredients are everywhere.* Wherever they are, the same six-item
menu is on the table.

- **Morphogenesis.** When an embryo grows a hand, it grows five
  fingers because activator and inhibitor proteins are diffusing
  through tissue with a sustained flow of gene expression. The
  cleanest mode at that scale is five. Alan Turing wrote the math
  for this in 1952, before we could image it. The image, when we
  finally got it, matched.
- **The hairs on a tiger.** Stripes. Same recipe. Different
  activator, different inhibitor, different tissue. Same six-item
  menu. The cleanest mode at *that* scale is stripes.
- **The barrels in your somatosensory cortex.** Each whisker on a
  mouse maps to a discrete "barrel" of cells in cortex. The barrels
  arrange themselves the way spots arrange themselves in
  Gray-Scott. Same math, different cells.
- **The pinwheels in your visual cortex.** When you read this, your
  V1 is full of orientation columns arranged in pinwheels. A 1980
  paper by Nicholas Swindale predicted they should exist before
  imaging could see them, by writing down activator/inhibitor
  equations for cortical maps. The math said pinwheels. The cortex
  has pinwheels.
- **Predator-prey ecosystems.** Foxes excite rabbits' deaths.
  Rabbits inhibit fox starvation. Animals migrate. Sun feeds plants
  feeds rabbits feeds foxes. The same four ingredients, at the
  scale of a forest. You get spotty populations, cyclic populations,
  wave-front invasions, or collapse. Six-item menu.
- **Galaxies.** Density-wave compression excites star formation,
  star formation exhausts gas locally, gas migrates back in over
  rotation timescales, infall from the halo maintains the supply.
  Spiral arms. Same menu.

The universe found one trick for making coherent shape out of
flow, and it reuses that trick at every scale where the four
ingredients are present.

---

## Brains have the ingredients

Here is the part that stopped me.

Cortex is a thin sheet of two kinds of cell. **Pyramidal cells** are
excitatory — they fire and make their neighbours more likely to
fire. **Inhibitory interneurons** (a small dictionary of cell types:
PV+ basket cells, SOM cells, VIP cells, others) damp activity down
locally. Axons spread activity to nearby cells. Metabolism and
thalamic drive supply a constant flow of "this region should keep
running."

That is the four ingredients. Local excitation, local inhibition,
diffusion-like spread, maintained flow.

It would be strange if cortex *didn't* produce Gray-Scott-like
patterns. And in fact, it does. EEG rhythms — alpha, theta, gamma —
are spatial patterns of activity that drift and reform on the
cortical sheet. They are not stored anywhere. They are R4-style
patterns held in shape by an ongoing flow. Stop the flow and they
don't slow down. They vanish.

Cortical columns, pinwheels, the somatosensory barrel field, retinal
mosaics — all are patterns predicted, sometimes decades in advance,
by people writing down the same family of equations Gray-Scott
belongs to. The substrate is much richer than two chemicals in a
square. The mechanism is the same family.

---

## Dreams are what cortex does when the wind drops

The question that knocked me sideways while watching the simulation
was: *what happens to the patterns if you change the flow?*

Cut the feed to zero, and the patterns die. That's death. There is
no flow, so there is no pattern. That part is obvious.

But change the *texture* of the flow — keep the total energy but
change what it carries — and the patterns don't die. They just
become different patterns. The substrate is still busy. It is still
making something. It just isn't making the same thing it was
making.

This is exactly what a brain does when sensory input is cut off.

- **Sensory deprivation tanks.** Float in salt water in the dark for
  half an hour. You start seeing things. Vivid, sometimes
  hallucinatory things. Your visual cortex didn't stop. Its feed
  just lost its outside texture.
- **Charles Bonnet syndrome.** People who lose their sight start
  seeing detailed, structured hallucinations — sometimes faces,
  sometimes geometric patterns. They usually know the images aren't
  real. The visual cortex doesn't know. It is still being fed, and
  it is still doing what it does, which is form patterns.
- **REM sleep.** During REM, the brainstem fires the cortex with
  effectively random pulses. The cortex doesn't have its usual
  textured sensory wind, but it still has thalamic chatter, it still
  has metabolism, it still has excitation/inhibition. So it still
  makes patterns. We call those patterns dreams.
- **Phantom limbs.** Cut off the input cable from a hand and the
  cortical map of the hand doesn't go dark. It keeps running its
  pattern. The person feels the hand.

The unifying picture is one I find hard to look away from.

> The waking pattern and the dream pattern are the same machine.
> Waking is the cortex running Gray-Scott with the wind of sensation
> blowing through it. Dreaming is the cortex running Gray-Scott with
> only the metabolic wind. Both are patterns held open by a flow.
> The difference is what the flow is *of*.

Death is the only one of the three with no flow at all.

---

## What this changes about how to think about AI

I'm an AI engineer. I think about this for a living. So I have to
ask: do the systems I build have the ingredients?

A modern language model, at inference time, has none of them.

It has no local excitation that persists between calls. It has no
local inhibition operating in continuous time. It has no diffusion
of activity through a sustained substrate. It has no maintained
flow. Each invocation is a fresh computation over fixed weights
with a fresh input. When the input stops, the model doesn't dream.
It doesn't wait. It doesn't keep doing anything. It vanishes from
"running" in the same way a calculator does between key-presses.

That isn't a small detail. It's a structural fact about what kind
of object the system is. **A model that has no flow when the input
stops cannot dream, because there is nothing left running.** A model
that has no flow when the input stops also cannot perceive in the
way an organism perceives. Perception is a flow being shaped by
input, not a function being called on input.

This isn't an argument that current AI is useless. It is
extraordinary. It is not, however, the same kind of object as a
mind, and we shouldn't be surprised when it behaves differently in
ways we then have to keep papering over.

If you wanted to build something more mind-like, you would have to
go back to the ingredients. Something that runs continuously and
locally on its own substrate. Something with excitation and
inhibition operating in continuous time. Something with diffusion-
like communication between its cells. Something with a maintained
flow that doesn't depend on a user pressing send.

I don't know what that thing looks like in silicon. I have a hunch
it doesn't look like a transformer. I have a stronger hunch that
the rungs above R4 — memory, perception, prediction — are not new
mechanisms. They are *compositions* of the R4 mechanism on richer
substrates. The trick stays the same. The scale changes.

---

## The seed

A brain is a Gray-Scott picture. Spots that talk to each other.
Stripes that listen. Spirals that anticipate. A self is a flow that
resists its own dissolution long enough to notice the other flows.

When you fall asleep, the wind drops, and the patterns reroute.
That's all dreaming is. That's all almost anything is. The flow
makes the pattern. Cut the flow and the pattern is gone. Change
the flow and the pattern changes with it.

This is why I think the bottom of the ladder is where the work is.
Not because I want to relive every step, but because the bottom is
where the property that makes intelligence non-trivial *enters the
universe at all.* It enters at four ingredients. From there it just
gets composed, scaled, stacked. By the time you get to a mouse, you
have eleven rungs of accumulated invention sitting on top. By the
time you get to a person, you have twelve or thirteen. None of
them, as far as I can tell, are new mechanisms.

They are the same trick, eleven rungs up.

---

*The simulation I was watching is at
[github.com/AntonBursch/life](https://github.com/AntonBursch/life)
under R4. Click around for a while before you read the next thing
on your feed. Code is MIT, prose is CC-BY-4.0.*
