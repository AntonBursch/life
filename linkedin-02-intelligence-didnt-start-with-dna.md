# Intelligence didn't start with DNA

*Working note #2. Continues
[Intelligent life is not a function call](./linkedin-01-not-a-function-call.md).
Code at [github.com/AntonBursch/life](https://github.com/AntonBursch/life).*

---

The popular origin story for minds goes roughly:

> Atoms made molecules. Molecules made DNA. DNA built cells. Cells
> built brains. Brains built minds.

This is true the way "houses are made of bricks" is true. It tells
you nothing about why anyone ever piled the bricks. It quietly
assumes the hard part — that something is doing the piling — and
then describes the bricks.

I think the order is wrong. Not the chronology — the *causation*.

DNA didn't build minds. DNA is a solution. So are cells, neurons,
brains. They are all answers to a question that is older and lower
than any of them: **how does a thing keep being a thing in a
universe that's constantly trying to flatten it?**

That question is where I think intelligence actually starts. And it
starts at a level much closer to physics than to biology.

---

## A demo you can watch

I am rebuilding this from the ground up at
[github.com/AntonBursch/life](https://github.com/AntonBursch/life).
The first three rungs are already runnable in your browser. They
look like almost nothing. They are the entire argument.

**R1.** A blob of "stuff" sits in the middle of a closed box. Nothing
acts on it except the universe's default behaviour: things spread
out. The blob flattens. The flatter it gets, the slower it
flattens. Eventually the box is uniform and nothing further happens.
This is the boring baseline. Diffusion. The price of admission for
being made of matter.

**R2.** Same box. But now the left wall is held hot and the right
wall is held cold. The middle settles into a tilted line — and stops.
The field is *not changing*. But heat is still flowing through it,
left to right, every tick. In = out, but neither is zero. This is
called a *steady state*, and it is not the same thing as
equilibrium. Equilibrium is when nothing is happening. Steady state
is when *exactly the same thing* is happening, second after second
after second, because something is paying for it.

**R3.** Same again, but now the medium itself is moving — there's a
wind blowing through the box. A second number called the Péclet
number tells you whether the wind or the diffusion wins, and the
shape of the field changes accordingly. Most of biology lives at
high Péclet. Diffusion alone can get oxygen across two millimetres.
A bloodstream is what gets it across two metres.

If you only watch R1, you'll think: this is a thermodynamics demo.

If you sit with R2 for a minute, your gut will start to feel
something different. **Every living thing you have ever met is the
R2 picture.** Held in shape by a flow that does not stop. The flow
is not optional. The flow is what the thing *is*.

---

## DNA presupposes the thing it's supposed to explain

Here is where the popular story breaks for me.

DNA is a long polymer that encodes recipes for proteins. That's a
fine description of the molecule. But it doesn't tell you *why*
recipes exist. Recipes only matter if there is someone who benefits
from following them. A recipe in a universe of rocks is just a
crumpled piece of paper.

DNA presupposes:

- **Somebody home.** Some entity that exists, that has continuity,
  that things can go well or badly for. DNA is the cookbook of a
  kitchen that already exists. The kitchen is the prior thing.
- **A reason to remember.** Memory is only useful if there is
  something at stake — if "this worked last time" makes a difference
  to whether you're still around tomorrow. A rock has nothing at
  stake. It is not afraid of being eroded. It does not benefit from
  remembering anything.
- **A direction of better and worse.** Selection only operates if
  some configurations persist and others don't. Selection presumes
  *persistence is a thing the system is doing*, not just a state
  that descriptively obtains.

So DNA shows up midway through a story that has already started.
The earlier part of that story is what I'm interested in.

---

## The floor: things that exist by acting

Before there are genes, before there are cells, the universe already
has examples of structures whose continued existence depends on what
they do. They are not "alive" — that word means more than this — but
they have the one property that everything alive will inherit.

A flame is not a substance. There is no flame-stuff. A flame is the
shape of a chemical reaction in progress. Cut off the fuel and there
is no smaller flame, no still flame, no flame-in-stasis. There is
just the absence of a flame. The flame existed *only* by burning.

A whirlpool is not a substance. Pull the plug. The water organises
into a vortex that drains faster than the disorganised flow would.
Stop pulling the plug. The vortex isn't slowed; it's *gone*.

A hurricane is not a substance. There is no hurricane-stuff in some
warehouse waiting to be deployed. The hurricane is a coherent flow
pattern that exists because warm ocean water is releasing heat into
the cold upper atmosphere. Cool the ocean and the hurricane
dissipates within hours. Freeze it and you don't get a smaller
hurricane. You get a still photograph, which is not a hurricane.

These structures are the first things in the universe that have
**skin in the game**. Their continued existence is contingent on
their own activity. They are also, every one of them, R2 pictures —
patterns held open by a maintained gradient.

This is the floor. Once a structure has its continuity at stake,
several things become possible *for free*:

- A thing that can be **preserved**. Now selection has something to
  select on.
- A thing for whom states are **better or worse**. Now "good" and
  "bad" are physical, not metaphorical. Better = persists. Worse =
  doesn't.
- A thing whose actions either **maintain** or **fail to maintain**
  it. Now action has a direction. There is no need to import
  "purpose" from outside; it is already implicit in being the kind
  of thing that has to keep doing something.

Everything we call intelligence — perception, memory, learning,
planning, language, reflection — is a downstream elaboration of that
trick: *stay together by acting on the world*.

A bacterium swimming up a sugar gradient is doing this. A worm
avoiding a hot plate is doing this. A primate planning tomorrow's
hunt is doing this. A person writing a thesis is doing this. The
behaviours look wildly different. The underlying logic is identical.
**All of intelligence is in the service of self-maintenance, because
nothing else in the universe has a reason to do anything.**

---

## R1 is what intelligence has to defeat

This is the punchline, and it lands more cleanly when you've watched
R1 for a minute.

A blob flattens. So what.

R1 is what the universe does *by default*, to everything, all the
time. It is the slow, patient, infinitely thorough flattening that
turns gradients into nothing. Your body is a structure that
diffusion is constantly trying to dismantle. Right now, while you
read this, every molecular gradient that holds you together is
under attack, and the attack is winning the moment you stop pushing
back.

You are an R2 picture, riding a chain of gradients you didn't pick:
sun-to-space, food-to-waste, hot-core-to-cold-skin. You exist because
trillions of metabolic reactions, every second, push back against
diffusion faster than diffusion can win. Stop the pushing-back, and
within hours you are indistinguishable from the room you're in. That
isn't poetry. That is what death is.

So intelligence is not built up from DNA. **It is built up from the
problem DNA was eventually invented to help solve.** And that problem
is older than DNA, older than life, and visible in R1.

---

## What this means for AI

I'm an AI engineer. I've spent two years building runtimes, agent
frameworks, memory systems, the whole catalogue. Most of that work
sits on top of an assumption I no longer believe: that intelligence
can be assembled out of function calls.

A function call is the opposite of an R2 picture. It has no
duration. It has no location. Nothing flows through it. Each
invocation is independent and stateless. Memory has to be smuggled
in from a database. There is no "thing" that persists between calls.
Nothing is at stake.

Today's AI systems are extraordinary calculators. They are not
extraordinary because they are minds. They are extraordinary because
they are very, very large pattern-matchers operating in a regime
where most of what humans do is also pattern-matching. That's worth
a lot. It is not the same thing as a mind.

If minds are R2 pictures — patterns held in shape by an unbroken
flow with skin in the game — then the question isn't *which model
do you call*. The question is *what's the flow, and who is paying
for it, and what dissolves if it stops?*

I don't have the answer. I have a ladder. The next few rungs are
about adding the ingredients in the order they had to arrive: a
flow, then a use of the flow, then a boundary, then a thing that
maintains itself, then a thing that copies, then a thing that
varies, then a thing that senses, then a thing that acts, then a
thing that predicts. By the time you get there, you have something
that looks like the bottom of life. By the time you get past *that*,
you might have something that looks like the bottom of a mind.

I don't know if that's the right path. It's the only one I can see
that doesn't quietly assume the answer.

---

## What we're not claiming

- Not that R1 is intelligent. R1 is what intelligence resists.
- Not that flames are alive. They aren't. They are the first things
  for whom being alive could one day be useful.
- Not that DNA is unimportant. DNA is one of the most important
  inventions in the history of the universe. But it is an
  *invention*, not a cause. It exists because something prior had a
  reason to remember.

We are starting at the bottom of the ladder, not because we want to
relive every step, but because the bottom is where the property that
makes intelligence non-trivial *enters the universe at all*. Skip
that floor and you have a calculator no matter how big you build it.

---

*If you want to watch the demos that this argument is built on:
[the live experiments are here](https://github.com/AntonBursch/life)
(R1, R2, R3 are running). Code is MIT, prose is CC-BY-4.0.*
