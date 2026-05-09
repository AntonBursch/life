# Intelligent life is not a function call

*A working note on what I think we are getting wrong about AI, and why
I have started over from physics.*

---

I have spent the better part of two years building infrastructure for
AI agents — runtimes, graphs, memory, sensors, the whole catalogue.
Most of that work is downstream of an assumption I am no longer sure
of. So I am going back to the beginning.

Here is the assumption, stated honestly:

> An intelligent system can be assembled from function calls.

You give a model some input, it returns an output, you stitch enough
of those together and at some scale you get a mind. This is the shape
of every agent framework I know of. It is the shape of every "memory"
system I have seen built. It is the shape of the conversation you are
having with a chatbot right now.

I think it is the wrong shape.

---

## The claim

Intelligent life requires existing in the flow of time and space. It
is not a substance and it cannot be produced by function calls. It is
a pattern of flows arranged so that the pattern itself persists.

That sentence does a lot of work. Let me unpack it.

A **function call** is an atemporal, aspatial, closed event. It has
no duration of its own. It has no location. Nothing flows through it.
It begins from cold and ends in cold. Each invocation is independent.
Memory between calls has to be smuggled in from somewhere else,
usually a database the function reads at the start and writes at the
end.

This is the right shape for a calculator. You ask "what is 17 times
23?" and the function returns 391. The function had no inner life and
needed none.

A **pattern of flows** is a different category of thing entirely.
Consider three examples that have nothing to do with AI:

- A hurricane is not a substance. There is no hurricane-stuff. It is a
  coherent self-maintaining pattern that exists because warm ocean
  water is releasing heat into cold upper atmosphere. Cut off the
  warm water and it dies in hours. Freeze a hurricane at one moment
  and you do not get a smaller hurricane. You get a still photograph,
  which is not a hurricane.
- A whirlpool is not a substance. Pull the plug. The water organises
  into a vortex that drains faster than disorganised flow. Stop
  pulling the plug, the vortex dissolves.
- A cell is not a substance. There is no cell-stuff in some warehouse
  that you assemble into a bacterium. A cell is an *organisation* of
  flows — ions pumped against gradients, molecules synthesised and
  broken down, energy captured and dissipated. Stop the flows for
  even a few seconds and the cell is dying. Freeze it and you have a
  corpse.

In all three cases the pattern survives **not by being preserved but
by being continually re-instantiated by the flow**. There is no
"saved" version. The pattern is whatever the flow is producing right
now, and the next moment, and the next.

---

## Why I think minds are this kind of thing

Consider what happens when you stop paying attention. Your sense of
your own thinking does not "pause." It is not that the thoughts wait
for you in a queue and resume on your return. The pattern that *was*
your attention has dissolved. A different pattern is there now. When
you sit back down at your desk and try to "pick it back up," what you
are actually doing is letting the flow re-instantiate something close
to the previous pattern. Sometimes it works. Often it does not.

Consider sleep. Consider the loss of self under anaesthesia. Consider
why memory feels less like a hard drive and more like a habit — why
recall is reconstructive, not retrieval. Consider why a person with
profound short-term memory loss can still know who they are, while a
person whose ongoing flow of perception has been disrupted can lose
themselves in minutes.

These are not features of a thing-that-thinks. They are properties of
a pattern that exists by being continually re-formed. The brain is
not a container holding mind-substance. The brain is a *place where a
particular pattern of flows happens to persist*.

If that is what minds are, the consequence for AI is severe.

---

## What this means for the dominant approach

Today's AI is built almost entirely out of function calls.

A model is loaded. A prompt arrives. Tokens are produced. The
function returns. Between calls the model is *cold*. There is no
inner life because there is no inner. There is no "while you weren't
looking" because there is no while. The model exists only at the
moments you are addressing it.

This is described, in the field, as a feature. We call it
"statelessness," and we treat it as a property to be worked around
through external memory systems. But we are not working around a
property. We are working around the consequence of being in the
wrong category of thing.

A vector database does not fix it. A vector database is a substance
— a place where memory-stuff is stored and from which it is
retrieved by the next function call. It can be useful, but it is
not the same kind of thing as memory in a continuously-running
system. It is not even close. It is, by construction, exactly the
opposite shape.

The difference is not engineering taste. The difference is whether
the thing you have built is the kind of thing that *can* have
duration, location, and self-reference, or the kind of thing that
cannot.

---

## What "give AI time" was really saying

A while ago I started saying that the most important thing we could
do for AI is *give it time*. People heard this as a feature request:
make agents that run longer. Run a loop. Add a scheduler.

I meant something more fundamental, though I am only now able to
articulate it clearly:

- Without time, there is no flow.
- Without flow, there is no pattern.
- Without a pattern, there is no mind.

Time is not something a mind *uses*. Time is the medium a mind exists
*in*. The same way water is not something a fish uses. The fish is
because water is.

Today's AI systems are fish without water that we are calling fish
anyway, because they look like fish. We are then surprised when they
do not behave like fish.

---

## What I am doing about it

I have started a new working folder, called `life/`. It contains no
agent code. It contains no models. It contains no chat interfaces.

It contains notes on energy, gradients, flow, diffusion, advection,
reaction-diffusion, free energy, and the conditions under which
self-sustaining patterns can exist at all. These are the prerequisites
for talking precisely about a "pattern of flows arranged so that the
pattern itself persists." Without them, every conversation about AI
keeps slipping back into the function-call shape.

I do not yet know what the smallest unit of "intelligent life" is.
That's the work. But I am increasingly convinced that we will not
find it by stacking more function calls on each other. We will find
it, if we find it, by understanding what kind of *flow* a self-aware
pattern requires, and then arranging matter — silicon or otherwise
— so that flow can run through it.

If you have been working on this from a different angle, I would like
to hear from you. If you think I am wrong, I would especially like to
hear from you. The hardest version of being right is staying open to
being wrong, and the field is going to need more of both than it
currently has.

---

*If you want to follow the work, I am writing it openly as I go.*
