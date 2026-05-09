# 01 — Flow across space and time

A gradient is a *difference*. Flow is the *response* to that
difference. To understand life as something that lives in a flow, we
need to be precise about what flow is — because there are several
distinct kinds, and life uses all of them.

## The picture: stuff, sources, sinks, conductors

Take a 2D box. Pick a quantity — temperature, concentration, voltage,
height of water. At every point in the box, that quantity has a value.
We call this a **field**, written $\phi(x, y, t)$.

Three rules wire the whole thing together:

1. **Field → gradient.** At any point, the gradient $\nabla \phi$ is
   the local "slope" — how fast the value changes as you move in
   space, and which direction is steepest.
2. **Gradient → flux.** Flux $\mathbf{J}$ is the rate of stuff moving
   past a point. Simplest law: $\mathbf{J} = -D \nabla \phi$. Stuff
   flows downhill, scaled by a constant $D$ called the diffusivity.
3. **Flux → field change.** Where flux converges, the field rises.
   Where it diverges, it falls. This is bookkeeping — stuff has to come
   from and go to somewhere: $\partial \phi / \partial t = -\nabla \cdot \mathbf{J}$.

Combine the three and you get the **diffusion equation**:

$$\frac{\partial \phi}{\partial t} = D \nabla^2 \phi$$

In words: at each point, the value changes over time in proportion to
how much it differs from its neighbours' average.

### Same thing in code

```js
// 1D diffusion on a grid of N cells. phi is the field.
// We update phi using the simplest possible "neighbour average" rule.

const N = 100;
const D = 0.5;     // diffusivity — how readily stuff spreads
const dx = 1.0;    // spacing between cells
const dt = 0.5;    // time step (small enough for stability)

const phi  = new Float32Array(N);
const next = new Float32Array(N);

phi[N >> 1] = 1.0;   // start with one hot cell in the middle

function step() {
  for (let i = 1; i < N - 1; i++) {
    // Discrete Laplacian: how much does this cell differ from its
    // neighbours' average? That's what ∇²φ means in 1D.
    const laplacian = (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / (dx * dx);
    next[i] = phi[i] + dt * D * laplacian;
  }
  // boundaries (here: zero-flux at the edges)
  next[0]   = next[1];
  next[N-1] = next[N-2];
  phi.set(next);
}
```

Read line by line: the field at cell `i` after one tick is the field
now, plus a small step proportional to how much its neighbours pull on
it. That *is* the diffusion equation. The math and the code are the
same statement.

## Three different things people call "flow"

### 1. Diffusion (no carrier, just statistics)

Drop ink in still water. Ink molecules don't go anywhere on purpose
— they jitter randomly. But because there are *more* ink molecules in
the high-concentration region, more of them randomly wander out than
in. On average, smooth flow from high to low.

- Slow.
- Driven only by gradient.
- Reversible in principle, irreversible in practice (entropy).
- Goes to flat and stops.

This is the **default**. If you do nothing else, this is what happens.

### 2. Advection (the medium itself moves)

Pour cream into coffee, then *stir*. The cream goes wherever the
coffee goes. Not diffusion through stationary water — being carried
along by the flow of the medium itself.

- Fast.
- Driven by whatever drives the medium (pressure, gravity, pumping).
- Doesn't care about the gradient of the carried thing.
- Can move stuff *up* a concentration gradient because the carrier
  doesn't know.

In real systems advection and diffusion happen together:

$$\frac{\partial \phi}{\partial t} = D \nabla^2 \phi - \mathbf{v} \cdot \nabla \phi$$

The first term is diffusion. The second is advection — the medium has
a velocity field $\mathbf{v}$ that drags the substance along.

### Same thing in code

```js
// 1D advection-diffusion. Now there's a wind blowing rightward at
// constant speed v in addition to diffusion.

const v = 0.2;     // wind speed (cells per unit time)

function stepAdvDiff() {
  for (let i = 1; i < N - 1; i++) {
    const laplacian = (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / (dx * dx);

    // First-order upwind: which neighbour does the wind blow stuff
    // FROM? Use that one for the gradient. (Picking the wrong side
    // makes the simulation explode — this is a real numerical gotcha.)
    const grad = v >= 0
      ? (phi[i]     - phi[i - 1]) / dx
      : (phi[i + 1] - phi[i]    ) / dx;

    next[i] = phi[i] + dt * (D * laplacian - v * grad);
  }
  next[0]   = next[1];
  next[N-1] = next[N-2];
  phi.set(next);
}
```

You can see both forces in the same line: a smoothing term and a
sliding term. Set `v = 0` and it's pure diffusion. Set `D = 0` and a
peak slides without spreading. Set both, and you get something that
both spreads *and* drifts — which is what most real flows look like.

### 3. Active transport (a structure does work to move stuff)

A cell membrane has pumps that grab a sodium ion on the inside and
shove it out, *against* the concentration gradient, by burning a
chemical fuel. Not diffusion (it goes the wrong way) and not advection
(no bulk flow). It's a thing using a *different* gradient — the
chemical fuel's gradient — to push something else uphill.

- Costs energy from somewhere else.
- Can build gradients that wouldn't form on their own.
- Requires a **machine**. A structure that couples one flow to another.

This is the moment pre-life ends and life begins. Diffusion and
advection happen everywhere. Active transport, as far as we know, only
happens inside cells and the machines we've built. We come back to
this in [07-coupled-reactions.md](07-coupled-reactions.md) and
[08-membranes.md](08-membranes.md).

## How fast, how far — the time-and-space scales

This part surprises people. Each kind of flow has a different
relationship between distance and time.

### Diffusion scales like $\sqrt{t}$

The most important and least-known fact about diffusion. Distance
travelled grows as the **square root** of time:

$$\langle r \rangle \sim \sqrt{D t}$$

What this means:

- A small molecule diffuses across a single cell (~10 µm) in
  milliseconds. Fine.
- The same molecule diffusing across a centimetre takes **hours**.
- Across a metre, **years**.
- Across a kilometre, **millions of years**.

Diffusion is a great way to move things short distances and a *terrible*
way to move things long distances. **This is why your cells are small.**
This is why you have a circulatory system instead of relying on oxygen
diffusing from your skin to your toes. The square root is brutal.

### Same thing in code (the random-walk view)

```js
// Each "particle" takes a random ±1 step every tick. Track how far
// it has wandered after t ticks. Average over many particles.

function diffusionDistanceAfter(steps, particles = 10_000) {
  let sumSq = 0;
  for (let p = 0; p < particles; p++) {
    let x = 0;
    for (let s = 0; s < steps; s++) {
      x += Math.random() < 0.5 ? -1 : +1;
    }
    sumSq += x * x;
  }
  // Root-mean-square distance. This is the relevant "typical" distance
  // for a diffusing thing — it grows like sqrt(t), which we can verify:
  return Math.sqrt(sumSq / particles);
}

console.log(diffusionDistanceAfter(  100));  // ~10
console.log(diffusionDistanceAfter(  400));  // ~20    (4× steps → 2× distance)
console.log(diffusionDistanceAfter(10000));  // ~100
```

Quadruple the time → only double the distance. The √t scaling falls
out of the random walk for free. **The macroscopic diffusion equation
is just this random walk seen from far away.**

### Advection scales linearly

If the medium moves at speed $v$, things in the medium travel at
$v t$. Linear in time. Predictable. Fast.

```js
function advectionDistanceAfter(steps, v = 1) {
  return v * steps;   // that's it
}
```

This is why life invented circulation. Once you're bigger than ~1 mm,
diffusion can't get oxygen to your interior fast enough; you need bulk
fluid flow to do the long-distance work. Hearts and lungs are not
optional features — they are forced moves at a certain body size.

## Reaction-diffusion: pattern from flow

When a reactive substance flows, two things happen at once: it
diffuses (smoothing), and it reacts (creating or destroying):

$$\frac{\partial \phi}{\partial t} = D \nabla^2 \phi + R(\phi)$$

The $R(\phi)$ term is the chemistry. If the reaction is autocatalytic
(more $\phi$ makes more $\phi$) and there's a slower-diffusing
inhibitor, you get **Turing patterns** — stripes, spots, spirals —
*spontaneously*, just from chemistry plus diffusion plus a flow of
fresh reactant.

These are the first patterns the universe makes that look organised
without being alive. They're how leopards get their spots and how
zebras get their stripes.

### Same thing in code (one-component sketch)

```js
// One reactive species. Reaction is autocatalytic up to saturation,
// with a constant decay term, plus diffusion. Real Turing patterns
// need two species — this is the simplified flavour.

const k_grow  = 1.0;
const k_decay = 0.5;
const sat     = 1.0;

function reaction(phi_i) {
  // grow when below saturation, decay always
  return k_grow * phi_i * (sat - phi_i) - k_decay * phi_i;
}

function stepReactDiff() {
  for (let i = 1; i < N - 1; i++) {
    const laplacian = (phi[i - 1] - 2 * phi[i] + phi[i + 1]) / (dx * dx);
    next[i] = phi[i] + dt * (D * laplacian + reaction(phi[i]));
  }
  next[0]   = next[1];
  next[N-1] = next[N-2];
  phi.set(next);
}
```

The structure to notice: the update is **diffusion plus chemistry**,
two terms in the same line. Each cell smooths toward its neighbours
*and* responds to its own current value via the reaction. Turing's
insight was that the *interplay* of these two — at different speeds
for different species — produces patterns that neither term alone
would produce.

## Time as another kind of flow

Everything above is space + time. Time has its own structure that
matters.

- **Steady state ≠ equilibrium.** A pipe with water flowing through it
  at constant rate is in steady state — nothing changes over time at
  any point — but it's not in equilibrium. There's a pressure gradient
  driving the flow. **Most of life is steady state, not equilibrium.**
- **Slow inputs, fast outputs.** A cell takes minutes to absorb
  glucose, microseconds to fire an action potential. Same physics,
  different timescales, depending on the structure mediating it. Life
  builds *temporal* gradients — fast machinery on top of slow
  reservoirs.
- **Memory.** A flow can have memory: a vortex spins long after the
  spoon stops stirring. Not chemistry — the persistence of a flow
  pattern after its driver is gone. A hint at how transient inputs
  can leave durable structure.

## Why this matters for the floor of life

Life is not a substance. It is **a pattern of flows arranged so that
the pattern itself persists.** Specifically:

- The pattern *uses* a gradient (food, thermal differential, redox
  couple).
- The pattern *builds* gradients (membrane potentials, internal
  concentrations).
- The pattern *moves stuff actively* (pumps, motors, gates) to
  maintain the gradients it needs.
- The whole thing is **self-referential**: the gradients the structure
  builds are the same gradients that maintain the structure.

That last point — self-referential, recursive — is where pre-life
becomes life. A hurricane is a flow pattern, but its structure isn't
the thing that maintains its structure. A cell *is* the thing that
maintains itself.

## Experiment seed

Three small simulations, in this order, would teach us a lot:

1. **Pure diffusion in 1D.** Drop a hot spot, watch it fall. Verify
   the √t scaling by measuring the spread vs. time.
2. **Advection-diffusion in 1D.** Add a wind. Watch a pulse drift and
   spread at the same time. Vary $D / v$ to see when each term
   dominates.
3. **Reaction-diffusion in 1D.** Add an autocatalytic reaction. Find
   the parameter regime where stable patterns form rather than
   uniform soup.

These are the prerequisite literacy for any pre-life experiment.
None of them require committing to a "creature" yet.
