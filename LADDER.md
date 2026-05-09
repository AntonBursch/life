# The ladder

> One experiment per rung. Each rung adds the smallest possible
> increment to the one below it. Each rung is run, watched, and
> written up before the next is started. If a rung surprises us, the
> ladder gets edited in place.

This is the operational version of [ARC.md](ARC.md). The arc says
*where we are going*. The ladder says *what we build next*.

## The rule for each rung

- Adds **one** new ingredient. Not two.
- Is **watchable in real time** and **manipulable while it runs**.
  Sliders, buttons, live readouts. If we can't see it move, we don't
  understand it.
- Has a named **claim** under test.
- Has a named **pass criterion** — what we have to see before moving up.
- Has a written-after **result** note saying what we actually saw and
  what surprised us.
- Uses the **simplest substrate that the claim allows**. The default
  for the early rungs is the native Rust core + WebSocket + browser
  viewer agreed to in the architecture (and a wasm build of the same
  core for the public site). For very small rungs, plain HTML/canvas
  is allowed if it tests the claim faithfully. The discipline is
  "smallest thing that tests the claim," not "smallest file count."
  This is new work; we are not bound to the `adam/experiments/`
  single-HTML rule. We took inspiration from it, not a contract.

If the result falsifies the claim, the rung does not collapse — it
gets a correction note. The ladder is the record of what we have
genuinely understood, not what we hoped for.

## The rungs

### R1 — Pure diffusion

- **Claim:** a gradient discharges itself by diffusion alone, and the
  spread grows like √t.
- **Build:** a 1D box. Inject heat at one end at t=0. Let it diffuse.
  Plot the field over time. Plot RMS spread vs time on log axes.
- **Pass:** the slope is 0.5 on log-log. Once the gradient flattens,
  nothing further happens.
- **Why:** anchors the meaning of "gradient" and "flow" before
  anything else. Establishes the boring baseline that every later
  rung is *not*.

### R2 — Driven diffusion (steady state)

- **Claim:** a continuous source at one end and sink at the other
  produces a *steady non-equilibrium state* — a gradient that does
  not flatten, because flow is being maintained.
- **Build:** add boundary conditions to R1. Hot side held hot, cold
  side held cold. Watch the field reach a linear profile and stay.
  Measure the throughput (energy in per tick = energy out per tick).
- **Pass:** field stops changing while flow continues. Inflow rate
  equals outflow rate. The gradient is *maintained* by the flow,
  not consumed.
- **Why:** the first appearance of "steady state ≠ equilibrium." The
  category that life lives in.

### R3 — Advection-diffusion

- **Claim:** add a wind, and the same gradient now drifts as well as
  spreads. The shape of "flow" depends on which mechanism dominates.
- **Build:** R2 plus a velocity field. Slider for wind speed. Slider
  for diffusivity.
- **Pass:** at high $D/v$ the field smooths; at low $D/v$ a pulse
  drifts mostly intact; in between it does both. Match the
  Péclet number's predictions qualitatively.
- **Why:** introduces "the medium itself moves" without yet asking
  for active transport. Sets up the contrast for R6.

### R4 — Reaction-diffusion (first pattern)

- **Claim:** a reactive substance plus diffusion plus a flow of
  fresh reactant produces *patterns*, not soup. A flow can hold a
  shape.
- **Build:** a 2D grid running Gray-Scott or similar. Source/sink
  for fuel and waste. Watch stripes / spots / spirals form.
- **Pass:** stable patterns form for some parameters; uniform soup
  forms for others. The boundary between regimes is visible.
- **Why:** the first time the universe does something *organised*
  with nothing but flow + chemistry + geometry. The floor of "life-
  like."

### R5 — Bénard-style emergent structure

- **Claim:** when a flow exceeds a threshold, the system spontaneously
  organises into a structure that dissipates the gradient *better*
  than the unorganised state. Below threshold, conduction. Above,
  convection.
- **Build:** a 2D fluid layer heated from below. Observe transition
  to convection cells as heating increases. Measure heat throughput
  before and after — it should jump.
- **Pass:** there is a clean threshold. The structured state really
  does dissipate more. Cells are visible.
- **Why:** patterns that exist *because* they are dissipating, that
  out-compete the boring state for gradient. The first hint of
  "selection pressure" without genes.

### R6 — Active transport

- **Claim:** a coupled-reaction "pump" can move stuff against a
  concentration gradient, by spending a different gradient. This is
  the move that pre-life chemistry cannot make and life always does.
- **Build:** two coupled species in a 1D box. Species A flows
  downhill. Species B is pumped uphill, but only where A is also
  flowing — the rates are linked. Energy comes from A's drop.
- **Pass:** B accumulates against its own gradient. Stop A's
  gradient and B's accumulation reverses by diffusion. The pump is
  a *use* of A's gradient, demonstrably.
- **Why:** the first machine. The first thing that "uses" a gradient
  for a purpose other than dissipating it.

### R7 — Spontaneous compartments

- **Claim:** under the right conditions, surfaces / membranes form
  *spontaneously* and persist as long as the flow conditions hold.
  Inside-vs-outside is not built; it falls out.
- **Build:** a phase-separation system (Cahn–Hilliard or similar)
  with a flow of "amphiphile" and "fuel." Watch droplets / bilayers
  form, persist, dissolve when fuel runs out.
- **Pass:** stable boundaries form under flow, dissolve without it.
  No compartment is hand-placed.
- **Why:** the boundary candidate from [life/README.md](README.md)
  becomes a result, not an assumption. Now we have *somewhere* for
  flows to be different inside vs outside.

### R8 — Compartment + active transport (proto-cell)

- **Claim:** a spontaneously-formed compartment can host an active
  pump. The pump builds an internal gradient. The compartment is
  now a tiny machine that *maintains a different state inside than
  outside* — the first thing that resembles a cell in shape.
- **Build:** combine R6 and R7. Pump operates only when wrapped in a
  membrane. Membrane only persists with fuel. The two co-depend.
- **Pass:** observe a state where compartment + pump together are
  more stable than either alone. Internal gradient measurable. Cut
  off external fuel → both dissolve together.
- **Why:** the first thing whose *structure is the cause of itself*
  in any meaningful sense. Pre-life on the slope toward life.

### R9 — Replication

- **Claim:** a compartment-with-pump can sometimes split, and each
  half can recover the pumped state. The pattern is now *reproducing*
  — not as a goal, as a side effect of growth + geometry.
- **Build:** add growth (a slow inflow of membrane material) to R8.
  When a compartment grows past a threshold, it splits. Each half
  must restore its internal gradient or die.
- **Pass:** at the right parameters, populations of compartments
  persist across many generations, with deaths and divisions
  balancing. At the wrong parameters, all die or all explode.
- **Why:** the first time the pattern outlives any of its instances.
  Identity becomes a property of the lineage, not the molecule.

### R10 — Variation and selection

- **Claim:** when replication is imperfect, lineages whose pumps
  dissipate the gradient better persist longer. Selection happens
  for free — no fitness function declared, just physics + variation
  + finite resources.
- **Build:** R9 with stochastic variation in pump efficiency at each
  split. Limited fuel. Track lineage survival.
- **Pass:** the average pump efficiency in the population rises over
  time. Slower pumps go extinct.
- **Why:** evolution as a *consequence* of the rungs below, not as a
  separate addition. We are now firmly in life territory.

### R11 — Sensing

- **Claim:** if a compartment can bias *which direction it grows*
  based on a local cue (e.g. higher fuel in one direction), its
  lineage dominates. Sensing emerges as a survival advantage on top
  of the existing physics.
- **Build:** R10 with a non-uniform fuel field. Allow some lineages
  to have a tiny "asymmetry" in growth that biases them up the
  gradient.
- **Pass:** sensing lineages out-compete blind ones in a non-uniform
  environment, are neutral in a uniform one.
- **Why:** the first time a pattern *reaches outward*. The ancestor
  of perception.

### R12 — Action

- **Claim:** a compartment that can *move itself* (eject mass,
  contract) toward higher fuel out-competes one that can only grow
  toward it. Motion is a more efficient way to ride the gradient.
- **Build:** R11 plus motility. Energetic cost for moving. Tradeoff
  between staying put and seeking.
- **Pass:** in environments with patchy fuel, moving lineages
  dominate. In uniform environments, sitting lineages dominate.
- **Why:** the first time the pattern *acts* on its world. The
  ancestor of behaviour.

### R13 — Internal model

- **Claim:** a compartment that *predicts* the gradient ahead of
  time — even slightly — beats one that only reacts. Prediction
  emerges as a metabolic shortcut.
- **Build:** R12 with a simple internal "expected gradient" state
  that updates from past experience. Compartments that act on the
  prediction (rather than the instantaneous reading) save energy.
- **Pass:** in temporally structured environments, predictors
  dominate. In random environments, they don't.
- **Why:** the first cognition. The simplest version of an inside
  modelling the outside.

### R14 — Embodied artificial mind (the destination)

- **Claim:** on real silicon, with real sensors and real actuators,
  a self-maintaining pattern of flows can be built that satisfies
  R1 through R13 in the physical world rather than in simulation.
- **Build:** the long project. Hardware, power, sensors, motors,
  the whole stack. Out of scope for `life/notes/`. The notes here
  are what justifies attempting it at all.
- **Pass:** the artefact persists, senses, acts, and predicts —
  and would fail to do any of those if its substrate were a
  function-call architecture instead of a continuous flow.
- **Why:** the destination from [ARC.md](ARC.md). The point of all
  the rungs below.

## Where we are on the ladder

R0. Reading. None of the above built yet.

When we start building, the first rung — R1, pure diffusion — will
live as a binary in the Rust workspace under `life/core/r01-diffusion/`,
driving the browser viewer under `life/viewer/` over the WebSocket
bridge, with the same core also compiling to wasm so the public site
can run it. The `adam/experiments/` single-HTML pattern was
clarifying for that project; this project is different work and gets
its own substrate.

## What this ladder buys us

Each rung is small enough to build in a sitting, watch, and be honest
about. Each rung's failure mode is interesting — if a rung doesn't
behave the way the rung-below predicted it would, we have learned
something about life that no amount of LLM tinkering could have told
us.

The whole thing is a falsifiable progression. If R5 doesn't show a
clean threshold, the "dissipative structure" framing weakens. If R10
doesn't show selection, "evolution falls out of physics" weakens. If
R13 doesn't show prediction's advantage, the framing of cognition as
metabolic shortcut weakens.

We are not committing to the conclusions. We are committing to the
discipline of climbing one rung at a time and seeing what holds.
