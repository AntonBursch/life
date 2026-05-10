// R5 — Bénard thermal convection. Streamfunction-vorticity, hot below.
import init, { WasmConvection2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2DTemperature } from "../viewer/canvas.js";

const W = 128;
const H = 32;
const KAPPA = 0.1;
const NU = 0.1;
const DX = 1.0;
const DT = 0.05;

// Heating presets in the "g" parameter. Critical onset for this geometry
// sits in the neighbourhood of g ≈ 0.01.
const PRESETS = {
    off:    { g: 0.000 },
    below:  { g: 0.005 },
    onset:  { g: 0.012 },
    cells:  { g: 0.025 },
    rolling:{ g: 0.060 },
};

const $ = (id) => document.getElementById(id);
const els = {
    canvas:   $("field"),
    play:     $("play"),
    reset:    $("reset"),
    speed:    $("speed"),
    speedV:   $("speedV"),
    gravity:  $("gravity"),
    gravityV: $("gravityV"),
    rT:       $("rT"),
    rNu:      $("rNu"),
    rEnergy:  $("rEnergy"),
    rPsi:     $("rPsi"),
    rRegime:  $("rRegime"),
    errSlot:  $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let gravity = parseFloat(els.gravity.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmConvection2D(W, H, KAPPA, NU, gravity, DX, DT);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(nu, energy) {
    if (nu < 1.02 && energy < 1e-5) return "conduction";
    if (nu < 1.1) return "perturbed";
    if (nu < 2.0) return "onset";
    if (nu < 4.0) return "convection cells";
    return "rolling boil";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const t = sim.temperature_field();
        drawField2DTemperature(ctx, width, height, t, sim.width, sim.height);

        els.rT.textContent = sim.time.toFixed(1);
        const nu = sim.nusselt;
        const energy = sim.mean_sq_vorticity;
        const psi = sim.max_abs_psi;
        els.rNu.textContent = nu.toFixed(3);
        els.rEnergy.textContent = energy.toExponential(2);
        els.rPsi.textContent = psi.toFixed(3);
        els.rRegime.textContent = regimeLabel(nu, energy);
    } else {
        ctx.clearRect(0, 0, width, height);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});

els.reset.addEventListener("click", () => {
    if (sim) sim.reset();
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.gravity.addEventListener("input", () => {
    gravity = parseFloat(els.gravity.value);
    els.gravityV.textContent = gravity.toFixed(4);
    if (sim) sim.set_gravity(gravity);
});

document.querySelectorAll("button[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
        const p = PRESETS[btn.dataset.preset];
        if (!p) return;
        gravity = p.g;
        els.gravity.value = String(gravity);
        els.gravityV.textContent = gravity.toFixed(4);
        if (sim) {
            sim.reset();
            sim.set_gravity(gravity);
        }
    });
});

// --- boot -----------------------------------------------------------------

init()
    .then(() => {
        buildSim();
        requestAnimationFrame(frame);
    })
    .catch((e) => {
        showError(
            `could not load wasm bundle: ${e.message ?? e}. ` +
                `did you run scripts/build-wasm.ps1 ?`,
        );
    });
