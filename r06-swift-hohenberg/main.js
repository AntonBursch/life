// R6 — Swift–Hohenberg. One scalar field, one PDE, one bifurcation knob.
import init, { WasmSwiftHohenberg2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2DDiverging } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 1.0;
const DT = 0.02;
const SEED_AMP = 0.05;

const PRESETS = {
    below:     { r: -0.20 },
    onset:     { r:  0.02 },
    stripes:   { r:  0.10 },
    labyrinth: { r:  0.30 },
    strong:    { r:  0.60 },
};

const $ = (id) => document.getElementById(id);
const els = {
    canvas:   $("field"),
    play:     $("play"),
    reset:    $("reset"),
    speed:    $("speed"),
    speedV:   $("speedV"),
    rval:     $("rval"),
    rvalV:    $("rvalV"),
    rT:       $("rT"),
    rR:       $("rR"),
    rVar:     $("rVar"),
    rMax:     $("rMax"),
    rRegime:  $("rRegime"),
    errSlot:  $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let rValue = parseFloat(els.rval.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmSwiftHohenberg2D(W, H, rValue, DX, DT);
        sim.seed_noise(SEED_AMP);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(r, maxAbs, variance) {
    if (r < -0.005) return "subcritical (decaying)";
    if (variance < 1e-4) return "uniform";
    if (maxAbs < 0.35) return "near onset";
    if (maxAbs < 0.75) return "stripes / labyrinth";
    return "saturated pattern";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const u = sim.u_field();
        // Symmetric colormap: scale by current max_abs so contrast is always good.
        const vmax = Math.max(sim.max_abs, 0.02);
        drawField2DDiverging(ctx, width, height, u, sim.width, sim.height, vmax);

        els.rT.textContent = sim.time.toFixed(1);
        els.rR.textContent = sim.r.toFixed(3);
        els.rVar.textContent = sim.variance.toExponential(2);
        els.rMax.textContent = sim.max_abs.toFixed(3);
        els.rRegime.textContent = regimeLabel(sim.r, sim.max_abs, sim.variance);
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
    if (sim) {
        sim.reset();
        sim.seed_noise(SEED_AMP);
    }
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.rval.addEventListener("input", () => {
    rValue = parseFloat(els.rval.value);
    els.rvalV.textContent = rValue.toFixed(3);
    if (sim) sim.set_r(rValue);
});

document.querySelectorAll("button[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
        const p = PRESETS[btn.dataset.preset];
        if (!p) return;
        rValue = p.r;
        els.rval.value = String(rValue);
        els.rvalV.textContent = rValue.toFixed(3);
        if (sim) {
            sim.set_r(rValue);
            sim.reset();
            sim.seed_noise(SEED_AMP);
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
