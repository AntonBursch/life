// R8 — Cahn–Hilliard phase separation. One conserved scalar field c.
import init, { WasmCahnHilliard2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2DDiverging } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 1.0;
const DT = 0.02;
const MOBILITY = 1.0;
const SEED_AMP = 0.05;

const $ = (id) => document.getElementById(id);
const els = {
    canvas:  $("field"),
    play:    $("play"),
    reseed:  $("reseed"),
    biased:  $("biased"),
    speed:   $("speed"),
    speedV:  $("speedV"),
    kval:    $("kval"),
    kvalV:   $("kvalV"),
    rT:      $("rT"),
    rMean:   $("rMean"),
    rDrift:  $("rDrift"),
    rVar:    $("rVar"),
    rBulk:   $("rBulk"),
    rF:      $("rF"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kappa = parseFloat(els.kval.value);
let seedCounter = 1;
let initialMean = 0;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCahnHilliard2D(W, H, MOBILITY, kappa, DX, DT);
        sim.seed_noise(SEED_AMP, 0.0, seedCounter);
        initialMean = sim.mean_c;
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function reseed(mean) {
    if (!sim) return;
    seedCounter = (seedCounter + 1) | 0;
    sim.seed_noise(SEED_AMP, mean, seedCounter);
    initialMean = sim.mean_c;
}

function regimeLabel(t, varC, bulk) {
    if (t < 1) return "smooth";
    if (varC < 0.05) return "spinodal grain";
    if (bulk < 0.55) return "early labyrinth";
    if (bulk < 0.75) return "coarsening";
    return "late coarsening";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const c = sim.c_field();
        // Diverging: c in roughly [-1, +1]. Saturate at 1.
        drawField2DDiverging(ctx, width, height, c, sim.width, sim.height, 1.0);

        els.rT.textContent = sim.time.toFixed(1);
        els.rMean.textContent = sim.mean_c.toExponential(2);
        els.rDrift.textContent = (sim.mean_c - initialMean).toExponential(2);
        els.rVar.textContent = sim.variance_c.toFixed(3);
        els.rBulk.textContent = sim.bulk_fraction.toFixed(3);
        els.rF.textContent = sim.free_energy.toFixed(4);
        els.rRegime.textContent = regimeLabel(
            sim.time,
            sim.variance_c,
            sim.bulk_fraction,
        );
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

els.reseed.addEventListener("click", () => reseed(0.0));
els.biased.addEventListener("click", () => reseed(0.3));

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.kval.addEventListener("input", () => {
    kappa = parseFloat(els.kval.value);
    els.kvalV.textContent = kappa.toFixed(2);
    if (sim) sim.set_kappa(kappa);
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
