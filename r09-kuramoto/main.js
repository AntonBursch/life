// R9 — Kuramoto. One phase per cell. Local 4-neighbour coupling.
import init, { WasmKuramoto2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2DPhase } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DT = 0.05;

const $ = (id) => document.getElementById(id);
const els = {
    canvas:  $("field"),
    play:    $("play"),
    reseed:  $("reseed"),
    newpop:  $("newpop"),
    kval:    $("kval"),
    kvalV:   $("kvalV"),
    sval:    $("sval"),
    svalV:   $("svalV"),
    speed:   $("speed"),
    speedV:  $("speedV"),
    rT:      $("rT"),
    rK:      $("rK"),
    rS:      $("rS"),
    rR:      $("rR"),
    rPsi:    $("rPsi"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let coupling = parseFloat(els.kval.value);
let sigma = parseFloat(els.sval.value);
let phaseSeed = 1;
let popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmKuramoto2D(W, H, coupling, DT);
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(r) {
    if (r < 0.2) return "incoherent";
    if (r < 0.6) return "partially synced";
    if (r < 0.9) return "locked (with defects)";
    return "fully locked";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const theta = sim.theta_field();
        drawField2DPhase(ctx, width, height, theta, sim.width, sim.height);

        const r = sim.order_parameter;
        els.rT.textContent   = sim.time.toFixed(1);
        els.rK.textContent   = sim.coupling.toFixed(2);
        els.rS.textContent   = sim.natural_freq_stddev.toFixed(3);
        els.rR.textContent   = r.toFixed(3);
        els.rPsi.textContent = sim.mean_phase.toFixed(2);
        els.rRegime.textContent = regimeLabel(r);
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

els.reseed.addEventListener("click", () => {
    if (!sim) return;
    phaseSeed = (phaseSeed + 1) | 0;
    sim.randomise_phases(phaseSeed);
    sim.reset_time();
});

els.newpop.addEventListener("click", () => {
    if (!sim) return;
    popSeed = (popSeed + 1) | 0;
    sim.set_natural_frequencies(sigma, popSeed);
    phaseSeed = (phaseSeed + 1) | 0;
    sim.randomise_phases(phaseSeed);
    sim.reset_time();
});

els.kval.addEventListener("input", () => {
    coupling = parseFloat(els.kval.value);
    els.kvalV.textContent = coupling.toFixed(2);
    if (sim) sim.set_coupling(coupling);
});

els.sval.addEventListener("input", () => {
    sigma = parseFloat(els.sval.value);
    els.svalV.textContent = sigma.toFixed(2);
    if (sim) sim.set_natural_frequencies(sigma, popSeed);
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
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
