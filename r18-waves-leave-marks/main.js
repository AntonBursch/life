// R18 -- Waves leave marks. Barkley spirals (R7) + threshold_event
// (new) -> per-cell event mask -> running counter + decaying trace.
import init, { WasmCoupledR18 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DTemperature,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;

const DIFFUSION = 1.0;
const A_VAL = 0.75;
const B_VAL = 0.06;
const EPS_DEFAULT = 0.02;
const DX = 1.0;
const DT = 0.05;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:     $("uCanvas"),
    evtCanvas:   $("evtCanvas"),
    traceCanvas: $("traceCanvas"),
    cntCanvas:   $("cntCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    resetMarks:  $("resetMarks"),
    kick:        $("kick"),
    thresh: $("thresh"), threshV: $("threshV"),
    tau:    $("tau"),    tauV:    $("tauV"),
    eps:    $("eps"),    epsV:    $("epsV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rT: $("rT"), rEvt: $("rEvt"), rCum: $("rCum"),
    rCov: $("rCov"), rMax: $("rMax"), rRate: $("rRate"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const uCtx     = els.uCanvas.getContext("2d");
const evtCtx   = els.evtCanvas.getContext("2d");
const traceCtx = els.traceCanvas.getContext("2d");
const cntCtx   = els.cntCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let threshold = parseFloat(els.thresh.value);
let tau = parseFloat(els.tau.value);
let epsVal = parseFloat(els.eps.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR18(
            W, H, DIFFUSION, A_VAL, B_VAL, epsVal, DX, DT, threshold,
        );
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(coverage, rate) {
    if (rate < 1e-4) return "cold (no firings)";
    if (coverage < 0.3) return "spiral seeding";
    if (coverage < 0.9) return "spiral sweeping";
    return "stable raster";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.evtCanvas);
        const f3 = fitCanvas(els.traceCanvas);
        const f4 = fitCanvas(els.cntCanvas);

        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        drawField2D(evtCtx, f2.width, f2.height, sim.events_field(), w, h, 1.0);
        drawField2D(traceCtx, f3.width, f3.height, sim.trace_field(tau), w, h, 1.0);

        const counts = sim.counts_field();
        const maxC = Math.max(1, sim.max_count());
        drawField2D(cntCtx, f4.width, f4.height, counts, w, h, maxC);

        els.rT.textContent = sim.tissue_time.toFixed(1);
        els.rEvt.textContent = String(sim.events_last_step);
        els.rCum.textContent = String(Math.round(sim.cumulative_event_count));
        const cov = sim.coverage();
        const rate = sim.mean_rate();
        els.rCov.textContent = (100 * cov).toFixed(0) + "%";
        els.rMax.textContent = String(sim.max_count());
        els.rRate.textContent = rate.toFixed(3);
        els.rRegime.textContent = regimeLabel(cov, rate);
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reseed.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_tissue();
    sim.seed_spiral();
});
els.resetMarks.addEventListener("click", () => sim?.reset_marks());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.thresh.addEventListener("input", () => {
    threshold = parseFloat(els.thresh.value);
    els.threshV.textContent = threshold.toFixed(2);
    if (sim) sim.set_threshold(threshold);
});
els.tau.addEventListener("input", () => {
    tau = parseFloat(els.tau.value);
    els.tauV.textContent = tau.toFixed(1);
});
els.eps.addEventListener("input", () => {
    epsVal = parseFloat(els.eps.value);
    els.epsV.textContent = epsVal.toFixed(3);
    if (sim) sim.set_eps(epsVal);
});
els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

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
