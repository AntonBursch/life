// R19 -- Memory of waves. Barkley (R7) + integrate_field (new)
// produces a leaky integral (low-pass) and a pure-accumulator
// dose meter on the same substrate. Phase A closer.
import init, { WasmCoupledR19 } from "../viewer/pkg/flow_wasm.js";
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
    uCanvas:    $("uCanvas"),
    avgCanvas:  $("avgCanvas"),
    doseCanvas: $("doseCanvas"),
    play:       $("play"),
    reseed:     $("reseed"),
    resetMem:   $("resetMem"),
    kick:       $("kick"),
    leak:  $("leak"),  leakV:  $("leakV"),
    eps:   $("eps"),   epsV:   $("epsV"),
    speed: $("speed"), speedV: $("speedV"),
    rT: $("rT"), rUm: $("rUm"), rTau: $("rTau"),
    rYp: $("rYp"), rYo: $("rYo"), rDmax: $("rDmax"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const uCtx    = els.uCanvas.getContext("2d");
const avgCtx  = els.avgCanvas.getContext("2d");
const doseCtx = els.doseCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let leakVal = parseFloat(els.leak.value);
let epsVal  = parseFloat(els.eps.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR19(
            W, H, DIFFUSION, A_VAL, B_VAL, epsVal, DX, DT, leakVal,
        );
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(t, tau, yPred, yObs) {
    if (t < tau) return "warming (t < τ)";
    const ratio = yPred > 0 ? yObs / yPred : 0;
    if (ratio < 0.7) return "approaching steady state";
    if (ratio < 1.3) return "near steady state";
    return "above prediction";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.avgCanvas);
        const f3 = fitCanvas(els.doseCanvas);

        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);

        const avgMax = Math.max(0.1, sim.avg_max() * 1.05);
        drawField2D(avgCtx, f2.width, f2.height, sim.avg_field(), w, h, avgMax);

        const doseMax = Math.max(1.0, sim.dose_max() * 1.05);
        drawField2D(doseCtx, f3.width, f3.height, sim.dose_field(), w, h, doseMax);

        const uM = sim.u_mean();
        const tau = leakVal > 0 ? 1.0 / leakVal : Infinity;
        const yPred = sim.avg_predicted();
        const yObs  = sim.avg_mean();
        const t = sim.tissue_time;

        els.rT.textContent = t.toFixed(1);
        els.rUm.textContent = uM.toFixed(3);
        els.rTau.textContent = isFinite(tau) ? tau.toFixed(2) : "∞";
        els.rYp.textContent = isFinite(yPred) ? yPred.toFixed(3) : "∞";
        els.rYo.textContent = yObs.toFixed(3);
        els.rDmax.textContent = sim.dose_max().toFixed(2);
        els.rRegime.textContent = regimeLabel(t, tau, yPred, yObs);
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
els.resetMem.addEventListener("click", () => sim?.reset_memory());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    if (sim) sim.set_leak(leakVal);
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
