// R24 -- Scar tissue. First closed-loop rung. Barkley reads a
// per-cell eps field built from a leaky integral of its own
// activity. New operator: modulate_parameter (parametrise).
import init, { WasmCoupledR24 } from "../viewer/pkg/flow_wasm.js";
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
const BASE_EPS = 0.02;
const DX = 1.0;
const DT = 0.05;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:   $("uCanvas"),
    memCanvas: $("memCanvas"),
    epsCanvas: $("epsCanvas"),
    play:       $("play"),
    reseedWave: $("reseedWave"),
    resetMem:   $("resetMem"),
    kick:       $("kick"),
    gain:   $("gain"),   gainV:   $("gainV"),
    leak:   $("leak"),   leakV:   $("leakV"),
    epsMax: $("epsMax"), epsMaxV: $("epsMaxV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rT: $("rT"), rEx: $("rEx"), rUm: $("rUm"), rTau: $("rTau"),
    rMm: $("rMm"), rMx: $("rMx"),
    rEm: $("rEm"), rEmx: $("rEmx"), rSc: $("rSc"),
    errSlot: $("errSlot"),
};

const uCtx   = els.uCanvas.getContext("2d");
const memCtx = els.memCanvas.getContext("2d");
const epsCtx = els.epsCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let gainVal = parseFloat(els.gain.value);
let leakVal = parseFloat(els.leak.value);
let epsMaxVal = parseFloat(els.epsMax.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR24(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            leakVal, gainVal, epsMaxVal,
        );
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.memCanvas);
        const f3 = fitCanvas(els.epsCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        const memMx = Math.max(0.5, sim.memory_max() * 1.05);
        drawField2D(memCtx, f2.width, f2.height, sim.memory_field(), w, h, memMx);
        // eps panel: normalise across [base, ceiling] so even small lifts read.
        const epsArr = sim.eps_field();
        const span = Math.max(1e-9, epsMaxVal - BASE_EPS);
        const display = new Float64Array(epsArr.length);
        for (let i = 0; i < epsArr.length; i++) {
            display[i] = Math.max(0, Math.min(1, (epsArr[i] - BASE_EPS) / span));
        }
        drawField2D(epsCtx, f3.width, f3.height, display, w, h, 1.0);

        const tau = leakVal > 0 ? 1.0 / leakVal : Infinity;
        els.rT.textContent   = sim.tissue_time.toFixed(1);
        els.rEx.textContent  = sim.excited_fraction().toFixed(3);
        els.rUm.textContent  = sim.u_mean().toFixed(3);
        els.rTau.textContent = isFinite(tau) ? tau.toFixed(2) : "∞";
        els.rMm.textContent  = sim.memory_mean().toFixed(3);
        els.rMx.textContent  = sim.memory_max().toFixed(3);
        els.rEm.textContent  = sim.eps_mean().toFixed(4);
        els.rEmx.textContent = sim.eps_max_current().toFixed(4);
        els.rSc.textContent  = (100 * sim.scar_fraction()).toFixed(1) + "%";
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reseedWave.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_tissue();
    sim.seed_spiral();
});
els.resetMem.addEventListener("click", () => sim?.reset_memory());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.gain.addEventListener("input", () => {
    gainVal = parseFloat(els.gain.value);
    els.gainV.textContent = gainVal.toFixed(3);
    sim?.set_gain(gainVal);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    sim?.set_leak(leakVal);
});
els.epsMax.addEventListener("input", () => {
    epsMaxVal = parseFloat(els.epsMax.value);
    els.epsMaxV.textContent = epsMaxVal.toFixed(2);
    sim?.set_eps_max(epsMaxVal);
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
