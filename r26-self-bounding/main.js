// R26 -- Self-bounding. Sharp bulk_gate turns memory into a binary
// wall mask; the wave carves and inhabits its own domain.
import init, { WasmCoupledR26 } from "../viewer/pkg/flow_wasm.js";
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
    uCanvas:    $("uCanvas"),
    memCanvas:  $("memCanvas"),
    wallCanvas: $("wallCanvas"),
    play:       $("play"),
    reseed:     $("reseed"),
    healWalls:  $("healWalls"),
    kick:       $("kick"),
    threshold: $("threshold"), thresholdV: $("thresholdV"),
    sharpness: $("sharpness"), sharpnessV: $("sharpnessV"),
    killEps:   $("killEps"),   killEpsV:   $("killEpsV"),
    leak:      $("leak"),      leakV:      $("leakV"),
    speed:     $("speed"),     speedV:     $("speedV"),
    rT: $("rT"), rEx: $("rEx"), rUm: $("rUm"), rWf: $("rWf"),
    rMm: $("rMm"), rMx: $("rMx"), rEm: $("rEm"), rTau: $("rTau"),
    errSlot: $("errSlot"),
};

const uCtx    = els.uCanvas.getContext("2d");
const memCtx  = els.memCanvas.getContext("2d");
const wallCtx = els.wallCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let thresholdVal = parseFloat(els.threshold.value);
let sharpnessVal = parseFloat(els.sharpness.value);
let killEpsVal   = parseFloat(els.killEps.value);
let leakVal      = parseFloat(els.leak.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR26(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            killEpsVal, thresholdVal, sharpnessVal, leakVal,
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
        const f3 = fitCanvas(els.wallCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        const memMx = Math.max(0.4, sim.memory_max() * 1.05);
        drawField2D(memCtx, f2.width, f2.height, sim.memory_field(), w, h, memMx);
        // wall mask: binary {0,1}
        drawField2D(wallCtx, f3.width, f3.height, sim.wall_mask(), w, h, 1.0);

        const tau = leakVal > 0 ? 1.0 / leakVal : Infinity;
        els.rT.textContent   = sim.tissue_time.toFixed(1);
        els.rEx.textContent  = sim.excited_fraction().toFixed(3);
        els.rUm.textContent  = sim.u_mean().toFixed(3);
        els.rWf.textContent  = (100 * sim.wall_fraction()).toFixed(1) + "%";
        els.rMm.textContent  = sim.memory_mean().toFixed(3);
        els.rMx.textContent  = sim.memory_max().toFixed(3);
        els.rEm.textContent  = sim.eps_mean().toFixed(4);
        els.rTau.textContent = isFinite(tau) ? tau.toFixed(2) : "∞";
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
els.healWalls.addEventListener("click", () => sim?.reset_memory());
els.kick.addEventListener("click", () => sim?.kick(W >> 1, H >> 1, 6, 1.0));

els.threshold.addEventListener("input", () => {
    thresholdVal = parseFloat(els.threshold.value);
    els.thresholdV.textContent = thresholdVal.toFixed(2);
    sim?.set_threshold(thresholdVal);
});
els.sharpness.addEventListener("input", () => {
    sharpnessVal = parseFloat(els.sharpness.value);
    els.sharpnessV.textContent = sharpnessVal.toFixed(2);
    sim?.set_sharpness(sharpnessVal);
});
els.killEps.addEventListener("input", () => {
    killEpsVal = parseFloat(els.killEps.value);
    els.killEpsV.textContent = killEpsVal.toFixed(3);
    sim?.set_kill_eps(killEpsVal);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    sim?.set_leak(leakVal);
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
