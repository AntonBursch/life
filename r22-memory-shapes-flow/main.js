// R22 -- Memory shapes flow. Barkley (R7) +
// integrate_field (R19) + gradient_field (R17) + advect_by (R17).
// No new operator. Phase-B composition.
import init, { WasmCoupledR22 } from "../viewer/pkg/flow_wasm.js";
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
const DYE_SIGMA = 12.0;
const DYE_AMP = 1.0;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:   $("uCanvas"),
    memCanvas: $("memCanvas"),
    dyeCanvas: $("dyeCanvas"),
    play:       $("play"),
    reseedWave: $("reseedWave"),
    resetMem:   $("resetMem"),
    reseedDye:  $("reseedDye"),
    resetDye:   $("resetDye"),
    kick:       $("kick"),
    leak:  $("leak"),  leakV:  $("leakV"),
    alpha: $("alpha"), alphaV: $("alphaV"),
    eps:   $("eps"),   epsV:   $("epsV"),
    speed: $("speed"), speedV: $("speedV"),
    rT: $("rT"), rTau: $("rTau"), rAlpha: $("rAlpha"),
    rMemMx: $("rMemMx"), rVmx: $("rVmx"),
    rDyeT: $("rDyeT"), rCx: $("rCx"), rCy: $("rCy"),
    errSlot: $("errSlot"),
};

const uCtx   = els.uCanvas.getContext("2d");
const memCtx = els.memCanvas.getContext("2d");
const dyeCtx = els.dyeCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let leakVal = parseFloat(els.leak.value);
let alphaVal = parseFloat(els.alpha.value);
let epsVal = parseFloat(els.eps.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function dropDefaultDye() {
    if (!sim) return;
    sim.seed_dye_blob((W * 3) / 4, H / 2, DYE_SIGMA, DYE_AMP);
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR22(
            W, H, DIFFUSION, A_VAL, B_VAL, epsVal, DX, DT, leakVal, alphaVal,
        );
        sim.seed_spiral();
        dropDefaultDye();
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
        const f3 = fitCanvas(els.dyeCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        const memMax = Math.max(0.5, sim.memory_max() * 1.05);
        drawField2D(memCtx, f2.width, f2.height, sim.memory_field(), w, h, memMax);
        const dyeMax = Math.max(0.1, sim.dye_max() * 1.05);
        drawField2D(dyeCtx, f3.width, f3.height, sim.dye_field(), w, h, dyeMax);

        const tau = leakVal > 0 ? 1.0 / leakVal : Infinity;
        els.rT.textContent     = sim.tissue_time.toFixed(1);
        els.rTau.textContent   = isFinite(tau) ? tau.toFixed(2) : "∞";
        els.rAlpha.textContent = alphaVal.toFixed(2);
        els.rMemMx.textContent = sim.memory_max().toFixed(3);
        els.rVmx.textContent   = sim.velocity_max().toFixed(3);
        els.rDyeT.textContent  = sim.dye_total().toFixed(1);
        els.rCx.textContent    = sim.dye_centroid_x().toFixed(2);
        els.rCy.textContent    = sim.dye_centroid_y().toFixed(2);
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
els.reseedDye.addEventListener("click", dropDefaultDye);
els.resetDye.addEventListener("click", () => sim?.reset_dye());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    sim?.set_leak(leakVal);
});
els.alpha.addEventListener("input", () => {
    alphaVal = parseFloat(els.alpha.value);
    els.alphaV.textContent = alphaVal.toFixed(1);
    sim?.set_alpha(alphaVal);
});
els.eps.addEventListener("input", () => {
    epsVal = parseFloat(els.eps.value);
    els.epsV.textContent = epsVal.toFixed(3);
    sim?.set_eps(epsVal);
});
els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

// click on the dye canvas to drop a dye blob at that location
els.dyeCanvas.addEventListener("click", (e) => {
    if (!sim) return;
    const rect = els.dyeCanvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const y = ((e.clientY - rect.top) / rect.height) * H;
    sim.seed_dye_blob(x, y, DYE_SIGMA, DYE_AMP);
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
