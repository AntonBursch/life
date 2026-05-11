// R23 -- Excitable courier. Barkley (R7) + gradient_field (R17) +
// advect_by (R17). No memory, no new operator. Phase-B composition.
import init, { WasmCoupledR23 } from "../viewer/pkg/flow_wasm.js";
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
const PAY_SIGMA = 10.0;
const PAY_AMP = 1.0;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:   $("uCanvas"),
    payCanvas: $("payCanvas"),
    play:         $("play"),
    reseedWave:   $("reseedWave"),
    dropPayload:  $("dropPayload"),
    resetPayload: $("resetPayload"),
    kick:         $("kick"),
    alpha: $("alpha"), alphaV: $("alphaV"),
    eps:   $("eps"),   epsV:   $("epsV"),
    speed: $("speed"), speedV: $("speedV"),
    rT: $("rT"), rAlpha: $("rAlpha"), rGx: $("rGx"),
    rPT: $("rPT"), rCx: $("rCx"), rCy: $("rCy"), rDrift: $("rDrift"),
    errSlot: $("errSlot"),
};

const uCtx   = els.uCanvas.getContext("2d");
const payCtx = els.payCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let alphaVal = parseFloat(els.alpha.value);
let epsVal = parseFloat(els.eps.value);
let initCx = (W / 4);
let initCy = (H / 4);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function dropDefaultPayload() {
    if (!sim) return;
    sim.seed_payload_blob(W / 4, H / 4, PAY_SIGMA, PAY_AMP);
    initCx = W / 4;
    initCy = H / 4;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR23(
            W, H, DIFFUSION, A_VAL, B_VAL, epsVal, DX, DT, alphaVal,
        );
        sim.seed_spiral();
        dropDefaultPayload();
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
        const f2 = fitCanvas(els.payCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        const pMax = Math.max(0.1, sim.payload_max() * 1.05);
        drawField2D(payCtx, f2.width, f2.height, sim.payload_field(), w, h, pMax);

        const cx = sim.payload_centroid_x();
        const cy = sim.payload_centroid_y();
        const drift = Math.sqrt((cx - initCx) ** 2 + (cy - initCy) ** 2);
        els.rT.textContent     = sim.tissue_time.toFixed(1);
        els.rAlpha.textContent = alphaVal.toFixed(2);
        els.rGx.textContent    = sim.grad_max().toFixed(3);
        els.rPT.textContent    = sim.payload_total().toFixed(1);
        els.rCx.textContent    = cx.toFixed(2);
        els.rCy.textContent    = cy.toFixed(2);
        els.rDrift.textContent = drift.toFixed(2);
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
els.dropPayload.addEventListener("click", dropDefaultPayload);
els.resetPayload.addEventListener("click", () => sim?.reset_payload());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
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

// click on payload canvas to drop a new blob
els.payCanvas.addEventListener("click", (e) => {
    if (!sim) return;
    const rect = els.payCanvas.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * W;
    const y = ((e.clientY - rect.top) / rect.height) * H;
    sim.seed_payload_blob(x, y, PAY_SIGMA, PAY_AMP);
    initCx = x;
    initCy = y;
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
