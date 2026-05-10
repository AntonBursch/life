// R1 — Pure 1D diffusion. Drives the wasm sim from the page.
import init, { WasmDiffusion1D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField1D } from "../viewer/canvas.js";

const N = 401; // grid cells
const DX = 1.0;
const DT = 0.5;
const VMAX = 1.5; // colour-scale ceiling

const $ = (id) => document.getElementById(id);
const els = {
    canvas: $("field"),
    play: $("play"),
    reset: $("reset"),
    step: $("step"),
    speed: $("speed"),
    speedV: $("speedV"),
    diff: $("diff"),
    diffV: $("diffV"),
    rT: $("rT"),
    rS: $("rS"),
    rTot: $("rTot"),
    rSig: $("rSig"),
    rSigPred: $("rSigPred"),
    errSlot: $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let D = parseFloat(els.diff.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function rebuildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmDiffusion1D(N, D, DX, DT);
        sim.seed_centre_pulse();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const phi = sim.phi();
        drawField1D(ctx, width, height, phi, VMAX);

        const t = sim.time;
        const sigma = sim.rms_spread;
        const sigmaPred = Math.sqrt(2 * D * t);

        els.rT.textContent = t.toFixed(2);
        els.rS.textContent = sim.len; // a stand-in; we track total time instead of step count for now
        els.rTot.textContent = sim.total.toFixed(6);
        els.rSig.textContent = sigma.toFixed(3);
        els.rSigPred.textContent = sigmaPred.toFixed(3);
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
        sim.seed_centre_pulse();
    }
});

els.step.addEventListener("click", () => {
    if (sim) {
        sim.step();
    }
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.diff.addEventListener("input", () => {
    D = parseFloat(els.diff.value);
    els.diffV.textContent = D.toFixed(2);
    rebuildSim();
});

// --- boot -----------------------------------------------------------------

init()
    .then(() => {
        rebuildSim();
        requestAnimationFrame(frame);
    })
    .catch((e) => {
        showError(
            `could not load wasm bundle: ${e.message ?? e}. ` +
                `did you run scripts/build-wasm.ps1 ?`,
        );
    });
