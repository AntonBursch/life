// R3 — Advection-diffusion. R2 plus a wind.
import init, { WasmAdvectionDiffusion1D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField1D } from "../viewer/canvas.js";

const N = 201;
const DX = 1.0;
const DT = 0.5;
const VMAX = 1.05;

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
    vel: $("vel"),
    velV: $("velV"),
    left: $("left"),
    leftV: $("leftV"),
    right: $("right"),
    rightV: $("rightV"),
    rT: $("rT"),
    rPe: $("rPe"),
    rRegime: $("rRegime"),
    rJL: $("rJL"),
    rJR: $("rJR"),
    rJRatio: $("rJRatio"),
    errSlot: $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let D = parseFloat(els.diff.value);
let v = parseFloat(els.vel.value);
let leftVal = parseFloat(els.left.value);
let rightVal = parseFloat(els.right.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function rebuildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmAdvectionDiffusion1D(N, D, v, DX, DT, leftVal, rightVal);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(pe) {
    const a = Math.abs(pe);
    if (a < 1) return "diffusion-dominated";
    if (a < 10) return "mixed";
    return "advection-dominated";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const phi = sim.phi();
        drawField1D(ctx, width, height, phi, VMAX);

        const pe = sim.peclet;
        const jl = sim.flux_left;
        const jr = sim.flux_right;

        els.rT.textContent = sim.time.toFixed(2);
        els.rPe.textContent = pe.toFixed(2);
        els.rRegime.textContent = regimeLabel(pe);
        els.rJL.textContent = jl.toFixed(6);
        els.rJR.textContent = jr.toFixed(6);
        els.rJRatio.textContent =
            Math.abs(jl) < 1e-9 ? "—" : (jr / jl).toFixed(4);
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
    if (sim) sim.reset_interior();
});

els.step.addEventListener("click", () => {
    if (sim) sim.step();
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.diff.addEventListener("input", () => {
    D = parseFloat(els.diff.value);
    els.diffV.textContent = D.toFixed(3);
    rebuildSim();
});

els.vel.addEventListener("input", () => {
    v = parseFloat(els.vel.value);
    els.velV.textContent = v.toFixed(3);
    rebuildSim();
});

els.left.addEventListener("input", () => {
    leftVal = parseFloat(els.left.value);
    els.leftV.textContent = leftVal.toFixed(2);
    rebuildSim();
});

els.right.addEventListener("input", () => {
    rightVal = parseFloat(els.right.value);
    els.rightV.textContent = rightVal.toFixed(2);
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
