// R31 -- Autocatalytic wall. The membrane self-assembles from a
// single kick. See ../THESIS.md.
import init, { WasmCoupledR31 } from "../viewer/pkg/flow_wasm.js";
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
const DT = 0.02;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas: $("uCanvas"),
    xCanvas: $("xCanvas"),
    bCanvas: $("bCanvas"),
    play:       $("play"),
    resetChem:  $("resetChem"),
    resetWave:  $("resetWave"),
    seedSpiral: $("seedSpiral"),
    lambda: $("lambda"), lambdaV: $("lambdaV"),
    vel:    $("vel"),    velV:    $("velV"),
    drive:  $("drive"),  driveV:  $("driveV"),
    supply: $("supply"), supplyV: $("supplyV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rT: $("rT"), rEx: $("rEx"), rXm: $("rXm"),
    rRing: $("rRing"), rCore: $("rCore"),
    rBm: $("rBm"),
    errSlot: $("errSlot"),
};

const uCtx = els.uCanvas.getContext("2d");
const xCtx = els.xCanvas.getContext("2d");
const bCtx = els.bCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let lambdaVal = parseFloat(els.lambda.value);
let velVal    = parseFloat(els.vel.value);
let driveVal  = parseFloat(els.drive.value);
let supplyVal = parseFloat(els.supply.value);
const uThrVal = 0.5;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR31(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            uThrVal, driveVal, velVal, supplyVal, lambdaVal,
        );
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

const X_VIS_MIN = 1.0;
const X_VIS_MAX = 3.0;

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.xCanvas);
        const f3 = fitCanvas(els.bCanvas);
        const w = sim.r31_width();
        const h = sim.r31_height();

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.r31_u_field(), w, h);

        const xRaw = sim.r31_x_field();
        const xVis = new Float64Array(xRaw.length);
        for (let i = 0; i < xRaw.length; i++) {
            xVis[i] = Math.max(0, xRaw[i] - X_VIS_MIN);
        }
        drawField2D(xCtx, f2.width, f2.height, xVis, w, h, X_VIS_MAX - X_VIS_MIN);

        drawField2D(bCtx, f3.width, f3.height, sim.r31_b_field(), w, h, 1.2);

        els.rT.textContent    = sim.r31_time().toFixed(1);
        els.rEx.textContent   = sim.r31_excited_fraction().toFixed(3);
        els.rXm.textContent   = sim.r31_x_mean().toFixed(3);
        els.rRing.textContent = (100 * sim.r31_x_high_fraction_ring()).toFixed(1) + "%";
        els.rCore.textContent = (100 * sim.r31_x_high_fraction_core()).toFixed(1) + "%";
        els.rBm.textContent   = sim.r31_b_mean().toFixed(3);
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.resetChem.addEventListener("click", () => {
    sim?.reset_chemistry();
});
els.resetWave.addEventListener("click", () => {
    sim?.reset_tissue();
});
els.seedSpiral.addEventListener("click", () => {
    sim?.seed_spiral();
});

els.lambda.addEventListener("input", () => {
    lambdaVal = parseFloat(els.lambda.value);
    els.lambdaV.textContent = lambdaVal.toFixed(2);
    sim?.set_lambda_b(lambdaVal);
});
els.vel.addEventListener("input", () => {
    velVal = parseFloat(els.vel.value);
    els.velV.textContent = velVal.toFixed(1);
    sim?.set_velocity(velVal);
});
els.drive.addEventListener("input", () => {
    driveVal = parseFloat(els.drive.value);
    els.driveV.textContent = driveVal.toFixed(1);
    sim?.set_drive(driveVal);
});
els.supply.addEventListener("input", () => {
    supplyVal = parseFloat(els.supply.value);
    els.supplyV.textContent = supplyVal.toFixed(2);
    sim?.set_supply(supplyVal);
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
