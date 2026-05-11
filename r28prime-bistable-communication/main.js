// R28' -- Bistable communication. The Schlogl species X is
// transported by a uniform velocity field; downstream cells
// commit to X_high via local chemistry, with no comparator
// anywhere. See ../THESIS.md.
import init, { WasmCoupledR28Prime } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DTemperature,
} from "../viewer/canvas.js";

// Wider grid than R27' to make the spatial channel visible.
const W = 160;
const H = 96;

const DIFFUSION = 1.0;
const A_VAL = 0.75;
const B_VAL = 0.06;
const BASE_EPS = 0.02;
const DX = 1.0;
const DT = 0.02;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:   $("uCanvas"),
    xCanvas:   $("xCanvas"),
    epsCanvas: $("epsCanvas"),
    play:      $("play"),
    reseed:    $("reseed"),
    killWave:  $("killWave"),
    healChem:  $("healChem"),
    kick:      $("kick"),
    vx:      $("vx"),      vxV:      $("vxV"),
    drive:   $("drive"),   driveV:   $("driveV"),
    uThr:    $("uThr"),    uThrV:    $("uThrV"),
    killEps: $("killEps"), killEpsV: $("killEpsV"),
    speed:   $("speed"),   speedV:   $("speedV"),
    rT: $("rT"), rEx: $("rEx"), rXm: $("rXm"),
    rHL: $("rHL"), rHR: $("rHR"), rEm: $("rEm"),
    errSlot: $("errSlot"),
};

const uCtx   = els.uCanvas.getContext("2d");
const xCtx   = els.xCanvas.getContext("2d");
const epsCtx = els.epsCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let vxVal      = parseFloat(els.vx.value);
let driveVal   = parseFloat(els.drive.value);
let uThrVal    = parseFloat(els.uThr.value);
let killEpsVal = parseFloat(els.killEps.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR28Prime(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            killEpsVal, uThrVal, driveVal, vxVal,
        );
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

// X colour range fixed at [1, 3] (the two Schlogl stable states).
const X_VIS_MIN = 1.0;
const X_VIS_MAX = 3.0;

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.xCanvas);
        const f3 = fitCanvas(els.epsCanvas);
        const w = sim.r28p_width();
        const h = sim.r28p_height();

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.r28p_u_field(), w, h);

        const xRaw = sim.r28p_x_field();
        const xVis = new Float64Array(xRaw.length);
        for (let i = 0; i < xRaw.length; i++) {
            xVis[i] = xRaw[i] - X_VIS_MIN;
        }
        drawField2D(xCtx, f2.width, f2.height, xVis, w, h, X_VIS_MAX - X_VIS_MIN);

        drawField2D(epsCtx, f3.width, f3.height, sim.r28p_eps_field(), w, h, killEpsVal);

        els.rT.textContent  = sim.r28p_time().toFixed(1);
        els.rEx.textContent = sim.r28p_excited_fraction().toFixed(3);
        els.rXm.textContent = sim.r28p_x_mean().toFixed(3);
        els.rHL.textContent = (100 * sim.r28p_x_high_fraction_left()).toFixed(1) + "%";
        els.rHR.textContent = (100 * sim.r28p_x_high_fraction_right()).toFixed(1) + "%";
        els.rEm.textContent = sim.r28p_eps_mean().toFixed(4);
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
els.killWave.addEventListener("click", () => sim?.reset_tissue());
els.healChem.addEventListener("click", () => sim?.reset_chemistry());
// Kick on the right half to probe the bistability there directly.
els.kick.addEventListener("click", () =>
    sim?.kick(Math.floor((W * 3) / 4), H >> 1, 6, 1.0),
);

els.vx.addEventListener("input", () => {
    vxVal = parseFloat(els.vx.value);
    els.vxV.textContent = vxVal.toFixed(1);
    sim?.set_velocity_x(vxVal);
});
els.drive.addEventListener("input", () => {
    driveVal = parseFloat(els.drive.value);
    els.driveV.textContent = driveVal.toFixed(1);
    sim?.set_drive(driveVal);
});
els.uThr.addEventListener("input", () => {
    uThrVal = parseFloat(els.uThr.value);
    els.uThrV.textContent = uThrVal.toFixed(2);
    sim?.set_u_thr(uThrVal);
});
els.killEps.addEventListener("input", () => {
    killEpsVal = parseFloat(els.killEps.value);
    els.killEpsV.textContent = killEpsVal.toFixed(3);
    sim?.set_kill_eps(killEpsVal);
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
