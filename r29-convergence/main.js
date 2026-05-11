// R29 -- Convergence. Two channels meet. Same composition as
// R28 (latch + advect + latch), but vx is position-dependent:
// left half flows right, right half flows left. Two spirals,
// one at each end; their wall_local fields advect inward and
// accumulate at the midline.
import init, { WasmCoupledR29 } from "../viewer/pkg/flow_wasm.js";
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
    uCanvas:           $("uCanvas"),
    memCanvas:         $("memCanvas"),
    wallLocalCanvas:   $("wallLocalCanvas"),
    transmittedCanvas: $("transmittedCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    killWave:    $("killWave"),
    healWalls:   $("healWalls"),
    sparkLeft:   $("sparkLeft"),
    sparkRight:  $("sparkRight"),
    sparkCenter: $("sparkCenter"),
    velocity: $("velocity"), velocityV: $("velocityV"),
    setThr:   $("setThr"),   setThrV:   $("setThrV"),
    resetThr: $("resetThr"), resetThrV: $("resetThrV"),
    killEps:  $("killEps"),  killEpsV:  $("killEpsV"),
    leak:     $("leak"),     leakV:     $("leakV"),
    speed:    $("speed"),    speedV:    $("speedV"),
    rT: $("rT"), rEx: $("rEx"),
    rWl: $("rWl"), rTr: $("rTr"),
    rWlM: $("rWlM"), rTrM: $("rTrM"),
    rMm: $("rMm"), rEm: $("rEm"),
    errSlot: $("errSlot"),
};

const uCtx   = els.uCanvas.getContext("2d");
const memCtx = els.memCanvas.getContext("2d");
const wlCtx  = els.wallLocalCanvas.getContext("2d");
const trCtx  = els.transmittedCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let velocityVal = parseFloat(els.velocity.value);
let setThrVal   = parseFloat(els.setThr.value);
let resetThrVal = parseFloat(els.resetThr.value);
let killEpsVal  = parseFloat(els.killEps.value);
let leakVal     = parseFloat(els.leak.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR29(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            killEpsVal, setThrVal, resetThrVal, leakVal, velocityVal,
        );
        sim.seed_two_sources();
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
        const f3 = fitCanvas(els.wallLocalCanvas);
        const f4 = fitCanvas(els.transmittedCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        drawField2D(memCtx, f2.width, f2.height, sim.memory_field(), w, h, 1.5);
        drawField2D(wlCtx,  f3.width, f3.height, sim.wall_local_field(),   w, h, 1.0);
        drawField2D(trCtx,  f4.width, f4.height, sim.transmitted_field(),  w, h, 1.0);

        els.rT.textContent   = sim.tissue_time.toFixed(1);
        els.rEx.textContent  = sim.excited_fraction().toFixed(3);
        els.rWl.textContent  = (100 * sim.wall_local_fraction()).toFixed(1) + "%";
        els.rTr.textContent  = (100 * sim.transmitted_fraction()).toFixed(1) + "%";
        els.rWlM.textContent = (100 * sim.midline_wall_local_fraction()).toFixed(1) + "%";
        els.rTrM.textContent = (100 * sim.midline_transmitted_fraction()).toFixed(1) + "%";
        els.rMm.textContent  = sim.memory_mean().toFixed(3);
        els.rEm.textContent  = sim.eps_mean().toFixed(4);
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reseed.addEventListener("click", () => sim?.seed_two_sources());
els.killWave.addEventListener("click", () => sim?.reset_tissue());
els.healWalls.addEventListener("click", () => sim?.reset_walls());
els.sparkLeft.addEventListener("click",
    () => sim?.kick(Math.floor(W / 5),       H >> 1, 6, 1.0));
els.sparkRight.addEventListener("click",
    () => sim?.kick(Math.floor((4 * W) / 5), H >> 1, 6, 1.0));
els.sparkCenter.addEventListener("click",
    () => sim?.kick(W >> 1, H >> 1, 6, 1.0));

els.velocity.addEventListener("input", () => {
    velocityVal = parseFloat(els.velocity.value);
    els.velocityV.textContent = velocityVal.toFixed(1);
    sim?.set_velocity(velocityVal);
});
els.setThr.addEventListener("input", () => {
    setThrVal = parseFloat(els.setThr.value);
    els.setThrV.textContent = setThrVal.toFixed(2);
    sim?.set_set_local(setThrVal);
});
els.resetThr.addEventListener("input", () => {
    resetThrVal = parseFloat(els.resetThr.value);
    els.resetThrV.textContent = resetThrVal.toFixed(2);
    sim?.set_reset_local(resetThrVal);
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
