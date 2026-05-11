// R25 -- Homeostasis. Global negative-feedback control on Barkley
// eps. Reuses the parametrise category from R24, scalar form.
import init, { WasmCoupledR25 } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2DTemperature } from "../viewer/canvas.js";

const W = 128;
const H = 128;

const DIFFUSION = 1.0;
const A_VAL = 0.75;
const B_VAL = 0.06;
const BASE_EPS = 0.02;
const DX = 1.0;
const DT = 0.05;
const WARMUP = 200;
const TRACE_CAP = 800;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:      $("uCanvas"),
    traceCanvas:  $("traceCanvas"),
    play:        $("play"),
    ctrl:        $("ctrl"),
    reseed:      $("reseed"),
    kick:        $("kick"),
    clearTrace:  $("clearTrace"),
    target: $("target"), targetV: $("targetV"),
    gain:   $("gain"),   gainV:   $("gainV"),
    leak:   $("leak"),   leakV:   $("leakV"),
    epsMax: $("epsMax"), epsMaxV: $("epsMaxV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rT: $("rT"), rEx: $("rEx"), rTgt: $("rTgt"), rErr: $("rErr"),
    rEps: $("rEps"), rOff: $("rOff"), rUm: $("rUm"), rCtl: $("rCtl"),
    errSlot: $("errSlot"),
};

const uCtx     = els.uCanvas.getContext("2d");
const traceCtx = els.traceCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let targetVal = parseFloat(els.target.value);
let gainVal   = parseFloat(els.gain.value);
let leakVal   = parseFloat(els.leak.value);
let epsMaxVal = parseFloat(els.epsMax.value);
let controllerOn = true;

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR25(
            W, H, DIFFUSION, A_VAL, B_VAL, BASE_EPS, DX, DT,
            targetVal, gainVal, epsMaxVal, WARMUP, TRACE_CAP,
        );
        sim.set_control_leak(leakVal);
        sim.set_controller_on(controllerOn);
        sim.seed_spiral();
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function drawTrace() {
    if (!sim) return;
    const f = fitCanvas(els.traceCanvas);
    const w = f.width;
    const h = f.height;
    traceCtx.fillStyle = "#000";
    traceCtx.fillRect(0, 0, w, h);

    const act = sim.trace_activity();
    const eps = sim.trace_eps();
    const n = act.length;
    if (n < 2) return;

    // Y-axes: left = activity (0..0.25), right = eps (BASE..epsMax).
    const actMax = 0.25;
    const epsLo = BASE_EPS;
    const epsHi = Math.max(epsMaxVal, BASE_EPS + 0.001);

    // gridlines
    traceCtx.strokeStyle = "#222";
    traceCtx.lineWidth = 1;
    traceCtx.beginPath();
    for (let i = 1; i < 5; i++) {
        const y = (h * i) / 5;
        traceCtx.moveTo(0, y); traceCtx.lineTo(w, y);
    }
    traceCtx.stroke();

    // target line (dashed)
    const yTgt = h - (targetVal / actMax) * h;
    traceCtx.strokeStyle = "#fa5";
    traceCtx.setLineDash([6, 4]);
    traceCtx.beginPath();
    traceCtx.moveTo(0, yTgt); traceCtx.lineTo(w, yTgt);
    traceCtx.stroke();
    traceCtx.setLineDash([]);

    // activity trace (cyan)
    traceCtx.strokeStyle = "#5cf";
    traceCtx.lineWidth = 1.5;
    traceCtx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const v = Math.max(0, Math.min(actMax, act[i]));
        const y = h - (v / actMax) * h;
        if (i === 0) traceCtx.moveTo(x, y); else traceCtx.lineTo(x, y);
    }
    traceCtx.stroke();

    // eps trace (red, right scale)
    traceCtx.strokeStyle = "#f55";
    traceCtx.lineWidth = 1.5;
    traceCtx.beginPath();
    for (let i = 0; i < n; i++) {
        const x = (i / (n - 1)) * w;
        const norm = (eps[i] - epsLo) / (epsHi - epsLo);
        const v = Math.max(0, Math.min(1, norm));
        const y = h - v * h;
        if (i === 0) traceCtx.moveTo(x, y); else traceCtx.lineTo(x, y);
    }
    traceCtx.stroke();
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const fu = fitCanvas(els.uCanvas);
        drawField2DTemperature(uCtx, fu.width, fu.height, sim.u_field(), sim.width, sim.height);
        drawTrace();

        const err = sim.error();
        els.rT.textContent   = sim.tissue_time.toFixed(1);
        els.rEx.textContent  = sim.excited_fraction().toFixed(3);
        els.rTgt.textContent = sim.target().toFixed(3);
        els.rErr.textContent = (err >= 0 ? "+" : "") + err.toFixed(3);
        els.rEps.textContent = sim.eps_global().toFixed(4);
        els.rOff.textContent = sim.eps_offset().toFixed(4);
        els.rUm.textContent  = sim.u_mean().toFixed(3);
        if (sim.warmup_remaining > 0) {
            els.rCtl.textContent = `warmup (${sim.warmup_remaining})`;
        } else {
            els.rCtl.textContent = sim.controller_active ? "active" : "off";
        }
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.ctrl.addEventListener("click", () => {
    controllerOn = !controllerOn;
    sim?.set_controller_on(controllerOn);
    els.ctrl.textContent = `Controller: ${controllerOn ? "on" : "off"}`;
});
els.reseed.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_tissue();
    sim.seed_spiral();
});
els.kick.addEventListener("click", () => sim?.kick(W >> 1, H >> 1, 8, 1.0));
els.clearTrace.addEventListener("click", () => sim?.clear_trace());

els.target.addEventListener("input", () => {
    targetVal = parseFloat(els.target.value);
    els.targetV.textContent = targetVal.toFixed(2);
    sim?.set_target(targetVal);
});
els.gain.addEventListener("input", () => {
    gainVal = parseFloat(els.gain.value);
    els.gainV.textContent = gainVal.toFixed(2);
    sim?.set_control_gain(gainVal);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    sim?.set_control_leak(leakVal);
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
