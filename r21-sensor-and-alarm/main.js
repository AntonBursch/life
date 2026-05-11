// R21 -- Sensor and alarm. Barkley (R7) + integrate_field (R19)
// + threshold_event (R18) + OR-latch. No new operator. Phase-B
// composition.
import init, { WasmCoupledR21 } from "../viewer/pkg/flow_wasm.js";
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

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:     $("uCanvas"),
    avgCanvas:   $("avgCanvas"),
    alarmCanvas: $("alarmCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    resetSensor: $("resetSensor"),
    resetAlarm:  $("resetAlarm"),
    kick:        $("kick"),
    leak:    $("leak"),    leakV:    $("leakV"),
    alarmTh: $("alarmTh"), alarmThV: $("alarmThV"),
    eps:     $("eps"),     epsV:     $("epsV"),
    speed:   $("speed"),   speedV:   $("speedV"),
    rT: $("rT"), rUm: $("rUm"), rTau: $("rTau"),
    rAvgM: $("rAvgM"), rAvgMx: $("rAvgMx"),
    rTripStep: $("rTripStep"), rTripCum: $("rTripCum"),
    rCov: $("rCov"),
    errSlot: $("errSlot"),
};

const uCtx     = els.uCanvas.getContext("2d");
const avgCtx   = els.avgCanvas.getContext("2d");
const alarmCtx = els.alarmCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let leakVal = parseFloat(els.leak.value);
let alarmThVal = parseFloat(els.alarmTh.value);
let epsVal = parseFloat(els.eps.value);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR21(
            W, H, DIFFUSION, A_VAL, B_VAL, epsVal, DX, DT, leakVal, alarmThVal,
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
        const f2 = fitCanvas(els.avgCanvas);
        const f3 = fitCanvas(els.alarmCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        const avgMax = Math.max(0.1, sim.avg_max() * 1.05);
        drawField2D(avgCtx, f2.width, f2.height, sim.avg_field(), w, h, avgMax);
        drawField2D(alarmCtx, f3.width, f3.height, sim.alarm_field(), w, h, 1.0);

        const t = sim.tissue_time;
        const tau = leakVal > 0 ? 1.0 / leakVal : Infinity;
        els.rT.textContent      = t.toFixed(1);
        els.rUm.textContent     = sim.u_mean().toFixed(3);
        els.rTau.textContent    = isFinite(tau) ? tau.toFixed(2) : "∞";
        els.rAvgM.textContent   = sim.avg_mean().toFixed(3);
        els.rAvgMx.textContent  = sim.avg_max().toFixed(3);
        els.rTripStep.textContent = String(sim.trips_last_step);
        els.rTripCum.textContent  = String(sim.cumulative_trip_count.toFixed(0));
        els.rCov.textContent    = (100 * sim.alarm_coverage()).toFixed(1) + "%";
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
els.resetSensor.addEventListener("click", () => sim?.reset_sensor());
els.resetAlarm.addEventListener("click", () => sim?.reset_alarm());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.leak.addEventListener("input", () => {
    leakVal = parseFloat(els.leak.value);
    els.leakV.textContent = leakVal.toFixed(2);
    sim?.set_leak(leakVal);
});
els.alarmTh.addEventListener("input", () => {
    alarmThVal = parseFloat(els.alarmTh.value);
    els.alarmThV.textContent = alarmThVal.toFixed(2);
    sim?.set_alarm_threshold(alarmThVal);
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
