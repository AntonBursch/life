// R20 -- Events seed matter. Barkley (R7) + threshold_event (R18)
// deposit V into autonomous Gray-Scott (R4) chemistry. Phase-B
// opener. No new operator.
import init, { WasmCoupledR20 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DTemperature,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:      $("uCanvas"),
    eventsCanvas: $("eventsCanvas"),
    vCanvas:      $("vCanvas"),
    play:         $("play"),
    waveToggle:   $("waveToggle"),
    reseed:       $("reseed"),
    resetChem:    $("resetChem"),
    kick:         $("kick"),
    threshold:    $("threshold"), thresholdV: $("thresholdV"),
    inject:       $("inject"),    injectV:    $("injectV"),
    eps:          $("eps"),       epsV:       $("epsV"),
    speed:        $("speed"),     speedV:     $("speedV"),
    rWave: $("rWave"), rT: $("rT"), rEvStep: $("rEvStep"),
    rEvCum: $("rEvCum"), rVmean: $("rVmean"), rCov: $("rCov"),
    errSlot: $("errSlot"),
};

const uCtx      = els.uCanvas.getContext("2d");
const eventsCtx = els.eventsCanvas.getContext("2d");
const vCtx      = els.vCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR20(W, H);
        sim.set_threshold(parseFloat(els.threshold.value));
        sim.set_inject(parseFloat(els.inject.value));
        sim.set_eps(parseFloat(els.eps.value));
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
        const f2 = fitCanvas(els.eventsCanvas);
        const f3 = fitCanvas(els.vCanvas);
        const w = sim.width;
        const h = sim.height;

        drawField2DTemperature(uCtx, f1.width, f1.height, sim.u_field(), w, h);
        drawField2D(eventsCtx, f2.width, f2.height, sim.events_field(), w, h, 1.0);
        drawField2DTemperature(vCtx, f3.width, f3.height, sim.chem_v_field(), w, h);

        els.rWave.textContent  = sim.wave_is_on ? "ON" : "OFF";
        els.rT.textContent     = sim.chem_time.toFixed(1);
        els.rEvStep.textContent = String(sim.events_last_step);
        els.rEvCum.textContent  = String(sim.cumulative_event_count.toFixed(0));
        els.rVmean.textContent  = sim.v_mean().toFixed(3);
        els.rCov.textContent    = (100 * sim.v_coverage()).toFixed(1) + "%";
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.waveToggle.addEventListener("click", () => {
    if (!sim) return;
    const next = !sim.wave_is_on;
    sim.set_wave_on(next);
    els.waveToggle.textContent = next ? "🌊 Wave: ON" : "🌊 Wave: OFF";
});
els.reseed.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_wave();
    sim.seed_spiral();
});
els.resetChem.addEventListener("click", () => sim?.reset_chem());
els.kick.addEventListener("click", () => {
    if (!sim) return;
    sim.kick(W >> 1, H >> 1, 6, 1.0);
});
els.threshold.addEventListener("input", () => {
    const v = parseFloat(els.threshold.value);
    els.thresholdV.textContent = v.toFixed(2);
    sim?.set_threshold(v);
});
els.inject.addEventListener("input", () => {
    const v = parseFloat(els.inject.value);
    els.injectV.textContent = v.toFixed(2);
    sim?.set_inject(v);
});
els.eps.addEventListener("input", () => {
    const v = parseFloat(els.eps.value);
    els.epsV.textContent = v.toFixed(3);
    sim?.set_eps(v);
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
