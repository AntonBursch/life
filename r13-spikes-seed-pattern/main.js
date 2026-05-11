// R13 -- Spikes seed pattern. Barkley + excitable_gate + Gray-Scott.
// Three panels: u (magma), feed (magma), V (magma).
import init, { WasmCoupledR13 } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2D } from "../viewer/canvas.js";

const W = 128;
const H = 128;

// Barkley
const BARK_DIFF = 1.0;
const BARK_A = 0.75;
const BARK_B = 0.01;
const BARK_EPS = 0.02;
const BARK_DX = 1.0;
const BARK_DT = 0.05;
const BARK_SUBSTEPS = 20; // 20 * 0.05 = 1.0, matches gs_dt

// Gray-Scott
const GS_DU = 0.16;
const GS_DV = 0.08;
const GS_DX = 1.0;
const GS_DT = 1.0;

const SHARPNESS = 0.1;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:    $("uCanvas"),
    feedCanvas: $("feedCanvas"),
    vCanvas:    $("vCanvas"),
    play:        $("play"),
    reseedSpiral: $("reseedSpiral"),
    reseedChem:   $("reseedChem"),
    flo:   $("flo"),  floV:  $("floV"),
    fhi:   $("fhi"),  fhiV:  $("fhiV"),
    kill:  $("kill"), killV: $("killV"),
    thr:   $("thr"),  thrV:  $("thrV"),
    speed: $("speed"), speedV: $("speedV"),
    rT:      $("rT"),
    rFire:   $("rFire"),
    rMeanV:  $("rMeanV"),
    rCov:    $("rCov"),
    rCorr:   $("rCorr"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const uCtx    = els.uCanvas.getContext("2d");
const feedCtx = els.feedCanvas.getContext("2d");
const vCtx    = els.vCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let fLo = parseFloat(els.flo.value);
let fHi = parseFloat(els.fhi.value);
let kill = parseFloat(els.kill.value);
let threshold = parseFloat(els.thr.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR13(
            W, H,
            BARK_DIFF, BARK_A, BARK_B, BARK_EPS, BARK_DX, BARK_DT,
            GS_DU, GS_DV, kill, GS_DX, GS_DT,
            fLo, fHi, threshold, SHARPNESS,
            BARK_SUBSTEPS,
        );
        sim.seed_spiral();
        sim.seed_blob(W >> 1, H >> 1, 4);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(meanV, cov, corr) {
    if (meanV < 0.005) return "cold";
    if (cov < 0.05) return "seeding";
    if (cov < 0.20) return "growing";
    if (corr > 0.05) return "wave-led blooms";
    return "saturated pattern";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const uFit    = fitCanvas(els.uCanvas);
        const feedFit = fitCanvas(els.feedCanvas);
        const vFit    = fitCanvas(els.vCanvas);

        const w = sim.width;
        const h = sim.height;
        const u = sim.u_field();
        const f = sim.feed_field();
        const v = sim.chem_v_field();

        drawField2D(uCtx, uFit.width, uFit.height, u, w, h, 1.0);
        drawField2D(feedCtx, feedFit.width, feedFit.height, f, w, h, fHi);
        drawField2D(vCtx, vFit.width, vFit.height, v, w, h, 0.5);

        const fire = sim.firing_fraction();
        const meanV = sim.mean_v();
        const cov = sim.v_coverage();
        const corr = sim.wave_pattern_correlation();
        els.rT.textContent = sim.chem_time.toFixed(0);
        els.rFire.textContent = (100 * fire).toFixed(1) + "%";
        els.rMeanV.textContent = meanV.toFixed(3);
        els.rCov.textContent = (100 * cov).toFixed(1) + "%";
        els.rCorr.textContent = (corr >= 0 ? "+" : "") + corr.toFixed(3);
        els.rRegime.textContent = regimeLabel(meanV, cov, corr);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});

els.reseedSpiral.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_excitable();
    sim.seed_spiral();
});

els.reseedChem.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_chem();
    sim.seed_blob(W >> 1, H >> 1, 4);
});

els.flo.addEventListener("input", () => {
    fLo = parseFloat(els.flo.value);
    els.floV.textContent = fLo.toFixed(3);
    if (sim) sim.set_f_lo(fLo);
});
els.fhi.addEventListener("input", () => {
    fHi = parseFloat(els.fhi.value);
    els.fhiV.textContent = fHi.toFixed(3);
    if (sim) sim.set_f_hi(fHi);
});
els.kill.addEventListener("input", () => {
    kill = parseFloat(els.kill.value);
    els.killV.textContent = kill.toFixed(3);
    if (sim) sim.set_kill(kill);
});
els.thr.addEventListener("input", () => {
    threshold = parseFloat(els.thr.value);
    els.thrV.textContent = threshold.toFixed(2);
    if (sim) sim.set_threshold(threshold);
});
els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

// --- boot -----------------------------------------------------------------

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
