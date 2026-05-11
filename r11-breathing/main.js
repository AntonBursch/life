// R11 — Phase drives reaction. A Kuramoto phase layer dictates the
// per-cell Gray-Scott feed rate. Three panels: phase, feed field, V.
import init, { WasmCoupledR11 } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2D, drawField2DPhase } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 1.0;
const DT_CHEM = 1.0;
const DT_PHASE = 0.05;

// Gray-Scott diffusion baseline (coral / spot regime).
const DU = 0.16;
const DV = 0.08;

const $ = (id) => document.getElementById(id);
const els = {
    phaseCanvas: $("phaseCanvas"),
    feedCanvas:  $("feedCanvas"),
    vCanvas:     $("vCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    reblob:      $("reblob"),
    coup:        $("coup"),
    coupV:       $("coupV"),
    sval:        $("sval"),
    svalV:       $("svalV"),
    fhi:         $("fhi"),
    fhiV:        $("fhiV"),
    flo:         $("flo"),
    floV:        $("floV"),
    kill:        $("kill"),
    killV:       $("killV"),
    speed:       $("speed"),
    speedV:      $("speedV"),
    rTc:     $("rTc"),
    rR:      $("rR"),
    rLoc:    $("rLoc"),
    rV:      $("rV"),
    rCov:    $("rCov"),
    rBr:     $("rBr"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const phaseCtx = els.phaseCanvas.getContext("2d");
const feedCtx  = els.feedCanvas.getContext("2d");
const vCtx     = els.vCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let coup = parseFloat(els.coup.value);
let sigma = parseFloat(els.sval.value);
let fHi = parseFloat(els.fhi.value);
let fLo = parseFloat(els.flo.value);
let kill = parseFloat(els.kill.value);
let phaseSeed = 1;
let popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR11(
            W, H,
            DU, DV, kill, DX, DT_CHEM,
            DT_PHASE, coup,
            fLo, fHi,
        );
        sim.seed_blob(W / 2 | 0, H / 2 | 0, 8);
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(r, loc, br) {
    if (r > 0.85) return "global breathing";
    if (loc > 0.5 && br > 0.05) return "patchy breathing";
    if (loc > 0.2) return "patchy sync";
    return "incoherent";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const phaseFit = fitCanvas(els.phaseCanvas);
        const feedFit  = fitCanvas(els.feedCanvas);
        const vFit     = fitCanvas(els.vCanvas);

        const w = sim.width;
        const h = sim.height;
        const theta = sim.theta_field();
        const f = sim.feed_field();
        const v = sim.v_field();

        drawField2DPhase(phaseCtx, phaseFit.width, phaseFit.height, theta, w, h);
        drawField2D(feedCtx, feedFit.width, feedFit.height, f, w, h, fHi);
        drawField2D(vCtx, vFit.width, vFit.height, v, w, h, 0.5);

        const r = sim.order_parameter;
        const loc = sim.local_correlation();
        const meanV = sim.total_v;
        const cov = sim.v_coverage;
        const br = sim.breathing_depth();

        els.rTc.textContent = sim.chem_time.toFixed(0);
        els.rR.textContent = r.toFixed(3);
        els.rLoc.textContent = loc.toFixed(3);
        els.rV.textContent = meanV.toFixed(3);
        els.rCov.textContent = cov.toFixed(3);
        els.rBr.textContent = br.toFixed(3);
        els.rRegime.textContent = regimeLabel(r, loc, br);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});

els.reseed.addEventListener("click", () => {
    if (!sim) return;
    phaseSeed = (phaseSeed + 1) | 0;
    sim.randomise_phases(phaseSeed);
});

els.reblob.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_chem();
    sim.seed_blob(W / 2 | 0, H / 2 | 0, 8);
});

els.coup.addEventListener("input", () => {
    coup = parseFloat(els.coup.value);
    els.coupV.textContent = coup.toFixed(1);
    if (sim) sim.set_coupling(coup);
});

els.sval.addEventListener("input", () => {
    sigma = parseFloat(els.sval.value);
    els.svalV.textContent = sigma.toFixed(2);
    if (sim) sim.set_natural_frequencies(sigma, popSeed);
});

els.fhi.addEventListener("input", () => {
    fHi = parseFloat(els.fhi.value);
    els.fhiV.textContent = fHi.toFixed(3);
    if (sim) sim.set_f_hi(fHi);
});

els.flo.addEventListener("input", () => {
    fLo = parseFloat(els.flo.value);
    els.floV.textContent = fLo.toFixed(3);
    if (sim) sim.set_f_lo(fLo);
});

els.kill.addEventListener("input", () => {
    kill = parseFloat(els.kill.value);
    els.killV.textContent = kill.toFixed(3);
    if (sim) sim.set_kill(kill);
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
