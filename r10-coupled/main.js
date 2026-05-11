// R10 — Coupled substrates. Barkley activator gates per-cell
// Kuramoto coupling. Three panels: activator, K-field, phase.
import init, { WasmCoupledR10 } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2D, drawField2DPhase } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 1.0;
const DT_TISSUE = 0.05;
const DT_PHASE = 0.05;

// Barkley spiral params (same as R7).
const BARKLEY_DIFF = 1.0;
const BARKLEY_A = 0.75;
const BARKLEY_B = 0.06;
const BARKLEY_EPS = 0.02;

const $ = (id) => document.getElementById(id);
const els = {
    actCanvas:   $("actCanvas"),
    gateCanvas:  $("gateCanvas"),
    phaseCanvas: $("phaseCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    respiral:    $("respiral"),
    khi:         $("khi"),
    khiV:        $("khiV"),
    klo:         $("klo"),
    kloV:        $("kloV"),
    sval:        $("sval"),
    svalV:       $("svalV"),
    speed:       $("speed"),
    speedV:      $("speedV"),
    rTt:     $("rTt"),
    rExc:    $("rExc"),
    rK:      $("rK"),
    rR:      $("rR"),
    rLoc:    $("rLoc"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const actCtx   = els.actCanvas.getContext("2d");
const gateCtx  = els.gateCanvas.getContext("2d");
const phaseCtx = els.phaseCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kHi = parseFloat(els.khi.value);
let kLo = parseFloat(els.klo.value);
let sigma = parseFloat(els.sval.value);
let phaseSeed = 1;
let popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR10(
            W, H,
            BARKLEY_DIFF, BARKLEY_A, BARKLEY_B, BARKLEY_EPS, DX, DT_TISSUE,
            DT_PHASE,
            kLo, kHi, 0.4, 0.15,
        );
        sim.seed_spiral();
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(loc, exc) {
    if (exc < 0.02) return "cold";
    if (loc < 0.15) return "uncoupled";
    if (loc < 0.5) return "patchy sync";
    if (loc < 0.8) return "wave-following";
    return "strongly bound";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const actFit   = fitCanvas(els.actCanvas);
        const gateFit  = fitCanvas(els.gateCanvas);
        const phaseFit = fitCanvas(els.phaseCanvas);

        const w = sim.width;
        const h = sim.height;
        const u = sim.u_field();
        const k = sim.k_coupling_field();
        const theta = sim.theta_field();

        // Activator: magma ramp, vmax = 1.
        drawField2D(actCtx, actFit.width, actFit.height, u, w, h, 1.0);
        // K-field: magma ramp, vmax = kHi so the gate's full range maps to colour.
        drawField2D(gateCtx, gateFit.width, gateFit.height, k, w, h, kHi);
        // Phase: cyclic colormap.
        drawField2DPhase(phaseCtx, phaseFit.width, phaseFit.height, theta, w, h);

        const r = sim.order_parameter;
        const loc = sim.local_correlation();
        const exc = sim.excited_fraction;
        els.rTt.textContent = sim.tissue_time.toFixed(1);
        els.rExc.textContent = exc.toFixed(3);
        els.rK.textContent = `${sim.k_lo.toFixed(2)} .. ${sim.k_hi.toFixed(2)}`;
        els.rR.textContent = r.toFixed(3);
        els.rLoc.textContent = loc.toFixed(3);
        els.rRegime.textContent = regimeLabel(loc, exc);
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

els.respiral.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_tissue();
    sim.seed_spiral();
});

els.khi.addEventListener("input", () => {
    kHi = parseFloat(els.khi.value);
    els.khiV.textContent = kHi.toFixed(1);
    if (sim) sim.set_k_hi(kHi);
});

els.klo.addEventListener("input", () => {
    kLo = parseFloat(els.klo.value);
    els.kloV.textContent = kLo.toFixed(2);
    if (sim) sim.set_k_lo(kLo);
});

els.sval.addEventListener("input", () => {
    sigma = parseFloat(els.sval.value);
    els.svalV.textContent = sigma.toFixed(2);
    if (sim) sim.set_natural_frequencies(sigma, popSeed);
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
