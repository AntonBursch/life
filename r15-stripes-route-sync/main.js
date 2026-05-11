// R15 -- Stripes route sync. Swift-Hohenberg (R6) -> bulk_gate ->
// Kuramoto (R9). Phase inherits the spatial geometry of the substrate.
import init, { WasmCoupledR15 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DDiverging,
    drawField2DPhase,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;

// Swift-Hohenberg (stability requires dt/dx^4 < 1/32 = 0.03125).
const SH_R = 0.3;
const SH_DX = 1.0;
const SH_DT = 0.025;

// Kuramoto
const PH_DT = 0.05;
const PHASE_SUBSTEPS = 5;

// bulk_gate defaults
const HALF_WIDTH = 0.3;
const SHARPNESS = 0.05;

const $ = (id) => document.getElementById(id);
const els = {
    uCanvas:     $("uCanvas"),
    kCanvas:     $("kCanvas"),
    phaseCanvas: $("phaseCanvas"),
    play:           $("play"),
    reseedPattern:  $("reseedPattern"),
    reseedPhases:   $("reseedPhases"),
    kstripe: $("kstripe"), kstripeV: $("kstripeV"),
    kgap:    $("kgap"),    kgapV:    $("kgapV"),
    hwidth:  $("hwidth"),  hwidthV:  $("hwidthV"),
    rparam:  $("rparam"),  rparamV:  $("rparamV"),
    sigma:   $("sigma"),   sigmaV:   $("sigmaV"),
    speed:   $("speed"),   speedV:   $("speedV"),
    rTu: $("rTu"), rStripe: $("rStripe"),
    rRs: $("rRs"), rRg: $("rRg"), rR: $("rR"),
    rLoc: $("rLoc"), rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const uCtx     = els.uCanvas.getContext("2d");
const kCtx     = els.kCanvas.getContext("2d");
const phaseCtx = els.phaseCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kStripe   = parseFloat(els.kstripe.value);
let kGap      = parseFloat(els.kgap.value);
let halfWidth = parseFloat(els.hwidth.value);
let rParam    = parseFloat(els.rparam.value);
let sigma     = parseFloat(els.sigma.value);
let phaseSeed = 42;
let patternSeed = 17; // not used as seed by SH (we just call seed_pattern amplitude)
let popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR15(
            W, H,
            rParam, SH_DX, SH_DT,
            PH_DT,
            kGap, kStripe, halfWidth, SHARPNESS,
            PHASE_SUBSTEPS,
        );
        sim.seed_pattern(0.3);
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(rs, rg, loc, stripeFrac) {
    if (stripeFrac < 0.05) return "no pattern";
    if (rs < 0.15 && rg < 0.15 && loc < 0.2) return "incoherent";
    if (rs > 0.4 && rg < 0.25) return "stripes locked, gaps drift";
    if (rs > 0.4 && rg > 0.4) return "globally locked";
    if (loc > 0.5 && rs < 0.4) return "locally locked, globally split";
    return "partial";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.uCanvas);
        const f2 = fitCanvas(els.kCanvas);
        const f3 = fitCanvas(els.phaseCanvas);

        const w = sim.width;
        const h = sim.height;

        // Pattern u: roughly in [-1.2, 1.2]. Diverging colormap.
        drawField2DDiverging(uCtx, f1.width, f1.height, sim.pattern_field(), w, h, 1.2);
        // K field: in [kGap, kStripe]. Linear ramp normalised by kStripe.
        const kMax = Math.max(kStripe, 0.01);
        drawField2D(kCtx, f2.width, f2.height, sim.k_coupling_field(), w, h, kMax);
        drawField2DPhase(phaseCtx, f3.width, f3.height, sim.theta_field(), w, h);

        const rs = sim.order_parameter_on_stripes();
        const rg = sim.order_parameter_in_gaps();
        const rGlobal = sim.order_parameter;
        const loc = sim.local_correlation();
        const stripeFrac = sim.stripe_fraction();

        els.rTu.textContent = sim.pattern_time.toFixed(1);
        els.rStripe.textContent = (100 * stripeFrac).toFixed(1) + "%";
        els.rRs.textContent = rs.toFixed(3);
        els.rRg.textContent = rg.toFixed(3);
        els.rR.textContent  = rGlobal.toFixed(3);
        els.rLoc.textContent = (loc >= 0 ? "+" : "") + loc.toFixed(3);
        els.rRegime.textContent = regimeLabel(rs, rg, loc, stripeFrac);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------
els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reseedPattern.addEventListener("click", () => {
    if (!sim) return;
    patternSeed = (patternSeed + 1) | 0;
    sim.seed_pattern(0.3);
});
els.reseedPhases.addEventListener("click", () => {
    if (!sim) return;
    phaseSeed = (phaseSeed + 1) | 0;
    sim.randomise_phases(phaseSeed);
});
els.kstripe.addEventListener("input", () => {
    kStripe = parseFloat(els.kstripe.value);
    els.kstripeV.textContent = kStripe.toFixed(1);
    if (sim) sim.set_k_stripe(kStripe);
});
els.kgap.addEventListener("input", () => {
    kGap = parseFloat(els.kgap.value);
    els.kgapV.textContent = kGap.toFixed(2);
    if (sim) sim.set_k_gap(kGap);
});
els.hwidth.addEventListener("input", () => {
    halfWidth = parseFloat(els.hwidth.value);
    els.hwidthV.textContent = halfWidth.toFixed(2);
    if (sim) sim.set_half_width(halfWidth);
});
els.rparam.addEventListener("input", () => {
    rParam = parseFloat(els.rparam.value);
    els.rparamV.textContent = rParam.toFixed(2);
    if (sim) sim.set_r(rParam);
});
els.sigma.addEventListener("input", () => {
    sigma = parseFloat(els.sigma.value);
    els.sigmaV.textContent = sigma.toFixed(2);
    if (sim) sim.set_natural_frequencies(sigma, popSeed);
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
