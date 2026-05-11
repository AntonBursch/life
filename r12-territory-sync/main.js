// R12 -- Territory shapes sync. CH territory + bulk_gate + Kuramoto.
// Three panels: phi (diverging), K (magma), theta (cyclic).
import init, { WasmCoupledR12 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DDiverging,
    drawField2DPhase,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 1.0;
const DT_TERRITORY = 0.05;
const DT_PHASE = 0.05;

// Cahn-Hilliard params (stable: mobility*kappa*dt/dx^4 = 0.0125 < 1/32).
const MOBILITY = 0.5;
const KAPPA = 0.5;

const $ = (id) => document.getElementById(id);
const els = {
    phiCanvas:   $("phiCanvas"),
    gateCanvas:  $("gateCanvas"),
    phaseCanvas: $("phaseCanvas"),
    play:        $("play"),
    reseed:      $("reseed"),
    reterritory: $("reterritory"),
    kbulk:       $("kbulk"),
    kbulkV:      $("kbulkV"),
    kwall:       $("kwall"),
    kwallV:      $("kwallV"),
    hwidth:      $("hwidth"),
    hwidthV:     $("hwidthV"),
    sval:        $("sval"),
    svalV:       $("svalV"),
    speed:       $("speed"),
    speedV:      $("speedV"),
    rTt:     $("rTt"),
    rR:      $("rR"),
    rRpos:   $("rRpos"),
    rRneg:   $("rRneg"),
    rCross:  $("rCross"),
    rLoc:    $("rLoc"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const phiCtx   = els.phiCanvas.getContext("2d");
const gateCtx  = els.gateCanvas.getContext("2d");
const phaseCtx = els.phaseCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kBulk = parseFloat(els.kbulk.value);
let kWall = parseFloat(els.kwall.value);
let halfWidth = parseFloat(els.hwidth.value);
let sigma = parseFloat(els.sval.value);
let phaseSeed = 1;
let territorySeed = 1;
let popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR12(
            W, H,
            MOBILITY, KAPPA, DX, DT_TERRITORY,
            DT_PHASE,
            kWall, kBulk, halfWidth, 0.1,
        );
        sim.seed_noise(0.05, 0.0, territorySeed);
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(rpos, rneg, cross) {
    const m = Math.min(rpos, rneg);
    if (m > 0.5 && Math.abs(cross) < 0.6) return "territorial sync";
    if (m > 0.3) return "two-domain locking";
    if (Math.max(rpos, rneg) > 0.3) return "one-sided lock";
    return "incoherent";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const phiFit   = fitCanvas(els.phiCanvas);
        const gateFit  = fitCanvas(els.gateCanvas);
        const phaseFit = fitCanvas(els.phaseCanvas);

        const w = sim.width;
        const h = sim.height;
        const phi = sim.phi_field();
        const k = sim.k_coupling_field();
        const theta = sim.theta_field();

        drawField2DDiverging(phiCtx, phiFit.width, phiFit.height, phi, w, h, 1.0);
        drawField2D(gateCtx, gateFit.width, gateFit.height, k, w, h, kBulk);
        drawField2DPhase(phaseCtx, phaseFit.width, phaseFit.height, theta, w, h);

        const r = sim.order_parameter;
        const rpos = sim.order_parameter_pos();
        const rneg = sim.order_parameter_neg();
        const cross = sim.cross_domain_alignment();
        const loc = sim.local_correlation();
        els.rTt.textContent = sim.territory_time.toFixed(1);
        els.rR.textContent = r.toFixed(3);
        els.rRpos.textContent = rpos.toFixed(3);
        els.rRneg.textContent = rneg.toFixed(3);
        els.rCross.textContent = cross.toFixed(3);
        els.rLoc.textContent = loc.toFixed(3);
        els.rRegime.textContent = regimeLabel(rpos, rneg, cross);
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

els.reterritory.addEventListener("click", () => {
    if (!sim) return;
    territorySeed = (territorySeed + 1) | 0;
    sim.reset_territory();
    sim.seed_noise(0.05, 0.0, territorySeed);
});

els.kbulk.addEventListener("input", () => {
    kBulk = parseFloat(els.kbulk.value);
    els.kbulkV.textContent = kBulk.toFixed(1);
    if (sim) sim.set_k_bulk(kBulk);
});

els.kwall.addEventListener("input", () => {
    kWall = parseFloat(els.kwall.value);
    els.kwallV.textContent = kWall.toFixed(2);
    if (sim) sim.set_k_wall(kWall);
});

els.hwidth.addEventListener("input", () => {
    halfWidth = parseFloat(els.hwidth.value);
    els.hwidthV.textContent = halfWidth.toFixed(2);
    if (sim) sim.set_half_width(halfWidth);
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
