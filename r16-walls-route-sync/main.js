// R16 -- Walls route sync. CH -> gradient_magnitude -> Kuramoto.
// First use of the gradient_magnitude operator: differentiate the
// territory, sync on the walls.
import init, { WasmCoupledR16 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DDiverging,
    drawField2DPhase,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;

// CH (stable: mobility*kappa*dt/dx^4 = 0.0125)
const MOBILITY = 0.5;
const KAPPA = 0.5;
const CH_DX = 1.0;
const CH_DT = 0.05;

const PH_DT = 0.05;
const PRE_EVOLVE = 400;

const $ = (id) => document.getElementById(id);
const els = {
    phiCanvas:   $("phiCanvas"),
    gradCanvas:  $("gradCanvas"),
    kCanvas:     $("kCanvas"),
    phaseCanvas: $("phaseCanvas"),
    play:           $("play"),
    reterritory:    $("reterritory"),
    reseedPhases:   $("reseedPhases"),
    kwall:  $("kwall"),  kwallV:  $("kwallV"),
    kbulk:  $("kbulk"),  kbulkV:  $("kbulkV"),
    gref:   $("gref"),   grefV:   $("grefV"),
    sharp:  $("sharp"),  sharpV:  $("sharpV"),
    sigma:  $("sigma"),  sigmaV:  $("sigmaV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rTt: $("rTt"), rTp: $("rTp"),
    rWall: $("rWall"),
    rRw: $("rRw"), rRb: $("rRb"), rR: $("rR"),
    rLoc: $("rLoc"), rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const phiCtx   = els.phiCanvas.getContext("2d");
const gradCtx  = els.gradCanvas.getContext("2d");
const kCtx     = els.kCanvas.getContext("2d");
const phaseCtx = els.phaseCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kWall  = parseFloat(els.kwall.value);
let kBulk  = parseFloat(els.kbulk.value);
let gRef   = parseFloat(els.gref.value);
let sharp  = parseFloat(els.sharp.value);
let sigma  = parseFloat(els.sigma.value);
let phaseSeed = 42;
let territorySeed = 1;
const popSeed = 17;

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR16(
            W, H,
            MOBILITY, KAPPA, CH_DX, CH_DT,
            PH_DT,
            kBulk, kWall, gRef, sharp,
        );
        sim.seed_territory(0.05, 0.0, territorySeed);
        sim.pre_evolve_territory(PRE_EVOLVE);
        sim.set_natural_frequencies(sigma, popSeed);
        sim.randomise_phases(phaseSeed);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(rw, rb, loc, wallFrac) {
    if (wallFrac < 0.03) return "no walls";
    if (rw < 0.1 && rb < 0.1 && loc < 0.15) return "incoherent";
    if (rw > rb + 0.15) return "walls locked, bulk drifts";
    if (rw > 0.4 && rb > 0.4) return "globally locked";
    if (loc > 0.4 && rw < 0.3) return "locally locked, globally split";
    return "partial";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.phiCanvas);
        const f2 = fitCanvas(els.gradCanvas);
        const f3 = fitCanvas(els.kCanvas);
        const f4 = fitCanvas(els.phaseCanvas);

        const w = sim.width;
        const h = sim.height;

        drawField2DDiverging(phiCtx, f1.width, f1.height, sim.phi_field(), w, h, 1.0);
        // |grad phi| is in roughly [0, 1.2]
        drawField2D(gradCtx, f2.width, f2.height, sim.grad_field(), w, h, 1.2);
        drawField2D(kCtx, f3.width, f3.height, sim.k_coupling_field(), w, h, Math.max(kWall, 0.01));
        drawField2DPhase(phaseCtx, f4.width, f4.height, sim.theta_field(), w, h);

        const rw = sim.order_parameter_on_walls();
        const rb = sim.order_parameter_in_bulk();
        const rg = sim.order_parameter;
        const loc = sim.local_correlation();
        const wallFrac = sim.wall_coverage();

        els.rTt.textContent = sim.territory_time.toFixed(1);
        els.rTp.textContent = sim.phase_time.toFixed(1);
        els.rWall.textContent = (100 * wallFrac).toFixed(1) + "%";
        els.rRw.textContent = rw.toFixed(3);
        els.rRb.textContent = rb.toFixed(3);
        els.rR.textContent  = rg.toFixed(3);
        els.rLoc.textContent = (loc >= 0 ? "+" : "") + loc.toFixed(3);
        els.rRegime.textContent = regimeLabel(rw, rb, loc, wallFrac);
    }
    requestAnimationFrame(frame);
}

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reterritory.addEventListener("click", () => {
    if (!sim) return;
    territorySeed = (territorySeed + 1) | 0;
    sim.reset_territory();
    sim.seed_territory(0.05, 0.0, territorySeed);
    sim.pre_evolve_territory(PRE_EVOLVE);
});
els.reseedPhases.addEventListener("click", () => {
    if (!sim) return;
    phaseSeed = (phaseSeed + 1) | 0;
    sim.randomise_phases(phaseSeed);
});
els.kwall.addEventListener("input", () => {
    kWall = parseFloat(els.kwall.value);
    els.kwallV.textContent = kWall.toFixed(1);
    if (sim) sim.set_k_wall(kWall);
});
els.kbulk.addEventListener("input", () => {
    kBulk = parseFloat(els.kbulk.value);
    els.kbulkV.textContent = kBulk.toFixed(2);
    if (sim) sim.set_k_bulk(kBulk);
});
els.gref.addEventListener("input", () => {
    gRef = parseFloat(els.gref.value);
    els.grefV.textContent = gRef.toFixed(2);
    if (sim) sim.set_grad_ref(gRef);
});
els.sharp.addEventListener("input", () => {
    sharp = parseFloat(els.sharp.value);
    els.sharpV.textContent = sharp.toFixed(2);
    if (sim) sim.set_sharp(sharp);
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
