// R14 -- Three-layer stack. CH -> Kuramoto -> Gray-Scott via
// bulk_gate and phase_to_scalar_field.
import init, { WasmCoupledR14 } from "../viewer/pkg/flow_wasm.js";
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

// Kuramoto
const PH_DT = 0.05;
const SIGMA = 0.1;

// Gray-Scott
const GS_DU = 0.16;
const GS_DV = 0.08;
const GS_DX = 1.0;
const GS_DT = 1.0;

const HALF_WIDTH = 0.5;
const SHARPNESS = 0.1;
const PHASE_SUBSTEPS = 20; // 20 * 0.05 = 1.0 matches gs dt

const $ = (id) => document.getElementById(id);
const els = {
    phiCanvas:   $("phiCanvas"),
    phaseCanvas: $("phaseCanvas"),
    feedCanvas:  $("feedCanvas"),
    vCanvas:     $("vCanvas"),
    play:        $("play"),
    reseedPhases: $("reseedPhases"),
    reterritory:  $("reterritory"),
    reseedChem:   $("reseedChem"),
    kbulk: $("kbulk"), kbulkV: $("kbulkV"),
    kwall: $("kwall"), kwallV: $("kwallV"),
    flo:   $("flo"),   floV:   $("floV"),
    fhi:   $("fhi"),   fhiV:   $("fhiV"),
    kill:  $("kill"),  killV:  $("killV"),
    speed: $("speed"), speedV: $("speedV"),
    rTt: $("rTt"), rTc: $("rTc"),
    rRpos: $("rRpos"), rRneg: $("rRneg"),
    rMeanV: $("rMeanV"), rCov: $("rCov"),
    rSplit: $("rSplit"), rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const phiCtx   = els.phiCanvas.getContext("2d");
const phaseCtx = els.phaseCanvas.getContext("2d");
const feedCtx  = els.feedCanvas.getContext("2d");
const vCtx     = els.vCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let kBulk = parseFloat(els.kbulk.value);
let kWall = parseFloat(els.kwall.value);
let fLo = parseFloat(els.flo.value);
let fHi = parseFloat(els.fhi.value);
let kill = parseFloat(els.kill.value);
let phaseSeed = 1;
let territorySeed = 1;
const popSeed = 17;

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR14(
            W, H,
            MOBILITY, KAPPA, CH_DX, CH_DT,
            PH_DT,
            GS_DU, GS_DV, kill, GS_DX, GS_DT,
            kWall, kBulk, HALF_WIDTH, SHARPNESS,
            fLo, fHi,
            PHASE_SUBSTEPS,
        );
        sim.seed_noise(0.05, 0.0, territorySeed);
        sim.set_natural_frequencies(SIGMA, popSeed);
        sim.randomise_phases(phaseSeed);
        sim.seed_blob(W >> 1, H >> 1, 4);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(rpos, rneg, cov, split) {
    if (cov < 0.02) return "cold";
    if (rpos < 0.2 && rneg < 0.2) return "incoherent feed";
    if (Math.abs(split) > 0.02) return "split bloom";
    if (cov < 0.15) return "growing";
    return "synchronous bloom";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.phiCanvas);
        const f2 = fitCanvas(els.phaseCanvas);
        const f3 = fitCanvas(els.feedCanvas);
        const f4 = fitCanvas(els.vCanvas);

        const w = sim.width;
        const h = sim.height;

        drawField2DDiverging(phiCtx, f1.width, f1.height, sim.phi_field(), w, h, 1.0);
        drawField2DPhase(phaseCtx, f2.width, f2.height, sim.theta_field(), w, h);
        drawField2D(feedCtx, f3.width, f3.height, sim.feed_field(), w, h, fHi);
        drawField2D(vCtx, f4.width, f4.height, sim.chem_v_field(), w, h, 0.5);

        const rpos = sim.order_parameter_pos();
        const rneg = sim.order_parameter_neg();
        const meanV = sim.mean_v();
        const cov = sim.v_coverage();
        const split = sim.bloom_split();
        els.rTt.textContent = sim.territory_time.toFixed(1);
        els.rTc.textContent = sim.chem_time.toFixed(0);
        els.rRpos.textContent = rpos.toFixed(3);
        els.rRneg.textContent = rneg.toFixed(3);
        els.rMeanV.textContent = meanV.toFixed(3);
        els.rCov.textContent = (100 * cov).toFixed(1) + "%";
        els.rSplit.textContent = (split >= 0 ? "+" : "") + split.toFixed(3);
        els.rRegime.textContent = regimeLabel(rpos, rneg, cov, split);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------
els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});
els.reseedPhases.addEventListener("click", () => {
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
els.reseedChem.addEventListener("click", () => {
    if (!sim) return;
    sim.reset_chem();
    sim.seed_blob(W >> 1, H >> 1, 4);
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
