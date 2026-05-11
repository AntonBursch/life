// R17 -- Territory carries dye. CH as stream function -> velocity ->
// advect_by transports passive dye along walls.
import init, { WasmCoupledR17 } from "../viewer/pkg/flow_wasm.js";
import {
    fitCanvas,
    drawField2D,
    drawField2DDiverging,
} from "../viewer/canvas.js";

const W = 128;
const H = 128;

const MOBILITY = 0.5;
const KAPPA = 0.5;
const CH_DX = 1.0;
const CH_DT = 0.05;
const PRE_EVOLVE = 400;

const $ = (id) => document.getElementById(id);
const els = {
    phiCanvas:   $("phiCanvas"),
    speedCanvas: $("speedCanvas"),
    vxCanvas:    $("vxCanvas"),
    dyeCanvas:   $("dyeCanvas"),
    play:        $("play"),
    reterritory: $("reterritory"),
    reseedH:     $("reseedH"),
    reseedV:     $("reseedV"),
    vscale: $("vscale"), vscaleV: $("vscaleV"),
    dtadv:  $("dtadv"),  dtadvV:  $("dtadvV"),
    bands:  $("bands"),  bandsV:  $("bandsV"),
    speed:  $("speed"),  speedV:  $("speedV"),
    rT: $("rT"), rSp: $("rSp"),
    rMass: $("rMass"), rVar: $("rVar"), rVarF: $("rVarF"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const phiCtx   = els.phiCanvas.getContext("2d");
const speedCtx = els.speedCanvas.getContext("2d");
const vxCtx    = els.vxCanvas.getContext("2d");
const dyeCtx   = els.dyeCanvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let vScale = parseFloat(els.vscale.value);
let dtAdv  = parseFloat(els.dtadv.value);
let bands  = parseInt(els.bands.value, 10);
let territorySeed = 1;
let initialMass = 1;
let initialVar = 1;

function showError(msg) { els.errSlot.innerHTML = `<div class="error">${msg}</div>`; }

function reseedDye(direction) {
    if (!sim) return;
    if (direction === "v") sim.seed_dye_stripes_vertical(bands);
    else sim.seed_dye_stripes(bands);
    initialMass = sim.dye_mass();
    initialVar  = Math.max(sim.dye_variance(), 1e-12);
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmCoupledR17(
            W, H,
            MOBILITY, KAPPA, CH_DX, CH_DT,
            dtAdv, vScale,
        );
        sim.seed_territory(0.05, 0.0, territorySeed);
        sim.pre_evolve_territory(PRE_EVOLVE);
        reseedDye("h");
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(speed, varRatio) {
    if (speed < 0.01) return "frozen";
    if (varRatio > 0.6) return "dye streaks forming";
    if (varRatio > 0.2) return "dye following walls";
    if (varRatio > 0.05) return "dye smoothing";
    return "dye smoothed to mean";
}

function frame() {
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }

        const f1 = fitCanvas(els.phiCanvas);
        const f2 = fitCanvas(els.speedCanvas);
        const f3 = fitCanvas(els.vxCanvas);
        const f4 = fitCanvas(els.dyeCanvas);

        const w = sim.width;
        const h = sim.height;

        drawField2DDiverging(phiCtx, f1.width, f1.height, sim.phi_field(), w, h, 1.0);
        const sf = sim.speed_field();
        const speedMax = Math.max(0.01, vScale * 1.2);
        drawField2D(speedCtx, f2.width, f2.height, sf, w, h, speedMax);
        drawField2DDiverging(vxCtx, f3.width, f3.height, sim.vx_field(), w, h, speedMax);
        drawField2D(dyeCtx, f4.width, f4.height, sim.dye_field(), w, h, 1.0);

        const meanSpeed = sim.mean_speed();
        const mass = sim.dye_mass();
        const variance = sim.dye_variance();
        const varRatio = variance / initialVar;

        els.rT.textContent = sim.territory_time.toFixed(1);
        els.rSp.textContent = meanSpeed.toFixed(3);
        els.rMass.textContent = (mass / initialMass).toFixed(3);
        els.rVar.textContent = variance.toFixed(4);
        els.rVarF.textContent = (100 * varRatio).toFixed(0) + "%";
        els.rRegime.textContent = regimeLabel(meanSpeed, varRatio);
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
    reseedDye("h");
});
els.reseedH.addEventListener("click", () => reseedDye("h"));
els.reseedV.addEventListener("click", () => reseedDye("v"));
els.vscale.addEventListener("input", () => {
    vScale = parseFloat(els.vscale.value);
    els.vscaleV.textContent = vScale.toFixed(1);
    if (sim) sim.set_v_scale(vScale);
});
els.dtadv.addEventListener("input", () => {
    dtAdv = parseFloat(els.dtadv.value);
    els.dtadvV.textContent = dtAdv.toFixed(2);
    if (sim) sim.set_dt_adv(dtAdv);
});
els.bands.addEventListener("input", () => {
    bands = parseInt(els.bands.value, 10);
    els.bandsV.textContent = String(bands);
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
