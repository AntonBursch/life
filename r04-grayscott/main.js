// R4 — Gray-Scott reaction-diffusion. 2D, two coupled fields, F/k slice.
import init, { WasmGrayScott2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas, drawField2D } from "../viewer/canvas.js";

const W = 160;
const H = 160;
const DU = 0.16;
const DV = 0.08;
const DX = 1.0;
const DT = 1.0;
const VMAX = 0.45;

const PRESETS = {
    coral: { feed: 0.0545, kill: 0.062 },
    spots: { feed: 0.0367, kill: 0.0649 },
    maze: { feed: 0.029, kill: 0.057 },
    uskate: { feed: 0.062, kill: 0.0609 },
    soup: { feed: 0.020, kill: 0.05 },
};

const $ = (id) => document.getElementById(id);
const els = {
    canvas: $("field"),
    play: $("play"),
    reset: $("reset"),
    seed: $("seed"),
    speed: $("speed"),
    speedV: $("speedV"),
    feed: $("feed"),
    feedV: $("feedV"),
    kill: $("kill"),
    killV: $("killV"),
    rT: $("rT"),
    rMean: $("rMean"),
    rMax: $("rMax"),
    rVar: $("rVar"),
    rRegime: $("rRegime"),
    errSlot: $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let feed = parseFloat(els.feed.value);
let kill = parseFloat(els.kill.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmGrayScott2D(W, H, DU, DV, feed, kill, DX, DT);
        sim.seed_blob(W >> 1, H >> 1, 10);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

function regimeLabel(meanV, varV) {
    if (meanV < 0.005 && varV < 1e-5) return "soup (V decayed)";
    if (varV < 1e-4) return "near-uniform";
    if (varV < 1e-3) return "weak pattern";
    return "pattern";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const v = sim.v_field();
        drawField2D(ctx, width, height, v, sim.width, sim.height, VMAX);

        els.rT.textContent = sim.time.toFixed(0);
        const meanV = sim.mean_v;
        const maxV = sim.max_v;
        const varV = sim.var_v;
        els.rMean.textContent = meanV.toFixed(4);
        els.rMax.textContent = maxV.toFixed(4);
        els.rVar.textContent = varV.toExponential(2);
        els.rRegime.textContent = regimeLabel(meanV, varV);
    } else {
        ctx.clearRect(0, 0, width, height);
    }
    requestAnimationFrame(frame);
}

// --- wiring ---------------------------------------------------------------

els.play.addEventListener("click", () => {
    running = !running;
    els.play.textContent = running ? "⏸ Pause" : "▶ Play";
});

els.reset.addEventListener("click", () => {
    if (sim) sim.reset();
});

els.seed.addEventListener("click", () => {
    if (sim) sim.seed_blob(W >> 1, H >> 1, 10);
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.feed.addEventListener("input", () => {
    feed = parseFloat(els.feed.value);
    els.feedV.textContent = feed.toFixed(4);
    if (sim) sim.set_feed(feed);
});

els.kill.addEventListener("input", () => {
    kill = parseFloat(els.kill.value);
    els.killV.textContent = kill.toFixed(4);
    if (sim) sim.set_kill(kill);
});

document.querySelectorAll("button[data-preset]").forEach((btn) => {
    btn.addEventListener("click", () => {
        const p = PRESETS[btn.dataset.preset];
        if (!p) return;
        feed = p.feed;
        kill = p.kill;
        els.feed.value = String(feed);
        els.kill.value = String(kill);
        els.feedV.textContent = feed.toFixed(4);
        els.killV.textContent = kill.toFixed(4);
        if (sim) {
            sim.reset();
            sim.set_feed(feed);
            sim.set_kill(kill);
            sim.seed_blob(W >> 1, H >> 1, 10);
        }
    });
});

els.canvas.addEventListener("click", (ev) => {
    if (!sim) return;
    const rect = els.canvas.getBoundingClientRect();
    const x = ev.clientX - rect.left;
    const y = ev.clientY - rect.top;
    const cx = Math.max(0, Math.min(W - 1, Math.floor((x / rect.width) * W)));
    const cy = Math.max(0, Math.min(H - 1, Math.floor((y / rect.height) * H)));
    sim.seed_blob(cx, cy, 6);
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
