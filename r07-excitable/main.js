// R7 — Barkley excitable medium. Two fields u (fast) and v (slow).
import init, { WasmBarkley2D } from "../viewer/pkg/flow_wasm.js";
import { fitCanvas } from "../viewer/canvas.js";

const W = 128;
const H = 128;
const DX = 0.4;
const DT = 0.01;
const DIFFUSION = 1.0;
const A_VAL = 0.75;

const $ = (id) => document.getElementById(id);
const els = {
    canvas:   $("field"),
    play:     $("play"),
    reset:    $("reset"),
    spiral:   $("spiral"),
    speed:    $("speed"),
    speedV:   $("speedV"),
    bval:     $("bval"),
    bvalV:    $("bvalV"),
    epsval:   $("epsval"),
    epsvalV:  $("epsvalV"),
    rT:       $("rT"),
    rMean:    $("rMean"),
    rVar:     $("rVar"),
    rFrac:    $("rFrac"),
    rRegime:  $("rRegime"),
    errSlot:  $("errSlot"),
};

const ctx = els.canvas.getContext("2d");

let sim = null;
let running = true;
let stepsPerFrame = parseInt(els.speed.value, 10);
let bValue = parseFloat(els.bval.value);
let epsValue = parseFloat(els.epsval.value);

function showError(msg) {
    els.errSlot.innerHTML = `<div class="error">${msg}</div>`;
}

function buildSim() {
    try {
        if (sim) sim.free?.();
        sim = new WasmBarkley2D(W, H, DIFFUSION, A_VAL, bValue, epsValue, DX, DT);
        els.errSlot.innerHTML = "";
    } catch (e) {
        showError(`could not build sim: ${e.message ?? e}`);
        sim = null;
    }
}

// Custom 2-field renderer:
//   u (fast, excited) -> hot ramp dark -> red -> yellow -> white.
//   v (slow, refractory) -> cool blue tail subtracted from u channel.
// Result: fronts are bright, refractory tails are blue, rest is near-black.
function drawExcitable(ctx, w, h, u, v, gw, gh) {
    if (!u.length || !v.length) {
        ctx.clearRect(0, 0, w, h);
        return;
    }
    if (!ctx._ebacking || ctx._ebacking.width !== gw || ctx._ebacking.height !== gh) {
        const off = document.createElement("canvas");
        off.width = gw;
        off.height = gh;
        ctx._ebacking = off;
        ctx._ebackingCtx = off.getContext("2d");
        ctx._ebackingImg = ctx._ebackingCtx.createImageData(gw, gh);
    }
    const img = ctx._ebackingImg;
    const data = img.data;
    for (let j = 0; j < gh; j++) {
        const row = j * gw;
        for (let i = 0; i < gw; i++) {
            let uu = u[row + i];
            let vv = v[row + i];
            if (uu < 0) uu = 0; else if (uu > 1) uu = 1;
            if (vv < 0) vv = 0; else if (vv > 1) vv = 1;
            // Hot ramp on u: dark -> red -> yellow -> white.
            // Three-stop piecewise linear.
            let r, g, b;
            if (uu < 0.33) {
                const s = uu / 0.33;
                r = Math.round(s * 220);
                g = Math.round(s * 30);
                b = Math.round(s * 20);
            } else if (uu < 0.66) {
                const s = (uu - 0.33) / 0.33;
                r = Math.round(220 + s * 35);
                g = Math.round(30 + s * 190);
                b = Math.round(20 + s * 30);
            } else {
                const s = (uu - 0.66) / 0.34;
                r = 255;
                g = Math.round(220 + s * 35);
                b = Math.round(50 + s * 205);
            }
            // Refractory tail tint: where u is low but v is high, add blue.
            const tail = Math.max(0, vv - uu);
            if (tail > 0.02) {
                const t = Math.min(1, tail * 1.6);
                r = Math.round(r * (1 - t * 0.85));
                g = Math.round(g * (1 - t * 0.4) + t * 40);
                b = Math.round(b + t * (180 - b));
            }
            const o = (row + i) * 4;
            data[o] = r;
            data[o + 1] = g;
            data[o + 2] = b;
            data[o + 3] = 255;
        }
    }
    ctx._ebackingCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(ctx._ebacking, 0, 0, w, h);
}

function regimeLabel(frac, varU) {
    if (varU < 1e-4 && frac < 0.001) return "rest";
    if (frac < 0.02) return "subthreshold / decaying";
    if (varU < 0.05) return "single wave";
    return "spiral / sustained";
}

function frame() {
    const { width, height } = fitCanvas(els.canvas);
    if (sim) {
        if (running && stepsPerFrame > 0) {
            sim.step_many(stepsPerFrame);
        }
        const u = sim.u_field();
        const v = sim.v_field();
        drawExcitable(ctx, width, height, u, v, sim.width, sim.height);

        els.rT.textContent = sim.time.toFixed(1);
        els.rMean.textContent = sim.mean_u.toFixed(3);
        els.rVar.textContent = sim.variance_u.toExponential(2);
        els.rFrac.textContent = sim.excited_fraction.toFixed(3);
        els.rRegime.textContent = regimeLabel(
            sim.excited_fraction,
            sim.variance_u,
        );
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

els.spiral.addEventListener("click", () => {
    if (sim) sim.seed_spiral();
});

els.speed.addEventListener("input", () => {
    stepsPerFrame = parseInt(els.speed.value, 10);
    els.speedV.textContent = String(stepsPerFrame);
});

els.bval.addEventListener("input", () => {
    bValue = parseFloat(els.bval.value);
    els.bvalV.textContent = bValue.toFixed(3);
    if (sim) sim.set_b(bValue);
});

els.epsval.addEventListener("input", () => {
    epsValue = parseFloat(els.epsval.value);
    els.epsvalV.textContent = epsValue.toFixed(3);
    if (sim) sim.set_eps(epsValue);
});

// Click-to-kick: suprathreshold disc at the click location.
els.canvas.addEventListener("click", (ev) => {
    if (!sim) return;
    const rect = els.canvas.getBoundingClientRect();
    const fx = (ev.clientX - rect.left) / rect.width;
    const fy = (ev.clientY - rect.top) / rect.height;
    const cx = Math.max(0, Math.min(sim.width - 1, Math.floor(fx * sim.width)));
    const cy = Math.max(0, Math.min(sim.height - 1, Math.floor(fy * sim.height)));
    sim.kick(cx, cy, 6, 0.8);
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
