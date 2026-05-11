// Tiny shared canvas helpers. Vanilla, no deps. ES module.

/**
 * Resize a canvas to its CSS size at devicePixelRatio. Returns the
 * effective drawing size in CSS pixels.
 */
export function fitCanvas(canvas) {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        const ctx = canvas.getContext("2d");
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    return { width: w, height: h };
}

/**
 * Draw a filled "field" plot of a 1D array onto the canvas.
 * Maps values linearly between [0, vmax].
 */
export function drawField1D(ctx, w, h, field, vmax) {
    ctx.clearRect(0, 0, w, h);
    if (!field.length) return;
    const cellW = w / field.length;
    const safeMax = vmax > 0 ? vmax : 1;
    // gradient strip beneath
    for (let i = 0; i < field.length; i++) {
        const t = Math.max(0, Math.min(1, field[i] / safeMax));
        const x = i * cellW;
        // colour from cool (low) to warm (high)
        const r = Math.round(20 + 200 * t);
        const g = Math.round(40 + 80 * t);
        const b = Math.round(120 + 80 * (1 - t));
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(x, 0, cellW + 1, h);
    }
    // outline curve on top so the shape is legible
    ctx.beginPath();
    for (let i = 0; i < field.length; i++) {
        const t = Math.max(0, Math.min(1, field[i] / safeMax));
        const x = i * cellW + cellW / 2;
        const y = h - t * h * 0.95 - 3;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = "rgba(255, 255, 255, 0.85)";
    ctx.lineWidth = 1.5;
    ctx.stroke();
}

/**
 * Draw a 2D scalar field onto the canvas. The field is `gw × gh` cells and
 * is rendered onto a backing canvas of that size with a colormap, then
 * scaled up to the visible canvas with nearest-neighbour interpolation.
 *
 * `vmax` is the upper end of the colour scale. Values >= vmax saturate to
 * the high end of the map; values <= 0 saturate to the low end.
 */
export function drawField2D(ctx, w, h, field, gw, gh, vmax) {
    if (!field.length) {
        ctx.clearRect(0, 0, w, h);
        return;
    }
    if (!ctx._backing || ctx._backing.width !== gw || ctx._backing.height !== gh) {
        const off = document.createElement("canvas");
        off.width = gw;
        off.height = gh;
        ctx._backing = off;
        ctx._backingCtx = off.getContext("2d");
        ctx._backingImg = ctx._backingCtx.createImageData(gw, gh);
    }
    const img = ctx._backingImg;
    const data = img.data;
    const safeMax = vmax > 0 ? vmax : 1;
    for (let i = 0; i < field.length; i++) {
        let t = field[i] / safeMax;
        if (t < 0) t = 0;
        else if (t > 1) t = 1;
        // Magma-ish ramp: black -> purple -> red -> orange -> yellow.
        const r = Math.round(255 * Math.min(1, t * 2.2));
        const g = Math.round(255 * Math.max(0, t * 1.6 - 0.5));
        const b = Math.round(255 * Math.max(0, Math.min(1, t * 2.4) - t * t * 1.8));
        const o = i * 4;
        data[o] = r;
        data[o + 1] = g;
        data[o + 2] = b;
        data[o + 3] = 255;
    }
    ctx._backingCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(ctx._backing, 0, 0, w, h);
}

/**
 * Draw a 2D temperature field with a blue (cold) -> red (hot) ramp.
 * Field values are expected in [0, 1] (0 = cold, 1 = hot).
 *
 * The field is rendered with row 0 at the BOTTOM of the canvas, so that
 * "hot at the bottom" reads correctly visually for Bénard-style convection.
 */
export function drawField2DTemperature(ctx, w, h, field, gw, gh) {
    if (!field.length) {
        ctx.clearRect(0, 0, w, h);
        return;
    }
    if (!ctx._tbacking || ctx._tbacking.width !== gw || ctx._tbacking.height !== gh) {
        const off = document.createElement("canvas");
        off.width = gw;
        off.height = gh;
        ctx._tbacking = off;
        ctx._tbackingCtx = off.getContext("2d");
        ctx._tbackingImg = ctx._tbackingCtx.createImageData(gw, gh);
    }
    const img = ctx._tbackingImg;
    const data = img.data;
    // Render with row 0 at bottom: input row j -> image row (gh-1-j).
    for (let j = 0; j < gh; j++) {
        const srcRow = j * gw;
        const dstRow = (gh - 1 - j) * gw;
        for (let i = 0; i < gw; i++) {
            let t = field[srcRow + i];
            if (t < 0) t = 0;
            else if (t > 1) t = 1;
            // Diverging cool/warm: deep blue at 0, white near 0.5, deep red at 1.
            // Two-piece linear ramp through near-white.
            let r, g, b;
            if (t < 0.5) {
                const s = t * 2.0;            // 0..1
                r = Math.round(35 + s * 220);
                g = Math.round(60 + s * 195);
                b = Math.round(170 + s * 85);
            } else {
                const s = (t - 0.5) * 2.0;    // 0..1
                r = Math.round(255);
                g = Math.round(255 - s * 200);
                b = Math.round(255 - s * 240);
            }
            const o = (dstRow + i) * 4;
            data[o] = r;
            data[o + 1] = g;
            data[o + 2] = b;
            data[o + 3] = 255;
        }
    }
    ctx._tbackingCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(ctx._tbacking, 0, 0, w, h);
}

/**
 * Draw a 2D scalar field with a symmetric diverging ramp:
 * negative values map to deep blue, zero to near-white, positive to deep red.
 * `vmax` sets the saturation magnitude; values outside [-vmax, +vmax] are clamped.
 */
export function drawField2DDiverging(ctx, w, h, field, gw, gh, vmax) {
    if (!field.length) {
        ctx.clearRect(0, 0, w, h);
        return;
    }
    if (!ctx._dbacking || ctx._dbacking.width !== gw || ctx._dbacking.height !== gh) {
        const off = document.createElement("canvas");
        off.width = gw;
        off.height = gh;
        ctx._dbacking = off;
        ctx._dbackingCtx = off.getContext("2d");
        ctx._dbackingImg = ctx._dbackingCtx.createImageData(gw, gh);
    }
    const img = ctx._dbackingImg;
    const data = img.data;
    const m = vmax > 1e-12 ? vmax : 1e-12;
    for (let j = 0; j < gh; j++) {
        const row = j * gw;
        for (let i = 0; i < gw; i++) {
            let v = field[row + i] / m;       // -1..+1
            if (v < -1) v = -1;
            else if (v > 1) v = 1;
            const t = 0.5 * (v + 1);          // 0..1
            let r, g, b;
            if (t < 0.5) {
                const s = t * 2.0;
                r = Math.round(35 + s * 220);
                g = Math.round(60 + s * 195);
                b = Math.round(170 + s * 85);
            } else {
                const s = (t - 0.5) * 2.0;
                r = 255;
                g = Math.round(255 - s * 200);
                b = Math.round(255 - s * 240);
            }
            const o = (row + i) * 4;
            data[o] = r;
            data[o + 1] = g;
            data[o + 2] = b;
            data[o + 3] = 255;
        }
    }
    ctx._dbackingCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(ctx._dbacking, 0, 0, w, h);
}

/**
 * Draw a 2D phase field (values in (-pi, pi]) with a cyclic colormap.
 * Hue cycles once over [-pi, +pi); same colour ↔ same phase.
 */
export function drawField2DPhase(ctx, w, h, field, gw, gh) {
    if (!field.length) {
        ctx.clearRect(0, 0, w, h);
        return;
    }
    if (!ctx._pbacking || ctx._pbacking.width !== gw || ctx._pbacking.height !== gh) {
        const off = document.createElement("canvas");
        off.width = gw;
        off.height = gh;
        ctx._pbacking = off;
        ctx._pbackingCtx = off.getContext("2d");
        ctx._pbackingImg = ctx._pbackingCtx.createImageData(gw, gh);
    }
    const img = ctx._pbackingImg;
    const data = img.data;
    const TAU = Math.PI * 2;
    for (let j = 0; j < gh; j++) {
        const row = j * gw;
        for (let i = 0; i < gw; i++) {
            const t = field[row + i];
            // Map (-pi, pi] -> [0, 1) hue, full saturation/value.
            let hue = (t + Math.PI) / TAU;
            if (hue < 0) hue += 1; else if (hue >= 1) hue -= 1;
            // HSV -> RGB with s=1, v=1.
            const h6 = hue * 6;
            const c = 255;
            const x = Math.round(c * (1 - Math.abs((h6 % 2) - 1)));
            let r, g, b;
            if      (h6 < 1) { r = c; g = x; b = 0; }
            else if (h6 < 2) { r = x; g = c; b = 0; }
            else if (h6 < 3) { r = 0; g = c; b = x; }
            else if (h6 < 4) { r = 0; g = x; b = c; }
            else if (h6 < 5) { r = x; g = 0; b = c; }
            else             { r = c; g = 0; b = x; }
            const o = (row + i) * 4;
            data[o] = r;
            data[o + 1] = g;
            data[o + 2] = b;
            data[o + 3] = 255;
        }
    }
    ctx._pbackingCtx.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(ctx._pbacking, 0, 0, w, h);
}
