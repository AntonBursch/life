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
