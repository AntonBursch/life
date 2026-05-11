//! R10 coupling primitives — operators that turn one substrate's
//! state into another substrate's control field.
//!
//! Each function here is intentionally tiny. They are the joints
//! between substrates: an excitable activator drives a per-cell
//! Kuramoto coupling, a phase field drives a reaction rate, etc.
//! Composition lives here so the substrate modules stay clean.

/// Smoothstep on Hermite cubic: maps `t` to a 0..1 ease curve.
#[inline]
fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    if edge1 <= edge0 {
        // Degenerate: collapse to a hard step at edge0.
        return if x >= edge0 { 1.0 } else { 0.0 };
    }
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Turn an excitable activator field `u` into a per-cell coupling
/// field, written into `out`.
///
/// Where the tissue is at rest (`u <= threshold - sharpness`) the
/// coupling is `k_lo`. Where the tissue is firing
/// (`u >= threshold + sharpness`) the coupling is `k_hi`. In the
/// transition band the coupling smoothsteps from `k_lo` to `k_hi`.
///
/// `sharpness` is the half-width of the transition band; smaller
/// values give a crisper gate. Must be `> 0`.
///
/// Returns Err if sizes mismatch, `sharpness <= 0`, or either
/// coupling endpoint is negative.
pub fn excitable_gate(
    u: &[f64],
    k_lo: f64,
    k_hi: f64,
    threshold: f64,
    sharpness: f64,
    out: &mut [f64],
) -> Result<(), CouplingError> {
    if u.len() != out.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if !(sharpness > 0.0) {
        return Err(CouplingError::BadParam {
            name: "sharpness",
            value: sharpness,
        });
    }
    if k_lo < 0.0 {
        return Err(CouplingError::BadParam {
            name: "k_lo",
            value: k_lo,
        });
    }
    if k_hi < 0.0 {
        return Err(CouplingError::BadParam {
            name: "k_hi",
            value: k_hi,
        });
    }
    let lo = threshold - sharpness;
    let hi = threshold + sharpness;
    for (ui, oi) in u.iter().zip(out.iter_mut()) {
        let g = smoothstep(lo, hi, *ui);
        *oi = k_lo + (k_hi - k_lo) * g;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CouplingError {
    SizeMismatch,
    BadParam { name: &'static str, value: f64 },
}

impl std::fmt::Display for CouplingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SizeMismatch => write!(f, "input and output fields differ in length"),
            Self::BadParam { name, value } => {
                write!(f, "{} is invalid (got {})", name, value)
            }
        }
    }
}

/// Map a phase field `theta` to a per-cell scalar control field, written
/// into `out`. Uses `g = 0.5 * (1 + cos(theta))`, which is `1` at the
/// peak of the cycle (`theta = 0`) and `0` at the trough
/// (`theta = ±pi`). The output is then `lo + (hi - lo) * g`.
///
/// This is the natural way to turn a phase clock into a rate
/// modulator: the modulated parameter oscillates between `lo` and `hi`
/// in time with the phase.
pub fn phase_to_scalar_field(
    theta: &[f64],
    lo: f64,
    hi: f64,
    out: &mut [f64],
) -> Result<(), CouplingError> {
    if theta.len() != out.len() {
        return Err(CouplingError::SizeMismatch);
    }
    for (t, o) in theta.iter().zip(out.iter_mut()) {
        let g = 0.5 * (1.0 + t.cos());
        *o = lo + (hi - lo) * g;
    }
    Ok(())
}

/// Map a scalar field whose interesting structure is its
/// *magnitude* (e.g. a phase-separated Cahn-Hilliard order
/// parameter sitting near +/-1 in the bulk and near 0 at the
/// walls) into a per-cell coupling. Cells where `|scalar|` is
/// well above `half_width` get `k_bulk`; cells where `|scalar|`
/// is well below `half_width` get `k_wall`; the transition is a
/// smoothstep of half-width `sharpness` around `half_width`.
///
/// This is the sign-blind cousin of `excitable_gate`. It lets a
/// territory substrate (where the sign of `scalar` distinguishes
/// two phases but both phases are equally "bulk") create
/// coupling islands whose walls are uncoupled.
pub fn bulk_gate(
    scalar: &[f64],
    k_wall: f64,
    k_bulk: f64,
    half_width: f64,
    sharpness: f64,
    out: &mut [f64],
) -> Result<(), CouplingError> {
    if scalar.len() != out.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if !(sharpness > 0.0) {
        return Err(CouplingError::BadParam { name: "sharpness", value: sharpness });
    }
    if !(half_width >= 0.0) {
        return Err(CouplingError::BadParam { name: "half_width", value: half_width });
    }
    if k_wall < 0.0 {
        return Err(CouplingError::BadParam { name: "k_wall", value: k_wall });
    }
    if k_bulk < 0.0 {
        return Err(CouplingError::BadParam { name: "k_bulk", value: k_bulk });
    }
    let lo = half_width - sharpness;
    let hi = half_width + sharpness;
    for (s, o) in scalar.iter().zip(out.iter_mut()) {
        let g = smoothstep(lo, hi, s.abs());
        *o = k_wall + (k_bulk - k_wall) * g;
    }
    Ok(())
}

/// Compute the magnitude of the spatial gradient of a scalar field
/// `phi` on a periodic `width x height` grid, written into `out`.
/// Uses centred differences with grid spacing `dx > 0`. The result
/// is `sqrt((dphi/dx)^2 + (dphi/dy)^2)` per cell.
///
/// This is the "differentiate" primitive of the operator alphabet:
/// it turns *where the bulk lives* into *where the boundaries
/// are*. Combined with downstream maps (linear, threshold, gate)
/// it lets edges drive downstream substrates -- walls of a
/// territory become rails of a chemistry, gradient flux becomes a
/// feed.
pub fn gradient_magnitude(
    phi: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    out: &mut [f64],
) -> Result<(), CouplingError> {
    if phi.len() != out.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if phi.len() != width * height {
        return Err(CouplingError::SizeMismatch);
    }
    if !(dx > 0.0) {
        return Err(CouplingError::BadParam { name: "dx", value: dx });
    }
    let inv_2dx = 0.5 / dx;
    for j in 0..height {
        let jn = if j == 0 { height - 1 } else { j - 1 };
        let js = if j + 1 == height { 0 } else { j + 1 };
        let row = j * width;
        let row_n = jn * width;
        let row_s = js * width;
        for i in 0..width {
            let iw = if i == 0 { width - 1 } else { i - 1 };
            let ie = if i + 1 == width { 0 } else { i + 1 };
            let gx = (phi[row + ie] - phi[row + iw]) * inv_2dx;
            let gy = (phi[row_s + i] - phi[row_n + i]) * inv_2dx;
            out[row + i] = (gx * gx + gy * gy).sqrt();
        }
    }
    Ok(())
}

/// Vector cousin of `gradient_magnitude`: write the per-cell
/// (dphi/dx, dphi/dy) components of a scalar field `phi` on a
/// periodic grid into `out_gx`, `out_gy`. Centred differences
/// with spacing `dx > 0`.
///
/// This is the read-vector primitive of the operator alphabet:
/// where `gradient_magnitude` collapses a gradient to a scalar
/// "wall intensity", `gradient_field` keeps the direction so it
/// can be used as a velocity for transport operators downstream.
pub fn gradient_field(
    phi: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    out_gx: &mut [f64],
    out_gy: &mut [f64],
) -> Result<(), CouplingError> {
    if phi.len() != out_gx.len() || phi.len() != out_gy.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if phi.len() != width * height {
        return Err(CouplingError::SizeMismatch);
    }
    if !(dx > 0.0) {
        return Err(CouplingError::BadParam { name: "dx", value: dx });
    }
    let inv_2dx = 0.5 / dx;
    for j in 0..height {
        let jn = if j == 0 { height - 1 } else { j - 1 };
        let js = if j + 1 == height { 0 } else { j + 1 };
        let row = j * width;
        let row_n = jn * width;
        let row_s = js * width;
        for i in 0..width {
            let iw = if i == 0 { width - 1 } else { i - 1 };
            let ie = if i + 1 == width { 0 } else { i + 1 };
            out_gx[row + i] = (phi[row + ie] - phi[row + iw]) * inv_2dx;
            out_gy[row + i] = (phi[row_s + i] - phi[row_n + i]) * inv_2dx;
        }
    }
    Ok(())
}

/// Transport a scalar `field` by a velocity field `(vx, vy)` for a
/// time step `dt`, on a periodic `width x height` grid with spacing
/// `dx > 0`. Writes the new field into `out`.
///
/// Implementation: semi-Lagrangian with bilinear interpolation.
/// For each cell `(i,j)` we trace back to the position
/// `(i - vx*dt/dx, j - vy*dt/dx)` and sample the old field there.
/// Periodic wrap is exact. Semi-Lagrangian is unconditionally
/// stable, so `dt` is a quality knob, not a stability knob.
///
/// This is the "transport" primitive of the operator alphabet --
/// the first operator that *moves* mass instead of *gating* or
/// *reading* it.
pub fn advect_by(
    field: &[f64],
    vx: &[f64],
    vy: &[f64],
    width: usize,
    height: usize,
    dx: f64,
    dt: f64,
    out: &mut [f64],
) -> Result<(), CouplingError> {
    let n = width * height;
    if field.len() != n || vx.len() != n || vy.len() != n || out.len() != n {
        return Err(CouplingError::SizeMismatch);
    }
    if !(dx > 0.0) {
        return Err(CouplingError::BadParam { name: "dx", value: dx });
    }
    if !(dt >= 0.0) {
        return Err(CouplingError::BadParam { name: "dt", value: dt });
    }
    let w = width as f64;
    let h = height as f64;
    let inv_dx = 1.0 / dx;
    for j in 0..height {
        let row = j * width;
        for i in 0..width {
            // Trace back in index space (one cell = dx).
            let mut x = i as f64 - vx[row + i] * dt * inv_dx;
            let mut y = j as f64 - vy[row + i] * dt * inv_dx;
            // Periodic wrap in continuous index space.
            x = x - (x / w).floor() * w;
            y = y - (y / h).floor() * h;
            let i0 = x.floor() as usize % width;
            let j0 = y.floor() as usize % height;
            let i1 = if i0 + 1 == width { 0 } else { i0 + 1 };
            let j1 = if j0 + 1 == height { 0 } else { j0 + 1 };
            let fx = x - x.floor();
            let fy = y - y.floor();
            let r0 = j0 * width;
            let r1 = j1 * width;
            let f00 = field[r0 + i0];
            let f10 = field[r0 + i1];
            let f01 = field[r1 + i0];
            let f11 = field[r1 + i1];
            let f0 = f00 * (1.0 - fx) + f10 * fx;
            let f1 = f01 * (1.0 - fx) + f11 * fx;
            out[row + i] = f0 * (1.0 - fy) + f1 * fy;
        }
    }
    Ok(())
}

/// Detect rising-edge threshold crossings: write `1` into `out[k]`
/// iff `prev[k] < threshold` and `curr[k] >= threshold`, otherwise
/// `0`. This is a *discretiser*: it turns a continuous field
/// trajectory into a per-cell event mask.
///
/// Operator alphabet category: "discretise". This is the first
/// operator whose output is symbolic (events) rather than
/// continuous (a field). Downstream you can sum events into a
/// counter, latch the time of the most recent event, or feed
/// them into a discrete process.
pub fn threshold_event(
    prev: &[f64],
    curr: &[f64],
    threshold: f64,
    out: &mut [u8],
) -> Result<(), CouplingError> {
    if prev.len() != curr.len() || prev.len() != out.len() {
        return Err(CouplingError::SizeMismatch);
    }
    for k in 0..prev.len() {
        out[k] = if prev[k] < threshold && curr[k] >= threshold { 1 } else { 0 };
    }
    Ok(())
}

/// Leaky integrator over a field. Advances `state` by one Euler
/// step of `dy/dt = input - leak * y`.
///
/// - `leak == 0`: pure accumulator. `state[k] += dt * input[k]`.
///   Useful for total exposure, dose, integrated flux.
/// - `leak  > 0`: low-pass filter / leaky integrate-and-fire.
///   Steady state is `state[k] = input[k] / leak`; the time
///   constant is `tau = 1 / leak`.
///
/// Operator alphabet category: "integrate" (continuous accumulator
/// with optional leak — dual to `threshold_event`, which is the
/// discretise operator).
pub fn integrate_field(
    input: &[f64],
    state: &mut [f64],
    dt: f64,
    leak: f64,
) -> Result<(), CouplingError> {
    if input.len() != state.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if !(dt >= 0.0) {
        return Err(CouplingError::BadParam { name: "dt", value: dt });
    }
    if !(leak >= 0.0) {
        return Err(CouplingError::BadParam { name: "leak", value: leak });
    }
    for k in 0..input.len() {
        state[k] += dt * (input[k] - leak * state[k]);
    }
    Ok(())
}

/// Affine map from a control field `x` into a substrate parameter
/// field `p`, with clamping. Writes `p[k] = clamp(base + gain * x[k], p_min, p_max)`.
///
/// This is the *parametrise* operator: it is how a field becomes a
/// local control over how a substrate behaves. Where every prior
/// operator either reads off a field, transports a field, or
/// reduces a field to events, this one writes back into the
/// parameter that the substrate's dynamics depend on. It is the
/// minimum hook required to close a feedback loop -- the past
/// (carried by `x`, typically a leaky integral) sets the future
/// (carried by `p`, a per-cell parameter that the next step of the
/// substrate will read).
///
/// Operator alphabet category: "parametrise" (the first new
/// category since Phase A; opens Phase C / cybernetic rungs).
pub fn modulate_parameter(
    x: &[f64],
    base: f64,
    gain: f64,
    p_min: f64,
    p_max: f64,
    p: &mut [f64],
) -> Result<(), CouplingError> {
    if x.len() != p.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if !(p_max >= p_min) {
        return Err(CouplingError::BadParam { name: "p_max", value: p_max });
    }
    for k in 0..x.len() {
        let raw = base + gain * x[k];
        p[k] = raw.clamp(p_min, p_max);
    }
    Ok(())
}

/// Per-cell hysteretic latch (Schmitt trigger). Each cell of
/// `state` is updated in place from `input` according to:
///   - if input[k] > set_threshold   -> state[k] = 1.0
///   - if input[k] < reset_threshold -> state[k] = 0.0
///   - otherwise                     -> state[k] unchanged
///
/// `set_threshold` must be greater than or equal to
/// `reset_threshold`. When they are equal this reduces to a plain
/// sign-of-(input - threshold) discretiser. When they differ, the
/// operator has its own state: a cell that has been "set" remains
/// set until input drops below `reset_threshold`, even if input
/// fluctuates between the two thresholds. The operator carries
/// memory of its own across calls -- this is the minimum required
/// for irreversibility / death-as-state.
///
/// Operator alphabet category: "latch" (stateful discretiser; new
/// in Phase D). This is the first operator whose output is not a
/// function of its current input alone -- it is a function of the
/// input *and* the operator's persistent state.
pub fn latch_field(
    state: &mut [f64],
    input: &[f64],
    set_threshold: f64,
    reset_threshold: f64,
) -> Result<(), CouplingError> {
    if state.len() != input.len() {
        return Err(CouplingError::SizeMismatch);
    }
    if !(set_threshold >= reset_threshold) {
        return Err(CouplingError::BadParam {
            name: "set_threshold",
            value: set_threshold,
        });
    }
    for k in 0..state.len() {
        let x = input[k];
        if x > set_threshold {
            state[k] = 1.0;
        } else if x < reset_threshold {
            state[k] = 0.0;
        }
        // Otherwise: hold (cell remains in whatever state it was in).
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_inputs() {
        let u = vec![0.0_f64; 9];
        let mut out_short = vec![0.0_f64; 8];
        assert_eq!(
            excitable_gate(&u, 0.1, 1.0, 0.5, 0.1, &mut out_short),
            Err(CouplingError::SizeMismatch)
        );
        let mut out = vec![0.0_f64; 9];
        assert!(matches!(
            excitable_gate(&u, 0.1, 1.0, 0.5, 0.0, &mut out).unwrap_err(),
            CouplingError::BadParam { name: "sharpness", .. }
        ));
        assert!(matches!(
            excitable_gate(&u, -0.1, 1.0, 0.5, 0.1, &mut out).unwrap_err(),
            CouplingError::BadParam { name: "k_lo", .. }
        ));
        assert!(matches!(
            excitable_gate(&u, 0.1, -1.0, 0.5, 0.1, &mut out).unwrap_err(),
            CouplingError::BadParam { name: "k_hi", .. }
        ));
    }

    #[test]
    fn gate_saturates_at_extremes() {
        let u = vec![-1.0, 0.0, 0.5, 1.0, 2.0];
        let mut out = vec![0.0_f64; 5];
        excitable_gate(&u, 0.2, 3.0, 0.5, 0.1, &mut out).unwrap();
        // Far below threshold -> k_lo.
        assert!((out[0] - 0.2).abs() < 1e-12);
        assert!((out[1] - 0.2).abs() < 1e-12);
        // Exactly at threshold -> midpoint.
        assert!((out[2] - (0.2 + (3.0 - 0.2) * 0.5)).abs() < 1e-12);
        // Well above threshold -> k_hi.
        assert!((out[3] - 3.0).abs() < 1e-12);
        assert!((out[4] - 3.0).abs() < 1e-12);
    }

    #[test]
    fn gate_is_monotone() {
        // For ascending u, the gate output should never decrease.
        let mut u = vec![0.0_f64; 64];
        for (i, ui) in u.iter_mut().enumerate() {
            *ui = -1.0 + 3.0 * (i as f64) / 63.0;
        }
        let mut out = vec![0.0_f64; 64];
        excitable_gate(&u, 0.1, 2.5, 0.4, 0.2, &mut out).unwrap();
        for w in out.windows(2) {
            assert!(w[1] >= w[0] - 1e-12, "non-monotone gate: {} -> {}", w[0], w[1]);
        }
    }

    #[test]
    fn phase_to_scalar_extremes() {
        let theta = vec![0.0, std::f64::consts::PI, -std::f64::consts::PI, 0.5 * std::f64::consts::PI];
        let mut out = vec![0.0_f64; 4];
        phase_to_scalar_field(&theta, 0.02, 0.06, &mut out).unwrap();
        // theta = 0 -> peak -> hi
        assert!((out[0] - 0.06).abs() < 1e-12);
        // theta = +-pi -> trough -> lo
        assert!((out[1] - 0.02).abs() < 1e-12);
        assert!((out[2] - 0.02).abs() < 1e-12);
        // theta = pi/2 -> midpoint
        assert!((out[3] - 0.04).abs() < 1e-12);
    }

    #[test]
    fn phase_to_scalar_size_mismatch() {
        let theta = vec![0.0; 5];
        let mut out = vec![0.0; 4];
        assert_eq!(
            phase_to_scalar_field(&theta, 0.0, 1.0, &mut out),
            Err(CouplingError::SizeMismatch)
        );
    }

    #[test]
    fn bulk_gate_extremes_and_walls() {
        // |scalar|=1 is well above half_width=0.5 -> k_bulk.
        // |scalar|=0 is well below -> k_wall. sign should not matter.
        let scalar = vec![-1.0, 1.0, 0.0, -0.0001];
        let mut out = vec![0.0; 4];
        bulk_gate(&scalar, 0.1, 5.0, 0.5, 0.05, &mut out).unwrap();
        assert!((out[0] - 5.0).abs() < 1e-9);
        assert!((out[1] - 5.0).abs() < 1e-9);
        assert!((out[2] - 0.1).abs() < 1e-9);
        assert!((out[3] - 0.1).abs() < 1e-9);
    }

    #[test]
    fn bulk_gate_rejects_bad_inputs() {
        let s = vec![0.0; 4];
        let mut o = vec![0.0; 4];
        assert!(matches!(
            bulk_gate(&s, 0.0, 1.0, 0.5, 0.0, &mut o),
            Err(CouplingError::BadParam { name: "sharpness", .. })
        ));
        assert!(matches!(
            bulk_gate(&s, 0.0, 1.0, -0.1, 0.05, &mut o),
            Err(CouplingError::BadParam { name: "half_width", .. })
        ));
        assert!(matches!(
            bulk_gate(&s, -1.0, 1.0, 0.5, 0.05, &mut o),
            Err(CouplingError::BadParam { name: "k_wall", .. })
        ));
        let mut wrong = vec![0.0; 3];
        assert_eq!(
            bulk_gate(&s, 0.0, 1.0, 0.5, 0.05, &mut wrong),
            Err(CouplingError::SizeMismatch)
        );
    }

    #[test]
    fn gate_drives_kuramoto_layer_into_partial_sync() {
        // a strong-coupling island; the rest stays weakly coupled.
        // Inside the patch, the local population should synchronise
        // measurably more than the global average.
        use crate::kuramoto::Kuramoto2D;
        let w = 48;
        let h = 48;
        let n = w * h;

        // Build a u-field: 1.0 inside a centred 20x20 patch, 0.0 elsewhere.
        let mut u = vec![0.0_f64; n];
        for j in 14..34 {
            for i in 14..34 {
                u[j * w + i] = 1.0;
            }
        }
        let mut k_field = vec![0.0_f64; n];
        excitable_gate(&u, 0.1, 3.0, 0.5, 0.1, &mut k_field).unwrap();

        let mut sim = Kuramoto2D::new(w, h, 0.0, 0.05).unwrap();
        sim.set_natural_frequencies(0.3, 11);
        sim.randomise_phases(13);

        for _ in 0..5000 {
            sim.step_with_coupling_field(&k_field).unwrap();
        }

        // Compute local r over a 16x16 window in the centre of the patch
        // vs a 16x16 window outside the patch.
        let local_r = |i0: usize, j0: usize| -> f64 {
            let mut cs = 0.0;
            let mut sn = 0.0;
            let theta = sim.theta();
            for j in j0..j0 + 16 {
                for i in i0..i0 + 16 {
                    let t = theta[j * w + i];
                    cs += t.cos();
                    sn += t.sin();
                }
            }
            (cs * cs + sn * sn).sqrt() / 256.0
        };
        let r_in = local_r(16, 16);
        let r_out = local_r(0, 0);
        assert!(
            r_in > r_out + 0.3,
            "gate failed to drive local sync: r_in={} r_out={}",
            r_in, r_out
        );
    }

    #[test]
    fn gradient_magnitude_flat_is_zero() {
        let w = 8;
        let h = 8;
        let phi = vec![0.7_f64; w * h];
        let mut out = vec![1.0_f64; w * h];
        gradient_magnitude(&phi, w, h, 1.0, &mut out).unwrap();
        for v in &out {
            assert!(v.abs() < 1e-12);
        }
    }

    #[test]
    fn gradient_magnitude_linear_ramp() {
        // phi[i,j] = i  -> dphi/dx = 1, dphi/dy = 0, |grad| = 1
        // everywhere except where the periodic wrap kicks in.
        let w = 16;
        let h = 4;
        let mut phi = vec![0.0_f64; w * h];
        for j in 0..h {
            for i in 0..w {
                phi[j * w + i] = i as f64;
            }
        }
        let mut out = vec![0.0_f64; w * h];
        gradient_magnitude(&phi, w, h, 1.0, &mut out).unwrap();
        // Interior columns should read |grad| ~= 1.
        for j in 0..h {
            for i in 1..w - 1 {
                let g = out[j * w + i];
                assert!((g - 1.0).abs() < 1e-12, "g={} at ({},{})", g, i, j);
            }
        }
    }

    #[test]
    fn gradient_magnitude_rejects_bad_inputs() {
        let phi = vec![0.0; 9];
        let mut out = vec![0.0; 8];
        assert_eq!(
            gradient_magnitude(&phi, 3, 3, 1.0, &mut out),
            Err(CouplingError::SizeMismatch)
        );
        let mut out2 = vec![0.0; 9];
        assert!(matches!(
            gradient_magnitude(&phi, 3, 3, 0.0, &mut out2).unwrap_err(),
            CouplingError::BadParam { name: "dx", .. }
        ));
    }

    #[test]
    fn gradient_field_linear_ramp() {
        // phi[i,j] = i  -> dphi/dx = 1, dphi/dy = 0
        let w = 16;
        let h = 4;
        let mut phi = vec![0.0_f64; w * h];
        for j in 0..h {
            for i in 0..w {
                phi[j * w + i] = i as f64;
            }
        }
        let mut gx = vec![0.0_f64; w * h];
        let mut gy = vec![0.0_f64; w * h];
        gradient_field(&phi, w, h, 1.0, &mut gx, &mut gy).unwrap();
        for j in 0..h {
            for i in 1..w - 1 {
                assert!((gx[j * w + i] - 1.0).abs() < 1e-12);
                assert!(gy[j * w + i].abs() < 1e-12);
            }
        }
    }

    #[test]
    fn advect_by_zero_velocity_is_identity() {
        let w = 8;
        let h = 8;
        let mut field = vec![0.0_f64; w * h];
        for k in 0..field.len() {
            field[k] = (k as f64) * 0.1;
        }
        let vx = vec![0.0_f64; w * h];
        let vy = vec![0.0_f64; w * h];
        let mut out = vec![999.0_f64; w * h];
        advect_by(&field, &vx, &vy, w, h, 1.0, 0.5, &mut out).unwrap();
        for k in 0..field.len() {
            assert!((out[k] - field[k]).abs() < 1e-12);
        }
    }

    #[test]
    fn advect_by_uniform_flow_translates() {
        // Uniform vx = 1, dt = 1, dx = 1 -> shift field by one cell to +x
        // (semi-Lagrangian samples at i - 1).
        let w = 8;
        let h = 4;
        let mut field = vec![0.0_f64; w * h];
        for j in 0..h {
            for i in 0..w {
                field[j * w + i] = i as f64;
            }
        }
        let vx = vec![1.0_f64; w * h];
        let vy = vec![0.0_f64; w * h];
        let mut out = vec![0.0_f64; w * h];
        advect_by(&field, &vx, &vy, w, h, 1.0, 1.0, &mut out).unwrap();
        for j in 0..h {
            for i in 0..w {
                // out[i] = field[i - 1 mod w]
                let src = if i == 0 { w - 1 } else { i - 1 };
                let expected = field[j * w + src];
                assert!(
                    (out[j * w + i] - expected).abs() < 1e-9,
                    "out[{},{}]={} expected {}", i, j, out[j * w + i], expected
                );
            }
        }
    }

    #[test]
    fn advect_by_rejects_bad_inputs() {
        let f = vec![0.0_f64; 9];
        let v = vec![0.0_f64; 9];
        let mut out_short = vec![0.0_f64; 8];
        assert_eq!(
            advect_by(&f, &v, &v, 3, 3, 1.0, 0.1, &mut out_short),
            Err(CouplingError::SizeMismatch)
        );
        let mut out = vec![0.0_f64; 9];
        assert!(matches!(
            advect_by(&f, &v, &v, 3, 3, 0.0, 0.1, &mut out).unwrap_err(),
            CouplingError::BadParam { name: "dx", .. }
        ));
        assert!(matches!(
            advect_by(&f, &v, &v, 3, 3, 1.0, -0.1, &mut out).unwrap_err(),
            CouplingError::BadParam { name: "dt", .. }
        ));
    }

    #[test]
    fn threshold_event_detects_rising_edges_only() {
        let prev = vec![0.0, 0.5, 0.6, 1.0, 0.3];
        let curr = vec![0.7, 0.7, 0.5, 0.9, 0.5];
        // threshold = 0.6
        //   0.0 -> 0.7 : rising across 0.6     -> 1
        //   0.5 -> 0.7 : rising across 0.6     -> 1
        //   0.6 -> 0.5 : prev already >=0.6    -> 0
        //   1.0 -> 0.9 : staying above         -> 0
        //   0.3 -> 0.5 : both below            -> 0
        let mut out = vec![9_u8; 5];
        threshold_event(&prev, &curr, 0.6, &mut out).unwrap();
        assert_eq!(out, vec![1, 1, 0, 0, 0]);
    }

    #[test]
    fn threshold_event_rejects_size_mismatch() {
        let prev = vec![0.0; 4];
        let curr = vec![0.0; 5];
        let mut out = vec![0; 4];
        assert_eq!(
            threshold_event(&prev, &curr, 0.5, &mut out),
            Err(CouplingError::SizeMismatch)
        );
    }

    #[test]
    fn integrate_field_pure_accumulator_grows_linearly() {
        // leak=0, constant input=1, dt=0.1, 10 steps -> state=1.0
        let input = vec![1.0; 3];
        let mut state = vec![0.0; 3];
        for _ in 0..10 {
            integrate_field(&input, &mut state, 0.1, 0.0).unwrap();
        }
        for s in &state {
            assert!((s - 1.0).abs() < 1e-12, "state {} != 1.0", s);
        }
    }

    #[test]
    fn integrate_field_leaky_approaches_steady_state() {
        // dy/dt = input - leak*y -> y_inf = input/leak
        // input=2, leak=1 -> y_inf = 2. Use small dt for stability.
        let input = vec![2.0; 1];
        let mut state = vec![0.0; 1];
        for _ in 0..2000 {
            integrate_field(&input, &mut state, 0.01, 1.0).unwrap();
        }
        assert!((state[0] - 2.0).abs() < 1e-6, "got {}", state[0]);
    }

    #[test]
    fn integrate_field_rejects_bad_inputs() {
        let input = vec![0.0; 3];
        let mut state = vec![0.0; 3];
        assert_eq!(
            integrate_field(&input, &mut vec![0.0; 4], 0.1, 0.0),
            Err(CouplingError::SizeMismatch)
        );
        assert!(matches!(
            integrate_field(&input, &mut state, -0.1, 0.0).unwrap_err(),
            CouplingError::BadParam { name: "dt", .. }
        ));
        assert!(matches!(
            integrate_field(&input, &mut state, 0.1, -1.0).unwrap_err(),
            CouplingError::BadParam { name: "leak", .. }
        ));
    }
}
