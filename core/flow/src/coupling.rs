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
}
