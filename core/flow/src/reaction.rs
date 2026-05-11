//! Reaction primitives — local-ODE steps on species fields.
//!
//! This module exists because R29 → R27′ named a substrate-honesty
//! rule (see `life/THESIS.md`): every operator that touches the
//! substrate must correspond to a real mechanism in nature.
//! `latch_field` violated that — a Schmitt trigger is electrical
//! engineering, not chemistry. Real biological irreversibility
//! comes from bistable reaction networks whose ODE has two stable
//! fixed points and a finite-cost barrier between them.
//!
//! `react_field` is the operator that does that honestly: it
//! advances a species concentration field by one tick of a given
//! local rate law. Composition with diffusion / advection /
//! parameter-modulation builds everything bigger out of real
//! kinetics.

use crate::coupling::CouplingError;

/// Advance a single species field by one tick of a local ODE,
/// using midpoint (RK2) for stability against stiff rate laws.
///
/// `rate(x)` returns dx/dt at a single cell given that cell's
/// current concentration. The rate function is therefore pure and
/// reaction-only; diffusion and transport happen elsewhere.
///
/// After the update each cell is clamped to `>= 0`; chemical
/// concentrations cannot be negative. There is no upper clamp —
/// runaway is a real result and we want to see it if it happens.
///
/// `dt` must be `> 0`. The caller is responsible for choosing a
/// `dt` small enough for the given rate law; if you need
/// sub-stepping, call `react_field` multiple times with `dt/n`.
pub fn react_field<F>(
    species: &mut [f64],
    rate: F,
    dt: f64,
) -> Result<(), CouplingError>
where
    F: Fn(f64) -> f64,
{
    if !(dt > 0.0) {
        return Err(CouplingError::BadParam {
            name: "dt",
            value: dt,
        });
    }
    for c in species.iter_mut() {
        let k1 = rate(*c);
        let mid = *c + 0.5 * dt * k1;
        let k2 = rate(mid);
        let next = *c + dt * k2;
        *c = if next < 0.0 { 0.0 } else { next };
    }
    Ok(())
}

/// Schlögl-model rate law:
///
/// ```text
/// A + 2X ⇌ 3X          (autocatalysis)
/// X ⇌ B                (decay)
/// ```
///
/// with mass-action kinetics
///
/// ```text
/// dX/dt = k1·A·X²  −  k2·X³  −  k3·X  +  k4·B
/// ```
///
/// `A` and `B` are reservoir concentrations (open boundary —
/// honest about being an open system). For
/// `k1·A = 6, k2 = 1, k3 = 11, k4·B = 6` the rate factors as
/// `−(X−1)(X−2)(X−3)`: stable fixed points at `X = 1` (low)
/// and `X = 3` (high), unstable separatrix at `X = 2`. This is
/// the canonical one-species chemical bistable; it predates and
/// is more honest than any Schmitt trigger.
///
/// Returns dX/dt as a closure of one variable.
#[inline]
pub fn schlogl_rate(k1a: f64, k2: f64, k3: f64, k4b: f64) -> impl Fn(f64) -> f64 {
    move |x: f64| k1a * x * x - k2 * x * x * x - k3 * x + k4b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn react_field_advances_one_species_toward_fixed_point() {
        // Simple linear relaxation dx/dt = -x; equilibrium 0.
        let mut field = vec![1.0; 4];
        for _ in 0..1000 {
            react_field(&mut field, |x| -x, 0.01).unwrap();
        }
        for &c in &field {
            assert!(c.abs() < 1e-3, "expected ~0, got {c}");
        }
    }

    #[test]
    fn react_field_clamps_non_negative() {
        // Forced negative drift: dx/dt = -10; without clamp this
        // would go very negative. With clamp it pins at 0.
        let mut field = vec![0.05];
        for _ in 0..10 {
            react_field(&mut field, |_x| -10.0, 0.1).unwrap();
        }
        assert!(field[0] >= 0.0, "got negative {}", field[0]);
        assert!(field[0] < 1e-9);
    }

    #[test]
    fn schlogl_has_two_stable_attractors() {
        // Bistable parameters: −(X−1)(X−2)(X−3).
        let rate = schlogl_rate(6.0, 1.0, 11.0, 6.0);

        // Start below the separatrix (X = 2): falls to low state at X ≈ 1.
        let mut lo = vec![1.5];
        for _ in 0..5000 {
            react_field(&mut lo, &rate, 0.01).unwrap();
        }
        assert!((lo[0] - 1.0).abs() < 0.05, "low attractor expected ≈1.0, got {}", lo[0]);

        // Start above the separatrix: rises to high state at X ≈ 3.
        let mut hi = vec![2.5];
        for _ in 0..5000 {
            react_field(&mut hi, &rate, 0.01).unwrap();
        }
        assert!((hi[0] - 3.0).abs() < 0.05, "high attractor expected ≈3.0, got {}", hi[0]);
    }

    #[test]
    fn schlogl_irreversible_under_no_forcing() {
        // Once in the high state, a cell stays there forever
        // under the bare rate law. This is what replaces
        // latch_field: irreversibility from chemistry, not from
        // a comparator.
        let rate = schlogl_rate(6.0, 1.0, 11.0, 6.0);
        let mut x = vec![3.0];
        for _ in 0..100_000 {
            react_field(&mut x, &rate, 0.01).unwrap();
        }
        assert!((x[0] - 3.0).abs() < 0.05, "high state lost: {}", x[0]);
    }

    #[test]
    fn react_field_rejects_bad_dt() {
        let mut field = vec![1.0];
        assert!(react_field(&mut field, |_| 0.0, 0.0).is_err());
        assert!(react_field(&mut field, |_| 0.0, -1.0).is_err());
    }
}
