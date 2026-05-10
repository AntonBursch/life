//! 1D advection-diffusion.
//!
//! Same field as `diffusion::Diffusion1D`, but the medium itself is moving
//! with velocity `v`. The continuous equation is
//!
//! ```text
//!   ∂φ/∂t = D · ∂²φ/∂x²  −  v · ∂φ/∂x
//! ```
//!
//! Two terms now compete:
//!
//! - the **diffusion** term (D · ∂²φ/∂x²) spreads gradients out, exactly
//!   like R1/R2.
//! - the **advection** term (−v · ∂φ/∂x) drifts the field downstream
//!   without changing its shape.
//!
//! Their relative strength is captured by the **Péclet number**:
//!
//! ```text
//!   Pe = v · L / D
//! ```
//!
//! At small `Pe` the field smooths and looks like R2 (linear steady state).
//! At large `Pe` advection wins: the field stays close to the inflow value
//! across most of the box and only drops sharply near the outflow wall.
//!
//! ## Discretisation
//!
//! - Diffusion: standard 3-point central difference (same as R1/R2).
//! - Advection: **first-order upwind**. The simplest scheme that stays
//!   stable for advection: read the gradient from the upwind side. This
//!   adds a small amount of numerical diffusion proportional to `|v| dx`,
//!   which is fine for this rung — we are after the *qualitative* contrast
//!   between Pe regimes, not a quantitative match.
//!
//! Stability requires both
//!
//! ```text
//!   D dt / dx²  ≤ 0.5         (diffusion CFL)
//!   |v| dt / dx ≤ 1            (advection CFL)
//! ```
//!
//! and conservatively their sum stays under 1 for the combined explicit
//! step. The constructor enforces this.

use crate::diffusion::BoundaryCondition;
use core::fmt;

#[derive(Debug)]
pub enum AdvDiffError {
    NonPositiveSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { ratio: f64 },
}

impl fmt::Display for AdvDiffError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AdvDiffError::NonPositiveSize => {
                write!(f, "grid size must be at least 2 cells")
            }
            AdvDiffError::NonPositive { name, value } => {
                write!(f, "{name} must be > 0, got {value}")
            }
            AdvDiffError::Unstable { ratio } => write!(
                f,
                "stability violated: D dt / dx^2 + |v| dt / dx = {ratio} (must be <= 1)"
            ),
        }
    }
}

impl std::error::Error for AdvDiffError {}

#[derive(Debug, Clone)]
pub struct AdvectionDiffusion1D {
    phi: Vec<f64>,
    next: Vec<f64>,
    diffusivity: f64,
    velocity: f64,
    dx: f64,
    dt: f64,
    boundary: BoundaryCondition,
    time: f64,
    steps: u64,
}

impl AdvectionDiffusion1D {
    pub fn new(
        n: usize,
        diffusivity: f64,
        velocity: f64,
        dx: f64,
        dt: f64,
        boundary: BoundaryCondition,
    ) -> Result<Self, AdvDiffError> {
        if n < 2 {
            return Err(AdvDiffError::NonPositiveSize);
        }
        if !(diffusivity > 0.0) {
            return Err(AdvDiffError::NonPositive {
                name: "diffusivity",
                value: diffusivity,
            });
        }
        if !(dx > 0.0) {
            return Err(AdvDiffError::NonPositive { name: "dx", value: dx });
        }
        if !(dt > 0.0) {
            return Err(AdvDiffError::NonPositive { name: "dt", value: dt });
        }
        let alpha = diffusivity * dt / (dx * dx);
        let beta = velocity.abs() * dt / dx;
        // Conservative combined stability bound for explicit upwind +
        // central diffusion. Both individual conditions are subsumed by
        // (alpha + beta) <= 1, with strict slack when alpha <= 0.5.
        let ratio = alpha + beta;
        if ratio > 1.0 || alpha > 0.5 {
            return Err(AdvDiffError::Unstable { ratio });
        }
        Ok(Self {
            phi: vec![0.0; n],
            next: vec![0.0; n],
            diffusivity,
            velocity,
            dx,
            dt,
            boundary,
            time: 0.0,
            steps: 0,
        })
    }

    pub fn len(&self) -> usize {
        self.phi.len()
    }

    pub fn is_empty(&self) -> bool {
        self.phi.is_empty()
    }

    pub fn phi(&self) -> &[f64] {
        &self.phi
    }

    pub fn phi_mut(&mut self) -> &mut [f64] {
        &mut self.phi
    }

    pub fn diffusivity(&self) -> f64 {
        self.diffusivity
    }

    pub fn velocity(&self) -> f64 {
        self.velocity
    }

    pub fn dx(&self) -> f64 {
        self.dx
    }

    pub fn dt(&self) -> f64 {
        self.dt
    }

    pub fn time(&self) -> f64 {
        self.time
    }

    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Péclet number Pe = v · L / D. Sign tracks `v`. Magnitude tells you
    /// which regime the field is in.
    pub fn peclet(&self) -> f64 {
        let l = (self.phi.len() - 1) as f64 * self.dx;
        self.velocity * l / self.diffusivity
    }

    /// Total mass on the grid.
    pub fn total(&self) -> f64 {
        self.phi.iter().sum()
    }

    /// Advance one step.
    pub fn step(&mut self) {
        let n = self.phi.len();
        let alpha = self.diffusivity * self.dt / (self.dx * self.dx);
        let v = self.velocity;
        let dt = self.dt;
        let dx = self.dx;

        for i in 1..n - 1 {
            let lap = self.phi[i - 1] - 2.0 * self.phi[i] + self.phi[i + 1];
            // Upwind advection: read the gradient from the side the wind
            // is coming from.
            let adv = if v >= 0.0 {
                -v * (self.phi[i] - self.phi[i - 1]) / dx
            } else {
                -v * (self.phi[i + 1] - self.phi[i]) / dx
            };
            self.next[i] = self.phi[i] + alpha * lap + dt * adv;
        }

        match self.boundary {
            BoundaryCondition::ZeroFlux => {
                let lap0 = -self.phi[0] + self.phi[1];
                self.next[0] = self.phi[0] + alpha * lap0;
                let lapn = self.phi[n - 2] - self.phi[n - 1];
                self.next[n - 1] = self.phi[n - 1] + alpha * lapn;
            }
            BoundaryCondition::Dirichlet => {
                self.next[0] = 0.0;
                self.next[n - 1] = 0.0;
            }
            BoundaryCondition::FixedPair { left, right } => {
                self.next[0] = left;
                self.next[n - 1] = right;
            }
        }

        std::mem::swap(&mut self.phi, &mut self.next);
        self.time += self.dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Total flux at the left interior face (between cells 0 and 1):
    /// `J = v · φ_upwind − D · ∂φ/∂x`. Sign convention: positive means
    /// flow rightward.
    ///
    /// We measure flux at the same kind of cell-face on both sides so the
    /// upwind scheme's discrete conservation law (`J_{i+1/2}` constant at
    /// steady state) gives `flux_left == flux_right` exactly.
    pub fn flux_left(&self) -> f64 {
        let upwind = if self.velocity >= 0.0 { self.phi[0] } else { self.phi[1] };
        let conv = self.velocity * upwind;
        let diff = -self.diffusivity * (self.phi[1] - self.phi[0]) / self.dx;
        conv + diff
    }

    /// Total flux at the right interior face (between cells n-2 and n-1).
    pub fn flux_right(&self) -> f64 {
        let n = self.phi.len();
        let upwind = if self.velocity >= 0.0 {
            self.phi[n - 2]
        } else {
            self.phi[n - 1]
        };
        let conv = self.velocity * upwind;
        let diff = -self.diffusivity * (self.phi[n - 1] - self.phi[n - 2]) / self.dx;
        conv + diff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_combination() {
        // alpha = 0.5 already, plus any non-zero advection should fail.
        let r = AdvectionDiffusion1D::new(
            10,
            1.0,
            0.5,
            1.0,
            1.0,
            BoundaryCondition::FixedPair { left: 1.0, right: 0.0 },
        );
        assert!(matches!(r, Err(AdvDiffError::Unstable { .. })));
    }

    #[test]
    fn zero_velocity_matches_linear_steady_state() {
        // With v=0 the system reduces to R2: linear steady state.
        let n = 101;
        let mut sim = AdvectionDiffusion1D::new(
            n,
            0.5,
            0.0,
            1.0,
            0.5,
            BoundaryCondition::FixedPair { left: 1.0, right: 0.0 },
        )
        .unwrap();
        sim.step_many(40_000);
        let mut max_err = 0.0_f64;
        for (i, &v) in sim.phi().iter().enumerate() {
            let expected = 1.0 - (i as f64) / ((n - 1) as f64);
            max_err = max_err.max((v - expected).abs());
        }
        assert!(
            max_err < 1e-3,
            "v=0 didn't reduce to linear: max err {max_err}"
        );
    }

    #[test]
    fn high_peclet_concentrates_near_inflow() {
        // With v > 0 and the left wall held at 1.0, the steady-state
        // profile bows toward the left — most of the box stays near 1
        // and the drop happens near the right wall.
        let n = 201;
        let mut sim = AdvectionDiffusion1D::new(
            n,
            0.05,
            0.4,
            1.0,
            0.5,
            BoundaryCondition::FixedPair { left: 1.0, right: 0.0 },
        )
        .unwrap();
        // Pe = 0.4 * 200 / 0.05 = 1600 — strongly advection-dominated.
        sim.step_many(20_000);
        // Midpoint should still be very close to 1.
        let mid = sim.phi()[n / 2];
        assert!(
            mid > 0.95,
            "high-Pe profile didn't stay flat near inflow: mid={mid}"
        );
        // Pure linear (R2) would give 0.5 at the midpoint, so 0.95+ is a
        // strong signal advection is winning.
    }

    #[test]
    fn steady_state_total_flux_is_conserved() {
        // At steady state, J(x) = v φ(x) − D ∂φ/∂x is constant in x.
        // Equivalently: flux_left == flux_right.
        let mut sim = AdvectionDiffusion1D::new(
            101,
            0.2,
            0.1,
            1.0,
            0.5,
            BoundaryCondition::FixedPair { left: 1.0, right: 0.0 },
        )
        .unwrap();
        sim.step_many(40_000);
        let jl = sim.flux_left();
        let jr = sim.flux_right();
        assert!(
            (jl - jr).abs() < 5e-4,
            "advection-diffusion flux mismatch: left {jl}, right {jr}"
        );
    }
}
