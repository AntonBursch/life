//! 1D diffusion on a uniform grid.
//!
//! The continuous equation is
//!
//! ```text
//!   ∂φ/∂t = D · ∂²φ/∂x²
//! ```
//!
//! Discretised with the simplest stable scheme (forward-time, centered-space):
//!
//! ```text
//!   φ_i^{n+1} = φ_i^n + dt · D · (φ_{i-1} - 2 φ_i + φ_{i+1}) / dx²
//! ```
//!
//! Stability requires `D · dt / dx² <= 0.5`. The constructor checks this and
//! returns an error if violated; `step()` assumes it.

use core::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryCondition {
    /// Zero-flux at both edges (Neumann). Nothing leaves the box.
    /// This is the R1 default — it lets us watch a closed system flatten.
    ZeroFlux,
    /// Dirichlet zero — both edges held at 0. Useful as a sanity check; a
    /// hot spot in the middle decays toward zero everywhere.
    Dirichlet,
}

#[derive(Debug)]
pub enum DiffusionError {
    NonPositiveSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { ratio: f64 },
}

impl fmt::Display for DiffusionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DiffusionError::NonPositiveSize => {
                write!(f, "grid size must be at least 2 cells")
            }
            DiffusionError::NonPositive { name, value } => {
                write!(f, "{name} must be > 0, got {value}")
            }
            DiffusionError::Unstable { ratio } => write!(
                f,
                "stability violated: D*dt/dx^2 = {ratio} (must be <= 0.5)"
            ),
        }
    }
}

impl std::error::Error for DiffusionError {}

/// 1D diffusion field on a uniform grid.
///
/// Holds the field `phi`, the parameters, and a scratch buffer for the next
/// step. Allocates once on construction and reuses on every step.
#[derive(Debug, Clone)]
pub struct Diffusion1D {
    phi: Vec<f64>,
    next: Vec<f64>,
    diffusivity: f64,
    dx: f64,
    dt: f64,
    boundary: BoundaryCondition,
    time: f64,
    steps: u64,
}

impl Diffusion1D {
    /// Build a new field of `n` cells initialised to zero.
    ///
    /// Returns an error if any parameter is non-positive or the chosen `dt`
    /// violates the explicit-scheme stability bound `D*dt/dx^2 <= 0.5`.
    pub fn new(
        n: usize,
        diffusivity: f64,
        dx: f64,
        dt: f64,
        boundary: BoundaryCondition,
    ) -> Result<Self, DiffusionError> {
        if n < 2 {
            return Err(DiffusionError::NonPositiveSize);
        }
        if !(diffusivity > 0.0) {
            return Err(DiffusionError::NonPositive {
                name: "diffusivity",
                value: diffusivity,
            });
        }
        if !(dx > 0.0) {
            return Err(DiffusionError::NonPositive { name: "dx", value: dx });
        }
        if !(dt > 0.0) {
            return Err(DiffusionError::NonPositive { name: "dt", value: dt });
        }
        let ratio = diffusivity * dt / (dx * dx);
        if ratio > 0.5 {
            return Err(DiffusionError::Unstable { ratio });
        }
        Ok(Self {
            phi: vec![0.0; n],
            next: vec![0.0; n],
            diffusivity,
            dx,
            dt,
            boundary,
            time: 0.0,
            steps: 0,
        })
    }

    /// Number of cells.
    pub fn len(&self) -> usize {
        self.phi.len()
    }

    pub fn is_empty(&self) -> bool {
        self.phi.is_empty()
    }

    /// Read-only view of the field.
    pub fn phi(&self) -> &[f64] {
        &self.phi
    }

    /// Mutable access to the initial field. Use this to set boundary
    /// conditions or seed an initial pulse before stepping.
    pub fn phi_mut(&mut self) -> &mut [f64] {
        &mut self.phi
    }

    pub fn diffusivity(&self) -> f64 {
        self.diffusivity
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

    /// Total of phi summed across all cells. Under zero-flux boundaries this
    /// should be conserved to floating-point precision.
    pub fn total(&self) -> f64 {
        self.phi.iter().sum()
    }

    /// Root-mean-square spread of mass around the mean position.
    /// For a delta-function initial condition, this should grow like `sqrt(2*D*t)`.
    pub fn rms_spread(&self) -> f64 {
        let total = self.total();
        if total <= 0.0 {
            return 0.0;
        }
        // mean position (cell index, not physical coordinate)
        let mean: f64 = self
            .phi
            .iter()
            .enumerate()
            .map(|(i, &v)| v * i as f64)
            .sum::<f64>()
            / total;
        let var: f64 = self
            .phi
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let d = i as f64 - mean;
                v * d * d
            })
            .sum::<f64>()
            / total;
        var.sqrt() * self.dx
    }

    /// Place a unit "delta" pulse at the centre of the grid. Exact mass is 1.
    pub fn seed_centre_pulse(&mut self) {
        for v in self.phi.iter_mut() {
            *v = 0.0;
        }
        let mid = self.phi.len() / 2;
        // unit mass concentrated in one cell — height is 1/dx so that
        // integrating gives 1.
        self.phi[mid] = 1.0 / self.dx;
        self.time = 0.0;
        self.steps = 0;
    }

    /// Advance one time step.
    pub fn step(&mut self) {
        let n = self.phi.len();
        let alpha = self.diffusivity * self.dt / (self.dx * self.dx);

        // Interior cells: standard 3-point stencil for the Laplacian.
        for i in 1..n - 1 {
            let lap = self.phi[i - 1] - 2.0 * self.phi[i] + self.phi[i + 1];
            self.next[i] = self.phi[i] + alpha * lap;
        }

        // Boundaries.
        match self.boundary {
            BoundaryCondition::ZeroFlux => {
                // ghost cell on each side mirrors its inner neighbour, so
                // d phi/dx = 0 at the wall. The Laplacian at the boundary
                // uses (phi[1] - phi[0]) / dx^2 effectively.
                let lap0 = -self.phi[0] + self.phi[1];
                self.next[0] = self.phi[0] + alpha * lap0;
                let lapn = self.phi[n - 2] - self.phi[n - 1];
                self.next[n - 1] = self.phi[n - 1] + alpha * lapn;
            }
            BoundaryCondition::Dirichlet => {
                self.next[0] = 0.0;
                self.next[n - 1] = 0.0;
            }
        }

        std::mem::swap(&mut self.phi, &mut self.next);
        self.time += self.dt;
        self.steps += 1;
    }

    /// Run `n` steps in a row.
    pub fn step_many(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    #[test]
    fn rejects_unstable_dt() {
        // alpha = 1.0 * 1.0 / (1.0 * 1.0) = 1.0 > 0.5, must fail
        let r = Diffusion1D::new(10, 1.0, 1.0, 1.0, BoundaryCondition::ZeroFlux);
        assert!(matches!(r, Err(DiffusionError::Unstable { .. })));
    }

    #[test]
    fn zero_flux_conserves_mass() {
        let mut sim =
            Diffusion1D::new(101, 0.5, 1.0, 0.5, BoundaryCondition::ZeroFlux).unwrap();
        sim.seed_centre_pulse();
        let initial_total = sim.total();
        for _ in 0..10_000 {
            sim.step();
        }
        // mass should be conserved under zero-flux to fp tolerance
        assert!(
            approx(sim.total(), initial_total, 1e-9),
            "total drifted: initial {initial_total} -> {}",
            sim.total()
        );
    }

    #[test]
    fn diffusion_flattens_in_time() {
        let mut sim =
            Diffusion1D::new(101, 0.5, 1.0, 0.5, BoundaryCondition::ZeroFlux).unwrap();
        sim.seed_centre_pulse();
        let peak_initial = sim.phi().iter().cloned().fold(f64::MIN, f64::max);
        for _ in 0..2_000 {
            sim.step();
        }
        let peak_after = sim.phi().iter().cloned().fold(f64::MIN, f64::max);
        assert!(
            peak_after < peak_initial,
            "peak did not decrease: {peak_initial} -> {peak_after}"
        );
    }

    #[test]
    fn rms_spread_grows_like_sqrt_t() {
        // The single-most-important check: variance should grow linearly in t.
        // For a delta-function initial condition, sigma(t) = sqrt(2 D t).
        let d = 0.5;
        let dx = 1.0;
        let dt = 0.5;
        let mut sim = Diffusion1D::new(401, d, dx, dt, BoundaryCondition::ZeroFlux).unwrap();
        sim.seed_centre_pulse();

        // Step long enough that the analytic Gaussian is the right limit, but
        // far short of the box edges so the boundaries don't matter.
        sim.step_many(100);
        let sigma_a = sim.rms_spread();
        let t_a = sim.time();

        sim.step_many(300); // 4x total time
        let sigma_b = sim.rms_spread();
        let t_b = sim.time();

        // sqrt(t) scaling: sigma_b / sigma_a should be sqrt(t_b / t_a)
        let ratio = sigma_b / sigma_a;
        let expected = (t_b / t_a).sqrt();
        assert!(
            (ratio - expected).abs() / expected < 0.05,
            "sqrt(t) scaling broke: sigma_a={sigma_a} sigma_b={sigma_b} ratio={ratio} expected={expected}"
        );
    }
}
