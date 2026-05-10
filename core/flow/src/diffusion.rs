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

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryCondition {
    /// Zero-flux at both edges (Neumann). Nothing leaves the box.
    /// This is the R1 default — it lets us watch a closed system flatten.
    ZeroFlux,
    /// Dirichlet zero — both edges held at 0. Useful as a sanity check; a
    /// hot spot in the middle decays toward zero everywhere.
    Dirichlet,
    /// Both edges clamped to fixed (but possibly different) values.
    /// This is R2: a continuous source on one side and sink on the other.
    /// The field reaches a steady linear profile and stops changing, while
    /// flux continues to flow from `left` to `right` (or vice versa).
    FixedPair { left: f64, right: f64 },
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
            BoundaryCondition::FixedPair { left, right } => {
                self.next[0] = left;
                self.next[n - 1] = right;
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

    /// Flux (rate of stuff crossing a wall, per unit area, per unit time)
    /// at the left boundary. Fick's law: `J = -D * d(phi)/dx`. A positive
    /// value means stuff is flowing into the box from the left wall (so when
    /// the left edge is hotter than the interior, this is positive).
    pub fn flux_left(&self) -> f64 {
        -self.diffusivity * (self.phi[1] - self.phi[0]) / self.dx
    }

    /// Flux at the right boundary. Same sign convention as `flux_left`: a
    /// positive value means stuff is flowing rightward through the wall. At
    /// R2 steady state, `flux_left` and `flux_right` converge to the same
    /// value — energy in equals energy out, and the gradient is *held open*
    /// by the through-flow.
    pub fn flux_right(&self) -> f64 {
        let n = self.phi.len();
        -self.diffusivity * (self.phi[n - 1] - self.phi[n - 2]) / self.dx
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
        // Peak of a delta pulse should decay monotonically toward the mean
        // until the field is essentially flat. Under zero-flux, the mean is
        // total / n, which stays put.
        let mut sim =
            Diffusion1D::new(101, 0.5, 1.0, 0.5, BoundaryCondition::ZeroFlux).unwrap();
        sim.seed_centre_pulse();
        let peak0 = sim.phi().iter().cloned().fold(0.0_f64, f64::max);
        sim.step_many(2_000);
        let peak1 = sim.phi().iter().cloned().fold(0.0_f64, f64::max);
        assert!(peak1 < peak0, "peak did not decay: {peak0} -> {peak1}");

        // After enough steps the spread should approach the half-width
        // of the box (uniform field).
        sim.step_many(20_000);
        let mean = sim.total() / sim.len() as f64;
        let max_dev = sim
            .phi()
            .iter()
            .map(|&v| (v - mean).abs())
            .fold(0.0_f64, f64::max);
        assert!(
            max_dev < 1e-3,
            "field did not flatten enough: max deviation {max_dev}"
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

    #[test]
    fn driven_reaches_linear_steady_state() {
        // Hot left, cold right — the analytic steady state is a straight line
        // from `left` down to `right`. We start from an initially flat zero
        // field and step long enough to settle.
        let n = 101;
        let left = 1.0;
        let right = 0.0;
        let mut sim = Diffusion1D::new(
            n,
            0.5,
            1.0,
            0.5,
            BoundaryCondition::FixedPair { left, right },
        )
        .unwrap();
        // The boundary values are imposed every step, so the initial interior
        // condition is mostly cosmetic — leave it at zero.
        sim.step_many(40_000);

        // Compare to the analytic linear profile.
        let mut max_err = 0.0_f64;
        for (i, &v) in sim.phi().iter().enumerate() {
            let expected = left + (right - left) * (i as f64) / ((n - 1) as f64);
            max_err = max_err.max((v - expected).abs());
        }
        assert!(
            max_err < 1e-3,
            "steady state did not match linear profile: max err {max_err}"
        );
    }

    #[test]
    fn steady_state_left_flux_equals_right_flux() {
        // Once the linear profile has settled, mass crossing the left wall
        // per tick must equal mass crossing the right wall per tick. That is
        // the formal statement of "energy in equals energy out".
        let mut sim = Diffusion1D::new(
            101,
            0.5,
            1.0,
            0.5,
            BoundaryCondition::FixedPair { left: 1.0, right: 0.0 },
        )
        .unwrap();
        sim.step_many(40_000);
        let jl = sim.flux_left();
        let jr = sim.flux_right();
        assert!(
            (jl - jr).abs() < 1e-5,
            "flux mismatch at steady state: left {jl}, right {jr}"
        );
        // And the value should match the analytic answer: J = D * (left - right) / L
        // where L = (n-1) * dx.
        let expected = 0.5 * (1.0 - 0.0) / 100.0;
        assert!(
            (jl - expected).abs() < 1e-4,
            "flux magnitude wrong: got {jl}, expected {expected}"
        );
    }
}
