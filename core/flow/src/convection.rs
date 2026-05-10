//! R5 — 2D thermal convection (Boussinesq, streamfunction–vorticity).
//!
//! A box of fluid is heated at the bottom and cooled at the top. Below a
//! threshold of heating, the heat just conducts through and the fluid stays
//! still. Above the threshold, the fluid spontaneously organises into
//! rolling convection cells that transport heat much faster than conduction
//! could. That transition is the point of this rung.
//!
//! We use the streamfunction–vorticity formulation of the 2D incompressible
//! Boussinesq equations:
//!
//! ```text
//! dT/dt + u . grad T  = kappa . laplacian(T)
//! dW/dt + u . grad W  = nu . laplacian(W) + g . dT/dx
//! laplacian(psi)      = -W
//! (u, v)              = (d psi / dy, -d psi / dx)
//! ```
//!
//! where T is temperature (0 at top, 1 at bottom), W is vorticity, psi is
//! the streamfunction, and `g` is a single buoyancy parameter that plays
//! the role of the Rayleigh number for this lattice.
//!
//! Geometry: width × height lattice. Horizontal direction is periodic.
//! Top and bottom are fixed-temperature (Dirichlet) with stress-free walls
//! (ψ = 0, ω = 0 at top and bottom).
//!
//! Time stepping is forward Euler with first-order upwinding for advection.
//! The Poisson equation for ψ is solved each step with a few sweeps of
//! red–black successive over-relaxation, warm-started from the previous
//! ψ — which is fine because the streamfunction changes little per step.

use core::fmt;

/// Construction-time validation errors.
#[derive(Debug, Clone, PartialEq)]
pub enum ConvectionError {
    /// width or height was zero.
    NonPositiveSize,
    /// A required positive parameter was non-positive.
    NonPositive { name: &'static str, value: f64 },
    /// Time step too large for the diffusion stability bound.
    Unstable { name: &'static str, ratio: f64 },
}

impl fmt::Display for ConvectionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConvectionError::NonPositiveSize => {
                write!(f, "width and height must be positive")
            }
            ConvectionError::NonPositive { name, value } => {
                write!(f, "{} must be positive (got {})", name, value)
            }
            ConvectionError::Unstable { name, ratio } => write!(
                f,
                "{} diffusion ratio dt/dx² = {:.4} exceeds 0.25",
                name, ratio
            ),
        }
    }
}

impl std::error::Error for ConvectionError {}

/// 2D Boussinesq convection in a width × height box.
///
/// The bottom row (j = 0) is held at temperature 1.0 (hot).
/// The top row (j = height - 1) is held at temperature 0.0 (cold).
/// The left and right edges are periodic.
#[derive(Debug)]
pub struct Convection2D {
    width: usize,
    height: usize,

    // Fields (row-major, index = j * width + i).
    t: Vec<f64>,
    t_next: Vec<f64>,
    omega: Vec<f64>,
    omega_next: Vec<f64>,
    psi: Vec<f64>,

    // Parameters.
    kappa: f64,   // thermal diffusivity
    nu: f64,      // kinematic viscosity
    gravity: f64, // buoyancy strength (Rayleigh-like)
    dx: f64,
    dt: f64,
    sor_sweeps: usize,
    sor_omega: f64,

    // Diagnostics.
    time: f64,
    steps: usize,
}

impl Convection2D {
    /// Build a new convection box.
    ///
    /// `kappa`, `nu`, `dx`, `dt` must all be positive. The forward-Euler
    /// stability bound `max(kappa, nu) · dt / dx² ≤ 0.25` is enforced.
    pub fn new(
        width: usize,
        height: usize,
        kappa: f64,
        nu: f64,
        gravity: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, ConvectionError> {
        if width == 0 || height == 0 {
            return Err(ConvectionError::NonPositiveSize);
        }
        if !(kappa > 0.0) {
            return Err(ConvectionError::NonPositive { name: "kappa", value: kappa });
        }
        if !(nu > 0.0) {
            return Err(ConvectionError::NonPositive { name: "nu", value: nu });
        }
        if !(dx > 0.0) {
            return Err(ConvectionError::NonPositive { name: "dx", value: dx });
        }
        if !(dt > 0.0) {
            return Err(ConvectionError::NonPositive { name: "dt", value: dt });
        }
        let r_t = kappa * dt / (dx * dx);
        if r_t > 0.25 {
            return Err(ConvectionError::Unstable { name: "kappa", ratio: r_t });
        }
        let r_n = nu * dt / (dx * dx);
        if r_n > 0.25 {
            return Err(ConvectionError::Unstable { name: "nu", ratio: r_n });
        }

        let n = width * height;
        let mut sim = Self {
            width,
            height,
            t: vec![0.0; n],
            t_next: vec![0.0; n],
            omega: vec![0.0; n],
            omega_next: vec![0.0; n],
            psi: vec![0.0; n],
            kappa,
            nu,
            gravity,
            dx,
            dt,
            sor_sweeps: 12,
            sor_omega: 1.7,
            time: 0.0,
            steps: 0,
        };
        sim.reset();
        Ok(sim)
    }

    /// Reset the box to the pure-conduction state plus a small symmetry-
    /// breaking perturbation at mid-height.
    pub fn reset(&mut self) {
        // Linear conduction profile T(y) = 1 - y/(H-1).
        let h_minus_1 = (self.height as f64 - 1.0).max(1.0);
        let w = self.width;
        for j in 0..self.height {
            let tj = 1.0 - (j as f64) / h_minus_1;
            for i in 0..w {
                self.t[j * w + i] = tj;
            }
        }
        // Tiny deterministic perturbation in the middle row to break symmetry.
        if self.height >= 3 {
            let j = self.height / 2;
            for i in 0..w {
                let phase = 2.0 * std::f64::consts::PI * (i as f64) / (w as f64);
                self.t[j * w + i] += 1e-3 * phase.sin();
            }
        }
        for v in self.omega.iter_mut() { *v = 0.0; }
        for v in self.omega_next.iter_mut() { *v = 0.0; }
        for v in self.psi.iter_mut() { *v = 0.0; }
        for v in self.t_next.iter_mut() { *v = 0.0; }
        self.time = 0.0;
        self.steps = 0;
        self.apply_t_bc();
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn time(&self) -> f64 { self.time }
    pub fn steps(&self) -> usize { self.steps }
    pub fn gravity(&self) -> f64 { self.gravity }
    pub fn kappa(&self) -> f64 { self.kappa }
    pub fn nu(&self) -> f64 { self.nu }

    pub fn set_gravity(&mut self, g: f64) {
        self.gravity = g;
    }

    pub fn temperature(&self) -> &[f64] { &self.t }
    pub fn vorticity(&self) -> &[f64] { &self.omega }
    pub fn streamfunction(&self) -> &[f64] { &self.psi }

    #[inline]
    fn idx(&self, i: usize, j: usize) -> usize {
        j * self.width + i
    }

    /// Periodic horizontal index.
    #[inline]
    fn wrap_x(&self, i: isize) -> usize {
        let w = self.width as isize;
        (((i % w) + w) % w) as usize
    }

    fn apply_t_bc(&mut self) {
        let bottom = 0;
        let top = self.height - 1;
        let w = self.width;
        for i in 0..w {
            self.t[bottom * w + i] = 1.0;
            self.t[top * w + i] = 0.0;
        }
    }

    fn apply_omega_bc(&mut self) {
        let bottom = 0;
        let top = self.height - 1;
        let w = self.width;
        for i in 0..w {
            self.omega[bottom * w + i] = 0.0;
            self.omega[top * w + i] = 0.0;
        }
    }

    fn apply_psi_bc(&mut self) {
        let bottom = 0;
        let top = self.height - 1;
        let w = self.width;
        for i in 0..w {
            self.psi[bottom * w + i] = 0.0;
            self.psi[top * w + i] = 0.0;
        }
    }

    /// Velocity at (i, j) from current streamfunction.
    /// u =  ∂ψ/∂y, v = -∂ψ/∂x. Centred differences.
    #[inline]
    fn velocity_at(&self, i: usize, j: usize) -> (f64, f64) {
        if j == 0 || j == self.height - 1 {
            // free-slip walls: velocity tangent only; we still need u for
            // advection at interior rows, but at the wall rows themselves
            // we never update t/ω because they're Dirichlet.
            return (0.0, 0.0);
        }
        let il = self.wrap_x(i as isize - 1);
        let ir = self.wrap_x(i as isize + 1);
        let u = (self.psi[self.idx(i, j + 1)] - self.psi[self.idx(i, j - 1)]) / (2.0 * self.dx);
        let v = -(self.psi[self.idx(ir, j)] - self.psi[self.idx(il, j)]) / (2.0 * self.dx);
        (u, v)
    }

    /// One red–black SOR sweep for ∇²ψ = -ω with ψ = 0 at top/bottom and
    /// periodic horizontal.
    fn poisson_sweep(&mut self) {
        let dx2 = self.dx * self.dx;
        let w = self.width;
        for color in 0..2 {
            for j in 1..self.height - 1 {
                for i in 0..w {
                    if (i + j) % 2 != color { continue; }
                    let il = self.wrap_x(i as isize - 1);
                    let ir = self.wrap_x(i as isize + 1);
                    let k = j * w + i;
                    let sum = self.psi[j * w + ir]
                        + self.psi[j * w + il]
                        + self.psi[(j + 1) * w + i]
                        + self.psi[(j - 1) * w + i];
                    let rhs = self.omega[k] * dx2;
                    let new_val = 0.25 * (sum + rhs);
                    self.psi[k] += self.sor_omega * (new_val - self.psi[k]);
                }
            }
        }
    }

    /// Advance the simulation by one time step.
    pub fn step(&mut self) {
        // 1. Update streamfunction from current vorticity (a few SOR sweeps,
        //    warm-started from previous ψ).
        self.apply_psi_bc();
        for _ in 0..self.sor_sweeps {
            self.poisson_sweep();
        }
        self.apply_psi_bc();

        let dx = self.dx;
        let dt = self.dt;
        let kappa = self.kappa;
        let nu = self.nu;
        let g = self.gravity;
        let w = self.width;
        let h = self.height;

        // 2. Advance T and ω on interior rows. Top and bottom rows are
        //    Dirichlet for T and ω.
        for j in 1..h - 1 {
            for i in 0..w {
                let il = self.wrap_x(i as isize - 1);
                let ir = self.wrap_x(i as isize + 1);
                let k_c = j * w + i;
                let k_l = j * w + il;
                let k_r = j * w + ir;
                let k_d = (j - 1) * w + i;
                let k_u = (j + 1) * w + i;
                let (u, v) = self.velocity_at(i, j);

                // Upwind advection of T.
                let t_c = self.t[k_c];
                let t_l = self.t[k_l];
                let t_r = self.t[k_r];
                let t_d = self.t[k_d];
                let t_u = self.t[k_u];
                let adv_tx = if u >= 0.0 {
                    u * (t_c - t_l) / dx
                } else {
                    u * (t_r - t_c) / dx
                };
                let adv_ty = if v >= 0.0 {
                    v * (t_c - t_d) / dx
                } else {
                    v * (t_u - t_c) / dx
                };
                let lap_t = (t_l + t_r + t_d + t_u - 4.0 * t_c) / (dx * dx);
                self.t_next[k_c] = t_c + dt * (-adv_tx - adv_ty + kappa * lap_t);

                // Upwind advection of ω with buoyancy source g·∂T/∂x.
                let w_c = self.omega[k_c];
                let w_l = self.omega[k_l];
                let w_r = self.omega[k_r];
                let w_d = self.omega[k_d];
                let w_u = self.omega[k_u];
                let adv_wx = if u >= 0.0 {
                    u * (w_c - w_l) / dx
                } else {
                    u * (w_r - w_c) / dx
                };
                let adv_wy = if v >= 0.0 {
                    v * (w_c - w_d) / dx
                } else {
                    v * (w_u - w_c) / dx
                };
                let lap_w = (w_l + w_r + w_d + w_u - 4.0 * w_c) / (dx * dx);
                let dtdx = (t_r - t_l) / (2.0 * dx);
                self.omega_next[k_c] =
                    w_c + dt * (-adv_wx - adv_wy + nu * lap_w + g * dtdx);
            }
        }

        // 3. Swap (interior rows only; boundary rows will be re-applied).
        std::mem::swap(&mut self.t, &mut self.t_next);
        std::mem::swap(&mut self.omega, &mut self.omega_next);
        self.apply_t_bc();
        self.apply_omega_bc();

        self.time += dt;
        self.steps += 1;
    }

    /// Advance by `n` time steps.
    pub fn step_many(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Nusselt number, computed as the horizontally-averaged temperature
    /// gradient at the top wall, divided by the pure-conduction value.
    ///
    /// `Nu = 1` is pure conduction. `Nu > 1` means the convective transport
    /// has steepened the wall gradient — more heat is getting through.
    pub fn nusselt(&self) -> f64 {
        if self.height < 2 { return 1.0; }
        let h_minus_1 = (self.height as f64 - 1.0).max(1.0);
        let conduction_grad = 1.0 / (h_minus_1 * self.dx);
        let top = self.height - 1;
        let mut sum = 0.0;
        for i in 0..self.width {
            // one-sided derivative at the top wall, pointing down (hot below)
            sum += (self.t[self.idx(i, top - 1)] - self.t[self.idx(i, top)]) / self.dx;
        }
        let mean_grad = sum / (self.width as f64);
        mean_grad / conduction_grad
    }

    /// Total kinetic energy proxy: ⟨ω²⟩. Zero in conduction, non-zero in
    /// convection. Useful for detecting the onset of motion.
    pub fn mean_sq_vorticity(&self) -> f64 {
        let mut s = 0.0;
        for v in &self.omega {
            s += v * v;
        }
        s / (self.omega.len() as f64)
    }

    /// Maximum absolute value of the streamfunction — a proxy for flow
    /// strength.
    pub fn max_abs_psi(&self) -> f64 {
        let mut m = 0.0f64;
        for v in &self.psi {
            let a = v.abs();
            if a > m { m = a; }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_kappa() {
        // dt/dx² = 1, kappa = 0.3 → ratio 0.3 > 0.25
        let err = Convection2D::new(8, 8, 0.3, 0.1, 0.0, 1.0, 1.0).unwrap_err();
        match err {
            ConvectionError::Unstable { name, .. } => assert_eq!(name, "kappa"),
            other => panic!("expected Unstable, got {:?}", other),
        }
    }

    #[test]
    fn zero_gravity_stays_at_conduction() {
        // With g = 0, the system has no buoyancy, so any initial perturbation
        // should diffuse away. Nu should sit at ~1 and vorticity should stay
        // at zero.
        let mut sim = Convection2D::new(32, 24, 0.1, 0.1, 0.0, 1.0, 0.5).unwrap();
        sim.step_many(2000);
        let nu = sim.nusselt();
        assert!((nu - 1.0).abs() < 0.05, "Nu = {} should be ~1 with g=0", nu);
        let energy = sim.mean_sq_vorticity();
        assert!(energy < 1e-6, "no buoyancy, no motion; got ⟨ω²⟩ = {}", energy);
    }

    #[test]
    fn strong_gravity_drives_convection() {
        // For a 24-tall box with κ = ν = 0.1, the critical Ra ≈ 657
        // corresponds to g ≈ 5.4e-4. We sit comfortably above it.
        let mut sim = Convection2D::new(64, 24, 0.1, 0.1, 0.05, 1.0, 0.05).unwrap();
        sim.step_many(8000);
        let nu = sim.nusselt();
        let energy = sim.mean_sq_vorticity();
        assert!(nu > 1.2, "convection should boost Nu well above 1; got {}", nu);
        assert!(energy > 1e-6, "convection should produce vorticity; got {}", energy);
    }

    #[test]
    fn fields_stay_finite() {
        let mut sim = Convection2D::new(48, 24, 0.1, 0.1, 0.02, 1.0, 0.05).unwrap();
        sim.step_many(4000);
        for v in sim.temperature() {
            assert!(v.is_finite(), "temperature went non-finite");
        }
        for v in sim.vorticity() {
            assert!(v.is_finite(), "vorticity went non-finite");
        }
        for v in sim.streamfunction() {
            assert!(v.is_finite(), "psi went non-finite");
        }
    }
}
