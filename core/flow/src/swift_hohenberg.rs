//! R6 — Swift–Hohenberg.
//!
//! The cleanest model of pattern formation in nature. One scalar field, one
//! PDE, one bifurcation knob:
//!
//! ```text
//!   du/dt = r*u - (1 + laplacian)^2 u - u^3
//! ```
//!
//! Below `r = 0`, perturbations decay; the field relaxes to zero. Above
//! `r = 0`, perturbations grow at a preferred wavelength `k_c = 1`
//! (wavelength `2*pi` in units of `dx`), and the cubic nonlinearity
//! saturates them. The result is stripes, hexagons, or labyrinths,
//! depending on initial conditions and how far above onset you are.
//!
//! Swift–Hohenberg is the *normal form* that many pattern-forming systems
//! reduce to near onset — Bénard convection, certain reaction-diffusion
//! systems, nonlinear optics, etc. This rung exhibits the universal
//! doorway that R4 and R5 each walked through with their own substrates.
//!
//! Discretisation: 5-point stencil for the Laplacian on a periodic grid;
//! the biharmonic is the Laplacian applied twice. Forward Euler in time.
//! The dominant linear eigenvalue is the biharmonic, whose magnitude on
//! the 5-point periodic stencil reaches `64/dx^4`, so forward-Euler
//! stability requires `dt <= dx^4 / 32`.

use core::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum SwiftHohenbergError {
    NonPositiveSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { ratio: f64 },
}

impl fmt::Display for SwiftHohenbergError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonPositiveSize => write!(f, "width and height must be positive"),
            Self::NonPositive { name, value } => {
                write!(f, "{} must be positive (got {})", name, value)
            }
            Self::Unstable { ratio } => write!(
                f,
                "biharmonic stability ratio dt/dx^4 = {:.4} exceeds 1/32",
                ratio
            ),
        }
    }
}

impl std::error::Error for SwiftHohenbergError {}

#[derive(Debug)]
pub struct SwiftHohenberg2D {
    width: usize,
    height: usize,
    u: Vec<f64>,
    lap: Vec<f64>,
    bih: Vec<f64>,
    u_next: Vec<f64>,
    r: f64,
    dx: f64,
    dt: f64,
    time: f64,
    steps: usize,
}

impl SwiftHohenberg2D {
    pub fn new(
        width: usize,
        height: usize,
        r: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, SwiftHohenbergError> {
        if width == 0 || height == 0 {
            return Err(SwiftHohenbergError::NonPositiveSize);
        }
        if !(dx > 0.0) {
            return Err(SwiftHohenbergError::NonPositive { name: "dx", value: dx });
        }
        if !(dt > 0.0) {
            return Err(SwiftHohenbergError::NonPositive { name: "dt", value: dt });
        }
        let dx4 = dx * dx * dx * dx;
        let ratio = dt / dx4;
        if ratio > 1.0 / 32.0 {
            return Err(SwiftHohenbergError::Unstable { ratio });
        }
        let n = width * height;
        Ok(Self {
            width,
            height,
            u: vec![0.0; n],
            lap: vec![0.0; n],
            bih: vec![0.0; n],
            u_next: vec![0.0; n],
            r,
            dx,
            dt,
            time: 0.0,
            steps: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn time(&self) -> f64 { self.time }
    pub fn steps(&self) -> usize { self.steps }
    pub fn r(&self) -> f64 { self.r }
    pub fn set_r(&mut self, r: f64) { self.r = r; }
    pub fn u(&self) -> &[f64] { &self.u }

    /// Reset to zero everywhere, then seed deterministic small-amplitude
    /// noise so the unstable modes have something to grow from.
    pub fn reset(&mut self) {
        for v in self.u.iter_mut() { *v = 0.0; }
        for v in self.u_next.iter_mut() { *v = 0.0; }
        self.time = 0.0;
        self.steps = 0;
        self.seed_noise(0.05);
    }

    /// Deterministic small noise. No rng dependency.
    pub fn seed_noise(&mut self, amplitude: f64) {
        let w = self.width;
        for j in 0..self.height {
            for i in 0..w {
                let h = hash_u32((i as u32).wrapping_mul(73856093) ^ (j as u32).wrapping_mul(19349663));
                let r = (h as f64 / u32::MAX as f64) * 2.0 - 1.0;
                self.u[j * w + i] += amplitude * r;
            }
        }
    }

    #[inline]
    fn wrap(idx: isize, n: usize) -> usize {
        let n = n as isize;
        (((idx % n) + n) % n) as usize
    }

    fn compute_laplacian(field: &[f64], out: &mut [f64], width: usize, height: usize, dx: f64) {
        let dx2 = dx * dx;
        for j in 0..height {
            let jm = Self::wrap(j as isize - 1, height);
            let jp = Self::wrap(j as isize + 1, height);
            for i in 0..width {
                let im = Self::wrap(i as isize - 1, width);
                let ip = Self::wrap(i as isize + 1, width);
                let c = field[j * width + i];
                let l = field[j * width + im];
                let r = field[j * width + ip];
                let d = field[jm * width + i];
                let u = field[jp * width + i];
                out[j * width + i] = (l + r + d + u - 4.0 * c) / dx2;
            }
        }
    }

    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;
        Self::compute_laplacian(&self.u, &mut self.lap, w, h, self.dx);
        Self::compute_laplacian(&self.lap, &mut self.bih, w, h, self.dx);
        let r = self.r;
        let dt = self.dt;
        for k in 0..self.u.len() {
            let u = self.u[k];
            let lap = self.lap[k];
            let bih = self.bih[k];
            // (1 + ∇²)² u  =  u + 2∇²u + ∇⁴u
            let lhs = u + 2.0 * lap + bih;
            let du = r * u - lhs - u * u * u;
            self.u_next[k] = u + dt * du;
        }
        std::mem::swap(&mut self.u, &mut self.u_next);
        self.time += dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: usize) {
        for _ in 0..n {
            self.step();
        }
    }

    /// Spatial mean of u.
    pub fn mean(&self) -> f64 {
        let s: f64 = self.u.iter().sum();
        s / (self.u.len() as f64)
    }

    /// Variance of u — zero in the uniform state, finite in the patterned state.
    pub fn variance(&self) -> f64 {
        let m = self.mean();
        let mut s = 0.0;
        for &v in &self.u {
            let d = v - m;
            s += d * d;
        }
        s / (self.u.len() as f64)
    }

    /// Peak absolute value, for the colormap.
    pub fn max_abs(&self) -> f64 {
        let mut m = 0.0f64;
        for &v in &self.u {
            let a = v.abs();
            if a > m { m = a; }
        }
        m
    }
}

#[inline]
fn hash_u32(mut x: u32) -> u32 {
    x = x.wrapping_add(0x9E3779B9);
    x ^= x >> 16;
    x = x.wrapping_mul(0x7FEB352D);
    x ^= x >> 15;
    x = x.wrapping_mul(0x846CA68B);
    x ^= x >> 16;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_dt() {
        // dx=1 -> dx^4=1, dt=0.05 -> ratio=0.05 > 1/32 = 0.03125
        let err = SwiftHohenberg2D::new(8, 8, 0.0, 1.0, 0.05).unwrap_err();
        match err {
            SwiftHohenbergError::Unstable { .. } => {}
            other => panic!("expected Unstable, got {:?}", other),
        }
    }

    #[test]
    fn below_onset_decays() {
        // r = -0.5: every mode decays.
        let mut sim = SwiftHohenberg2D::new(32, 32, -0.5, 1.0, 0.02).unwrap();
        sim.seed_noise(0.5);
        let v0 = sim.variance();
        sim.step_many(5000);
        let v1 = sim.variance();
        assert!(v1 < v0 * 0.05, "below onset, variance should decay: {} -> {}", v0, v1);
    }

    #[test]
    fn above_onset_grows() {
        // r = 0.5: well above onset.
        let mut sim = SwiftHohenberg2D::new(64, 64, 0.5, 1.0, 0.02).unwrap();
        sim.seed_noise(0.01);
        sim.step_many(10000);
        let v = sim.variance();
        assert!(v > 0.05, "above onset, pattern should emerge: var={}", v);
        // Stays bounded thanks to the cubic.
        assert!(sim.max_abs() < 5.0, "pattern should saturate, got max_abs={}", sim.max_abs());
    }

    #[test]
    fn fields_stay_finite() {
        let mut sim = SwiftHohenberg2D::new(48, 48, 0.3, 1.0, 0.02).unwrap();
        sim.seed_noise(0.05);
        sim.step_many(5000);
        for v in sim.u() {
            assert!(v.is_finite(), "u went non-finite");
        }
    }
}
