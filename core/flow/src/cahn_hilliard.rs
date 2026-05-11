//! R8 — Cahn–Hilliard phase separation on a 2D periodic grid.
//!
//! The contrast rung to R7. R7 needs nothing maintained from outside —
//! the medium is autonomous and excitable, and spirals keep spinning
//! using only the standing energy stored in resting cells. R8 is the
//! opposite extreme: a *conservative* field. The total amount of stuff
//! is fixed forever. There is no driving force, no pump, no boundary
//! flux. And yet the field organises itself into domains and the
//! domains keep growing, slowly, forever. Coarsening without drive.
//!
//! ```text
//!   mu  = c^3 - c - kappa * lap(c)        (chemical potential)
//!   dc/dt = M * lap(mu)
//! ```
//!
//! `c` is a conserved order parameter in roughly `[-1, +1]`; the bulk
//! phases sit at `c = +/- 1`. The free energy `f(c) = (c^2 - 1)^2 / 4`
//! is a double well. Uniform `c = 0` is linearly unstable: small noise
//! grows into domains. Once domains form, the interface energy
//! `kappa |grad c|^2` drives them to merge — small domains get eaten by
//! large ones — but mass conservation forbids the field from just going
//! to a single phase: it has to ferry material through the interfaces.
//! Average domain size grows like `t^(1/3)` (Lifshitz–Slyozov).
//!
//! Discretisation: 5-point Laplacian (applied twice), forward Euler.
//! Mass is conserved exactly to machine precision because the update
//! is a Laplacian of something (a discrete divergence of a flux).
//! Forward-Euler stability of the 4th-order term needs roughly
//! `M * kappa * dt / dx^4 <= 1/32` (the discrete biharmonic has
//! eigenvalues up to `64/dx^4`).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CahnHilliardError {
    InvalidSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { ratio: f64 },
}

impl fmt::Display for CahnHilliardError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize => write!(f, "width and height must be >= 4"),
            Self::NonPositive { name, value } => {
                write!(f, "{} must be > 0 (got {})", name, value)
            }
            Self::Unstable { ratio } => write!(
                f,
                "stability ratio M*kappa*dt/dx^4 = {:.4} exceeds 1/32 (0.03125)",
                ratio
            ),
        }
    }
}

/// 2D Cahn–Hilliard with periodic boundaries.
pub struct CahnHilliard2D {
    width: usize,
    height: usize,
    mobility: f64,
    kappa: f64,
    dx: f64,
    dt: f64,
    c: Vec<f64>,
    mu: Vec<f64>,
    scratch: Vec<f64>,
    time: f64,
    steps: u64,
}

impl CahnHilliard2D {
    pub fn new(
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, CahnHilliardError> {
        if width < 4 || height < 4 {
            return Err(CahnHilliardError::InvalidSize);
        }
        for (name, value) in [
            ("mobility", mobility),
            ("kappa", kappa),
            ("dx", dx),
            ("dt", dt),
        ] {
            if !(value > 0.0) {
                return Err(CahnHilliardError::NonPositive { name, value });
            }
        }
        let ratio = mobility * kappa * dt / dx.powi(4);
        if ratio > 1.0 / 32.0 {
            return Err(CahnHilliardError::Unstable { ratio });
        }
        let n = width * height;
        Ok(Self {
            width,
            height,
            mobility,
            kappa,
            dx,
            dt,
            c: vec![0.0; n],
            mu: vec![0.0; n],
            scratch: vec![0.0; n],
            time: 0.0,
            steps: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn time(&self) -> f64 { self.time }
    pub fn steps(&self) -> u64 { self.steps }
    pub fn mobility(&self) -> f64 { self.mobility }
    pub fn kappa(&self) -> f64 { self.kappa }
    pub fn c(&self) -> &[f64] { &self.c }

    pub fn set_mobility(&mut self, m: f64) {
        if m > 0.0 { self.mobility = m; }
    }
    pub fn set_kappa(&mut self, k: f64) {
        if k > 0.0 { self.kappa = k; }
    }

    pub fn reset(&mut self) {
        for c in self.c.iter_mut() { *c = 0.0; }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Seed with small zero-mean noise around a target mean.
    /// `seed` makes it deterministic.
    pub fn seed_noise(&mut self, amplitude: f64, mean: f64, seed: u64) {
        let mut s = seed.wrapping_add(0x9E3779B97F4A7C15);
        for c in self.c.iter_mut() {
            // xorshift64
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let r = ((s >> 11) as f64) / ((1u64 << 53) as f64); // [0,1)
            *c = mean + amplitude * (2.0 * r - 1.0);
        }
        // Re-centre exactly on `mean` so noise is zero-mean.
        let n = self.c.len() as f64;
        let actual_mean: f64 = self.c.iter().sum::<f64>() / n;
        let shift = mean - actual_mean;
        for c in self.c.iter_mut() { *c += shift; }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Periodic 5-point Laplacian.
    fn lap_into(
        width: usize,
        height: usize,
        dx: f64,
        src: &[f64],
        dst: &mut [f64],
    ) {
        let w = width;
        let h = height;
        let inv_dx2 = 1.0 / (dx * dx);
        for j in 0..h {
            let jm = if j == 0 { h - 1 } else { j - 1 };
            let jp = if j == h - 1 { 0 } else { j + 1 };
            let row = j * w;
            let row_m = jm * w;
            let row_p = jp * w;
            for i in 0..w {
                let im = if i == 0 { w - 1 } else { i - 1 };
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let c0 = src[row + i];
                let lap = src[row + im] + src[row + ip]
                    + src[row_m + i] + src[row_p + i]
                    - 4.0 * c0;
                dst[row + i] = lap * inv_dx2;
            }
        }
    }

    pub fn step(&mut self) {
        // mu = c^3 - c - kappa * lap(c)
        // 1) scratch = lap(c)
        Self::lap_into(self.width, self.height, self.dx, &self.c, &mut self.scratch);
        // 2) mu = c^3 - c - kappa * scratch
        let kappa = self.kappa;
        for k in 0..self.c.len() {
            let c = self.c[k];
            self.mu[k] = c * c * c - c - kappa * self.scratch[k];
        }
        // 3) scratch = lap(mu)
        Self::lap_into(self.width, self.height, self.dx, &self.mu, &mut self.scratch);
        // 4) c += dt * M * scratch
        let coeff = self.dt * self.mobility;
        for k in 0..self.c.len() {
            self.c[k] += coeff * self.scratch[k];
        }
        self.time += self.dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn mean_c(&self) -> f64 {
        if self.c.is_empty() { return 0.0; }
        self.c.iter().sum::<f64>() / (self.c.len() as f64)
    }

    pub fn variance_c(&self) -> f64 {
        let n = self.c.len();
        if n == 0 { return 0.0; }
        let m = self.mean_c();
        let mut acc = 0.0;
        for &c in &self.c {
            let d = c - m;
            acc += d * d;
        }
        acc / (n as f64)
    }

    pub fn max_abs_c(&self) -> f64 {
        let mut m = 0.0_f64;
        for &c in &self.c { if c.abs() > m { m = c.abs(); } }
        m
    }

    /// Fraction of cells with `|c| > 0.5` — proxy for "in one of the two
    /// bulk phases", as opposed to sitting in an interface.
    pub fn bulk_fraction(&self) -> f64 {
        if self.c.is_empty() { return 0.0; }
        let mut n = 0usize;
        for &c in &self.c { if c.abs() > 0.5 { n += 1; } }
        n as f64 / self.c.len() as f64
    }

    /// Total free energy per cell:
    ///   `(c^2 - 1)^2 / 4  +  (kappa/2) * |grad c|^2`
    /// Coarsening means this monotonically decreases.
    pub fn free_energy(&self) -> f64 {
        let w = self.width;
        let h = self.height;
        let inv_dx = 1.0 / self.dx;
        let mut acc = 0.0;
        for j in 0..h {
            let jp = if j == h - 1 { 0 } else { j + 1 };
            for i in 0..w {
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let c0 = self.c[j * w + i];
                let cx = self.c[j * w + ip];
                let cy = self.c[jp * w + i];
                let gx = (cx - c0) * inv_dx;
                let gy = (cy - c0) * inv_dx;
                let bulk = (c0 * c0 - 1.0).powi(2) / 4.0;
                let grad = 0.5 * self.kappa * (gx * gx + gy * gy);
                acc += bulk + grad;
            }
        }
        acc / (w as f64 * h as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_dt() {
        // M=1, kappa=1, dx=1, dt=0.1 -> ratio 0.1 > 1/32
        let err = CahnHilliard2D::new(32, 32, 1.0, 1.0, 1.0, 0.1)
            .map(|_| ())
            .unwrap_err();
        assert!(matches!(err, CahnHilliardError::Unstable { .. }));
    }

    #[test]
    fn uniform_field_stays_uniform() {
        // c = +1 is a stable bulk phase: mu = 1 - 1 - 0 = 0, dc/dt = 0.
        let mut sim = CahnHilliard2D::new(32, 32, 1.0, 1.0, 1.0, 0.02).unwrap();
        for c in sim.c.iter_mut() { *c = 1.0; }
        sim.step_many(500);
        let dev: f64 = sim.c.iter().map(|&c| (c - 1.0).abs()).fold(0.0, f64::max);
        assert!(dev < 1e-9, "c=+1 drifted by {}", dev);
    }

    #[test]
    fn mass_is_conserved() {
        let mut sim = CahnHilliard2D::new(48, 48, 1.0, 1.0, 1.0, 0.02).unwrap();
        sim.seed_noise(0.05, 0.0, 12345);
        let m0 = sim.mean_c();
        sim.step_many(5000);
        let m1 = sim.mean_c();
        assert!((m1 - m0).abs() < 1e-9, "mean drifted: {} -> {}", m0, m1);
    }

    #[test]
    fn random_init_coarsens() {
        // From low-amplitude zero-mean noise, variance should grow into the
        // bulk wells (toward ~1) as the system phase-separates.
        let mut sim = CahnHilliard2D::new(64, 64, 1.0, 1.0, 1.0, 0.02).unwrap();
        sim.seed_noise(0.05, 0.0, 7);
        let v0 = sim.variance_c();
        sim.step_many(10_000);
        let v1 = sim.variance_c();
        assert!(v1 > v0 * 50.0, "variance did not grow: {} -> {}", v0, v1);
        assert!(sim.bulk_fraction() > 0.6, "domains did not form: bulk_fraction={}", sim.bulk_fraction());
    }

    #[test]
    fn free_energy_decreases() {
        let mut sim = CahnHilliard2D::new(64, 64, 1.0, 1.0, 1.0, 0.02).unwrap();
        sim.seed_noise(0.05, 0.0, 99);
        sim.step_many(500); // get past the initial linearised blow-up
        let f0 = sim.free_energy();
        sim.step_many(10_000);
        let f1 = sim.free_energy();
        assert!(f1 < f0, "free energy did not decrease: {} -> {}", f0, f1);
    }
}
