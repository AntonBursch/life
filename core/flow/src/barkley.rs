//! R7 — Excitable media (Barkley model) on a 2D periodic grid.
//!
//! The first rung where pattern has *direction* and *history*, not just
//! shape. R4 made stationary spots; R5 made stationary cells; R6 stripped
//! the pattern-forming move to its bones. None of them go anywhere.
//! Excitable media do — and once you break a wavefront, the broken end
//! curls into a spiral that keeps spinning forever.
//!
//! ```text
//!   du/dt = D * lap(u) + (1/eps) * u * (1 - u) * (u - (v + b) / a)
//!   dv/dt = u - v
//! ```
//!
//! `u` is the fast "voltage-like" variable; `v` is the slow recovery
//! variable. Rest state is `(u,v) = (0,0)`. A kick to `u` above threshold
//! `(v+b)/a` triggers a stereotyped excursion: rapid upstroke to ~1,
//! plateau while `v` rises, then `u` collapses as `v` lifts the
//! threshold above `u`, then `v` slowly relaxes back. Couple cells
//! through diffusion of `u` and the excursion propagates as a wave.
//!
//! Discretisation: 5-point Laplacian, forward Euler. Stability needs
//! both `D*dt/dx^2 <= 0.25` (diffusion) and `dt <= eps` (stiff reaction).

use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BarkleyError {
    InvalidSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { ratio: f64 },
}

impl fmt::Display for BarkleyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize => write!(f, "width and height must be >= 4"),
            Self::NonPositive { name, value } => {
                write!(f, "{} must be > 0 (got {})", name, value)
            }
            Self::Unstable { ratio } => write!(
                f,
                "diffusion stability ratio D*dt/dx^2 = {:.4} exceeds 0.25",
                ratio
            ),
        }
    }
}

/// 2D Barkley excitable medium with periodic boundaries.
pub struct Barkley2D {
    width: usize,
    height: usize,
    diffusion: f64,
    a: f64,
    b: f64,
    eps: f64,
    dx: f64,
    dt: f64,
    u: Vec<f64>,
    v: Vec<f64>,
    lap: Vec<f64>,
    u_next: Vec<f64>,
    v_next: Vec<f64>,
    time: f64,
    steps: usize,
}

impl Barkley2D {
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        eps: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, BarkleyError> {
        if width < 4 || height < 4 {
            return Err(BarkleyError::InvalidSize);
        }
        for (name, value) in [
            ("diffusion", diffusion),
            ("a", a),
            ("eps", eps),
            ("dx", dx),
            ("dt", dt),
        ] {
            if value <= 0.0 {
                return Err(BarkleyError::NonPositive { name, value });
            }
        }
        let ratio = diffusion * dt / (dx * dx);
        if ratio > 0.25 {
            return Err(BarkleyError::Unstable { ratio });
        }
        let n = width * height;
        Ok(Self {
            width,
            height,
            diffusion,
            a,
            b,
            eps,
            dx,
            dt,
            u: vec![0.0; n],
            v: vec![0.0; n],
            lap: vec![0.0; n],
            u_next: vec![0.0; n],
            v_next: vec![0.0; n],
            time: 0.0,
            steps: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn time(&self) -> f64 { self.time }
    pub fn steps(&self) -> usize { self.steps }
    pub fn a(&self) -> f64 { self.a }
    pub fn b(&self) -> f64 { self.b }
    pub fn eps(&self) -> f64 { self.eps }
    pub fn set_a(&mut self, a: f64) { if a > 0.0 { self.a = a; } }
    pub fn set_b(&mut self, b: f64) { self.b = b; }
    pub fn set_eps(&mut self, eps: f64) { if eps > 0.0 { self.eps = eps; } }
    pub fn u(&self) -> &[f64] { &self.u }
    pub fn v(&self) -> &[f64] { &self.v }

    pub fn reset(&mut self) {
        for x in &mut self.u { *x = 0.0; }
        for x in &mut self.v { *x = 0.0; }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Seed a broken wavefront that curls into a single spiral.
    /// Left strip is excited (u≈1); recovery is loaded in the top half
    /// (v large) so the wave can only propagate into the bottom half.
    /// The corner where excited meets refractory is the spiral tip.
    pub fn seed_spiral(&mut self) {
        let w = self.width;
        let h = self.height;
        let w4 = w / 4;
        for j in 0..h {
            for i in 0..w {
                let k = j * w + i;
                // A vertical bar of excited cells just left of centre.
                // Use 0.8 not 1.0 so the reaction term doesn't sit on its
                // zero at u=1 (which would freeze the kicked region).
                self.u[k] = if i >= w4 && i < w / 2 { 0.8 } else { 0.0 };
                // Top half is refractory: v above the excitation threshold.
                self.v[k] = if j < h / 2 { 0.4 } else { 0.0 };
            }
        }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Drop a circular suprathreshold kick into u at (cx, cy).
    /// From rest, this nucleates a target-pattern (ring) wave.
    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        let w = self.width as i32;
        let h = self.height as i32;
        let cx = cx as i32;
        let cy = cy as i32;
        let r2 = (radius as i32).pow(2);
        for dj in -(radius as i32)..=(radius as i32) {
            for di in -(radius as i32)..=(radius as i32) {
                if di * di + dj * dj > r2 { continue; }
                let i = ((cx + di).rem_euclid(w)) as usize;
                let j = ((cy + dj).rem_euclid(h)) as usize;
                self.u[j * self.width + i] = amplitude;
            }
        }
    }

    fn compute_laplacian(&mut self) {
        let w = self.width;
        let h = self.height;
        let inv_dx2 = 1.0 / (self.dx * self.dx);
        for j in 0..h {
            let jm = if j == 0 { h - 1 } else { j - 1 };
            let jp = if j + 1 == h { 0 } else { j + 1 };
            for i in 0..w {
                let im = if i == 0 { w - 1 } else { i - 1 };
                let ip = if i + 1 == w { 0 } else { i + 1 };
                let c = self.u[j * w + i];
                let l = self.u[j * w + im] + self.u[j * w + ip]
                    + self.u[jm * w + i] + self.u[jp * w + i] - 4.0 * c;
                self.lap[j * w + i] = l * inv_dx2;
            }
        }
    }

    pub fn step(&mut self) {
        self.compute_laplacian();
        let n = self.u.len();
        let dt = self.dt;
        let d = self.diffusion;
        let a = self.a;
        let b = self.b;
        let inv_eps = 1.0 / self.eps;
        for k in 0..n {
            let u = self.u[k];
            let v = self.v[k];
            let du = d * self.lap[k] + inv_eps * u * (1.0 - u) * (u - (v + b) / a);
            let dv = u - v;
            self.u_next[k] = u + dt * du;
            self.v_next[k] = v + dt * dv;
        }
        std::mem::swap(&mut self.u, &mut self.u_next);
        std::mem::swap(&mut self.v, &mut self.v_next);
        self.time += dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: usize) {
        for _ in 0..n { self.step(); }
    }

    /// Fraction of cells whose `u` is above the half-excitation level.
    /// Proxy for "how much wavefront is in the box."
    pub fn excited_fraction(&self) -> f64 {
        let mut c = 0usize;
        for &x in &self.u { if x > 0.5 { c += 1; } }
        c as f64 / self.u.len() as f64
    }

    pub fn mean_u(&self) -> f64 {
        let s: f64 = self.u.iter().sum();
        s / self.u.len() as f64
    }

    pub fn max_abs_u(&self) -> f64 {
        let mut m = 0.0_f64;
        for &x in &self.u { let a = x.abs(); if a > m { m = a; } }
        m
    }

    /// Spatial variance of `u`. Zero iff the medium is spatially uniform.
    pub fn variance_u(&self) -> f64 {
        let m = self.mean_u();
        let mut s = 0.0;
        for &x in &self.u { let d = x - m; s += d * d; }
        s / self.u.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_dt() {
        // dx=0.4, D=1: dt must be <= 0.04. Pass 0.1.
        let err = Barkley2D::new(16, 16, 1.0, 0.75, 0.01, 0.02, 0.4, 0.1)
            .map(|_| ()).unwrap_err();
        match err {
            BarkleyError::Unstable { .. } => {}
            other => panic!("expected Unstable, got {:?}", other),
        }
    }

    #[test]
    fn rest_state_stays_at_rest() {
        // (u, v) = (0, 0) is a fixed point: u*(1-u)*(u-b/a) = 0 at u=0,
        // and du/dv requires u=v=0 as well.
        let mut sim = Barkley2D::new(32, 32, 1.0, 0.75, 0.01, 0.02, 0.4, 0.01).unwrap();
        sim.step_many(1000);
        assert!(sim.max_abs_u() < 1e-6, "rest must be stationary, max|u|={}", sim.max_abs_u());
    }

    #[test]
    fn subthreshold_kick_dies() {
        // Threshold from rest (v=0) is u = b/a = 0.0133. A tiny kick is
        // below threshold and should not propagate.
        let mut sim = Barkley2D::new(40, 40, 1.0, 0.75, 0.01, 0.02, 0.4, 0.01).unwrap();
        sim.kick(20, 20, 2, 0.005);
        sim.step_many(2000);
        assert!(sim.excited_fraction() < 0.02,
            "subthreshold kick should not propagate, frac={}", sim.excited_fraction());
    }

    #[test]
    fn suprathreshold_kick_propagates() {
        // A strong kick at the centre should excite a ring wave that
        // visibly spreads. After enough time, there must be both excited
        // and non-excited cells in the box.
        let mut sim = Barkley2D::new(64, 64, 1.0, 0.75, 0.01, 0.02, 0.4, 0.01).unwrap();
        sim.kick(32, 32, 6, 0.8);
        sim.step_many(500);
        let var = sim.variance_u();
        assert!(var > 0.001,
            "ring wave should keep the medium non-uniform, var={}", var);
        for &x in sim.u() { assert!(x.is_finite()); }
        for &x in sim.v() { assert!(x.is_finite()); }
    }

    #[test]
    fn spiral_seed_produces_sustained_activity() {
        // Spiral seed should still be lively long after the initial
        // transient — that is the whole point of excitable media.
        let mut sim = Barkley2D::new(80, 80, 1.0, 0.75, 0.01, 0.02, 0.4, 0.01).unwrap();
        sim.seed_spiral();
        sim.step_many(5000);
        let var = sim.variance_u();
        assert!(var > 0.01,
            "spiral should sustain spatial non-uniformity, var={}", var);
        for &x in sim.u() { assert!(x.is_finite()); }
    }
}
