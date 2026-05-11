//! R9 — Kuramoto oscillators on a 2D periodic grid.
//!
//! The fourth substrate. R7 was excitation, R8 was conservation. R9
//! replaces both with *phase*. Every cell is a clock. Each clock has
//! its own slightly different natural frequency, drawn once at the
//! start and frozen for life. Connect each clock to its four neighbours
//! with a coupling `K`. For small `K` the clocks tick at their own
//! rates; the population is incoherent. Above a critical coupling
//! some of the clocks lock onto a common rhythm. Above a higher
//! coupling almost all of them lock. The phase field becomes a sheet
//! of synchronised time.
//!
//! Per-cell update (lattice Kuramoto, local coupling):
//!
//! ```text
//!   dtheta_i/dt = omega_i + (K / 4) * sum_{j ~ i} sin(theta_j - theta_i)
//! ```
//!
//! The macroscopic order parameter
//!
//! ```text
//!   r * exp(i*psi) = (1/N) * sum_i exp(i*theta_i)
//! ```
//!
//! collapses the field to a single complex number; `r` in [0,1] is the
//! synchrony level (0 = incoherent, 1 = fully phase-locked), `psi` is
//! the mean phase.
//!
//! Discretisation: forward Euler. The natural-frequency distribution
//! is zero-mean Gaussian with prescribed standard deviation.

use std::fmt;
use std::f64::consts::{PI, TAU};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KuramotoError {
    InvalidSize,
    NonPositive { name: &'static str, value: f64 },
    Negative { name: &'static str, value: f64 },
}

impl fmt::Display for KuramotoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSize => write!(f, "width and height must be >= 4"),
            Self::NonPositive { name, value } => {
                write!(f, "{} must be > 0 (got {})", name, value)
            }
            Self::Negative { name, value } => {
                write!(f, "{} must be >= 0 (got {})", name, value)
            }
        }
    }
}

/// 2D Kuramoto with local (4-neighbour) coupling on a periodic grid.
pub struct Kuramoto2D {
    width: usize,
    height: usize,
    coupling: f64,
    dt: f64,
    theta: Vec<f64>,
    omega: Vec<f64>,
    scratch: Vec<f64>,
    time: f64,
    steps: u64,
}

impl Kuramoto2D {
    pub fn new(
        width: usize,
        height: usize,
        coupling: f64,
        dt: f64,
    ) -> Result<Self, KuramotoError> {
        if width < 4 || height < 4 {
            return Err(KuramotoError::InvalidSize);
        }
        if !(dt > 0.0) {
            return Err(KuramotoError::NonPositive { name: "dt", value: dt });
        }
        if coupling < 0.0 {
            return Err(KuramotoError::Negative {
                name: "coupling",
                value: coupling,
            });
        }
        let n = width * height;
        Ok(Self {
            width,
            height,
            coupling,
            dt,
            theta: vec![0.0; n],
            omega: vec![0.0; n],
            scratch: vec![0.0; n],
            time: 0.0,
            steps: 0,
        })
    }

    pub fn width(&self) -> usize { self.width }
    pub fn height(&self) -> usize { self.height }
    pub fn time(&self) -> f64 { self.time }
    pub fn steps(&self) -> u64 { self.steps }
    pub fn coupling(&self) -> f64 { self.coupling }
    pub fn theta(&self) -> &[f64] { &self.theta }
    pub fn omega(&self) -> &[f64] { &self.omega }

    pub fn set_coupling(&mut self, k: f64) {
        if k >= 0.0 { self.coupling = k; }
    }

    /// Reset time/steps but keep theta and omega.
    pub fn reset_time(&mut self) {
        self.time = 0.0;
        self.steps = 0;
    }

    /// Re-randomise phases in `[-pi, pi)`. `omega` is left untouched.
    pub fn randomise_phases(&mut self, seed: u64) {
        let mut s = seed.wrapping_add(0xD1B54A32D192ED03);
        for t in self.theta.iter_mut() {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            let r = ((s >> 11) as f64) / ((1u64 << 53) as f64); // [0,1)
            *t = (2.0 * r - 1.0) * PI;
        }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Draw a fresh frozen frequency distribution: Gaussian, mean 0,
    /// stddev `sigma`. Box–Muller from a xorshift stream.
    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u64) {
        let mut s = seed.wrapping_add(0x94D049BB133111EB);
        let mut next_uniform = || -> f64 {
            s ^= s << 13; s ^= s >> 7; s ^= s << 17;
            // Avoid exact zero for log() input.
            (((s >> 11) as f64) + 1.0) / ((1u64 << 53) as f64 + 1.0)
        };
        let mut i = 0;
        let n = self.omega.len();
        while i < n {
            let u1 = next_uniform();
            let u2 = next_uniform();
            let mag = sigma * (-2.0 * u1.ln()).sqrt();
            self.omega[i] = mag * (TAU * u2).cos();
            if i + 1 < n {
                self.omega[i + 1] = mag * (TAU * u2).sin();
            }
            i += 2;
        }
        // Re-centre to exactly zero mean so any drift is purely from coupling.
        let mean: f64 = self.omega.iter().sum::<f64>() / (n as f64);
        for w in self.omega.iter_mut() { *w -= mean; }
    }

    /// Set every oscillator to the same natural frequency.
    pub fn set_uniform_frequency(&mut self, omega: f64) {
        for w in self.omega.iter_mut() { *w = omega; }
    }

    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;
        let k_over_4 = self.coupling * 0.25;
        // Compute dtheta into scratch.
        for j in 0..h {
            let jm = if j == 0 { h - 1 } else { j - 1 };
            let jp = if j == h - 1 { 0 } else { j + 1 };
            let row = j * w;
            let row_m = jm * w;
            let row_p = jp * w;
            for i in 0..w {
                let im = if i == 0 { w - 1 } else { i - 1 };
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let t0 = self.theta[row + i];
                let s = (self.theta[row + im] - t0).sin()
                    + (self.theta[row + ip] - t0).sin()
                    + (self.theta[row_m + i] - t0).sin()
                    + (self.theta[row_p + i] - t0).sin();
                self.scratch[row + i] = self.omega[row + i] + k_over_4 * s;
            }
        }
        // Forward Euler, then wrap into (-pi, pi].
        let dt = self.dt;
        for k in 0..self.theta.len() {
            let mut t = self.theta[k] + dt * self.scratch[k];
            // Wrap to (-pi, pi]
            if t > PI || t <= -PI {
                t = ((t + PI).rem_euclid(TAU)) - PI;
            }
            self.theta[k] = t;
        }
        self.time += dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    /// One forward-Euler step with a *per-cell* coupling field. Cell `i`
    /// pulls toward its 4 neighbours with strength `k_field[i] / 4`.
    /// Returns Err if the field size is wrong or contains negatives.
    /// This is the composition primitive used by R10: a separate
    /// substrate (e.g. an excitable activator) computes `k_field`, then
    /// drives the phase layer through it.
    pub fn step_with_coupling_field(
        &mut self,
        k_field: &[f64],
    ) -> Result<(), KuramotoError> {
        if k_field.len() != self.theta.len() {
            return Err(KuramotoError::InvalidSize);
        }
        let w = self.width;
        let h = self.height;
        for j in 0..h {
            let jm = if j == 0 { h - 1 } else { j - 1 };
            let jp = if j == h - 1 { 0 } else { j + 1 };
            let row = j * w;
            let row_m = jm * w;
            let row_p = jp * w;
            for i in 0..w {
                let im = if i == 0 { w - 1 } else { i - 1 };
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let idx = row + i;
                let t0 = self.theta[idx];
                let k_local = k_field[idx];
                if k_local < 0.0 {
                    return Err(KuramotoError::Negative {
                        name: "k_field[i]",
                        value: k_local,
                    });
                }
                let s = (self.theta[row + im] - t0).sin()
                    + (self.theta[row + ip] - t0).sin()
                    + (self.theta[row_m + i] - t0).sin()
                    + (self.theta[row_p + i] - t0).sin();
                self.scratch[idx] = self.omega[idx] + 0.25 * k_local * s;
            }
        }
        let dt = self.dt;
        for k in 0..self.theta.len() {
            let mut t = self.theta[k] + dt * self.scratch[k];
            if t > PI || t <= -PI {
                t = ((t + PI).rem_euclid(TAU)) - PI;
            }
            self.theta[k] = t;
        }
        self.time += dt;
        self.steps += 1;
        Ok(())
    }

    /// Global order parameter `r` in `[0, 1]`.
    pub fn order_parameter(&self) -> f64 {
        let n = self.theta.len();
        if n == 0 { return 0.0; }
        let mut cs = 0.0;
        let mut sn = 0.0;
        for &t in &self.theta { cs += t.cos(); sn += t.sin(); }
        ((cs * cs + sn * sn).sqrt()) / (n as f64)
    }

    /// Mean phase `psi` in `(-pi, pi]`.
    pub fn mean_phase(&self) -> f64 {
        let mut cs = 0.0;
        let mut sn = 0.0;
        for &t in &self.theta { cs += t.cos(); sn += t.sin(); }
        sn.atan2(cs)
    }

    /// Circular variance: `1 - r`. 0 = locked, 1 = perfectly spread.
    pub fn circular_variance(&self) -> f64 {
        1.0 - self.order_parameter()
    }

    pub fn natural_freq_stddev(&self) -> f64 {
        let n = self.omega.len();
        if n == 0 { return 0.0; }
        let m: f64 = self.omega.iter().sum::<f64>() / (n as f64);
        let mut acc = 0.0;
        for &w in &self.omega {
            let d = w - m;
            acc += d * d;
        }
        (acc / (n as f64)).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_bad_inputs() {
        assert!(matches!(
            Kuramoto2D::new(3, 8, 1.0, 0.05).map(|_| ()).unwrap_err(),
            KuramotoError::InvalidSize
        ));
        assert!(matches!(
            Kuramoto2D::new(16, 16, 1.0, 0.0).map(|_| ()).unwrap_err(),
            KuramotoError::NonPositive { .. }
        ));
        assert!(matches!(
            Kuramoto2D::new(16, 16, -0.1, 0.05).map(|_| ()).unwrap_err(),
            KuramotoError::Negative { .. }
        ));
    }

    #[test]
    fn order_parameter_bounded() {
        let mut sim = Kuramoto2D::new(32, 32, 1.0, 0.05).unwrap();
        sim.randomise_phases(1);
        let r = sim.order_parameter();
        assert!(r >= 0.0 && r <= 1.0 + 1e-12, "r = {}", r);
    }

    #[test]
    fn identical_oscillators_lock_from_perturbed_uniform() {
        // sigma = 0 and a small zero-mean phase ripple. The uniform
        // phase-locked state is the linearly stable fixed point.
        // (Note: from *fully random* initial phases the 2D torus can
        // trap long-lived vortex pairs; that is real physics, not a
        // test issue. We test stability of the locked state itself.)
        let mut sim = Kuramoto2D::new(32, 32, 1.0, 0.05).unwrap();
        sim.set_uniform_frequency(0.0);
        sim.randomise_phases(7);
        // shrink the ripple by hand to within a small angle
        for t in 0..32 * 32 {
            let p = sim.theta[t];
            sim.theta[t] = 0.05 * p;
        }
        let r0 = sim.order_parameter();
        sim.step_many(1000);
        let r1 = sim.order_parameter();
        assert!(r0 > 0.99 && r1 > 0.999, "did not lock: r0={}, r1={}", r0, r1);
    }

    #[test]
    fn identical_oscillators_anneal_toward_sync() {
        // From fully random phases, vortex defects coarsen slowly.
        // Over a long run, r should climb substantially.
        let mut sim = Kuramoto2D::new(32, 32, 1.0, 0.05).unwrap();
        sim.set_uniform_frequency(0.0);
        sim.randomise_phases(7);
        let r0 = sim.order_parameter();
        sim.step_many(8000);
        let r1 = sim.order_parameter();
        assert!(r1 > r0 + 0.3, "r did not climb: {} -> {}", r0, r1);
    }

    #[test]
    fn zero_coupling_stays_incoherent() {
        // K = 0: every cell ticks at its own frequency. Phases drift
        // apart; r decays toward ~0 (Bessel-like residual on a finite
        // lattice).
        let mut sim = Kuramoto2D::new(32, 32, 0.0, 0.05).unwrap();
        sim.set_natural_frequencies(0.5, 42);
        sim.randomise_phases(11);
        sim.step_many(5000);
        assert!(
            sim.order_parameter() < 0.1,
            "expected near-incoherent, got r={}",
            sim.order_parameter()
        );
    }

    #[test]
    fn strong_coupling_locks_heterogeneous_population() {
        // sigma = 0.3, K = 3.0 is well above the critical coupling
        // for local 2D Kuramoto; we should see clear partial sync.
        let mut sim = Kuramoto2D::new(32, 32, 3.0, 0.05).unwrap();
        sim.set_natural_frequencies(0.3, 1234);
        sim.randomise_phases(99);
        let r0 = sim.order_parameter();
        sim.step_many(4000);
        let r1 = sim.order_parameter();
        assert!(
            r1 > 0.5 && r1 > r0 + 0.3,
            "did not partially sync: r0={}, r1={}",
            r0,
            r1
        );
    }

    #[test]
    fn phases_stay_wrapped() {
        let mut sim = Kuramoto2D::new(16, 16, 0.5, 0.05).unwrap();
        sim.set_uniform_frequency(2.0); // big drift
        sim.randomise_phases(3);
        sim.step_many(10_000);
        let max_abs = sim
            .theta()
            .iter()
            .map(|t| t.abs())
            .fold(0.0_f64, f64::max);
        assert!(max_abs <= PI + 1e-9, "phase escaped wrap: {}", max_abs);
    }
}
