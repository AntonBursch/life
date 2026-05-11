//! 2D Gray-Scott reaction-diffusion.
//!
//! Two coupled fields U and V on a uniform 2D grid with periodic boundaries:
//!
//! ```text
//!   ∂U/∂t =  Du · ∇²U  -  U V² + F (1 - U)
//!   ∂V/∂t =  Dv · ∇²V  +  U V² - (F + k) V
//! ```
//!
//! U is the "fuel," constantly being fed in at rate F and consumed by the
//! reaction `U + 2V → 3V`. V is the "product," autocatalytic in itself, and
//! removed at rate F + k. With a fresh-fuel source plus a sink for the
//! product, the box does not relax to soup — it organises into spots,
//! stripes, and spirals. This is the first rung where the universe makes a
//! shape with nothing but flow + chemistry + geometry.
//!
//! Discretisation: forward Euler in time, 5-point laplacian in space, with
//! periodic boundary conditions. Stability for the explicit Euler step is
//!
//! ```text
//!   D · dt / dx²  ≤  1/4
//! ```
//!
//! for each diffusivity D. The constructor enforces this.

use core::fmt;

#[derive(Debug)]
pub enum GrayScottError {
    NonPositiveSize,
    NonPositive { name: &'static str, value: f64 },
    Unstable { name: &'static str, ratio: f64 },
}

impl fmt::Display for GrayScottError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GrayScottError::NonPositiveSize => {
                write!(f, "grid width and height must each be at least 3 cells")
            }
            GrayScottError::NonPositive { name, value } => {
                write!(f, "{name} must be > 0, got {value}")
            }
            GrayScottError::Unstable { name, ratio } => write!(
                f,
                "stability violated: {name} · dt / dx² = {ratio} (must be <= 0.25)"
            ),
        }
    }
}

impl std::error::Error for GrayScottError {}

/// 2D Gray-Scott reaction-diffusion field with periodic boundaries.
#[derive(Debug, Clone)]
pub struct GrayScott2D {
    width: usize,
    height: usize,
    u: Vec<f64>,
    v: Vec<f64>,
    u_next: Vec<f64>,
    v_next: Vec<f64>,
    du: f64,
    dv: f64,
    feed: f64,
    kill: f64,
    dx: f64,
    dt: f64,
    time: f64,
    steps: u64,
}

impl GrayScott2D {
    /// Build a fresh grid. `u` is initialised to 1.0 everywhere, `v` to 0.0,
    /// which is the trivial fixed point — without a seed perturbation, no
    /// pattern will appear. Call [`seed_blob`](Self::seed_blob) to kick it
    /// out of that state.
    pub fn new(
        width: usize,
        height: usize,
        du: f64,
        dv: f64,
        feed: f64,
        kill: f64,
        dx: f64,
        dt: f64,
    ) -> Result<Self, GrayScottError> {
        if width < 3 || height < 3 {
            return Err(GrayScottError::NonPositiveSize);
        }
        if du <= 0.0 {
            return Err(GrayScottError::NonPositive {
                name: "du",
                value: du,
            });
        }
        if dv <= 0.0 {
            return Err(GrayScottError::NonPositive {
                name: "dv",
                value: dv,
            });
        }
        if dx <= 0.0 {
            return Err(GrayScottError::NonPositive {
                name: "dx",
                value: dx,
            });
        }
        if dt <= 0.0 {
            return Err(GrayScottError::NonPositive {
                name: "dt",
                value: dt,
            });
        }
        let alpha_u = du * dt / (dx * dx);
        if alpha_u > 0.25 {
            return Err(GrayScottError::Unstable {
                name: "du",
                ratio: alpha_u,
            });
        }
        let alpha_v = dv * dt / (dx * dx);
        if alpha_v > 0.25 {
            return Err(GrayScottError::Unstable {
                name: "dv",
                ratio: alpha_v,
            });
        }

        let n = width * height;
        Ok(Self {
            width,
            height,
            u: vec![1.0; n],
            v: vec![0.0; n],
            u_next: vec![0.0; n],
            v_next: vec![0.0; n],
            du,
            dv,
            feed,
            kill,
            dx,
            dt,
            time: 0.0,
            steps: 0,
        })
    }

    #[inline]
    pub fn width(&self) -> usize {
        self.width
    }

    #[inline]
    pub fn height(&self) -> usize {
        self.height
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.u.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.u.is_empty()
    }

    #[inline]
    pub fn time(&self) -> f64 {
        self.time
    }

    #[inline]
    pub fn steps(&self) -> u64 {
        self.steps
    }

    #[inline]
    pub fn feed(&self) -> f64 {
        self.feed
    }

    #[inline]
    pub fn kill(&self) -> f64 {
        self.kill
    }

    /// Live tuning hook. Allows the viewer to slide F and k while running.
    pub fn set_feed(&mut self, feed: f64) {
        self.feed = feed;
    }

    /// Live tuning hook for the kill rate.
    pub fn set_kill(&mut self, kill: f64) {
        self.kill = kill;
    }

    pub fn u(&self) -> &[f64] {
        &self.u
    }

    pub fn v(&self) -> &[f64] {
        &self.v
    }

    pub fn u_mut(&mut self) -> &mut [f64] {
        &mut self.u
    }

    pub fn v_mut(&mut self) -> &mut [f64] {
        &mut self.v
    }

    /// Reset U to 1.0 everywhere, V to 0.0. The trivial fixed point.
    pub fn reset(&mut self) {
        for u in self.u.iter_mut() {
            *u = 1.0;
        }
        for v in self.v.iter_mut() {
            *v = 0.0;
        }
        self.time = 0.0;
        self.steps = 0;
    }

    /// Knock a square patch of side `2r+1` around `(cx, cy)` toward the
    /// non-trivial state: U≈0.5, V≈0.25 plus a small jitter. The jitter is
    /// deterministic — a hash of the cell index — so a "seed" is
    /// reproducible without pulling in an rng dependency.
    pub fn seed_blob(&mut self, cx: usize, cy: usize, r: usize) {
        let w = self.width;
        let h = self.height;
        let x0 = cx.saturating_sub(r);
        let y0 = cy.saturating_sub(r);
        let x1 = (cx + r + 1).min(w);
        let y1 = (cy + r + 1).min(h);
        for j in y0..y1 {
            for i in x0..x1 {
                let idx = j * w + i;
                // Cheap, deterministic jitter in [-0.05, 0.05].
                let h32 = (idx.wrapping_mul(2654435761)) as u32;
                let jitter = (h32 as f64 / u32::MAX as f64 - 0.5) * 0.10;
                self.u[idx] = 0.5 + jitter;
                self.v[idx] = 0.25 + jitter;
            }
        }
    }

    /// Forward Euler step with periodic boundary conditions.
    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;
        let dx2 = self.dx * self.dx;
        let cu = self.du * self.dt / dx2;
        let cv = self.dv * self.dt / dx2;
        let f = self.feed;
        let k = self.kill;
        let dt = self.dt;

        for j in 0..h {
            let jn = if j == 0 { h - 1 } else { j - 1 };
            let js = if j + 1 == h { 0 } else { j + 1 };
            let row = j * w;
            let row_n = jn * w;
            let row_s = js * w;
            for i in 0..w {
                let iw = if i == 0 { w - 1 } else { i - 1 };
                let ie = if i + 1 == w { 0 } else { i + 1 };
                let c = row + i;
                let u_c = self.u[c];
                let v_c = self.v[c];
                let lap_u =
                    self.u[row + iw] + self.u[row + ie] + self.u[row_n + i] + self.u[row_s + i]
                        - 4.0 * u_c;
                let lap_v =
                    self.v[row + iw] + self.v[row + ie] + self.v[row_n + i] + self.v[row_s + i]
                        - 4.0 * v_c;
                let uvv = u_c * v_c * v_c;
                self.u_next[c] = u_c + dt * (self.du * lap_u / dx2 - uvv + f * (1.0 - u_c));
                self.v_next[c] = v_c + dt * (self.dv * lap_v / dx2 + uvv - (f + k) * v_c);
                // The two coefficients cu/cv aren't used directly here because
                // we want a single multiply per term; they're computed above
                // only as a documentation of where stability comes from.
                let _ = (cu, cv);
            }
        }

        core::mem::swap(&mut self.u, &mut self.u_next);
        core::mem::swap(&mut self.v, &mut self.v_next);
        self.time += self.dt;
        self.steps += 1;
    }

    pub fn step_many(&mut self, n: u64) {
        for _ in 0..n {
            self.step();
        }
    }

    /// One forward-Euler step with a *per-cell* feed rate. The kill rate
    /// stays uniform. Composition primitive: a separate substrate (e.g. a
    /// Kuramoto phase layer) computes `feed_field`, then drives the
    /// chemistry through it.
    pub fn step_with_feed_field(
        &mut self,
        feed_field: &[f64],
    ) -> Result<(), GrayScottError> {
        if feed_field.len() != self.u.len() {
            return Err(GrayScottError::NonPositiveSize);
        }
        let w = self.width;
        let h = self.height;
        let dx2 = self.dx * self.dx;
        let k = self.kill;
        let dt = self.dt;

        for j in 0..h {
            let jn = if j == 0 { h - 1 } else { j - 1 };
            let js = if j + 1 == h { 0 } else { j + 1 };
            let row = j * w;
            let row_n = jn * w;
            let row_s = js * w;
            for i in 0..w {
                let iw = if i == 0 { w - 1 } else { i - 1 };
                let ie = if i + 1 == w { 0 } else { i + 1 };
                let c = row + i;
                let u_c = self.u[c];
                let v_c = self.v[c];
                let lap_u = self.u[row + iw] + self.u[row + ie]
                    + self.u[row_n + i] + self.u[row_s + i] - 4.0 * u_c;
                let lap_v = self.v[row + iw] + self.v[row + ie]
                    + self.v[row_n + i] + self.v[row_s + i] - 4.0 * v_c;
                let uvv = u_c * v_c * v_c;
                let f = feed_field[c];
                self.u_next[c] = u_c + dt * (self.du * lap_u / dx2 - uvv + f * (1.0 - u_c));
                self.v_next[c] = v_c + dt * (self.dv * lap_v / dx2 + uvv - (f + k) * v_c);
            }
        }

        core::mem::swap(&mut self.u, &mut self.u_next);
        core::mem::swap(&mut self.v, &mut self.v_next);
        self.time += self.dt;
        self.steps += 1;
        Ok(())
    }

    /// Mean of V across the grid. Useful as a one-number summary of how
    /// much "product" the system is sustaining.
    pub fn mean_v(&self) -> f64 {
        if self.v.is_empty() {
            return 0.0;
        }
        let s: f64 = self.v.iter().sum();
        s / self.v.len() as f64
    }

    pub fn max_v(&self) -> f64 {
        self.v.iter().copied().fold(0.0_f64, f64::max)
    }

    /// Variance of V — a crude proxy for "is there a pattern?" Uniform
    /// soup has var ≈ 0; spots/stripes/spirals push it up.
    pub fn var_v(&self) -> f64 {
        let n = self.v.len();
        if n == 0 {
            return 0.0;
        }
        let mean = self.mean_v();
        let s: f64 = self.v.iter().map(|x| (x - mean) * (x - mean)).sum();
        s / n as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_unstable_du() {
        // du=1.0, dx=1.0, dt=1.0 -> ratio 1.0, way over 0.25.
        let err = GrayScott2D::new(16, 16, 1.0, 0.5, 0.04, 0.06, 1.0, 1.0).unwrap_err();
        match err {
            GrayScottError::Unstable { name, .. } => assert_eq!(name, "du"),
            other => panic!("expected Unstable, got {other:?}"),
        }
    }

    #[test]
    fn trivial_fixed_point_stays_trivial() {
        // Without a seed perturbation, U=1, V=0 is a fixed point. It must
        // stay there exactly.
        let mut sim = GrayScott2D::new(32, 32, 0.16, 0.08, 0.04, 0.06, 1.0, 1.0).unwrap();
        sim.step_many(500);
        for &u in sim.u() {
            assert!((u - 1.0).abs() < 1e-12, "u drifted from 1.0: {u}");
        }
        for &v in sim.v() {
            assert!(v.abs() < 1e-12, "v drifted from 0.0: {v}");
        }
    }

    #[test]
    fn seed_produces_pattern() {
        // With classic "coral" parameters (F=0.0545, k=0.062) and a centred
        // seed, the variance of V should be clearly non-zero after enough
        // steps. We're not asserting *which* pattern; just that the box has
        // left the soup state.
        let mut sim = GrayScott2D::new(64, 64, 0.16, 0.08, 0.0545, 0.062, 1.0, 1.0).unwrap();
        sim.seed_blob(32, 32, 8);
        sim.step_many(6000);
        let var = sim.var_v();
        assert!(var > 1e-4, "variance of V too low for a patterned state: {var}");
        assert!(sim.max_v() > 0.1, "V never grew");
    }

    #[test]
    fn fields_stay_finite() {
        // Belt-and-braces: nothing NaNs or blows up under classic params.
        let mut sim = GrayScott2D::new(48, 48, 0.16, 0.08, 0.04, 0.06, 1.0, 1.0).unwrap();
        sim.seed_blob(24, 24, 4);
        sim.step_many(2000);
        for &u in sim.u() {
            assert!(u.is_finite() && (-0.1..=1.5).contains(&u), "u out of range: {u}");
        }
        for &v in sim.v() {
            assert!(v.is_finite() && (-0.1..=1.5).contains(&v), "v out of range: {v}");
        }
    }
}
