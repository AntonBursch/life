//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{AdvectionDiffusion1D, BoundaryCondition, Convection2D, Diffusion1D, GrayScott2D, SwiftHohenberg2D};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct WasmDiffusion1D {
    inner: Diffusion1D,
}

#[wasm_bindgen]
impl WasmDiffusion1D {
    /// Construct. Throws a JS error if parameters are invalid or unstable.
    #[wasm_bindgen(constructor)]
    pub fn new(n: usize, diffusivity: f64, dx: f64, dt: f64) -> Result<WasmDiffusion1D, JsError> {
        let inner = Diffusion1D::new(n, diffusivity, dx, dt, BoundaryCondition::ZeroFlux)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn seed_centre_pulse(&mut self) {
        self.inner.seed_centre_pulse();
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        self.inner.step_many(n as u64);
    }

    /// Length of the field.
    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Simulated time elapsed.
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    /// Conserved total (under zero-flux boundaries).
    #[wasm_bindgen(getter)]
    pub fn total(&self) -> f64 {
        self.inner.total()
    }

    /// RMS spread, the metric we use to verify sqrt(t) scaling.
    #[wasm_bindgen(getter)]
    pub fn rms_spread(&self) -> f64 {
        self.inner.rms_spread()
    }

    /// Copy the field into a freshly-allocated `Float64Array`. The viewer
    /// uses this every frame; allocations are unavoidable here unless we
    /// move to memory views, which we'll consider when a rung pushes us.
    pub fn phi(&self) -> Vec<f64> {
        self.inner.phi().to_vec()
    }
}

/// R2 — driven diffusion. Same field, same equation, but the left and
/// right ends are held at fixed values. The system settles into a steady
/// linear gradient that does not flatten because flux is being maintained.
#[wasm_bindgen]
pub struct WasmDriven1D {
    inner: Diffusion1D,
    left: f64,
    right: f64,
}

#[wasm_bindgen]
impl WasmDriven1D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n: usize,
        diffusivity: f64,
        dx: f64,
        dt: f64,
        left: f64,
        right: f64,
    ) -> Result<WasmDriven1D, JsError> {
        let inner = Diffusion1D::new(
            n,
            diffusivity,
            dx,
            dt,
            BoundaryCondition::FixedPair { left, right },
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner, left, right })
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        self.inner.step_many(n as u64);
    }

    /// Reset the interior field to zero. Boundaries are imposed on the next
    /// step so we don't write to them here.
    pub fn reset_interior(&mut self) {
        let phi = self.inner.phi_mut();
        let n = phi.len();
        for v in phi.iter_mut().take(n - 1).skip(1) {
            *v = 0.0;
        }
    }

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    #[wasm_bindgen(getter)]
    pub fn flux_left(&self) -> f64 {
        self.inner.flux_left()
    }

    #[wasm_bindgen(getter)]
    pub fn flux_right(&self) -> f64 {
        self.inner.flux_right()
    }

    /// Boundary values currently in force.
    #[wasm_bindgen(getter)]
    pub fn left(&self) -> f64 {
        self.left
    }

    #[wasm_bindgen(getter)]
    pub fn right(&self) -> f64 {
        self.right
    }

    pub fn phi(&self) -> Vec<f64> {
        self.inner.phi().to_vec()
    }
}

/// R3 — advection-diffusion. Same field as R2, but the medium itself is
/// moving with velocity `v`. The Péclet number `v · L / D` controls which
/// term dominates: small Pe looks like R2 (linear), large Pe pushes the
/// field toward the inflow value across most of the box.
#[wasm_bindgen]
pub struct WasmAdvectionDiffusion1D {
    inner: AdvectionDiffusion1D,
    left: f64,
    right: f64,
}

#[wasm_bindgen]
impl WasmAdvectionDiffusion1D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        n: usize,
        diffusivity: f64,
        velocity: f64,
        dx: f64,
        dt: f64,
        left: f64,
        right: f64,
    ) -> Result<WasmAdvectionDiffusion1D, JsError> {
        let inner = AdvectionDiffusion1D::new(
            n,
            diffusivity,
            velocity,
            dx,
            dt,
            BoundaryCondition::FixedPair { left, right },
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner, left, right })
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        self.inner.step_many(n as u64);
    }

    pub fn reset_interior(&mut self) {
        let phi = self.inner.phi_mut();
        let n = phi.len();
        for v in phi.iter_mut().take(n - 1).skip(1) {
            *v = 0.0;
        }
    }

    #[wasm_bindgen(getter)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    #[wasm_bindgen(getter)]
    pub fn flux_left(&self) -> f64 {
        self.inner.flux_left()
    }

    #[wasm_bindgen(getter)]
    pub fn flux_right(&self) -> f64 {
        self.inner.flux_right()
    }

    #[wasm_bindgen(getter)]
    pub fn peclet(&self) -> f64 {
        self.inner.peclet()
    }

    #[wasm_bindgen(getter)]
    pub fn left(&self) -> f64 {
        self.left
    }

    #[wasm_bindgen(getter)]
    pub fn right(&self) -> f64 {
        self.right
    }

    pub fn phi(&self) -> Vec<f64> {
        self.inner.phi().to_vec()
    }
}


/// R4 — 2D Gray-Scott reaction-diffusion. Two coupled fields with a fresh
/// feed of U and a sink for V. Above the relax-to-soup boundary, the box
/// spontaneously organises into spots, stripes, or spirals.
#[wasm_bindgen]
pub struct WasmGrayScott2D {
    inner: GrayScott2D,
}

#[wasm_bindgen]
impl WasmGrayScott2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        du: f64,
        dv: f64,
        feed: f64,
        kill: f64,
        dx: f64,
        dt: f64,
    ) -> Result<WasmGrayScott2D, JsError> {
        let inner = GrayScott2D::new(width, height, du, dv, feed, kill, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        self.inner.step_many(n as u64);
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn seed_blob(&mut self, cx: usize, cy: usize, r: usize) {
        self.inner.seed_blob(cx, cy, r);
    }

    /// Live tuning hooks — the viewer slides F and k while running.
    pub fn set_feed(&mut self, feed: f64) {
        self.inner.set_feed(feed);
    }

    pub fn set_kill(&mut self, kill: f64) {
        self.inner.set_kill(kill);
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    #[wasm_bindgen(getter)]
    pub fn feed(&self) -> f64 {
        self.inner.feed()
    }

    #[wasm_bindgen(getter)]
    pub fn kill(&self) -> f64 {
        self.inner.kill()
    }

    #[wasm_bindgen(getter)]
    pub fn mean_v(&self) -> f64 {
        self.inner.mean_v()
    }

    #[wasm_bindgen(getter)]
    pub fn max_v(&self) -> f64 {
        self.inner.max_v()
    }

    #[wasm_bindgen(getter)]
    pub fn var_v(&self) -> f64 {
        self.inner.var_v()
    }

    /// Copy the V field into a `Float64Array`. The viewer maps this onto an
    /// RGBA bitmap every frame.
    pub fn v_field(&self) -> Vec<f64> {
        self.inner.v().to_vec()
    }
}

/// R5 — 2D Boussinesq thermal convection. A box of fluid heated below.
/// Below threshold: pure conduction, Nu = 1. Above: convection cells, Nu > 1.
#[wasm_bindgen]
pub struct WasmConvection2D {
    inner: Convection2D,
}

#[wasm_bindgen]
impl WasmConvection2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        kappa: f64,
        nu: f64,
        gravity: f64,
        dx: f64,
        dt: f64,
    ) -> Result<WasmConvection2D, JsError> {
        let inner = Convection2D::new(width, height, kappa, nu, gravity, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n {
            self.inner.step();
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn set_gravity(&mut self, g: f64) {
        self.inner.set_gravity(g);
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    #[wasm_bindgen(getter)]
    pub fn gravity(&self) -> f64 {
        self.inner.gravity()
    }

    #[wasm_bindgen(getter)]
    pub fn nusselt(&self) -> f64 {
        self.inner.nusselt()
    }

    #[wasm_bindgen(getter)]
    pub fn mean_sq_vorticity(&self) -> f64 {
        self.inner.mean_sq_vorticity()
    }

    #[wasm_bindgen(getter)]
    pub fn max_abs_psi(&self) -> f64 {
        self.inner.max_abs_psi()
    }

    /// Temperature field, row-major, length = width * height.
    pub fn temperature_field(&self) -> Vec<f64> {
        self.inner.temperature().to_vec()
    }

    /// Streamfunction field, for drawing flow contours.
    pub fn psi_field(&self) -> Vec<f64> {
        self.inner.streamfunction().to_vec()
    }
}

/// R6 — Swift–Hohenberg. One scalar field, one PDE, one bifurcation knob.
/// Below `r=0` the field decays to zero. Above, finite-wavelength patterns
/// emerge: stripes, then labyrinths, then hexagonal cells as `r` grows.
#[wasm_bindgen]
pub struct WasmSwiftHohenberg2D {
    inner: SwiftHohenberg2D,
}

#[wasm_bindgen]
impl WasmSwiftHohenberg2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        r: f64,
        dx: f64,
        dt: f64,
    ) -> Result<WasmSwiftHohenberg2D, JsError> {
        let inner = SwiftHohenberg2D::new(width, height, r, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) {
        self.inner.step();
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n {
            self.inner.step();
        }
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn seed_noise(&mut self, amplitude: f64) {
        self.inner.seed_noise(amplitude);
    }

    pub fn set_r(&mut self, r: f64) {
        self.inner.set_r(r);
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize {
        self.inner.width()
    }

    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize {
        self.inner.height()
    }

    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time()
    }

    #[wasm_bindgen(getter)]
    pub fn r(&self) -> f64 {
        self.inner.r()
    }

    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        self.inner.mean()
    }

    #[wasm_bindgen(getter)]
    pub fn variance(&self) -> f64 {
        self.inner.variance()
    }

    #[wasm_bindgen(getter)]
    pub fn max_abs(&self) -> f64 {
        self.inner.max_abs()
    }

    /// The scalar field `u`, row-major, length = width * height.
    pub fn u_field(&self) -> Vec<f64> {
        self.inner.u().to_vec()
    }
}
