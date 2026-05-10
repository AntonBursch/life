//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{AdvectionDiffusion1D, BoundaryCondition, Diffusion1D};
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
