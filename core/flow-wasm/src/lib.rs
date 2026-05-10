//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{BoundaryCondition, Diffusion1D};
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
