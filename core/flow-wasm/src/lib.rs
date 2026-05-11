//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{excitable_gate, AdvectionDiffusion1D, Barkley2D, BoundaryCondition, CahnHilliard2D, Convection2D, Diffusion1D, GrayScott2D, Kuramoto2D, SwiftHohenberg2D};
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

/// R7 — Barkley excitable medium. Two fields, fast `u` and slow `v`.
/// A suprathreshold kick triggers a stereotyped excursion that propagates
/// as a wave; a broken wavefront curls into a sustained spiral.
#[wasm_bindgen]
pub struct WasmBarkley2D {
    inner: Barkley2D,
}

#[wasm_bindgen]
impl WasmBarkley2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        eps: f64,
        dx: f64,
        dt: f64,
    ) -> Result<WasmBarkley2D, JsError> {
        let inner = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) { self.inner.step(); }
    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.inner.step(); }
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    pub fn seed_spiral(&mut self) { self.inner.seed_spiral(); }
    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.inner.kick(cx, cy, radius, amplitude);
    }
    pub fn set_a(&mut self, a: f64) { self.inner.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.inner.set_b(b); }
    pub fn set_eps(&mut self, eps: f64) { self.inner.set_eps(eps); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.inner.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.inner.height() }
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 { self.inner.time() }
    #[wasm_bindgen(getter)]
    pub fn a(&self) -> f64 { self.inner.a() }
    #[wasm_bindgen(getter)]
    pub fn b(&self) -> f64 { self.inner.b() }
    #[wasm_bindgen(getter)]
    pub fn eps(&self) -> f64 { self.inner.eps() }
    #[wasm_bindgen(getter)]
    pub fn mean_u(&self) -> f64 { self.inner.mean_u() }
    #[wasm_bindgen(getter)]
    pub fn variance_u(&self) -> f64 { self.inner.variance_u() }
    #[wasm_bindgen(getter)]
    pub fn max_abs_u(&self) -> f64 { self.inner.max_abs_u() }
    #[wasm_bindgen(getter)]
    pub fn excited_fraction(&self) -> f64 { self.inner.excited_fraction() }

    pub fn u_field(&self) -> Vec<f64> { self.inner.u().to_vec() }
    pub fn v_field(&self) -> Vec<f64> { self.inner.v().to_vec() }
}

/// R8 — Cahn–Hilliard phase separation. One conserved scalar field `c`
/// that splits into two bulk phases and coarsens forever without drive.
#[wasm_bindgen]
pub struct WasmCahnHilliard2D {
    inner: CahnHilliard2D,
}

#[wasm_bindgen]
impl WasmCahnHilliard2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        dx: f64,
        dt: f64,
    ) -> Result<WasmCahnHilliard2D, JsError> {
        let inner = CahnHilliard2D::new(width, height, mobility, kappa, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) { self.inner.step(); }
    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.inner.step(); }
    }
    pub fn reset(&mut self) { self.inner.reset(); }
    pub fn seed_noise(&mut self, amplitude: f64, mean: f64, seed: u32) {
        self.inner.seed_noise(amplitude, mean, seed as u64);
    }
    pub fn set_mobility(&mut self, m: f64) { self.inner.set_mobility(m); }
    pub fn set_kappa(&mut self, k: f64) { self.inner.set_kappa(k); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.inner.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.inner.height() }
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 { self.inner.time() }
    #[wasm_bindgen(getter)]
    pub fn mobility(&self) -> f64 { self.inner.mobility() }
    #[wasm_bindgen(getter)]
    pub fn kappa(&self) -> f64 { self.inner.kappa() }
    #[wasm_bindgen(getter)]
    pub fn mean_c(&self) -> f64 { self.inner.mean_c() }
    #[wasm_bindgen(getter)]
    pub fn variance_c(&self) -> f64 { self.inner.variance_c() }
    #[wasm_bindgen(getter)]
    pub fn max_abs_c(&self) -> f64 { self.inner.max_abs_c() }
    #[wasm_bindgen(getter)]
    pub fn bulk_fraction(&self) -> f64 { self.inner.bulk_fraction() }
    #[wasm_bindgen(getter)]
    pub fn free_energy(&self) -> f64 { self.inner.free_energy() }

    pub fn c_field(&self) -> Vec<f64> { self.inner.c().to_vec() }
}

/// R9 — Kuramoto phase oscillators on a 2D periodic grid with local
/// 4-neighbour coupling.
#[wasm_bindgen]
pub struct WasmKuramoto2D {
    inner: Kuramoto2D,
}

#[wasm_bindgen]
impl WasmKuramoto2D {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        coupling: f64,
        dt: f64,
    ) -> Result<WasmKuramoto2D, JsError> {
        let inner = Kuramoto2D::new(width, height, coupling, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self { inner })
    }

    pub fn step(&mut self) { self.inner.step(); }
    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.inner.step(); }
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.inner.randomise_phases(seed as u64);
    }
    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.inner.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn set_uniform_frequency(&mut self, omega: f64) {
        self.inner.set_uniform_frequency(omega);
    }
    pub fn set_coupling(&mut self, k: f64) { self.inner.set_coupling(k); }
    pub fn reset_time(&mut self) { self.inner.reset_time(); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.inner.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.inner.height() }
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 { self.inner.time() }
    #[wasm_bindgen(getter)]
    pub fn coupling(&self) -> f64 { self.inner.coupling() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.inner.order_parameter() }
    #[wasm_bindgen(getter)]
    pub fn mean_phase(&self) -> f64 { self.inner.mean_phase() }
    #[wasm_bindgen(getter)]
    pub fn circular_variance(&self) -> f64 { self.inner.circular_variance() }
    #[wasm_bindgen(getter)]
    pub fn natural_freq_stddev(&self) -> f64 { self.inner.natural_freq_stddev() }

    pub fn theta_field(&self) -> Vec<f64> { self.inner.theta().to_vec() }
    pub fn omega_field(&self) -> Vec<f64> { self.inner.omega().to_vec() }
}

/// R10 — Coupled substrates. A Barkley excitable layer's activator
/// gates a per-cell coupling field for a Kuramoto phase layer. Each
/// `step` advances both substrates together: Barkley.step(), then
/// `excitable_gate(u, k_lo, k_hi, threshold, sharpness)`, then
/// Kuramoto.step_with_coupling_field(k_field). The viewer can read
/// either the activator field, the phase field, or the live coupling
/// field.
#[wasm_bindgen]
pub struct WasmCoupledR10 {
    tissue: Barkley2D,
    phase: Kuramoto2D,
    k_field: Vec<f64>,
    k_lo: f64,
    k_hi: f64,
    threshold: f64,
    sharpness: f64,
}

#[wasm_bindgen]
impl WasmCoupledR10 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Barkley params
        diffusion: f64,
        a: f64,
        b: f64,
        eps: f64,
        dx: f64,
        dt_tissue: f64,
        // Kuramoto params
        dt_phase: f64,
        // Gate params
        k_lo: f64,
        k_hi: f64,
        threshold: f64,
        sharpness: f64,
    ) -> Result<WasmCoupledR10, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt_tissue)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, 0.0, dt_phase)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            tissue,
            phase,
            k_field: vec![0.0; width * height],
            k_lo,
            k_hi,
            threshold,
            sharpness,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step();
        let _ = excitable_gate(
            self.tissue.u(),
            self.k_lo,
            self.k_hi,
            self.threshold,
            self.sharpness,
            &mut self.k_field,
        );
        let _ = self.phase.step_with_coupling_field(&self.k_field);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }
    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }

    pub fn set_k_lo(&mut self, v: f64) { if v >= 0.0 { self.k_lo = v; } }
    pub fn set_k_hi(&mut self, v: f64) { if v >= 0.0 { self.k_hi = v; } }
    pub fn set_threshold(&mut self, v: f64) { self.threshold = v; }
    pub fn set_sharpness(&mut self, v: f64) { if v > 0.0 { self.sharpness = v; } }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.tissue.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.tissue.height() }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }
    #[wasm_bindgen(getter)]
    pub fn phase_time(&self) -> f64 { self.phase.time() }
    #[wasm_bindgen(getter)]
    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }
    #[wasm_bindgen(getter)]
    pub fn mean_phase(&self) -> f64 { self.phase.mean_phase() }
    #[wasm_bindgen(getter)]
    pub fn k_lo(&self) -> f64 { self.k_lo }
    #[wasm_bindgen(getter)]
    pub fn k_hi(&self) -> f64 { self.k_hi }
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 { self.threshold }
    #[wasm_bindgen(getter)]
    pub fn sharpness(&self) -> f64 { self.sharpness }

    /// Mean cos(theta_i - theta_j) over 4-neighbour pairs. The R10
    /// diagnostic: high local correlation with near-zero global r
    /// means neighbours are marching together but the canvas as a
    /// whole hasn't locked. That is exactly what excitation-gated
    /// coupling produces.
    pub fn local_correlation(&self) -> f64 {
        let w = self.phase.width();
        let h = self.phase.height();
        let theta = self.phase.theta();
        let mut acc = 0.0;
        let mut count = 0usize;
        for j in 0..h {
            let jp = if j == h - 1 { 0 } else { j + 1 };
            for i in 0..w {
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let t0 = theta[j * w + i];
                acc += (theta[j * w + ip] - t0).cos();
                acc += (theta[jp * w + i] - t0).cos();
                count += 2;
            }
        }
        if count == 0 { 0.0 } else { acc / (count as f64) }
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
    pub fn k_coupling_field(&self) -> Vec<f64> { self.k_field.clone() }
}
