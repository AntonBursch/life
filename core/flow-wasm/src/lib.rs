//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{excitable_gate, phase_to_scalar_field, bulk_gate, gradient_magnitude, gradient_field, advect_by, threshold_event, integrate_field, AdvectionDiffusion1D, Barkley2D, BoundaryCondition, CahnHilliard2D, Convection2D, Diffusion1D, GrayScott2D, Kuramoto2D, SwiftHohenberg2D};
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

// =====================================================================
// R11: phase drives reaction. A Kuramoto phase layer modulates the
// per-cell Gray-Scott feed rate via `phase_to_scalar_field`. When the
// phase layer locks, the chemistry breathes in unison; when it does
// not, regions starve and grow at independent times. Reverse arrow of
// R10.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR11 {
    chem: GrayScott2D,
    phase: Kuramoto2D,
    feed_field: Vec<f64>,
    f_lo: f64,
    f_hi: f64,
    // ring of recent mean(V) samples for the "breathing depth" readout
    v_mean_ring: Vec<f64>,
    v_mean_idx: usize,
    v_mean_filled: bool,
}

#[wasm_bindgen]
impl WasmCoupledR11 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Gray-Scott params
        du: f64,
        dv: f64,
        kill: f64,
        dx: f64,
        dt_chem: f64,
        // Kuramoto params
        dt_phase: f64,
        coupling: f64,
        // Feed-modulation envelope
        f_lo: f64,
        f_hi: f64,
    ) -> Result<WasmCoupledR11, JsError> {
        let f_seed = 0.5 * (f_lo + f_hi);
        let chem = GrayScott2D::new(width, height, du, dv, f_seed, kill, dx, dt_chem)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, coupling, dt_phase)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            chem,
            phase,
            feed_field: vec![f_seed; width * height],
            f_lo,
            f_hi,
            v_mean_ring: vec![0.0; 240],
            v_mean_idx: 0,
            v_mean_filled: false,
        })
    }

    pub fn step(&mut self) {
        self.phase.step();
        let _ = phase_to_scalar_field(
            self.phase.theta(),
            self.f_lo,
            self.f_hi,
            &mut self.feed_field,
        );
        let _ = self.chem.step_with_feed_field(&self.feed_field);

        // Track recent mean(V).
        let v = self.chem.v();
        let n = v.len() as f64;
        let m: f64 = v.iter().sum::<f64>() / n;
        self.v_mean_ring[self.v_mean_idx] = m;
        self.v_mean_idx = (self.v_mean_idx + 1) % self.v_mean_ring.len();
        if self.v_mean_idx == 0 { self.v_mean_filled = true; }
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_blob(&mut self, cx: usize, cy: usize, r: usize) {
        self.chem.seed_blob(cx, cy, r);
    }
    pub fn reset_chem(&mut self) { self.chem.reset(); }

    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }
    pub fn set_coupling(&mut self, k: f64) { self.phase.set_coupling(k); }
    pub fn set_f_lo(&mut self, v: f64) { if v >= 0.0 { self.f_lo = v; } }
    pub fn set_f_hi(&mut self, v: f64) { if v >= 0.0 { self.f_hi = v; } }
    pub fn set_kill(&mut self, v: f64) { self.chem.set_kill(v); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.chem.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.chem.height() }
    #[wasm_bindgen(getter)]
    pub fn phase_time(&self) -> f64 { self.phase.time() }
    #[wasm_bindgen(getter)]
    pub fn chem_time(&self) -> f64 { self.chem.time() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }
    #[wasm_bindgen(getter)]
    pub fn f_lo(&self) -> f64 { self.f_lo }
    #[wasm_bindgen(getter)]
    pub fn f_hi(&self) -> f64 { self.f_hi }
    #[wasm_bindgen(getter)]
    pub fn coupling(&self) -> f64 { self.phase.coupling() }
    #[wasm_bindgen(getter)]
    pub fn total_v(&self) -> f64 {
        let v = self.chem.v();
        v.iter().sum::<f64>() / (v.len() as f64)
    }
    #[wasm_bindgen(getter)]
    pub fn v_coverage(&self) -> f64 {
        let v = self.chem.v();
        v.iter().filter(|x| **x > 0.2).count() as f64 / (v.len() as f64)
    }

    /// Standard deviation of mean(V) over the last ~240 steps,
    /// normalised by its mean. This is the "breathing depth": when
    /// phases lock, total V rises and falls together and this number
    /// grows; when they don't, it stays small.
    pub fn breathing_depth(&self) -> f64 {
        let len = if self.v_mean_filled { self.v_mean_ring.len() } else { self.v_mean_idx };
        if len < 8 { return 0.0; }
        let slice = &self.v_mean_ring[..len];
        let mean: f64 = slice.iter().sum::<f64>() / (len as f64);
        if mean.abs() < 1e-9 { return 0.0; }
        let var: f64 = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (len as f64);
        var.sqrt() / mean.abs()
    }

    /// Local phase correlation, as in R10.
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

    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
    pub fn feed_field(&self) -> Vec<f64> { self.feed_field.clone() }
    pub fn v_field(&self) -> Vec<f64> { self.chem.v().to_vec() }
    pub fn u_field(&self) -> Vec<f64> { self.chem.u().to_vec() }
}

// =====================================================================
// R12: territory shapes sync. A Cahn-Hilliard domain field phi gates a
// Kuramoto coupling field through the *same* operator R10 used
// (`excitable_gate`), with threshold = 0. Cells inside positive
// domains see high coupling and lock together. Cells inside negative
// domains do the same independently. The domain walls are sync walls.
// As the territory coarsens, the sync map coarsens with it.
//
// Demonstrates operator reuse: one composition operator + two new
// substrate slots = new phenomenon, no new core math.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR12 {
    territory: CahnHilliard2D,
    phase: Kuramoto2D,
    k_field: Vec<f64>,
    k_wall: f64,
    k_bulk: f64,
    half_width: f64,
    sharpness: f64,
}

#[wasm_bindgen]
impl WasmCoupledR12 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Cahn-Hilliard params
        mobility: f64,
        kappa: f64,
        dx: f64,
        dt_territory: f64,
        // Kuramoto params
        dt_phase: f64,
        // Bulk-gate params: walls (|phi|<half_width) -> k_wall,
        // bulks (|phi|>half_width) -> k_bulk.
        k_wall: f64,
        k_bulk: f64,
        half_width: f64,
        sharpness: f64,
    ) -> Result<WasmCoupledR12, JsError> {
        let territory = CahnHilliard2D::new(width, height, mobility, kappa, dx, dt_territory)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, 0.0, dt_phase)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            territory,
            phase,
            k_field: vec![0.0; width * height],
            k_wall,
            k_bulk,
            half_width,
            sharpness,
        })
    }

    pub fn step(&mut self) {
        self.territory.step();
        let _ = bulk_gate(
            self.territory.c(),
            self.k_wall,
            self.k_bulk,
            self.half_width,
            self.sharpness,
            &mut self.k_field,
        );
        let _ = self.phase.step_with_coupling_field(&self.k_field);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_noise(&mut self, amplitude: f64, mean: f64, seed: u32) {
        self.territory.seed_noise(amplitude, mean, seed as u64);
    }
    pub fn reset_territory(&mut self) { self.territory.reset(); }

    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }
    pub fn set_k_wall(&mut self, v: f64) { if v >= 0.0 { self.k_wall = v; } }
    pub fn set_k_bulk(&mut self, v: f64) { if v >= 0.0 { self.k_bulk = v; } }
    pub fn set_half_width(&mut self, v: f64) { if v >= 0.0 { self.half_width = v; } }
    pub fn set_sharpness(&mut self, v: f64) { if v > 0.0 { self.sharpness = v; } }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.territory.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.territory.height() }
    #[wasm_bindgen(getter)]
    pub fn territory_time(&self) -> f64 { self.territory.time() }
    #[wasm_bindgen(getter)]
    pub fn phase_time(&self) -> f64 { self.phase.time() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }
    #[wasm_bindgen(getter)]
    pub fn k_wall(&self) -> f64 { self.k_wall }
    #[wasm_bindgen(getter)]
    pub fn k_bulk(&self) -> f64 { self.k_bulk }
    #[wasm_bindgen(getter)]
    pub fn half_width(&self) -> f64 { self.half_width }

    /// Mean cos(theta_i - theta_j) over 4-neighbour pairs.
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

    /// Per-domain order parameter: |mean(e^{i theta})| computed
    /// separately over phi > 0 and phi < 0. When the territory has
    /// coarsened, each domain is its own Kuramoto population and
    /// these climb independently while the global r stays small.
    pub fn order_parameter_pos(&self) -> f64 {
        let phi = self.territory.c();
        let theta = self.phase.theta();
        let mut cs = 0.0_f64;
        let mut sn = 0.0_f64;
        let mut n = 0_usize;
        for (p, t) in phi.iter().zip(theta.iter()) {
            if *p > 0.0 { cs += t.cos(); sn += t.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / (n as f64) }
    }

    pub fn order_parameter_neg(&self) -> f64 {
        let phi = self.territory.c();
        let theta = self.phase.theta();
        let mut cs = 0.0_f64;
        let mut sn = 0.0_f64;
        let mut n = 0_usize;
        for (p, t) in phi.iter().zip(theta.iter()) {
            if *p < 0.0 { cs += t.cos(); sn += t.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / (n as f64) }
    }

    /// Cosine of the phase difference between the two domains'
    /// mean phases. Near +1 => both domains accidentally aligned;
    /// near -1 => phase-opposed; near 0 => independent, which is
    /// the headline regime of this rung.
    pub fn cross_domain_alignment(&self) -> f64 {
        let phi = self.territory.c();
        let theta = self.phase.theta();
        let (mut cp, mut sp, mut np_) = (0.0_f64, 0.0_f64, 0_usize);
        let (mut cn, mut sn, mut nn_) = (0.0_f64, 0.0_f64, 0_usize);
        for (p, t) in phi.iter().zip(theta.iter()) {
            if *p > 0.0 { cp += t.cos(); sp += t.sin(); np_ += 1; }
            else if *p < 0.0 { cn += t.cos(); sn += t.sin(); nn_ += 1; }
        }
        if np_ == 0 || nn_ == 0 { return 0.0; }
        let mean_p = sp.atan2(cp);
        let mean_n = sn.atan2(cn);
        (mean_p - mean_n).cos()
    }

    pub fn phi_field(&self) -> Vec<f64> { self.territory.c().to_vec() }
    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
    pub fn k_coupling_field(&self) -> Vec<f64> { self.k_field.clone() }
}

// =====================================================================
// R13: spikes seed pattern. A Barkley excitable layer (R7) drives the
// per-cell Gray-Scott feed (R4) through `excitable_gate`. Where the
// excitable medium is at rest, the feed is starved -> chemistry decays.
// Where a spiral wave fires, the feed jumps -> chemistry grows in the
// wake of the wave. Spirals trace patterns onto a chemical canvas.
//
// This rung makes the missing R7 -> R4 arrow and is the third use of
// `excitable_gate` (after R10 and the unused-here template). One
// operator, three different jobs: confirms the alphabet thesis.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR13 {
    excitable: Barkley2D,
    chem: GrayScott2D,
    feed_field: Vec<f64>,
    f_lo: f64,
    f_hi: f64,
    threshold: f64,
    sharpness: f64,
    bark_substeps: u32,
}

#[wasm_bindgen]
impl WasmCoupledR13 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Barkley params
        bark_diffusion: f64,
        bark_a: f64,
        bark_b: f64,
        bark_eps: f64,
        bark_dx: f64,
        bark_dt: f64,
        // Gray-Scott params (feed comes from gate; kill is fixed)
        gs_du: f64,
        gs_dv: f64,
        gs_kill: f64,
        gs_dx: f64,
        gs_dt: f64,
        // Excitable-gate params: u below threshold -> f_lo, above -> f_hi
        f_lo: f64,
        f_hi: f64,
        threshold: f64,
        sharpness: f64,
        // Number of Barkley substeps per Gray-Scott step (balances dts)
        bark_substeps: u32,
    ) -> Result<WasmCoupledR13, JsError> {
        let excitable = Barkley2D::new(width, height, bark_diffusion, bark_a, bark_b, bark_eps, bark_dx, bark_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let f_seed = 0.5 * (f_lo + f_hi);
        let chem = GrayScott2D::new(width, height, gs_du, gs_dv, f_seed, gs_kill, gs_dx, gs_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            excitable,
            chem,
            feed_field: vec![f_seed; width * height],
            f_lo,
            f_hi,
            threshold,
            sharpness,
            bark_substeps: bark_substeps.max(1),
        })
    }

    pub fn step(&mut self) {
        for _ in 0..self.bark_substeps {
            self.excitable.step();
        }
        let _ = excitable_gate(
            self.excitable.u(),
            self.f_lo,
            self.f_hi,
            self.threshold,
            self.sharpness,
            &mut self.feed_field,
        );
        let _ = self.chem.step_with_feed_field(&self.feed_field);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.excitable.seed_spiral(); }
    pub fn reset_excitable(&mut self) { self.excitable.reset(); }
    pub fn seed_blob(&mut self, cx: usize, cy: usize, r: usize) {
        self.chem.seed_blob(cx, cy, r);
    }
    pub fn reset_chem(&mut self) { self.chem.reset(); }

    pub fn set_f_lo(&mut self, v: f64) { if v >= 0.0 { self.f_lo = v; } }
    pub fn set_f_hi(&mut self, v: f64) { if v >= 0.0 { self.f_hi = v; } }
    pub fn set_threshold(&mut self, v: f64) { self.threshold = v; }
    pub fn set_sharpness(&mut self, v: f64) { if v > 0.0 { self.sharpness = v; } }
    pub fn set_kill(&mut self, v: f64) { self.chem.set_kill(v); }
    pub fn set_bark_substeps(&mut self, n: u32) { self.bark_substeps = n.max(1); }
    pub fn set_bark_a(&mut self, v: f64) { self.excitable.set_a(v); }
    pub fn set_bark_b(&mut self, v: f64) { self.excitable.set_b(v); }
    pub fn set_bark_eps(&mut self, v: f64) { self.excitable.set_eps(v); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.chem.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.chem.height() }
    #[wasm_bindgen(getter)]
    pub fn excitable_time(&self) -> f64 { self.excitable.time() }
    #[wasm_bindgen(getter)]
    pub fn chem_time(&self) -> f64 { self.chem.time() }
    #[wasm_bindgen(getter)]
    pub fn f_lo(&self) -> f64 { self.f_lo }
    #[wasm_bindgen(getter)]
    pub fn f_hi(&self) -> f64 { self.f_hi }
    #[wasm_bindgen(getter)]
    pub fn threshold(&self) -> f64 { self.threshold }

    /// Mean chemistry density. Useful as a "is anything alive" needle.
    pub fn mean_v(&self) -> f64 {
        let v = self.chem.v();
        v.iter().sum::<f64>() / (v.len() as f64)
    }

    /// Fraction of cells with V > 0.2 — the "pattern coverage" needle.
    pub fn v_coverage(&self) -> f64 {
        let v = self.chem.v();
        v.iter().filter(|x| **x > 0.2).count() as f64 / (v.len() as f64)
    }

    /// Fraction of cells currently spiking (excitable.u above threshold).
    pub fn firing_fraction(&self) -> f64 {
        let u = self.excitable.u();
        let t = self.threshold;
        u.iter().filter(|x| **x > t).count() as f64 / (u.len() as f64)
    }

    /// Spatial correlation of the feed_field with the chemistry V.
    /// When chemistry tracks the wave (grows where wave just fired and
    /// hasn't yet decayed), this is positive. When it lags far behind,
    /// it falls. Pearson correlation, 4-byte-cheap.
    pub fn wave_pattern_correlation(&self) -> f64 {
        let f = &self.feed_field;
        let v = self.chem.v();
        let n = f.len() as f64;
        if n < 2.0 { return 0.0; }
        let fm: f64 = f.iter().sum::<f64>() / n;
        let vm: f64 = v.iter().sum::<f64>() / n;
        let mut num = 0.0_f64;
        let mut df2 = 0.0_f64;
        let mut dv2 = 0.0_f64;
        for (fi, vi) in f.iter().zip(v.iter()) {
            let df = fi - fm;
            let dv = vi - vm;
            num += df * dv;
            df2 += df * df;
            dv2 += dv * dv;
        }
        let denom = (df2 * dv2).sqrt();
        if denom < 1e-12 { 0.0 } else { num / denom }
    }

    pub fn u_field(&self) -> Vec<f64> { self.excitable.u().to_vec() }
    pub fn v_excitable_field(&self) -> Vec<f64> { self.excitable.v().to_vec() }
    pub fn feed_field(&self) -> Vec<f64> { self.feed_field.clone() }
    pub fn chem_v_field(&self) -> Vec<f64> { self.chem.v().to_vec() }
    pub fn chem_u_field(&self) -> Vec<f64> { self.chem.u().to_vec() }
}

// =====================================================================
// R14: three-layer stack. CH (R8) -> Kuramoto (R9) -> Gray-Scott (R4).
//
//   territory.phi  --bulk_gate-->            k_field (per-cell K)
//   k_field        --kuramoto.step-->        theta (phase per cell)
//   theta          --phase_to_scalar_field-> feed_field (per-cell F)
//   feed_field     --gs.step_with_feed-->    chemistry V
//
// No new operators, no new substrates. The point of this rung is
// that the alphabet composes end-to-end: stack three substrates and
// two operators, get one phenomenon that depends on all three. The
// territory segments which Kuramoto cluster locks; that cluster's
// collective phase paces a region of chemistry feed; so the walls
// of the territory become walls of bloom timing in the chemistry.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR14 {
    territory: CahnHilliard2D,
    phase: Kuramoto2D,
    chem: GrayScott2D,
    k_field: Vec<f64>,
    feed_field: Vec<f64>,
    k_wall: f64,
    k_bulk: f64,
    half_width: f64,
    sharpness: f64,
    f_lo: f64,
    f_hi: f64,
    phase_substeps: u32,
}

#[wasm_bindgen]
impl WasmCoupledR14 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Cahn-Hilliard
        mobility: f64,
        kappa: f64,
        ch_dx: f64,
        ch_dt: f64,
        // Kuramoto
        ph_dt: f64,
        // Gray-Scott
        gs_du: f64,
        gs_dv: f64,
        gs_kill: f64,
        gs_dx: f64,
        gs_dt: f64,
        // bulk_gate (CH -> K)
        k_wall: f64,
        k_bulk: f64,
        half_width: f64,
        sharpness: f64,
        // phase_to_scalar_field (theta -> feed)
        f_lo: f64,
        f_hi: f64,
        // Number of Kuramoto substeps per Gray-Scott step
        phase_substeps: u32,
    ) -> Result<WasmCoupledR14, JsError> {
        let territory = CahnHilliard2D::new(width, height, mobility, kappa, ch_dx, ch_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, 0.0, ph_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let f_seed = 0.5 * (f_lo + f_hi);
        let chem = GrayScott2D::new(width, height, gs_du, gs_dv, f_seed, gs_kill, gs_dx, gs_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            territory, phase, chem,
            k_field: vec![0.0; n],
            feed_field: vec![f_seed; n],
            k_wall, k_bulk, half_width, sharpness,
            f_lo, f_hi,
            phase_substeps: phase_substeps.max(1),
        })
    }

    pub fn step(&mut self) {
        // Slow layer: territory.
        self.territory.step();
        let _ = bulk_gate(
            self.territory.c(),
            self.k_wall, self.k_bulk, self.half_width, self.sharpness,
            &mut self.k_field,
        );
        // Mid layer: phase, several substeps per gs step.
        for _ in 0..self.phase_substeps {
            let _ = self.phase.step_with_coupling_field(&self.k_field);
        }
        // Operator 2: theta -> feed.
        let _ = phase_to_scalar_field(self.phase.theta(), self.f_lo, self.f_hi, &mut self.feed_field);
        // Fast layer: chemistry.
        let _ = self.chem.step_with_feed_field(&self.feed_field);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_noise(&mut self, amplitude: f64, mean: f64, seed: u32) {
        self.territory.seed_noise(amplitude, mean, seed as u64);
    }
    pub fn reset_territory(&mut self) { self.territory.reset(); }
    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }
    pub fn seed_blob(&mut self, cx: usize, cy: usize, r: usize) {
        self.chem.seed_blob(cx, cy, r);
    }
    pub fn reset_chem(&mut self) { self.chem.reset(); }

    pub fn set_k_wall(&mut self, v: f64) { if v >= 0.0 { self.k_wall = v; } }
    pub fn set_k_bulk(&mut self, v: f64) { if v >= 0.0 { self.k_bulk = v; } }
    pub fn set_half_width(&mut self, v: f64) { if v >= 0.0 { self.half_width = v; } }
    pub fn set_sharpness(&mut self, v: f64) { if v > 0.0 { self.sharpness = v; } }
    pub fn set_f_lo(&mut self, v: f64) { if v >= 0.0 { self.f_lo = v; } }
    pub fn set_f_hi(&mut self, v: f64) { if v >= 0.0 { self.f_hi = v; } }
    pub fn set_kill(&mut self, v: f64) { self.chem.set_kill(v); }
    pub fn set_phase_substeps(&mut self, n: u32) { self.phase_substeps = n.max(1); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.chem.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.chem.height() }
    #[wasm_bindgen(getter)]
    pub fn territory_time(&self) -> f64 { self.territory.time() }
    #[wasm_bindgen(getter)]
    pub fn chem_time(&self) -> f64 { self.chem.time() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }

    pub fn order_parameter_pos(&self) -> f64 {
        let phi = self.territory.c();
        let theta = self.phase.theta();
        let (mut cs, mut sn, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (p, t) in phi.iter().zip(theta.iter()) {
            if *p > 0.0 { cs += t.cos(); sn += t.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / n as f64 }
    }

    pub fn order_parameter_neg(&self) -> f64 {
        let phi = self.territory.c();
        let theta = self.phase.theta();
        let (mut cs, mut sn, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (p, t) in phi.iter().zip(theta.iter()) {
            if *p < 0.0 { cs += t.cos(); sn += t.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / n as f64 }
    }

    pub fn mean_v(&self) -> f64 {
        let v = self.chem.v();
        v.iter().sum::<f64>() / (v.len() as f64)
    }

    pub fn v_coverage(&self) -> f64 {
        let v = self.chem.v();
        v.iter().filter(|x| **x > 0.2).count() as f64 / (v.len() as f64)
    }

    /// Mean V over phi>0 minus mean V over phi<0. Nonzero means the
    /// two territorial halves are blooming at different mean levels,
    /// which only happens if phase-locking within each bulk has
    /// produced a different mean phase, which only happens because the
    /// territory routed coupling that way. Three-layer fingerprint.
    pub fn bloom_split(&self) -> f64 {
        let phi = self.territory.c();
        let v = self.chem.v();
        let (mut sp, mut np_, mut sn, mut nn_) = (0.0_f64, 0_usize, 0.0_f64, 0_usize);
        for (p, vi) in phi.iter().zip(v.iter()) {
            if *p > 0.0 { sp += vi; np_ += 1; }
            else if *p < 0.0 { sn += vi; nn_ += 1; }
        }
        if np_ == 0 || nn_ == 0 { return 0.0; }
        (sp / np_ as f64) - (sn / nn_ as f64)
    }

    pub fn phi_field(&self) -> Vec<f64> { self.territory.c().to_vec() }
    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
    pub fn k_coupling_field(&self) -> Vec<f64> { self.k_field.clone() }
    pub fn feed_field(&self) -> Vec<f64> { self.feed_field.clone() }
    pub fn chem_v_field(&self) -> Vec<f64> { self.chem.v().to_vec() }
}

// =====================================================================
// R15: stripes route sync. Swift-Hohenberg (R6) self-organises a
// striped/spotted pattern whose |u| sits near a positive peak inside
// each stripe and crosses zero between stripes. bulk_gate (the same
// operator as R12) reads |u| and produces a per-cell Kuramoto coupling
// that is high inside stripes and low between them. Kuramoto then
// locks along the SH stripes and drifts in the gaps.
//
// Third use of bulk_gate (after R12, R14). New substrate pair (R6 ^
// R9) that no previous rung has touched. Confirms the operator as a
// reusable primitive across distinct geometries.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR15 {
    pattern: SwiftHohenberg2D,
    phase: Kuramoto2D,
    k_field: Vec<f64>,
    k_gap: f64,
    k_stripe: f64,
    half_width: f64,
    sharpness: f64,
    phase_substeps: u32,
}

#[wasm_bindgen]
impl WasmCoupledR15 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Swift-Hohenberg
        sh_r: f64,
        sh_dx: f64,
        sh_dt: f64,
        // Kuramoto
        ph_dt: f64,
        // bulk_gate
        k_gap: f64,
        k_stripe: f64,
        half_width: f64,
        sharpness: f64,
        phase_substeps: u32,
    ) -> Result<WasmCoupledR15, JsError> {
        let pattern = SwiftHohenberg2D::new(width, height, sh_r, sh_dx, sh_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, 0.0, ph_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            pattern, phase,
            k_field: vec![0.0; width * height],
            k_gap, k_stripe, half_width, sharpness,
            phase_substeps: phase_substeps.max(1),
        })
    }

    pub fn step(&mut self) {
        self.pattern.step();
        let _ = bulk_gate(
            self.pattern.u(),
            self.k_gap, self.k_stripe, self.half_width, self.sharpness,
            &mut self.k_field,
        );
        for _ in 0..self.phase_substeps {
            let _ = self.phase.step_with_coupling_field(&self.k_field);
        }
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_pattern(&mut self, amplitude: f64) {
        self.pattern.reset();
        self.pattern.seed_noise(amplitude);
    }
    pub fn reset_pattern(&mut self) { self.pattern.reset(); }
    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }
    pub fn set_r(&mut self, v: f64) { self.pattern.set_r(v); }
    pub fn set_k_gap(&mut self, v: f64) { if v >= 0.0 { self.k_gap = v; } }
    pub fn set_k_stripe(&mut self, v: f64) { if v >= 0.0 { self.k_stripe = v; } }
    pub fn set_half_width(&mut self, v: f64) { if v >= 0.0 { self.half_width = v; } }
    pub fn set_sharpness(&mut self, v: f64) { if v > 0.0 { self.sharpness = v; } }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.pattern.width() }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.pattern.height() }
    #[wasm_bindgen(getter)]
    pub fn pattern_time(&self) -> f64 { self.pattern.time() }
    #[wasm_bindgen(getter)]
    pub fn phase_time(&self) -> f64 { self.phase.time() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }

    /// Local phase correlation, 4-neighbour.
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

    /// Order parameter restricted to stripe cells (|u| above half_width).
    pub fn order_parameter_on_stripes(&self) -> f64 {
        let u = self.pattern.u();
        let theta = self.phase.theta();
        let (mut cs, mut sn, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (ui, ti) in u.iter().zip(theta.iter()) {
            if ui.abs() > self.half_width { cs += ti.cos(); sn += ti.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / n as f64 }
    }

    pub fn order_parameter_in_gaps(&self) -> f64 {
        let u = self.pattern.u();
        let theta = self.phase.theta();
        let (mut cs, mut sn, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (ui, ti) in u.iter().zip(theta.iter()) {
            if ui.abs() < self.half_width { cs += ti.cos(); sn += ti.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (cs * cs + sn * sn).sqrt() / n as f64 }
    }

    /// Fraction of cells classified as "stripe" (|u| above half_width).
    pub fn stripe_fraction(&self) -> f64 {
        let u = self.pattern.u();
        let hw = self.half_width;
        u.iter().filter(|x| x.abs() > hw).count() as f64 / u.len() as f64
    }

    pub fn pattern_field(&self) -> Vec<f64> { self.pattern.u().to_vec() }
    pub fn k_coupling_field(&self) -> Vec<f64> { self.k_field.clone() }
    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
}





// =====================================================================
// R16: walls route sync. Same substrate pair as R12 (Cahn-Hilliard +
// Kuramoto) but routed through the new operator `gradient_magnitude`
// instead of `bulk_gate`. R12 syncs in the +/-1 bulk regions; R16
// syncs on the walls. The operator inverts which spatial feature is
// the synced backbone.
//
// First use of gradient_magnitude. Operator alphabet category:
// "differentiate" -- expose boundaries rather than bulk.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR16 {
    territory: CahnHilliard2D,
    phase: Kuramoto2D,
    grad: Vec<f64>,
    k_field: Vec<f64>,
    k_bulk: f64,
    k_wall: f64,
    grad_ref: f64,
    sharp: f64,
    width: usize,
    height: usize,
    dx: f64,
}

#[wasm_bindgen]
impl WasmCoupledR16 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        // Cahn-Hilliard
        mobility: f64,
        kappa: f64,
        ch_dx: f64,
        ch_dt: f64,
        // Kuramoto
        ph_dt: f64,
        // remap: smoothstep gate around grad_ref +/- sharp
        k_bulk: f64,
        k_wall: f64,
        grad_ref: f64,
        sharp: f64,
    ) -> Result<WasmCoupledR16, JsError> {
        let territory = CahnHilliard2D::new(width, height, mobility, kappa, ch_dx, ch_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let phase = Kuramoto2D::new(width, height, 0.0, ph_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            territory, phase,
            grad: vec![0.0; n],
            k_field: vec![k_bulk; n],
            k_bulk, k_wall,
            grad_ref: if grad_ref > 0.0 { grad_ref } else { 1.0 },
            sharp: if sharp > 0.0 { sharp } else { 0.1 },
            width, height, dx: ch_dx,
        })
    }

    pub fn step(&mut self) {
        self.territory.step();
        let _ = gradient_magnitude(
            self.territory.c(),
            self.width, self.height, self.dx,
            &mut self.grad,
        );
        let span = self.k_wall - self.k_bulk;
        let edge0 = self.grad_ref - self.sharp;
        let edge1 = self.grad_ref + self.sharp;
        let inv = if edge1 > edge0 { 1.0 / (edge1 - edge0) } else { 0.0 };
        for (g, kk) in self.grad.iter().zip(self.k_field.iter_mut()) {
            let t = ((*g - edge0) * inv).clamp(0.0, 1.0);
            let s = t * t * (3.0 - 2.0 * t);
            *kk = self.k_bulk + span * s;
        }
        let _ = self.phase.step_with_coupling_field(&self.k_field);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_territory(&mut self, amplitude: f64, mean: f64, seed: u32) {
        self.territory.seed_noise(amplitude, mean, seed as u64);
    }
    pub fn reset_territory(&mut self) {
        self.territory.reset();
    }
    pub fn pre_evolve_territory(&mut self, n: u32) {
        for _ in 0..n { self.territory.step(); }
    }
    pub fn set_natural_frequencies(&mut self, sigma: f64, seed: u32) {
        self.phase.set_natural_frequencies(sigma, seed as u64);
    }
    pub fn randomise_phases(&mut self, seed: u32) {
        self.phase.randomise_phases(seed as u64);
    }

    pub fn set_k_bulk(&mut self, v: f64) { if v >= 0.0 { self.k_bulk = v; } }
    pub fn set_k_wall(&mut self, v: f64) { if v >= 0.0 { self.k_wall = v; } }
    pub fn set_grad_ref(&mut self, v: f64) { if v > 0.0 { self.grad_ref = v; } }
    pub fn set_sharp(&mut self, v: f64) { if v > 0.0 { self.sharp = v; } }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn territory_time(&self) -> f64 { self.territory.time() }
    #[wasm_bindgen(getter)]
    pub fn phase_time(&self) -> f64 { self.phase.time() }
    #[wasm_bindgen(getter)]
    pub fn order_parameter(&self) -> f64 { self.phase.order_parameter() }

    pub fn mean_grad(&self) -> f64 {
        if self.grad.is_empty() { return 0.0; }
        self.grad.iter().sum::<f64>() / self.grad.len() as f64
    }

    pub fn wall_coverage(&self) -> f64 {
        let thresh = 0.5 * self.grad_ref;
        let n = self.grad.iter().filter(|g| **g > thresh).count();
        n as f64 / self.grad.len() as f64
    }

    /// Order parameter restricted to wall cells (|g| > 0.5*grad_ref).
    pub fn order_parameter_on_walls(&self) -> f64 {
        let thresh = 0.5 * self.grad_ref;
        let theta = self.phase.theta();
        let (mut c, mut s, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (gi, ti) in self.grad.iter().zip(theta.iter()) {
            if *gi > thresh { c += ti.cos(); s += ti.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (c * c + s * s).sqrt() / n as f64 }
    }

    pub fn order_parameter_in_bulk(&self) -> f64 {
        let thresh = 0.5 * self.grad_ref;
        let theta = self.phase.theta();
        let (mut c, mut s, mut n) = (0.0_f64, 0.0_f64, 0_usize);
        for (gi, ti) in self.grad.iter().zip(theta.iter()) {
            if *gi <= thresh { c += ti.cos(); s += ti.sin(); n += 1; }
        }
        if n == 0 { 0.0 } else { (c * c + s * s).sqrt() / n as f64 }
    }

    pub fn local_correlation(&self) -> f64 {
        let w = self.phase.width();
        let h = self.phase.height();
        let theta = self.phase.theta();
        let mut acc = 0.0;
        let mut cnt = 0_usize;
        for j in 0..h {
            let jp = if j == h - 1 { 0 } else { j + 1 };
            for i in 0..w {
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let t0 = theta[j * w + i];
                acc += (theta[j * w + ip] - t0).cos();
                acc += (theta[jp * w + i] - t0).cos();
                cnt += 2;
            }
        }
        if cnt == 0 { 0.0 } else { acc / cnt as f64 }
    }

    pub fn phi_field(&self) -> Vec<f64> { self.territory.c().to_vec() }
    pub fn grad_field(&self) -> Vec<f64> { self.grad.clone() }
    pub fn k_coupling_field(&self) -> Vec<f64> { self.k_field.clone() }
    pub fn theta_field(&self) -> Vec<f64> { self.phase.theta().to_vec() }
}

// =====================================================================
// R17: territory carries dye. CH (R8) phi field is treated as a
// stream function. gradient_field reads (dphi/dx, dphi/dy); rotating
// 90 degrees gives an incompressible velocity v = (dphi/dy, -dphi/dx).
// advect_by carries a passive dye along the streamlines, which are
// the level sets of phi. Walls of the territory become rivers; dye
// circulates along them rather than crossing them.
//
// First use of advect_by (transport primitive) and gradient_field
// (vector read). Operator alphabet category: "transport".
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR17 {
    territory: CahnHilliard2D,
    dye: Vec<f64>,
    dye_next: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    v_scale: f64,
    dt_adv: f64,
    width: usize,
    height: usize,
    dx: f64,
}

#[wasm_bindgen]
impl WasmCoupledR17 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        mobility: f64,
        kappa: f64,
        ch_dx: f64,
        ch_dt: f64,
        dt_adv: f64,
        v_scale: f64,
    ) -> Result<WasmCoupledR17, JsError> {
        let territory = CahnHilliard2D::new(width, height, mobility, kappa, ch_dx, ch_dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            territory,
            dye: vec![0.5; n],
            dye_next: vec![0.0; n],
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            v_scale,
            dt_adv,
            width, height, dx: ch_dx,
        })
    }

    pub fn step(&mut self) {
        self.territory.step();
        let _ = gradient_field(
            self.territory.c(),
            self.width, self.height, self.dx,
            &mut self.gx, &mut self.gy,
        );
        let s = self.v_scale;
        for k in 0..self.gx.len() {
            self.vx[k] =  s * self.gy[k];
            self.vy[k] = -s * self.gx[k];
        }
        let _ = advect_by(
            &self.dye, &self.vx, &self.vy,
            self.width, self.height, self.dx, self.dt_adv,
            &mut self.dye_next,
        );
        std::mem::swap(&mut self.dye, &mut self.dye_next);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_territory(&mut self, amplitude: f64, mean: f64, seed: u32) {
        self.territory.seed_noise(amplitude, mean, seed as u64);
    }
    pub fn reset_territory(&mut self) {
        self.territory.reset();
    }
    pub fn pre_evolve_territory(&mut self, n: u32) {
        for _ in 0..n { self.territory.step(); }
    }

    /// Seed dye as horizontal stripes (sin) with `bands` periods.
    pub fn seed_dye_stripes(&mut self, bands: f64) {
        for j in 0..self.height {
            let s = (j as f64 / self.height as f64 * std::f64::consts::TAU * bands).sin();
            let v = (s + 1.0) * 0.5;
            for i in 0..self.width {
                self.dye[j * self.width + i] = v;
            }
        }
    }

    /// Seed dye as vertical stripes.
    pub fn seed_dye_stripes_vertical(&mut self, bands: f64) {
        for i in 0..self.width {
            let s = (i as f64 / self.width as f64 * std::f64::consts::TAU * bands).sin();
            let v = (s + 1.0) * 0.5;
            for j in 0..self.height {
                self.dye[j * self.width + i] = v;
            }
        }
    }

    pub fn set_v_scale(&mut self, v: f64) { self.v_scale = v; }
    pub fn set_dt_adv(&mut self, v: f64) { if v >= 0.0 { self.dt_adv = v; } }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn territory_time(&self) -> f64 { self.territory.time() }

    /// Total dye mass (sum). Should stay near initial value.
    pub fn dye_mass(&self) -> f64 {
        self.dye.iter().sum()
    }

    /// Variance of the dye field -- shrinks as numerical diffusion
    /// erodes the stripe structure.
    pub fn dye_variance(&self) -> f64 {
        if self.dye.is_empty() { return 0.0; }
        let mean = self.dye.iter().sum::<f64>() / self.dye.len() as f64;
        self.dye.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / self.dye.len() as f64
    }

    /// Mean speed of the velocity field -- diagnostic for how
    /// strongly the territory is currently transporting.
    pub fn mean_speed(&self) -> f64 {
        if self.vx.is_empty() { return 0.0; }
        let mut s = 0.0;
        for k in 0..self.vx.len() {
            s += (self.vx[k] * self.vx[k] + self.vy[k] * self.vy[k]).sqrt();
        }
        s / self.vx.len() as f64
    }

    pub fn phi_field(&self) -> Vec<f64> { self.territory.c().to_vec() }
    pub fn vx_field(&self) -> Vec<f64> { self.vx.clone() }
    pub fn vy_field(&self) -> Vec<f64> { self.vy.clone() }
    pub fn speed_field(&self) -> Vec<f64> {
        self.vx.iter().zip(self.vy.iter())
            .map(|(a, b)| (a * a + b * b).sqrt())
            .collect()
    }
    pub fn dye_field(&self) -> Vec<f64> { self.dye.clone() }
}

// =====================================================================
// R18: waves leave marks. Barkley (R7) propagates spiral / target
// waves; the new operator threshold_event watches u against a
// threshold and emits a 1 wherever u rises through it on this
// step. A per-cell counter accumulates those events into a firing-
// rate map; a last-fire field latches the most recent event time
// for a decaying-trace visualisation.
//
// First use of threshold_event -- operator alphabet's first
// "discretise" primitive. Continuous field in, symbolic events
// out.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR18 {
    tissue: Barkley2D,
    prev_u: Vec<f64>,
    events: Vec<u8>,
    counts: Vec<u32>,
    last_fire: Vec<f64>,
    threshold: f64,
    events_this_step: u32,
    cumulative_events: u64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR18 {
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
        threshold: f64,
    ) -> Result<WasmCoupledR18, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            prev_u: tissue.u().to_vec(),
            tissue,
            events: vec![0; n],
            counts: vec![0; n],
            last_fire: vec![-1.0; n],
            threshold,
            events_this_step: 0,
            cumulative_events: 0,
            width, height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step();
        let _ = threshold_event(
            &self.prev_u,
            self.tissue.u(),
            self.threshold,
            &mut self.events,
        );
        self.events_this_step = 0;
        let t = self.tissue.time();
        for k in 0..self.events.len() {
            if self.events[k] == 1 {
                self.counts[k] = self.counts[k].saturating_add(1);
                self.last_fire[k] = t;
                self.events_this_step += 1;
            }
        }
        self.cumulative_events += self.events_this_step as u64;
        self.prev_u.copy_from_slice(self.tissue.u());
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) {
        self.tissue.seed_spiral();
        self.prev_u.copy_from_slice(self.tissue.u());
        self.reset_marks();
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
        self.prev_u.copy_from_slice(self.tissue.u());
    }

    pub fn reset_tissue(&mut self) {
        self.tissue.reset();
        self.prev_u.copy_from_slice(self.tissue.u());
        self.reset_marks();
    }

    pub fn reset_marks(&mut self) {
        for c in &mut self.counts { *c = 0; }
        for t in &mut self.last_fire { *t = -1.0; }
        self.events_this_step = 0;
        self.cumulative_events = 0;
    }

    pub fn set_threshold(&mut self, v: f64) { self.threshold = v; }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }
    pub fn set_eps(&mut self, e: f64) { self.tissue.set_eps(e); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }
    #[wasm_bindgen(getter)]
    pub fn events_last_step(&self) -> u32 { self.events_this_step }
    #[wasm_bindgen(getter)]
    pub fn cumulative_event_count(&self) -> f64 { self.cumulative_events as f64 }

    /// Fraction of cells that have fired at least once.
    pub fn coverage(&self) -> f64 {
        let hit = self.counts.iter().filter(|c| **c > 0).count();
        hit as f64 / self.counts.len() as f64
    }

    /// Maximum per-cell event count.
    pub fn max_count(&self) -> u32 {
        *self.counts.iter().max().unwrap_or(&0)
    }

    /// Mean firing rate per cell (events / time), averaged over
    /// cells that have fired at least once.
    pub fn mean_rate(&self) -> f64 {
        let t = self.tissue.time().max(1e-9);
        let hit: usize = self.counts.iter().filter(|c| **c > 0).count();
        if hit == 0 { return 0.0; }
        let sum: u64 = self.counts.iter().map(|c| *c as u64).sum();
        sum as f64 / hit as f64 / t
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn v_field(&self) -> Vec<f64> { self.tissue.v().to_vec() }
    pub fn events_field(&self) -> Vec<f64> {
        // 0/1 -> f64 for the shared field renderer.
        self.events.iter().map(|e| *e as f64).collect()
    }
    pub fn counts_field(&self) -> Vec<f64> {
        self.counts.iter().map(|c| *c as f64).collect()
    }
    /// Decaying trace: exp(-(t - last_fire)/tau) per cell, in [0,1].
    pub fn trace_field(&self, tau: f64) -> Vec<f64> {
        let t = self.tissue.time();
        let tau = if tau > 0.0 { tau } else { 1.0 };
        self.last_fire.iter().map(|lf| {
            if *lf < 0.0 { 0.0 } else { (-((t - lf) / tau)).exp() }
        }).collect()
    }
}

// =====================================================================
// R19: memory of waves. Final Phase-A rung. Barkley (R7) drives
// two leaky integrators with different leaks. integrate_field is
// the alphabet's "integrate" primitive -- dual to R18's
// discretise. dy/dt = u - leak*y. leak=0: pure dose meter.
// leak>0: low-pass with time constant tau = 1/leak; steady state
// y_inf = <u>/leak.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR19 {
    tissue: Barkley2D,
    dose: Vec<f64>,            // leak = 0  (pure accumulator)
    avg: Vec<f64>,             // leak > 0  (leaky low-pass)
    leak: f64,
    dt_integrate: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR19 {
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
        leak: f64,
    ) -> Result<WasmCoupledR19, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            dose: vec![0.0; n],
            avg: vec![0.0; n],
            leak,
            dt_integrate: dt,
            width, height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step();
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.dose, self.dt_integrate, 0.0);
        let _ = integrate_field(u, &mut self.avg,  self.dt_integrate, self.leak);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) {
        self.tissue.seed_spiral();
        self.reset_memory();
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn reset_tissue(&mut self) {
        self.tissue.reset();
        self.reset_memory();
    }

    pub fn reset_memory(&mut self) {
        for v in &mut self.dose { *v = 0.0; }
        for v in &mut self.avg  { *v = 0.0; }
    }

    pub fn set_leak(&mut self, leak: f64) {
        self.leak = leak.max(0.0);
    }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }
    pub fn set_eps(&mut self, e: f64) { self.tissue.set_eps(e); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }
    #[wasm_bindgen(getter)]
    pub fn leak_value(&self) -> f64 { self.leak }

    pub fn u_mean(&self) -> f64 {
        let u = self.tissue.u();
        u.iter().sum::<f64>() / u.len() as f64
    }
    pub fn dose_max(&self) -> f64 {
        self.dose.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn dose_mean(&self) -> f64 {
        self.dose.iter().sum::<f64>() / self.dose.len() as f64
    }
    pub fn avg_max(&self) -> f64 {
        self.avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn avg_mean(&self) -> f64 {
        self.avg.iter().sum::<f64>() / self.avg.len() as f64
    }
    /// Predicted steady-state of leaky integrator: <u>/leak.
    pub fn avg_predicted(&self) -> f64 {
        if self.leak <= 0.0 { f64::INFINITY }
        else { self.u_mean() / self.leak }
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn dose_field(&self) -> Vec<f64> { self.dose.clone() }
    pub fn avg_field(&self) -> Vec<f64> { self.avg.clone() }
}

// =====================================================================
// R20: events seed matter. Phase-B opener. Barkley (R7) spikes via
// threshold_event (R18's operator) deposit V into autonomous
// Gray-Scott (R4) chemistry in the stable-spots regime. Each event
// seeds an isolated spot only if the cell and its 4 neighbours are
// empty -- this keeps spots spaced apart so the chemistry has U
// left to feed them, instead of coating the whole grid and
// starving itself. Once seeded, GS runs by itself: turning the
// wave off does *not* collapse the pattern. The wave was creative,
// not modulatory. No new operator -- first Phase-B composition.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR20 {
    wave: Barkley2D,
    chem: GrayScott2D,
    prev_u: Vec<f64>,
    events: Vec<u8>,
    threshold: f64,
    inject_v: f64,
    empty_v: f64,
    wave_on: bool,
    bark_substeps: u32,
    last_events: u32,
    cumulative_events: u64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR20 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
    ) -> Result<WasmCoupledR20, JsError> {
        let wave = Barkley2D::new(width, height, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05)
            .map_err(|e| JsError::new(&e.to_string()))?;
        // Stable spots regime: f=0.030, k=0.062.
        let chem = GrayScott2D::new(width, height, 0.16, 0.08, 0.030, 0.062, 1.0, 1.0)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            prev_u: wave.u().to_vec(),
            wave,
            chem,
            events: vec![0; n],
            threshold: 0.4,
            inject_v: 0.5,
            empty_v: 0.05,
            wave_on: true,
            bark_substeps: 20,
            last_events: 0,
            cumulative_events: 0,
            width,
            height,
        })
    }

    /// One chemistry step. The wave (if on) is advanced
    /// `bark_substeps` times; each substep emits rising-edge
    /// events that seed isolated empty cells in V.
    pub fn step(&mut self) {
        let w = self.width;
        let h = self.height;
        let mut step_events = 0u32;
        if self.wave_on {
            for _ in 0..self.bark_substeps {
                self.prev_u.copy_from_slice(self.wave.u());
                self.wave.step();
                let u = self.wave.u();
                let _ = threshold_event(&self.prev_u, u, self.threshold, &mut self.events);
                for j in 0..h {
                    let jn = if j == 0 { h - 1 } else { j - 1 };
                    let js = if j + 1 == h { 0 } else { j + 1 };
                    for i in 0..w {
                        let idx = j * w + i;
                        if self.events[idx] != 1 {
                            continue;
                        }
                        let iw = if i == 0 { w - 1 } else { i - 1 };
                        let ie = if i + 1 == w { 0 } else { i + 1 };
                        let v = self.chem.v();
                        if v[idx] < self.empty_v
                            && v[j * w + iw] < self.empty_v
                            && v[j * w + ie] < self.empty_v
                            && v[jn * w + i] < self.empty_v
                            && v[js * w + i] < self.empty_v
                        {
                            self.chem.v_mut()[idx] = self.inject_v;
                            step_events += 1;
                        }
                    }
                }
            }
        }
        self.last_events = step_events;
        self.cumulative_events += step_events as u64;
        self.chem.step();
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) {
        self.wave.seed_spiral();
        self.prev_u.copy_from_slice(self.wave.u());
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.wave.kick(cx, cy, radius, amplitude);
        self.prev_u.copy_from_slice(self.wave.u());
    }

    pub fn reset_wave(&mut self) {
        self.wave.reset();
        self.prev_u.copy_from_slice(self.wave.u());
    }

    pub fn reset_chem(&mut self) {
        self.chem.reset();
        self.cumulative_events = 0;
        self.last_events = 0;
    }

    pub fn set_wave_on(&mut self, on: bool) { self.wave_on = on; }
    pub fn set_threshold(&mut self, v: f64) { self.threshold = v; }
    pub fn set_inject(&mut self, v: f64) { self.inject_v = v.clamp(0.0, 1.0); }
    pub fn set_empty(&mut self, v: f64) { self.empty_v = v.clamp(0.0, 1.0); }
    pub fn set_a(&mut self, a: f64) { self.wave.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.wave.set_b(b); }
    pub fn set_eps(&mut self, e: f64) { self.wave.set_eps(e); }
    pub fn set_feed(&mut self, f: f64) { self.chem.set_feed(f); }
    pub fn set_kill(&mut self, k: f64) { self.chem.set_kill(k); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn wave_time(&self) -> f64 { self.wave.time() }
    #[wasm_bindgen(getter)]
    pub fn chem_time(&self) -> f64 { self.chem.time() }
    #[wasm_bindgen(getter)]
    pub fn wave_is_on(&self) -> bool { self.wave_on }
    #[wasm_bindgen(getter)]
    pub fn events_last_step(&self) -> u32 { self.last_events }
    #[wasm_bindgen(getter)]
    pub fn cumulative_event_count(&self) -> f64 { self.cumulative_events as f64 }

    pub fn v_mean(&self) -> f64 { self.chem.mean_v() }
    pub fn v_max(&self) -> f64 { self.chem.max_v() }
    /// Fraction of cells with V > 0.2 (a "lit spot").
    pub fn v_coverage(&self) -> f64 {
        let v = self.chem.v();
        let lit = v.iter().filter(|x| **x > 0.2).count();
        lit as f64 / v.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.wave.u().to_vec() }
    pub fn events_field(&self) -> Vec<f64> {
        self.events.iter().map(|e| *e as f64).collect()
    }
    pub fn chem_v_field(&self) -> Vec<f64> { self.chem.v().to_vec() }
    pub fn chem_u_field(&self) -> Vec<f64> { self.chem.u().to_vec() }
}

// =====================================================================
// R21: sensor and alarm. Phase-B composition. No new operator.
// Chains integrate_field (R19) -> threshold_event (R18) and adds a
// latch. The leaky average smooths the raw Barkley activator u
// over time-constant tau = 1/leak; a threshold_event on the
// *averaged* field fires only when integrated exposure crosses a
// trip level; an OR-latch keeps the alarm bit set once raised.
// Difference from R18: R18 fires on every wave passage at the raw
// signal; R21 fires only when activity has been *sustained*.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR21 {
    tissue: Barkley2D,
    avg: Vec<f64>,
    prev_avg: Vec<f64>,
    events: Vec<u8>,
    alarm: Vec<u8>,
    leak: f64,
    alarm_threshold: f64,
    dt_integrate: f64,
    cumulative_trips: u64,
    last_trips: u32,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR21 {
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
        leak: f64,
        alarm_threshold: f64,
    ) -> Result<WasmCoupledR21, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            avg: vec![0.0; n],
            prev_avg: vec![0.0; n],
            events: vec![0; n],
            alarm: vec![0; n],
            leak,
            alarm_threshold,
            dt_integrate: dt,
            cumulative_trips: 0,
            last_trips: 0,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step();
        let u = self.tissue.u();
        self.prev_avg.copy_from_slice(&self.avg);
        let _ = integrate_field(u, &mut self.avg, self.dt_integrate, self.leak);
        let _ = threshold_event(
            &self.prev_avg,
            &self.avg,
            self.alarm_threshold,
            &mut self.events,
        );
        let mut trips = 0u32;
        for k in 0..self.events.len() {
            if self.events[k] == 1 && self.alarm[k] == 0 {
                self.alarm[k] = 1;
                trips += 1;
            }
        }
        self.last_trips = trips;
        self.cumulative_trips += trips as u64;
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) {
        self.tissue.seed_spiral();
        self.reset_sensor();
        self.reset_alarm();
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn reset_tissue(&mut self) {
        self.tissue.reset();
        self.reset_sensor();
        self.reset_alarm();
    }

    pub fn reset_sensor(&mut self) {
        for v in &mut self.avg { *v = 0.0; }
        for v in &mut self.prev_avg { *v = 0.0; }
    }

    pub fn reset_alarm(&mut self) {
        for a in &mut self.alarm { *a = 0; }
        self.cumulative_trips = 0;
        self.last_trips = 0;
    }

    pub fn set_leak(&mut self, leak: f64) { self.leak = leak.max(0.0); }
    pub fn set_alarm_threshold(&mut self, th: f64) { self.alarm_threshold = th; }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }
    pub fn set_eps(&mut self, e: f64) { self.tissue.set_eps(e); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }
    #[wasm_bindgen(getter)]
    pub fn leak_value(&self) -> f64 { self.leak }
    #[wasm_bindgen(getter)]
    pub fn alarm_threshold_value(&self) -> f64 { self.alarm_threshold }
    #[wasm_bindgen(getter)]
    pub fn trips_last_step(&self) -> u32 { self.last_trips }
    #[wasm_bindgen(getter)]
    pub fn cumulative_trip_count(&self) -> f64 { self.cumulative_trips as f64 }

    pub fn u_mean(&self) -> f64 {
        let u = self.tissue.u();
        u.iter().sum::<f64>() / u.len() as f64
    }
    pub fn avg_mean(&self) -> f64 {
        self.avg.iter().sum::<f64>() / self.avg.len() as f64
    }
    pub fn avg_max(&self) -> f64 {
        self.avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn alarm_coverage(&self) -> f64 {
        let lit = self.alarm.iter().filter(|x| **x == 1).count();
        lit as f64 / self.alarm.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn avg_field(&self) -> Vec<f64> { self.avg.clone() }
    pub fn alarm_field(&self) -> Vec<f64> {
        self.alarm.iter().map(|a| *a as f64).collect()
    }
}

// =====================================================================
// WasmCoupledR22 -- Memory shapes flow. Phase B (B3).
//
// integrate_field(u, memory, dt, leak)   leaky integral of Barkley u
// gradient_field(memory, ...)            memory -> velocity vector
// advect_by(dye, alpha*grad, ...)        dye transported by velocity
//
// First rung where past activity shapes present motion. The dye does
// not see u; it sees only grad(memory) -- the wave's integrated
// trace.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR22 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    gx: Vec<f64>,
    gy: Vec<f64>,
    dye: Vec<f64>,
    dye_next: Vec<f64>,
    leak: f64,
    alpha: f64,
    dt_step: f64,
    dx: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR22 {
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
        leak: f64,
        alpha: f64,
    ) -> Result<WasmCoupledR22, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            memory: vec![0.0; n],
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            dye: vec![0.0; n],
            dye_next: vec![0.0; n],
            leak,
            alpha,
            dt_step: dt,
            dx,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step();
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = gradient_field(&self.memory, self.width, self.height, self.dx, &mut self.gx, &mut self.gy);
        for k in 0..self.gx.len() {
            self.gx[k] *= self.alpha;
            self.gy[k] *= self.alpha;
        }
        let _ = advect_by(
            &self.dye, &self.gx, &self.gy,
            self.width, self.height, self.dx, self.dt_step,
            &mut self.dye_next,
        );
        std::mem::swap(&mut self.dye, &mut self.dye_next);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Place a Gaussian blob of dye at (cx, cy) with std dev `sigma`
    /// and peak amplitude `amp`. Adds on top of existing dye.
    pub fn seed_dye_blob(&mut self, cx: f64, cy: f64, sigma: f64, amp: f64) {
        let s2 = (sigma * sigma).max(1e-6);
        for j in 0..self.height {
            for i in 0..self.width {
                let dxp = i as f64 - cx;
                let dyp = j as f64 - cy;
                let r2 = dxp * dxp + dyp * dyp;
                self.dye[j * self.width + i] += amp * (-r2 / (2.0 * s2)).exp();
            }
        }
    }

    pub fn reset_tissue(&mut self) { self.tissue.reset(); }
    pub fn reset_memory(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
    }
    pub fn reset_dye(&mut self) {
        for v in &mut self.dye { *v = 0.0; }
        for v in &mut self.dye_next { *v = 0.0; }
    }

    pub fn set_leak(&mut self, leak: f64) { self.leak = leak.max(0.0); }
    pub fn set_alpha(&mut self, alpha: f64) { self.alpha = alpha; }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }
    pub fn set_eps(&mut self, e: f64) { self.tissue.set_eps(e); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }

    pub fn u_mean(&self) -> f64 {
        let u = self.tissue.u();
        u.iter().sum::<f64>() / u.len() as f64
    }
    pub fn memory_max(&self) -> f64 {
        self.memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    pub fn velocity_max(&self) -> f64 {
        (0..self.gx.len())
            .map(|k| (self.gx[k] * self.gx[k] + self.gy[k] * self.gy[k]).sqrt())
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }
    pub fn dye_total(&self) -> f64 { self.dye.iter().sum() }
    pub fn dye_max(&self) -> f64 {
        self.dye.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn dye_centroid_x(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..self.height {
            for i in 0..self.width {
                let d = self.dye[j * self.width + i];
                num += i as f64 * d;
                den += d;
            }
        }
        if den > 1e-12 { num / den } else { 0.0 }
    }
    pub fn dye_centroid_y(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..self.height {
            for i in 0..self.width {
                let d = self.dye[j * self.width + i];
                num += j as f64 * d;
                den += d;
            }
        }
        if den > 1e-12 { num / den } else { 0.0 }
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn dye_field(&self) -> Vec<f64> { self.dye.clone() }
    pub fn vx_field(&self) -> Vec<f64> { self.gx.clone() }
    pub fn vy_field(&self) -> Vec<f64> { self.gy.clone() }
}
