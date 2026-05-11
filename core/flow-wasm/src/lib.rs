//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{excitable_gate, phase_to_scalar_field, bulk_gate, gradient_magnitude, gradient_field, advect_by, threshold_event, integrate_field, modulate_parameter, latch_field, react_field, schlogl_rate, AdvectionDiffusion1D, Barkley2D, BoundaryCondition, CahnHilliard2D, Convection2D, Diffusion1D, GrayScott2D, Kuramoto2D, SwiftHohenberg2D};
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

// =====================================================================
// WasmCoupledR23 -- Excitable courier. Phase B (B4).
//
// gradient_field(u, ...)                  velocity = alpha * grad u
// advect_by(payload, alpha*grad(u), ...)  transport
//
// Difference from R22: NO memory. Velocity is the instantaneous
// slope of the wave, not its integrated trace. The wave carries
// the payload as it passes. Spiral chirality breaks symmetry,
// kicks accumulate coherently.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR23 {
    tissue: Barkley2D,
    gx: Vec<f64>,
    gy: Vec<f64>,
    payload: Vec<f64>,
    payload_next: Vec<f64>,
    alpha: f64,
    dt_step: f64,
    dx: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR23 {
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
        alpha: f64,
    ) -> Result<WasmCoupledR23, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            gx: vec![0.0; n],
            gy: vec![0.0; n],
            payload: vec![0.0; n],
            payload_next: vec![0.0; n],
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
        let _ = gradient_field(u, self.width, self.height, self.dx, &mut self.gx, &mut self.gy);
        for k in 0..self.gx.len() {
            self.gx[k] *= self.alpha;
            self.gy[k] *= self.alpha;
        }
        let _ = advect_by(
            &self.payload, &self.gx, &self.gy,
            self.width, self.height, self.dx, self.dt_step,
            &mut self.payload_next,
        );
        std::mem::swap(&mut self.payload, &mut self.payload_next);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn seed_payload_blob(&mut self, cx: f64, cy: f64, sigma: f64, amp: f64) {
        let s2 = (sigma * sigma).max(1e-6);
        for j in 0..self.height {
            for i in 0..self.width {
                let dxp = i as f64 - cx;
                let dyp = j as f64 - cy;
                let r2 = dxp * dxp + dyp * dyp;
                self.payload[j * self.width + i] += amp * (-r2 / (2.0 * s2)).exp();
            }
        }
    }

    pub fn reset_tissue(&mut self) { self.tissue.reset(); }
    pub fn reset_payload(&mut self) {
        for v in &mut self.payload { *v = 0.0; }
        for v in &mut self.payload_next { *v = 0.0; }
    }

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
    pub fn grad_max(&self) -> f64 {
        (0..self.gx.len())
            .map(|k| (self.gx[k] * self.gx[k] + self.gy[k] * self.gy[k]).sqrt())
            .fold(f64::NEG_INFINITY, f64::max)
            .max(0.0)
    }
    pub fn payload_total(&self) -> f64 { self.payload.iter().sum() }
    pub fn payload_max(&self) -> f64 {
        self.payload.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn payload_centroid_x(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..self.height {
            for i in 0..self.width {
                let p = self.payload[j * self.width + i];
                num += i as f64 * p;
                den += p;
            }
        }
        if den > 1e-12 { num / den } else { 0.0 }
    }
    pub fn payload_centroid_y(&self) -> f64 {
        let mut num = 0.0;
        let mut den = 0.0;
        for j in 0..self.height {
            for i in 0..self.width {
                let p = self.payload[j * self.width + i];
                num += j as f64 * p;
                den += p;
            }
        }
        if den > 1e-12 { num / den } else { 0.0 }
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn payload_field(&self) -> Vec<f64> { self.payload.clone() }
}

// =====================================================================
// WasmCoupledR24 -- Scar tissue. Phase C opener (C1).
//
// The first closed-loop / cybernetic rung. A derived field writes
// back into the substrate's own parameter:
//
//   Barkley.step_with_eps_field(eps_field)        substrate reads
//   integrate_field(u, memory, ..., leak)         R19
//   modulate_parameter(memory, ..., eps_field)    new operator
//
// History changes responsiveness. Where the wave has been, the
// tissue takes longer to recover, the wave is pushed away from
// its own trail.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR24 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    leak: f64,
    gain: f64,
    eps_min: f64,
    eps_max: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR24 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        leak: f64,
        gain: f64,
        eps_max: f64,
    ) -> Result<WasmCoupledR24, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            memory: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            leak,
            gain,
            eps_min: base_eps,
            eps_max,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.eps_min);
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = modulate_parameter(
            &self.memory,
            self.base_eps,
            self.gain,
            self.eps_min,
            self.eps_max,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn reset_tissue(&mut self) { self.tissue.reset(); }
    pub fn reset_memory(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_base_eps(&mut self, e: f64) {
        self.base_eps = e;
        self.eps_min = e;
        self.tissue.set_eps(e);
    }
    pub fn set_leak(&mut self, leak: f64) { self.leak = leak.max(0.0); }
    pub fn set_gain(&mut self, gain: f64) { self.gain = gain; }
    pub fn set_eps_max(&mut self, m: f64) { self.eps_max = m.max(self.eps_min); }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

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
    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn memory_max(&self) -> f64 {
        self.memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    pub fn eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }
    pub fn eps_max_current(&self) -> f64 {
        self.eps_field.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    /// Fraction of cells whose eps has been raised above 1.5 * base_eps.
    /// A "how much scar tissue" readout.
    pub fn scar_fraction(&self) -> f64 {
        let thresh = self.base_eps * 1.5;
        let lit = self.eps_field.iter().filter(|&&e| e > thresh).count();
        lit as f64 / self.eps_field.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR25 -- Homeostasis. Phase C, second rung.
//
// First *negative-feedback* control loop. Total Barkley activity
// (excited_fraction) drives a single global eps offset to hold the
// system at a chosen target activity. Reuses the parametrise idea
// from R24 but as a scalar: one number controls the whole tissue.
//
// Control law (each step, after warmup):
//   err        = excited_fraction(u) - target
//   eps_offset = clamp((1 - leak*dt)*eps_offset + k*err*dt, 0, eps_max-base)
//   sim.set_eps(base + eps_offset)
//
// More activity than target -> eps rises -> wave gets harder
// to sustain -> activity falls. Less activity -> offset relaxes
// back. Classic leaky integrator controller on a non-linear plant.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR25 {
    tissue: Barkley2D,
    base_eps: f64,
    eps_offset: f64,
    target: f64,
    control_gain: f64,
    control_leak: f64,
    eps_max: f64,
    dt_step: f64,
    warmup_left: u32,
    controller_on: bool,
    width: usize,
    height: usize,
    // Rolling trace of (activity, eps_global) for the viewer plot.
    trace_activity: Vec<f64>,
    trace_eps: Vec<f64>,
    trace_cap: usize,
}

#[wasm_bindgen]
impl WasmCoupledR25 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        target: f64,
        control_gain: f64,
        eps_max: f64,
        warmup: u32,
        trace_cap: usize,
    ) -> Result<WasmCoupledR25, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(Self {
            tissue,
            base_eps,
            eps_offset: 0.0,
            target,
            control_gain,
            control_leak: 0.5,
            eps_max,
            dt_step: dt,
            warmup_left: warmup,
            controller_on: true,
            width,
            height,
            trace_activity: Vec::with_capacity(trace_cap),
            trace_eps: Vec::with_capacity(trace_cap),
            trace_cap,
        })
    }

    pub fn step(&mut self) {
        let activity = self.tissue.excited_fraction();
        let mut eps_global = self.base_eps + self.eps_offset;
        if self.warmup_left > 0 {
            self.warmup_left -= 1;
        } else if self.controller_on {
            let err = activity - self.target;
            let proposed = (1.0 - self.control_leak * self.dt_step) * self.eps_offset
                + self.control_gain * err * self.dt_step;
            let max_off = (self.eps_max - self.base_eps).max(0.0);
            self.eps_offset = proposed.max(0.0).min(max_off);
            eps_global = self.base_eps + self.eps_offset;
            self.tissue.set_eps(eps_global);
        }
        self.tissue.step();
        // Record trace (sub-sample by always pushing -- caller can
        // step_many at modest rates so the buffer is meaningful).
        if self.trace_cap > 0 {
            if self.trace_activity.len() >= self.trace_cap {
                self.trace_activity.remove(0);
                self.trace_eps.remove(0);
            }
            self.trace_activity.push(activity);
            self.trace_eps.push(eps_global);
        }
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn reset_tissue(&mut self) {
        self.tissue.reset();
        self.eps_offset = 0.0;
        self.tissue.set_eps(self.base_eps);
        self.trace_activity.clear();
        self.trace_eps.clear();
    }
    pub fn reset_controller(&mut self) {
        self.eps_offset = 0.0;
        self.tissue.set_eps(self.base_eps);
    }
    pub fn clear_trace(&mut self) {
        self.trace_activity.clear();
        self.trace_eps.clear();
    }

    pub fn set_target(&mut self, t: f64) { self.target = t; }
    pub fn set_control_gain(&mut self, k: f64) { self.control_gain = k; }
    pub fn set_control_leak(&mut self, l: f64) { self.control_leak = l.max(0.0); }
    pub fn set_eps_max(&mut self, m: f64) { self.eps_max = m.max(self.base_eps); }
    pub fn set_controller_on(&mut self, on: bool) {
        self.controller_on = on;
        if !on {
            self.eps_offset = 0.0;
            self.tissue.set_eps(self.base_eps);
        }
    }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }
    #[wasm_bindgen(getter)]
    pub fn warmup_remaining(&self) -> u32 { self.warmup_left }
    #[wasm_bindgen(getter)]
    pub fn controller_active(&self) -> bool { self.controller_on && self.warmup_left == 0 }

    pub fn u_mean(&self) -> f64 {
        let u = self.tissue.u();
        u.iter().sum::<f64>() / u.len() as f64
    }
    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn eps_global(&self) -> f64 { self.base_eps + self.eps_offset }
    pub fn eps_offset(&self) -> f64 { self.eps_offset }
    pub fn target(&self) -> f64 { self.target }
    pub fn error(&self) -> f64 { self.tissue.excited_fraction() - self.target }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn trace_activity(&self) -> Vec<f64> { self.trace_activity.clone() }
    pub fn trace_eps(&self) -> Vec<f64> { self.trace_eps.clone() }
}

// =====================================================================
// WasmCoupledR26 -- Self-bounding. Phase C, third rung.
//
// Memory passes through bulk_gate with a *sharp* sigmoid: where
// memory exceeds the threshold, the cell becomes a wall (eps =
// kill_eps); below threshold, normal tissue (eps = base). The
// wave does not equilibrate with itself -- it carves a topological
// partition. Same chain skeleton as R24, but bulk_gate replaces
// modulate_parameter and sharpness becomes the qualitative knob.
//
// Chain:
//   Barkley.step_with_eps_field(eps)
//   integrate_field(u, memory, dt, leak)
//   bulk_gate(memory, base, kill_eps, threshold, sharpness, eps)
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR26 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    threshold: f64,
    sharpness: f64,
    leak: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR26 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        threshold: f64,
        sharpness: f64,
        leak: f64,
    ) -> Result<WasmCoupledR26, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            memory: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            threshold,
            sharpness,
            leak,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = bulk_gate(
            &self.memory,
            self.base_eps,
            self.kill_eps,
            self.threshold,
            self.sharpness,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn reset_tissue(&mut self) { self.tissue.reset(); }
    pub fn reset_memory(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_threshold(&mut self, t: f64) { self.threshold = t.max(0.0); }
    pub fn set_sharpness(&mut self, s: f64) { self.sharpness = s.max(1e-4); }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }
    pub fn set_leak(&mut self, l: f64) { self.leak = l.max(0.0); }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

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
    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn memory_max(&self) -> f64 {
        self.memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    /// Fraction of cells whose memory exceeds the threshold -- the
    /// "wall fraction". This is the wave's self-built boundary.
    pub fn wall_fraction(&self) -> f64 {
        let thr = self.threshold;
        let lit = self.memory.iter().filter(|&&m| m > thr).count();
        lit as f64 / self.memory.len() as f64
    }
    pub fn eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }
    pub fn eps_max_current(&self) -> f64 {
        self.eps_field.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
    /// Binary wall mask: 1.0 where memory > threshold, 0.0 elsewhere.
    /// Cheaper render path than normalising eps_field.
    pub fn wall_mask(&self) -> Vec<f64> {
        self.memory.iter()
            .map(|&m| if m > self.threshold { 1.0 } else { 0.0 })
            .collect()
    }
}

// =====================================================================
// WasmCoupledR27 -- Latched death. Phase D, first rung.
//
// First operator with persistent state of its own: latch_field.
// In R26 walls only persisted because the wave kept writing them;
// kill the wave and the walls dissolve. In R27 a per-cell Schmitt
// trigger holds wall state: once a cell crosses set_threshold it
// becomes a wall and (with reset_threshold = 0) remains one forever,
// regardless of subsequent activity. The structure outlives the
// process that built it.
//
// Chain:
//   Barkley.step_with_eps_field(eps)
//   integrate_field(u, memory, dt, leak)            -- R19 op
//   latch_field(wall_state, memory, set, reset)     -- NEW op (latch category)
//   modulate_parameter(wall_state, ...) -> eps      -- R24 op
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR27 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    wall_state: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    set_threshold: f64,
    reset_threshold: f64,
    leak: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR27 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        set_threshold: f64,
        reset_threshold: f64,
        leak: f64,
    ) -> Result<WasmCoupledR27, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            memory: vec![0.0; n],
            wall_state: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            set_threshold,
            reset_threshold,
            leak,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = latch_field(
            &mut self.wall_state,
            &self.memory,
            self.set_threshold,
            self.reset_threshold,
        );
        let _ = modulate_parameter(
            &self.wall_state,
            self.base_eps,
            self.kill_eps - self.base_eps,
            self.base_eps,
            self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Kill the wave (zero u and v) but leave wall_state intact.
    /// This is the headline R27 demo: structure outlives process.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// "Heal" walls: clear both memory and the latch. Explicit reset,
    /// independent of any input.
    pub fn reset_walls(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
        for v in &mut self.wall_state { *v = 0.0; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_set_threshold(&mut self, t: f64) {
        self.set_threshold = t.max(self.reset_threshold);
    }
    pub fn set_reset_threshold(&mut self, t: f64) {
        self.reset_threshold = t.min(self.set_threshold).max(0.0);
    }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }
    pub fn set_leak(&mut self, l: f64) { self.leak = l.max(0.0); }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

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
    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn memory_max(&self) -> f64 {
        self.memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max).max(0.0)
    }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    /// Fraction of cells currently latched into the wall state.
    /// Persistent: even after the wave is killed, this stays put.
    pub fn wall_fraction(&self) -> f64 {
        self.wall_state.iter().sum::<f64>() / self.wall_state.len() as f64
    }
    pub fn eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn wall_field(&self) -> Vec<f64> { self.wall_state.clone() }
    pub fn eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR28 -- Communication. Phase D, second rung.
//
// Pure composition (no new operators). Builds an information
// channel by composing latch_field + advect_by + latch_field:
//
//   memory      = integrate_field(u)                       -- R19
//   wall_local  = latch_field(memory, set_local, reset_local)  -- R27 op
//   tmp         = advect_by(transmitted, vx, vy, dt)       -- R6 op
//   transmitted = latch_field(tmp, wall_local,             -- R27 op,
//                             set=0.5, reset=-1.0)            one-way
//   eps         = modulate_parameter(transmitted, ...)     -- R24 op
//
// The local latch carries R27's death-as-state: a cell where the
// wave has been long enough is permanently a wall. The transmitted
// field is a second one-way latch whose state is the *advected*
// history of the local latch -- it never erases and it drifts
// rightward at velocity v. eps is driven by the TRANSMITTED field,
// not the local one, so each cell experiences walls that
// originated some time ago at position x - v*t. Information from
// one region's history arrives in another region's present without
// any wave ever travelling between them.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR28 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    wall_local: Vec<f64>,
    transmitted: Vec<f64>,
    advect_tmp: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    set_local: f64,
    reset_local: f64,
    leak: f64,
    dx_step: f64,
    dt_step: f64,
    velocity: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR28 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        set_local: f64,
        reset_local: f64,
        leak: f64,
        velocity: f64,
    ) -> Result<WasmCoupledR28, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        Ok(Self {
            tissue,
            memory: vec![0.0; n],
            wall_local: vec![0.0; n],
            transmitted: vec![0.0; n],
            advect_tmp: vec![0.0; n],
            vx: vec![velocity; n],
            vy: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            set_local,
            reset_local,
            leak,
            dx_step: dx,
            dt_step: dt,
            velocity,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = latch_field(
            &mut self.wall_local,
            &self.memory,
            self.set_local,
            self.reset_local,
        );
        // Transport the transmitted-walls field rightward by v*dt.
        let _ = advect_by(
            &self.transmitted,
            &self.vx,
            &self.vy,
            self.width,
            self.height,
            self.dx_step,
            self.dt_step,
            &mut self.advect_tmp,
        );
        std::mem::swap(&mut self.transmitted, &mut self.advect_tmp);
        // One-way latch: any cell where wall_local fires turns transmitted
        // on permanently. reset = -1.0 means transmitted never erases.
        let _ = latch_field(
            &mut self.transmitted,
            &self.wall_local,
            0.5,
            -1.0,
        );
        let _ = modulate_parameter(
            &self.transmitted,
            self.base_eps,
            self.kill_eps - self.base_eps,
            self.base_eps,
            self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Kill the wave. Both wall_local and transmitted are unaffected.
    /// Watch transmitted continue to drift after the wave is gone --
    /// the channel keeps delivering messages from the dead spiral.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Clear memory and both latches, restoring fresh tissue everywhere.
    pub fn reset_walls(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
        for v in &mut self.wall_local { *v = 0.0; }
        for v in &mut self.transmitted { *v = 0.0; }
        for v in &mut self.advect_tmp { *v = 0.0; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_velocity(&mut self, v: f64) {
        self.velocity = v;
        for cell in &mut self.vx { *cell = v; }
    }
    pub fn set_set_local(&mut self, t: f64) {
        self.set_local = t.max(self.reset_local);
    }
    pub fn set_reset_local(&mut self, t: f64) {
        self.reset_local = t.min(self.set_local).max(0.0);
    }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }
    pub fn set_leak(&mut self, l: f64) { self.leak = l.max(0.0); }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }

    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    pub fn wall_local_fraction(&self) -> f64 {
        self.wall_local.iter().sum::<f64>() / self.wall_local.len() as f64
    }
    pub fn transmitted_fraction(&self) -> f64 {
        // Count "on" cells (transmitted > 0.5 after latch).
        let n_on = self.transmitted.iter().filter(|&&x| x > 0.5).count();
        n_on as f64 / self.transmitted.len() as f64
    }
    /// Fraction of cells in the right half of the grid whose transmitted
    /// field is on. Useful for verifying that information has crossed
    /// the midline -- the local walls did not.
    pub fn transmitted_right_fraction(&self) -> f64 {
        let w = self.width;
        let h = self.height;
        let mid = w / 2;
        let mut on = 0usize;
        let mut total = 0usize;
        for j in 0..h {
            let row = j * w;
            for i in mid..w {
                total += 1;
                if self.transmitted[row + i] > 0.5 { on += 1; }
            }
        }
        if total == 0 { 0.0 } else { on as f64 / total as f64 }
    }
    pub fn wall_local_right_fraction(&self) -> f64 {
        let w = self.width;
        let h = self.height;
        let mid = w / 2;
        let mut on = 0usize;
        let mut total = 0usize;
        for j in 0..h {
            let row = j * w;
            for i in mid..w {
                total += 1;
                if self.wall_local[row + i] > 0.5 { on += 1; }
            }
        }
        if total == 0 { 0.0 } else { on as f64 / total as f64 }
    }
    pub fn eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn wall_local_field(&self) -> Vec<f64> { self.wall_local.clone() }
    pub fn transmitted_field(&self) -> Vec<f64> { self.transmitted.clone() }
    pub fn eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR29 -- Convergence. Phase D, third rung.
//
// Still no new operator. Same composition as R28 -- latch_field
// composed with advect_by composed with latch_field -- but the
// velocity field is position-dependent:
//
//   vx[i,j] = +v   if i <  W/2     (left half flows rightward)
//   vx[i,j] = -v   if i >= W/2     (right half flows leftward)
//
// Two spirals are seeded at opposite ends. Each spiral builds
// wall_local in its footprint. Those local walls are advected
// *inward* and latched permanently into transmitted. The two
// channels converge: cells near the midline receive walls that
// arrived from both directions. Kill both waves and the centre
// remains permanently walled by messages from sources that no
// longer exist -- a permanent record of having been spoken to
// from both sides.
//
// Communication composes. Two channels through the same medium
// do not interfere structurally; they OR into the same
// transmitted-latch field. The midline carries a superposition
// of histories.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR29 {
    tissue: Barkley2D,
    memory: Vec<f64>,
    wall_local: Vec<f64>,
    transmitted: Vec<f64>,
    advect_tmp: Vec<f64>,
    vx: Vec<f64>,
    vy: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    set_local: f64,
    reset_local: f64,
    leak: f64,
    dx_step: f64,
    dt_step: f64,
    velocity: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR29 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        set_local: f64,
        reset_local: f64,
        leak: f64,
        velocity: f64,
    ) -> Result<WasmCoupledR29, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let mut s = Self {
            tissue,
            memory: vec![0.0; n],
            wall_local: vec![0.0; n],
            transmitted: vec![0.0; n],
            advect_tmp: vec![0.0; n],
            vx: vec![0.0; n],
            vy: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            set_local,
            reset_local,
            leak,
            dx_step: dx,
            dt_step: dt,
            velocity,
            width,
            height,
        };
        s.rebuild_velocity_field();
        Ok(s)
    }

    fn rebuild_velocity_field(&mut self) {
        let w = self.width;
        let h = self.height;
        let mid = w / 2;
        for j in 0..h {
            let row = j * w;
            for i in 0..w {
                // Left half flows right (+v), right half flows left (-v).
                // Cells AT the midline get -v so they sample slightly
                // to the right -- which is the symmetric mirror of
                // the cell just before the midline sampling slightly
                // to its left. The midline is the collision boundary.
                let v = if i < mid { self.velocity } else { -self.velocity };
                self.vx[row + i] = v;
                self.vy[row + i] = 0.0;
            }
        }
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);
        let u = self.tissue.u();
        let _ = integrate_field(u, &mut self.memory, self.dt_step, self.leak);
        let _ = latch_field(
            &mut self.wall_local,
            &self.memory,
            self.set_local,
            self.reset_local,
        );
        let _ = advect_by(
            &self.transmitted,
            &self.vx,
            &self.vy,
            self.width,
            self.height,
            self.dx_step,
            self.dt_step,
            &mut self.advect_tmp,
        );
        std::mem::swap(&mut self.transmitted, &mut self.advect_tmp);
        let _ = latch_field(
            &mut self.transmitted,
            &self.wall_local,
            0.5,
            -1.0,
        );
        let _ = modulate_parameter(
            &self.transmitted,
            self.base_eps,
            self.kill_eps - self.base_eps,
            self.base_eps,
            self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Seed two spirals, one on each side, facing each other.
    /// Done as two kicks; the substrate's excitable dynamics turn
    /// localised excitations into self-sustaining spirals after a
    /// few hundred steps.
    pub fn seed_two_sources(&mut self) {
        self.tissue.reset();
        let w = self.width;
        let h = self.height;
        let r = ((w.min(h) as f64) * 0.06) as usize + 2;
        self.tissue.kick(w / 5,        h / 2, r, 1.0);
        self.tissue.kick(4 * w / 5,    h / 2, r, 1.0);
    }

    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    pub fn reset_walls(&mut self) {
        for v in &mut self.memory { *v = 0.0; }
        for v in &mut self.wall_local { *v = 0.0; }
        for v in &mut self.transmitted { *v = 0.0; }
        for v in &mut self.advect_tmp { *v = 0.0; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_velocity(&mut self, v: f64) {
        self.velocity = v;
        self.rebuild_velocity_field();
    }
    pub fn set_set_local(&mut self, t: f64) {
        self.set_local = t.max(self.reset_local);
    }
    pub fn set_reset_local(&mut self, t: f64) {
        self.reset_local = t.min(self.set_local).max(0.0);
    }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }
    pub fn set_leak(&mut self, l: f64) { self.leak = l.max(0.0); }
    pub fn set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn set_b(&mut self, b: f64) { self.tissue.set_b(b); }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> usize { self.width }
    #[wasm_bindgen(getter)]
    pub fn height(&self) -> usize { self.height }
    #[wasm_bindgen(getter)]
    pub fn tissue_time(&self) -> f64 { self.tissue.time() }

    pub fn excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn memory_mean(&self) -> f64 {
        self.memory.iter().sum::<f64>() / self.memory.len() as f64
    }
    pub fn wall_local_fraction(&self) -> f64 {
        self.wall_local.iter().sum::<f64>() / self.wall_local.len() as f64
    }
    pub fn transmitted_fraction(&self) -> f64 {
        let n_on = self.transmitted.iter().filter(|&&x| x > 0.5).count();
        n_on as f64 / self.transmitted.len() as f64
    }
    /// Fraction of midline cells (a vertical strip of width ~10% W
    /// centred at W/2) whose transmitted field is on. This is the
    /// "collision zone" -- walls arriving here originated from both
    /// the left and the right channels.
    pub fn midline_transmitted_fraction(&self) -> f64 {
        let w = self.width;
        let h = self.height;
        let mid = w / 2;
        let half_band = (w / 20).max(1);
        let lo = mid.saturating_sub(half_band);
        let hi = (mid + half_band).min(w);
        let mut on = 0usize;
        let mut total = 0usize;
        for j in 0..h {
            let row = j * w;
            for i in lo..hi {
                total += 1;
                if self.transmitted[row + i] > 0.5 { on += 1; }
            }
        }
        if total == 0 { 0.0 } else { on as f64 / total as f64 }
    }
    pub fn midline_wall_local_fraction(&self) -> f64 {
        let w = self.width;
        let h = self.height;
        let mid = w / 2;
        let half_band = (w / 20).max(1);
        let lo = mid.saturating_sub(half_band);
        let hi = (mid + half_band).min(w);
        let mut on = 0usize;
        let mut total = 0usize;
        for j in 0..h {
            let row = j * w;
            for i in lo..hi {
                total += 1;
                if self.wall_local[row + i] > 0.5 { on += 1; }
            }
        }
        if total == 0 { 0.0 } else { on as f64 / total as f64 }
    }
    pub fn eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn memory_field(&self) -> Vec<f64> { self.memory.clone() }
    pub fn wall_local_field(&self) -> Vec<f64> { self.wall_local.clone() }
    pub fn transmitted_field(&self) -> Vec<f64> { self.transmitted.clone() }
    pub fn eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}


// =====================================================================
// WasmCoupledR27Prime -- Bistable death. Phase E, first rung.
//
// Substrate-honest rebuild of R27. See life/THESIS.md.
//
// R27 used latch_field (Schmitt trigger) to make walls irreversible.
// Nature has no Schmitt triggers. Nature has bistable reaction
// networks. R27' replaces the latch with the Schlogl model:
//
//   A + 2X -> 3X         (autocatalysis)
//   X -> B               (decay)
//
//   dX/dt = k1A*X^2 - k2*X^3 - k3*X + k4B  +  D * [u - u_thr]+
//
// With (k1A, k2, k3, k4B) = (6, 1, 11, 6) the bare rate factors
// as -(X-1)(X-2)(X-3): low stable at X=1, separatrix at X=2,
// high stable at X=3. The drive term is real chemistry too --
// the wave activator above threshold boosts the autocatalysis
// rate locally. No comparators, no free knobs.
//
// Chain:
//   1. Barkley.step_with_eps_field(eps_field)
//   2. react_field(X, schlogl + wave_drive)     -- NEW Phase E op
//   3. modulate_parameter(X-X_low, ...) -> eps  -- input is now a
//      real species concentration, not a memory label.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR27Prime {
    tissue: Barkley2D,
    species_x: Vec<f64>,
    x_shifted: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    k1a: f64,
    k2: f64,
    k3: f64,
    k4b: f64,
    x_low: f64,
    x_high: f64,
    u_thr: f64,
    drive: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR27Prime {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        u_thr: f64,
        drive: f64,
    ) -> Result<WasmCoupledR27Prime, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let x_low = 1.0_f64;
        let x_high = 3.0_f64;
        Ok(Self {
            tissue,
            species_x: vec![x_low; n],
            x_shifted: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            k1a: 6.0, k2: 1.0, k3: 11.0, k4b: 6.0,
            x_low, x_high,
            u_thr,
            drive,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);

        let u = self.tissue.u();
        let bare = schlogl_rate(self.k1a, self.k2, self.k3, self.k4b);
        let n = self.species_x.len();
        for k in 0..n {
            let drive_k = self.drive * (u[k] - self.u_thr).max(0.0);
            let mut one = [self.species_x[k]];
            let _ = react_field(
                &mut one,
                |x| bare(x) + drive_k,
                self.dt_step,
            );
            self.species_x[k] = one[0];
        }

        let gain = (self.kill_eps - self.base_eps) / (self.x_high - self.x_low);
        for k in 0..n {
            self.x_shifted[k] = self.species_x[k] - self.x_low;
        }
        let _ = modulate_parameter(
            &self.x_shifted,
            self.base_eps, gain,
            self.base_eps, self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Kill the wave (zero u and v) but leave the X field intact.
    /// Headline R27' demo: structure outlives process, from real
    /// chemistry. Every cell past X=2 falls into the high attractor
    /// on its own and stays there with no input.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Reset chemistry: X back to its low stable state everywhere.
    pub fn reset_chemistry(&mut self) {
        for v in &mut self.species_x { *v = self.x_low; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_drive(&mut self, d: f64) { self.drive = d.max(0.0); }
    pub fn set_u_thr(&mut self, t: f64) { self.u_thr = t.clamp(0.0, 1.5); }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }
    pub fn r27p_set_a(&mut self, a: f64) { self.tissue.set_a(a); }
    pub fn r27p_set_b(&mut self, b: f64) { self.tissue.set_b(b); }

    pub fn r27p_width(&self) -> usize { self.width }
    pub fn r27p_height(&self) -> usize { self.height }
    pub fn r27p_time(&self) -> f64 { self.tissue.time() }
    pub fn r27p_x_low(&self) -> f64 { self.x_low }
    pub fn r27p_x_high(&self) -> f64 { self.x_high }

    pub fn r27p_excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn x_mean(&self) -> f64 {
        self.species_x.iter().sum::<f64>() / self.species_x.len() as f64
    }
    /// Fraction of cells past the unstable separatrix (X > 2).
    /// These cells will fall into the high attractor on their own
    /// with zero forcing -- chemical irreversibility.
    pub fn x_high_fraction(&self) -> f64 {
        let n = self.species_x.len() as f64;
        self.species_x.iter().filter(|&&x| x > 2.0).count() as f64 / n
    }
    pub fn r27p_eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn r27p_u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn x_field(&self) -> Vec<f64> { self.species_x.clone() }
    pub fn r27p_eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR28Prime -- Bistable communication. Phase E, second rung.
//
// Substrate-honest rebuild of R28. See life/THESIS.md.
//
// R28 used `latch_field -> advect_by(latched_bit) -> latch_field`
// to make walls travel without their builder. The advected field
// was a Schmitt-trigger output -- a label, not a substance.
// Nature transports reactants, not labels.
//
// R28' uses the R27' Schlogl species X (a real concentration) and
// advects *it* by a uniform velocity field. The wave's footprint
// commits cells on the left to X = 3; advection carries that mass
// rightward; downstream cells receive inflow that lifts them past
// the X = 2 separatrix, after which their own Schlogl dynamics
// commit them autonomously. Communication is mass-action plus
// transport. No comparators anywhere.
//
// Chain each tick:
//   1. Barkley.step_with_eps_field(eps_field)
//   2. react_field(X, schlogl + wave_drive)      -- local chemistry
//   3. advect_by(X, vx, vy, dt)                  -- transport of X
//   4. modulate_parameter(X - X_low, ...) -> eps -- refractoriness
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR28Prime {
    tissue: Barkley2D,
    species_x: Vec<f64>,
    x_tmp: Vec<f64>,
    x_shifted: Vec<f64>,
    vx_field: Vec<f64>,
    vy_field: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    k1a: f64,
    k2: f64,
    k3: f64,
    k4b: f64,
    x_low: f64,
    x_high: f64,
    u_thr: f64,
    drive: f64,
    velocity_x: f64,
    velocity_y: f64,
    dx_step: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR28Prime {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        u_thr: f64,
        drive: f64,
        velocity: f64,
    ) -> Result<WasmCoupledR28Prime, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let x_low = 1.0_f64;
        let x_high = 3.0_f64;
        Ok(Self {
            tissue,
            species_x: vec![x_low; n],
            x_tmp: vec![0.0; n],
            x_shifted: vec![0.0; n],
            vx_field: vec![velocity; n],
            vy_field: vec![0.0; n],
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            k1a: 6.0, k2: 1.0, k3: 11.0, k4b: 6.0,
            x_low, x_high,
            u_thr,
            drive,
            velocity_x: velocity,
            velocity_y: 0.0,
            dx_step: dx,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        // 1. Barkley substrate.
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);

        // 2. Local Schlogl reaction with wave-coupled drive.
        let u = self.tissue.u();
        let bare = schlogl_rate(self.k1a, self.k2, self.k3, self.k4b);
        let n = self.species_x.len();
        for k in 0..n {
            let drive_k = self.drive * (u[k] - self.u_thr).max(0.0);
            let mut one = [self.species_x[k]];
            let _ = react_field(
                &mut one,
                |x| bare(x) + drive_k,
                self.dt_step,
            );
            self.species_x[k] = one[0];
        }

        // 3. Transport X by (vx, vy). Semi-Lagrangian, unconditionally stable.
        let _ = advect_by(
            &self.species_x,
            &self.vx_field, &self.vy_field,
            self.width, self.height,
            self.dx_step, self.dt_step,
            &mut self.x_tmp,
        );
        std::mem::swap(&mut self.species_x, &mut self.x_tmp);

        // 4. eps from X (refractoriness proportional to species).
        let gain = (self.kill_eps - self.base_eps) / (self.x_high - self.x_low);
        for k in 0..n {
            self.x_shifted[k] = self.species_x[k] - self.x_low;
        }
        let _ = modulate_parameter(
            &self.x_shifted,
            self.base_eps, gain,
            self.base_eps, self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Kill the wave; leave the X field and its in-flight transport intact.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Reset chemistry: X back to its low stable state everywhere.
    pub fn reset_chemistry(&mut self) {
        for v in &mut self.species_x { *v = self.x_low; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_drive(&mut self, d: f64) { self.drive = d.max(0.0); }
    pub fn set_u_thr(&mut self, t: f64) { self.u_thr = t.clamp(0.0, 1.5); }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }

    /// Set uniform horizontal velocity (cells / time unit). Negative = leftward.
    pub fn set_velocity_x(&mut self, vx: f64) {
        self.velocity_x = vx;
        for v in &mut self.vx_field { *v = vx; }
    }
    pub fn set_velocity_y(&mut self, vy: f64) {
        self.velocity_y = vy;
        for v in &mut self.vy_field { *v = vy; }
    }

    pub fn r28p_width(&self) -> usize { self.width }
    pub fn r28p_height(&self) -> usize { self.height }
    pub fn r28p_time(&self) -> f64 { self.tissue.time() }
    pub fn r28p_x_low(&self) -> f64 { self.x_low }
    pub fn r28p_x_high(&self) -> f64 { self.x_high }
    pub fn r28p_velocity_x(&self) -> f64 { self.velocity_x }

    pub fn r28p_excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn r28p_x_mean(&self) -> f64 {
        self.species_x.iter().sum::<f64>() / self.species_x.len() as f64
    }
    /// Global fraction of cells past the X = 2 separatrix.
    pub fn r28p_x_high_fraction(&self) -> f64 {
        let n = self.species_x.len() as f64;
        self.species_x.iter().filter(|&&x| x > 2.0).count() as f64 / n
    }
    /// Right-half fraction past the separatrix -- cells the spiral
    /// likely never reached. The "communication" metric.
    pub fn r28p_x_high_fraction_right(&self) -> f64 {
        let mid = self.width / 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in mid..self.width {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    /// Left-half fraction past the separatrix -- where the spiral is.
    pub fn r28p_x_high_fraction_left(&self) -> f64 {
        let mid = self.width / 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..mid {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    pub fn r28p_eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn r28p_u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn r28p_x_field(&self) -> Vec<f64> { self.species_x.clone() }
    pub fn r28p_eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR29Prime -- Bistable convergence. Phase E, third rung.
//
// Substrate-honest rebuild of R29. See life/THESIS.md.
//
// Same composition as R28' (react + advect + parametrise) with
// one change: the velocity field is position-dependent. The left
// half flows rightward at +v, the right half flows leftward at
// -v. Two source reservoirs clamp X at X_high (an honest
// Dirichlet boundary condition -- a continuously-supplied
// metabolic reservoir). Reactant from both sides converges on
// the midline.
//
// The honest finding: while the clamp is active, the midline
// reaches commitment ABOVE either source alone (the OR property
// of two converging channels). When the clamp is released,
// advection drains committed cells and the midline returns to
// X = 1 -- in an open advecting system, real chemistry cannot
// hold structure without ongoing supply. R29's 'permanent record'
// was an artifact of latch_field, not of nature.
//
// Chain each tick:
//   1. (if clamp_on) species_x[source_disks] = X_high
//   2. Barkley.step_with_eps_field(eps_field)
//   3. react_field(X, schlogl + wave_drive)
//   4. advect_by(X, vx, vy, dt)            -- vx position-dependent
//   5. modulate_parameter(X - X_low, ...) -> eps
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR29Prime {
    tissue: Barkley2D,
    species_x: Vec<f64>,
    x_tmp: Vec<f64>,
    x_shifted: Vec<f64>,
    vx_field: Vec<f64>,
    vy_field: Vec<f64>,
    eps_field: Vec<f64>,
    source_mask: Vec<bool>,
    base_eps: f64,
    kill_eps: f64,
    k1a: f64,
    k2: f64,
    k3: f64,
    k4b: f64,
    x_low: f64,
    x_high: f64,
    u_thr: f64,
    drive: f64,
    velocity: f64,
    source_radius: usize,
    clamp_on: bool,
    dx_step: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR29Prime {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        u_thr: f64,
        drive: f64,
        velocity: f64,
        source_radius: usize,
    ) -> Result<WasmCoupledR29Prime, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let x_low = 1.0_f64;
        let x_high = 3.0_f64;

        // Position-dependent velocity field: +v on left half, -v on right half.
        let mid = width / 2;
        let mut vx_field = vec![0.0_f64; n];
        for j in 0..height {
            let row = j * width;
            for i in 0..width {
                vx_field[row + i] = if i < mid { velocity } else { -velocity };
            }
        }

        // Source-disk masks at (W/5, H/2) and (4W/5, H/2).
        let src_l = (width / 5, height / 2);
        let src_r = (4 * width / 5, height / 2);
        let mut source_mask = vec![false; n];
        let r2 = (source_radius as i32).pow(2);
        let wi = width as i32;
        let hi = height as i32;
        for (cx, cy) in [src_l, src_r] {
            let cxi = cx as i32;
            let cyi = cy as i32;
            for dj in -(source_radius as i32)..=(source_radius as i32) {
                for di in -(source_radius as i32)..=(source_radius as i32) {
                    if di * di + dj * dj > r2 { continue; }
                    let i = ((cxi + di).rem_euclid(wi)) as usize;
                    let j = ((cyi + dj).rem_euclid(hi)) as usize;
                    source_mask[j * width + i] = true;
                }
            }
        }

        Ok(Self {
            tissue,
            species_x: vec![x_low; n],
            x_tmp: vec![0.0; n],
            x_shifted: vec![0.0; n],
            vx_field,
            vy_field: vec![0.0; n],
            eps_field: vec![base_eps; n],
            source_mask,
            base_eps,
            kill_eps,
            k1a: 6.0, k2: 1.0, k3: 11.0, k4b: 6.0,
            x_low, x_high,
            u_thr,
            drive,
            velocity,
            source_radius,
            clamp_on: true,
            dx_step: dx,
            dt_step: dt,
            width,
            height,
        })
    }

    pub fn step(&mut self) {
        // 1. Source clamp.
        if self.clamp_on {
            for k in 0..self.species_x.len() {
                if self.source_mask[k] {
                    self.species_x[k] = self.x_high;
                }
            }
        }

        // 2. Barkley substrate.
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);

        // 3. Schlogl reaction with wave-coupled drive.
        let u = self.tissue.u();
        let bare = schlogl_rate(self.k1a, self.k2, self.k3, self.k4b);
        let n = self.species_x.len();
        for k in 0..n {
            let drive_k = self.drive * (u[k] - self.u_thr).max(0.0);
            let mut one = [self.species_x[k]];
            let _ = react_field(
                &mut one,
                |x| bare(x) + drive_k,
                self.dt_step,
            );
            self.species_x[k] = one[0];
        }

        // 4. Position-dependent transport.
        let _ = advect_by(
            &self.species_x,
            &self.vx_field, &self.vy_field,
            self.width, self.height,
            self.dx_step, self.dt_step,
            &mut self.x_tmp,
        );
        std::mem::swap(&mut self.species_x, &mut self.x_tmp);

        // 5. eps from X.
        let gain = (self.kill_eps - self.base_eps) / (self.x_high - self.x_low);
        for k in 0..n {
            self.x_shifted[k] = self.species_x[k] - self.x_low;
        }
        let _ = modulate_parameter(
            &self.x_shifted,
            self.base_eps, gain,
            self.base_eps, self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Zero the Barkley substrate. X field and source clamp persist.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Reset chemistry: X back to its low stable state everywhere.
    /// (If the clamp is still on, source disks will be re-clamped on
    /// the next step.)
    pub fn reset_chemistry(&mut self) {
        for v in &mut self.species_x { *v = self.x_low; }
        for v in &mut self.eps_field { *v = self.base_eps; }
    }

    pub fn set_drive(&mut self, d: f64) { self.drive = d.max(0.0); }
    pub fn set_u_thr(&mut self, t: f64) { self.u_thr = t.clamp(0.0, 1.5); }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }

    /// Set the magnitude of the inward velocity. Stored as
    /// +velocity on the left half, -velocity on the right half.
    pub fn set_velocity(&mut self, v: f64) {
        self.velocity = v;
        let mid = self.width / 2;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..self.width {
                self.vx_field[row + i] = if i < mid { v } else { -v };
            }
        }
    }

    /// Turn the source-reservoir clamp on or off. The headline test:
    /// release the clamp and watch the midline drain.
    pub fn set_clamp_on(&mut self, on: bool) { self.clamp_on = on; }
    pub fn r29p_clamp_on(&self) -> bool { self.clamp_on }

    pub fn r29p_width(&self) -> usize { self.width }
    pub fn r29p_height(&self) -> usize { self.height }
    pub fn r29p_time(&self) -> f64 { self.tissue.time() }
    pub fn r29p_x_low(&self) -> f64 { self.x_low }
    pub fn r29p_x_high(&self) -> f64 { self.x_high }
    pub fn r29p_velocity(&self) -> f64 { self.velocity }

    pub fn r29p_excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn r29p_x_mean(&self) -> f64 {
        self.species_x.iter().sum::<f64>() / self.species_x.len() as f64
    }
    pub fn r29p_x_high_fraction(&self) -> f64 {
        let n = self.species_x.len() as f64;
        self.species_x.iter().filter(|&&x| x > 2.0).count() as f64 / n
    }
    /// Left-half (i < W/2) X_high fraction.
    pub fn r29p_x_high_fraction_left(&self) -> f64 {
        let mid = self.width / 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..mid {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    /// Right-half (i >= W/2) X_high fraction.
    pub fn r29p_x_high_fraction_right(&self) -> f64 {
        let mid = self.width / 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in mid..self.width {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    /// Midline-band (i in [mid-W/20, mid+W/20]) X_high fraction.
    /// This is the "accumulator" -- two channels OR into here.
    pub fn r29p_x_high_fraction_mid(&self) -> f64 {
        let mid = self.width / 2;
        let half = (self.width / 20).max(1);
        let lo = mid - half;
        let hi_idx = mid + half;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in lo..hi_idx {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    pub fn r29p_eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn r29p_u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn r29p_x_field(&self) -> Vec<f64> { self.species_x.clone() }
    pub fn r29p_eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR30 -- Enclosure. Phase E, fourth rung.
//
// A closed boundary made of real chemistry, maintained by a
// downhill flow of fuel. See life/THESIS.md.
//
// Two coupled species fields on top of the Barkley wave:
//   X -- Schlogl bistable (R27' chemistry), but the constant
//        reservoir term k4*B is replaced by a SPATIAL field
//        B(x). Where B is high, the bistable has X = 1 / 2 / 3
//        fixed points; where B is low, only the low fp survives.
//   B -- fuel field. Clamped at the grid boundary to B_supply
//        (Dirichlet BC). First-order consumption at rate
//        LAMBDA_B everywhere (the downhill that pays for the
//        structure). Advected inward by a 4-way radial velocity
//        field.
//
// On construction X is seeded high in a wall-thick ring at the
// boundary. While supply > 0, B remains high enough there for
// the chemistry to hold X at its high fp, and eps stays high
// in the wall -> the Barkley spiral inside is enclosed. When
// supply drops to 0 (set_supply(0.0)), the ring's B is washed
// out by advection within ~50 t.u. and the wall drains: the
// bistable's high fp disappears under it. Cleanest honest
// substrate "membrane".
//
// Chain each tick:
//   1. Dirichlet clamp on a 2-cell border: b[edge] = supply.
//   2. Barkley.step_with_eps_field(eps).
//   3. react_field on X with rate(x; B_local, drive_k):
//        dX/dt = k1A x^2 - k2 x^3 - k3 x + k4 B(x) + drive.
//   4. react_field on B with rate(b) = -LAMBDA_B * b
//      (consumption; the downhill).
//   5. advect_by(B, vx, vy)   -- radial-inward velocity field.
//   6. modulate_parameter(X - X_low, ...) -> eps.
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR30 {
    tissue: Barkley2D,
    species_x: Vec<f64>,
    b_field: Vec<f64>,
    b_tmp: Vec<f64>,
    x_shifted: Vec<f64>,
    vx_field: Vec<f64>,
    vy_field: Vec<f64>,
    eps_field: Vec<f64>,
    base_eps: f64,
    kill_eps: f64,
    k1a: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    x_low: f64,
    x_high: f64,
    u_thr: f64,
    drive: f64,
    velocity: f64,
    supply: f64,
    lambda_b: f64,
    wall_thick: usize,
    dx_step: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR30 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        kill_eps: f64,
        u_thr: f64,
        drive: f64,
        velocity: f64,
        supply: f64,
        lambda_b: f64,
        wall_thick: usize,
    ) -> Result<WasmCoupledR30, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let x_low = 1.0_f64;
        let x_high = 3.0_f64;

        // Radial-inward 4-way velocity field.
        let midx = width / 2;
        let midy = height / 2;
        let mut vx_field = vec![0.0_f64; n];
        let mut vy_field = vec![0.0_f64; n];
        for j in 0..height {
            let row = j * width;
            for i in 0..width {
                vx_field[row + i] = if i < midx { velocity } else { -velocity };
                vy_field[row + i] = if j < midy { velocity } else { -velocity };
            }
        }

        // Seed X high in the wall ring at the boundary.
        let mut species_x = vec![x_low; n];
        for j in 0..height {
            let row = j * width;
            for i in 0..width {
                let near_edge =
                    i < wall_thick
                    || i >= width - wall_thick
                    || j < wall_thick
                    || j >= height - wall_thick;
                if near_edge {
                    species_x[row + i] = x_high;
                }
            }
        }

        Ok(Self {
            tissue,
            species_x,
            b_field: vec![supply; n],
            b_tmp: vec![0.0; n],
            x_shifted: vec![0.0; n],
            vx_field,
            vy_field,
            eps_field: vec![base_eps; n],
            base_eps,
            kill_eps,
            k1a: 6.0, k2: 1.0, k3: 11.0, k4: 6.0,
            x_low, x_high,
            u_thr,
            drive,
            velocity,
            supply,
            lambda_b,
            wall_thick,
            dx_step: dx,
            dt_step: dt,
            width,
            height,
        })
    }

    fn clamp_border(&mut self) {
        let w = self.width;
        let h = self.height;
        let s = self.supply;
        for i in 0..w {
            self.b_field[i] = s;
            self.b_field[w + i] = s;
            self.b_field[(h - 1) * w + i] = s;
            self.b_field[(h - 2) * w + i] = s;
        }
        for j in 0..h {
            let row = j * w;
            self.b_field[row] = s;
            self.b_field[row + 1] = s;
            self.b_field[row + w - 1] = s;
            self.b_field[row + w - 2] = s;
        }
    }

    pub fn step(&mut self) {
        // 1. Dirichlet clamp.
        self.clamp_border();

        // 2. Barkley wave step.
        self.tissue.step_with_eps_field(&self.eps_field, self.base_eps);

        // 3. Schlogl reaction with local B and wave drive.
        let u = self.tissue.u();
        let n = self.species_x.len();
        for k in 0..n {
            let drive_k = self.drive * (u[k] - self.u_thr).max(0.0);
            let b_k = self.b_field[k];
            let k1a = self.k1a; let k2 = self.k2;
            let k3 = self.k3; let k4 = self.k4;
            let rate = |x: f64| {
                k1a * x * x - k2 * x * x * x - k3 * x + k4 * b_k + drive_k
            };
            let mut one = [self.species_x[k]];
            let _ = react_field(&mut one, rate, self.dt_step);
            self.species_x[k] = one[0];
        }

        // 4. Fuel consumption.
        let lam = self.lambda_b;
        let _ = react_field(
            &mut self.b_field,
            |b| -lam * b,
            self.dt_step,
        );

        // 5. Radial-inward advection of B.
        let _ = advect_by(
            &self.b_field,
            &self.vx_field, &self.vy_field,
            self.width, self.height,
            self.dx_step, self.dt_step,
            &mut self.b_tmp,
        );
        std::mem::swap(&mut self.b_field, &mut self.b_tmp);

        // 6. eps from X.
        let gain = (self.kill_eps - self.base_eps) / (self.x_high - self.x_low);
        for k in 0..n {
            self.x_shifted[k] = self.species_x[k] - self.x_low;
        }
        let _ = modulate_parameter(
            &self.x_shifted,
            self.base_eps, gain,
            self.base_eps, self.kill_eps,
            &mut self.eps_field,
        );
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Reset X and B to baseline + re-seed the wall ring.
    /// The supply stays at its current value.
    pub fn reset_chemistry(&mut self) {
        let n = self.species_x.len();
        for k in 0..n {
            self.species_x[k] = self.x_low;
            self.b_field[k] = self.supply;
            self.eps_field[k] = self.base_eps;
        }
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..self.width {
                let near_edge =
                    i < self.wall_thick
                    || i >= self.width - self.wall_thick
                    || j < self.wall_thick
                    || j >= self.height - self.wall_thick;
                if near_edge {
                    self.species_x[row + i] = self.x_high;
                }
            }
        }
    }

    pub fn set_drive(&mut self, d: f64) { self.drive = d.max(0.0); }
    pub fn set_u_thr(&mut self, t: f64) { self.u_thr = t.clamp(0.0, 1.5); }
    pub fn set_kill_eps(&mut self, e: f64) { self.kill_eps = e.max(self.base_eps); }

    /// Set the magnitude of the inward velocity. Stored as
    /// +velocity on the inward halves.
    pub fn set_velocity(&mut self, v: f64) {
        self.velocity = v;
        let midx = self.width / 2;
        let midy = self.height / 2;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..self.width {
                self.vx_field[row + i] = if i < midx { v } else { -v };
                self.vy_field[row + i] = if j < midy { v } else { -v };
            }
        }
    }

    /// Set the boundary supply concentration. Drop to 0.0 to see
    /// the wall drain -- the honest "spends a downhill flow" test.
    pub fn set_supply(&mut self, s: f64) { self.supply = s.max(0.0); }

    /// Set the first-order fuel consumption rate.
    pub fn set_lambda_b(&mut self, l: f64) { self.lambda_b = l.max(0.0); }

    pub fn r30_width(&self) -> usize { self.width }
    pub fn r30_height(&self) -> usize { self.height }
    pub fn r30_time(&self) -> f64 { self.tissue.time() }
    pub fn r30_x_low(&self) -> f64 { self.x_low }
    pub fn r30_x_high(&self) -> f64 { self.x_high }
    pub fn r30_supply(&self) -> f64 { self.supply }
    pub fn r30_velocity(&self) -> f64 { self.velocity }
    pub fn r30_lambda_b(&self) -> f64 { self.lambda_b }
    pub fn r30_wall_thick(&self) -> usize { self.wall_thick }

    pub fn r30_excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn r30_x_mean(&self) -> f64 {
        self.species_x.iter().sum::<f64>() / self.species_x.len() as f64
    }
    pub fn r30_b_mean(&self) -> f64 {
        self.b_field.iter().sum::<f64>() / self.b_field.len() as f64
    }
    pub fn r30_x_high_fraction(&self) -> f64 {
        let n = self.species_x.len() as f64;
        self.species_x.iter().filter(|&&x| x > 2.0).count() as f64 / n
    }
    /// Cells in the outer ring (within ring_w of the edge) that
    /// are committed to X_high. This is the wall fraction.
    pub fn r30_x_high_fraction_ring(&self) -> f64 {
        let ring_w = self.wall_thick * 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..self.width {
                let in_ring =
                    i < ring_w
                    || i >= self.width - ring_w
                    || j < ring_w
                    || j >= self.height - ring_w;
                if in_ring {
                    total += 1;
                    if self.species_x[row + i] > 2.0 { hi += 1; }
                }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    /// Cells in the interior that are committed to X_high. This
    /// should stay at 0 -- the wave is enclosed.
    pub fn r30_x_high_fraction_core(&self) -> f64 {
        let ring_w = self.wall_thick * 2;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in ring_w..(self.height - ring_w) {
            let row = j * self.width;
            for i in ring_w..(self.width - ring_w) {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }
    pub fn r30_eps_mean(&self) -> f64 {
        self.eps_field.iter().sum::<f64>() / self.eps_field.len() as f64
    }

    pub fn r30_u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn r30_x_field(&self) -> Vec<f64> { self.species_x.clone() }
    pub fn r30_b_field(&self) -> Vec<f64> { self.b_field.clone() }
    pub fn r30_eps_field(&self) -> Vec<f64> { self.eps_field.clone() }
}

// =====================================================================
// WasmCoupledR31 -- Autocatalytic wall. Phase E, fifth rung.
//
// Substrate-honest self-assembly. See life/THESIS.md.
//
// Difference from R30:
//  - No wall seed at t=0. X starts at X_LOW everywhere.
//  - The Barkley spiral at the centre is the only driver. As its
//    waves sweep outward, the wave drive transiently lifts X past
//    the Schlogl separatrix. In cells with high local B (the
//    fuel-rich ring near the boundary) the high fp X = 3 exists,
//    so committed cells STAY committed after the wave passes. In
//    the fuel-starved core the high fp does not exist; X relaxes
//    back to X_LOW.
//  - No modulate_parameter -- eps is held at base everywhere. The
//    wave is the autocatalytic propagator; X chemistry is the
//    commit-gate; B is the spatial selector. Decoupling eps from
//    X lets us isolate the assembly question (R32 will restore
//    the feedback and study stability).
//
// No new operator. Same alphabet (react_field, advect_by, Barkley
// step), one fewer in the loop.
//
// Chain each tick:
//   1. clamp_border(B, B_supply)
//   2. Barkley.step()                              -- uniform eps
//   3. react_field(X, schlogl(local B) + wave_drive)
//   4. react_field(B, -lambda_B * B)               -- fuel consumption
//   5. advect_by(B, vx, vy, dt)                    -- inward transport
// =====================================================================
#[wasm_bindgen]
pub struct WasmCoupledR31 {
    tissue: Barkley2D,
    species_x: Vec<f64>,
    b_field: Vec<f64>,
    b_tmp: Vec<f64>,
    vx_field: Vec<f64>,
    vy_field: Vec<f64>,
    k1a: f64,
    k2: f64,
    k3: f64,
    k4: f64,
    x_low: f64,
    x_high: f64,
    u_thr: f64,
    drive: f64,
    velocity: f64,
    supply: f64,
    lambda_b: f64,
    dx_step: f64,
    dt_step: f64,
    width: usize,
    height: usize,
}

#[wasm_bindgen]
impl WasmCoupledR31 {
    #[wasm_bindgen(constructor)]
    pub fn new(
        width: usize,
        height: usize,
        diffusion: f64,
        a: f64,
        b: f64,
        base_eps: f64,
        dx: f64,
        dt: f64,
        u_thr: f64,
        drive: f64,
        velocity: f64,
        supply: f64,
        lambda_b: f64,
    ) -> Result<WasmCoupledR31, JsError> {
        let tissue = Barkley2D::new(width, height, diffusion, a, b, base_eps, dx, dt)
            .map_err(|e| JsError::new(&e.to_string()))?;
        let n = width * height;
        let x_low = 1.0_f64;
        let x_high = 3.0_f64;
        let mut vx_field = vec![0.0_f64; n];
        let mut vy_field = vec![0.0_f64; n];
        let midx = width / 2;
        let midy = height / 2;
        for j in 0..height {
            let row = j * width;
            for i in 0..width {
                vx_field[row + i] = if i < midx { velocity } else { -velocity };
                vy_field[row + i] = if j < midy { velocity } else { -velocity };
            }
        }
        Ok(Self {
            tissue,
            species_x: vec![x_low; n],
            b_field: vec![supply; n],
            b_tmp: vec![0.0; n],
            vx_field,
            vy_field,
            k1a: 6.0, k2: 1.0, k3: 11.0, k4: 6.0,
            x_low, x_high,
            u_thr,
            drive,
            velocity,
            supply,
            lambda_b,
            dx_step: dx,
            dt_step: dt,
            width,
            height,
        })
    }

    fn clamp_border_inplace(&mut self) {
        let w = self.width;
        let h = self.height;
        let s = self.supply;
        for i in 0..w {
            self.b_field[i] = s;
            self.b_field[w + i] = s;
            self.b_field[(h - 1) * w + i] = s;
            self.b_field[(h - 2) * w + i] = s;
        }
        for j in 0..h {
            let row = j * w;
            self.b_field[row] = s;
            self.b_field[row + 1] = s;
            self.b_field[row + w - 1] = s;
            self.b_field[row + w - 2] = s;
        }
    }

    pub fn step(&mut self) {
        // 1. Dirichlet boundary for B.
        self.clamp_border_inplace();

        // 2. Wave with uniform eps (no feedback in this rung).
        self.tissue.step();

        // 3. Schlogl reaction for X with local B and wave drive.
        let u = self.tissue.u();
        let n = self.species_x.len();
        for k in 0..n {
            let drive_k = self.drive * (u[k] - self.u_thr).max(0.0);
            let b_k = self.b_field[k];
            let k1a = self.k1a; let k2 = self.k2; let k3 = self.k3; let k4 = self.k4;
            let rate_x = |x: f64| {
                k1a * x * x - k2 * x * x * x - k3 * x + k4 * b_k + drive_k
            };
            let mut one = [self.species_x[k]];
            let _ = react_field(&mut one, rate_x, self.dt_step);
            self.species_x[k] = one[0];
        }

        // 4. Fuel consumption (the downhill).
        let lambda = self.lambda_b;
        let _ = react_field(&mut self.b_field, |b| -lambda * b, self.dt_step);

        // 5. Inward advection of B.
        let _ = advect_by(
            &self.b_field,
            &self.vx_field, &self.vy_field,
            self.width, self.height,
            self.dx_step, self.dt_step,
            &mut self.b_tmp,
        );
        std::mem::swap(&mut self.b_field, &mut self.b_tmp);
    }

    pub fn step_many(&mut self, n: u32) {
        for _ in 0..n { self.step(); }
    }

    pub fn seed_spiral(&mut self) { self.tissue.seed_spiral(); }

    pub fn kick(&mut self, cx: usize, cy: usize, radius: usize, amplitude: f64) {
        self.tissue.kick(cx, cy, radius, amplitude);
    }

    /// Kill the wave; X and B remain intact.
    pub fn reset_tissue(&mut self) { self.tissue.reset(); }

    /// Reset chemistry: X back to X_low, B back to supply everywhere.
    pub fn reset_chemistry(&mut self) {
        for v in &mut self.species_x { *v = self.x_low; }
        for v in &mut self.b_field { *v = self.supply; }
        for v in &mut self.b_tmp { *v = 0.0; }
    }

    pub fn set_drive(&mut self, d: f64) { self.drive = d.max(0.0); }
    pub fn set_u_thr(&mut self, t: f64) { self.u_thr = t.clamp(0.0, 1.5); }
    pub fn set_supply(&mut self, s: f64) { self.supply = s.max(0.0); }
    pub fn set_lambda_b(&mut self, l: f64) { self.lambda_b = l.max(0.0); }
    pub fn set_velocity(&mut self, v: f64) {
        self.velocity = v;
        let midx = self.width / 2;
        let midy = self.height / 2;
        for j in 0..self.height {
            let row = j * self.width;
            for i in 0..self.width {
                self.vx_field[row + i] = if i < midx { v } else { -v };
                self.vy_field[row + i] = if j < midy { v } else { -v };
            }
        }
    }

    pub fn r31_width(&self) -> usize { self.width }
    pub fn r31_height(&self) -> usize { self.height }
    pub fn r31_time(&self) -> f64 { self.tissue.time() }
    pub fn r31_x_low(&self) -> f64 { self.x_low }
    pub fn r31_x_high(&self) -> f64 { self.x_high }
    pub fn r31_supply(&self) -> f64 { self.supply }
    pub fn r31_velocity(&self) -> f64 { self.velocity }
    pub fn r31_lambda_b(&self) -> f64 { self.lambda_b }

    pub fn r31_excited_fraction(&self) -> f64 { self.tissue.excited_fraction() }
    pub fn r31_x_mean(&self) -> f64 {
        self.species_x.iter().sum::<f64>() / self.species_x.len() as f64
    }
    pub fn r31_b_mean(&self) -> f64 {
        self.b_field.iter().sum::<f64>() / self.b_field.len() as f64
    }

    /// Fraction of cells in the outer ring (within ring_w = 12 cells
    /// of the boundary) past the X = 2 separatrix.
    pub fn r31_x_high_fraction_ring(&self) -> f64 {
        let ring_w = 12usize;
        let w = self.width; let h = self.height;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in 0..h {
            let row = j * w;
            for i in 0..w {
                let in_core = i >= ring_w && i < w - ring_w
                    && j >= ring_w && j < h - ring_w;
                if !in_core {
                    total += 1;
                    if self.species_x[row + i] > 2.0 { hi += 1; }
                }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }

    /// Fraction of cells in the core (interior, beyond ring_w from the
    /// boundary) past X = 2. Should stay near zero.
    pub fn r31_x_high_fraction_core(&self) -> f64 {
        let ring_w = 12usize;
        let w = self.width; let h = self.height;
        let mut hi = 0usize;
        let mut total = 0usize;
        for j in ring_w..(h - ring_w) {
            let row = j * w;
            for i in ring_w..(w - ring_w) {
                total += 1;
                if self.species_x[row + i] > 2.0 { hi += 1; }
            }
        }
        if total == 0 { 0.0 } else { hi as f64 / total as f64 }
    }

    pub fn r31_u_field(&self) -> Vec<f64> { self.tissue.u().to_vec() }
    pub fn r31_x_field(&self) -> Vec<f64> { self.species_x.clone() }
    pub fn r31_b_field(&self) -> Vec<f64> { self.b_field.clone() }
}
