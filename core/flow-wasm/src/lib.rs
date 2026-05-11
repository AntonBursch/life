//! wasm-bindgen wrappers around `flow`.
//!
//! Thin layer; the real math lives in `flow`. This crate only handles the
//! JS/Rust interop and exposes the diffusion field in a form a Canvas
//! renderer can consume cheaply.

use flow::{excitable_gate, phase_to_scalar_field, bulk_gate, AdvectionDiffusion1D, Barkley2D, BoundaryCondition, CahnHilliard2D, Convection2D, Diffusion1D, GrayScott2D, Kuramoto2D, SwiftHohenberg2D};
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


