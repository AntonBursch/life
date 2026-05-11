//! R31 -- Autocatalytic wall. The boundary self-assembles from a
//! single transient kick at the center. Phase E, fifth rung.
//!
//! Difference from R30:
//!  - No wall seed at t=0. X starts at X_LOW everywhere.
//!  - The Barkley spiral at the center is the only driver. As
//!    its waves sweep outward, the wave drive transiently lifts
//!    X past the Schlogl separatrix. In cells with high local B
//!    (the fuel-rich ring near the boundary) the high fp X = 3
//!    exists, so committed cells STAY committed after the wave
//!    passes. In the fuel-starved core the high fp does not
//!    exist; X relaxes back to X_LOW.
//!  - No modulate_parameter -- eps stays at BASE_EPS everywhere.
//!    Reason: we want to isolate the autocatalysis question.
//!    Wave-as-delivery + chemistry-as-commit-gate. R32 will
//!    re-introduce eps feedback and study stability.
//!
//! No new operator. Same alphabet (react_field, advect_by,
//! Barkley2D step), just one fewer operator in the loop.

use flow::{advect_by, react_field, Barkley2D};

const W: usize = 128;
const H: usize = 128;

// Schlogl chemistry (R27' values).
const K1A: f64 = 6.0;
const K2: f64 = 1.0;
const K3: f64 = 11.0;
const K4: f64 = 6.0;
const X_LOW: f64 = 1.0;

// Wave coupling -- a stronger drive than R27'/R30 because we are
// asking a single wave passage to push X past the separatrix
// from rest, with no help from a pre-seeded ring.
const U_THR: f64 = 0.5;
const DRIVE: f64 = 8.0;

// eps is held constant -- no feedback to the wave.
const BASE_EPS: f64 = 0.02;

// Fuel field B (R30 values).
const B_SUPPLY: f64 = 1.0;
const LAMBDA_B: f64 = 0.10;
const VELOCITY: f64 = 1.0;

const DX: f64 = 1.0;
const DT: f64 = 0.02;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, BASE_EPS, DX, DT).unwrap();

    // Single kick: a spiral seeded once at the center. This is
    // the only thing that distinguishes the initial state from
    // a uniform field at X = X_LOW. No wall is seeded.
    sim.seed_spiral();

    let n = W * H;
    let mut species_x = vec![X_LOW; n];
    let mut b_field = vec![B_SUPPLY; n];
    let mut b_tmp = vec![0.0_f64; n];

    let midx = W / 2;
    let midy = H / 2;
    let mut vx = vec![0.0_f64; n];
    let mut vy = vec![0.0_f64; n];
    for j in 0..H {
        let row = j * W;
        for i in 0..W {
            vx[row + i] = if i < midx { VELOCITY } else { -VELOCITY };
            vy[row + i] = if j < midy { VELOCITY } else { -VELOCITY };
        }
    }

    let clamp_border = |b: &mut [f64], supply: f64| {
        for i in 0..W {
            b[i] = supply;
            b[W + i] = supply;
            b[(H - 1) * W + i] = supply;
            b[(H - 2) * W + i] = supply;
        }
        for j in 0..H {
            let row = j * W;
            b[row] = supply;
            b[row + 1] = supply;
            b[row + W - 1] = supply;
            b[row + W - 2] = supply;
        }
    };

    println!("R31 autocatalytic wall: ring self-assembles from a single kick.");
    println!("Grid {}x{}, |v|={} c/t.u. radially inward, lambda_B={}, dt={}.",
        W, H, VELOCITY, LAMBDA_B, DT);
    println!("No wall seeded. Spiral at center is the only driver.");
    println!("  step   t   excited  <X>  X_hi_ring  X_hi_core  <B>  B_ring  B_core");
    println!("  ----  ---  -------  ---  ---------  ---------  ---  ------  ------");

    let ring_w = 12usize;
    let core_lo = ring_w;
    let core_hi_x = W - ring_w;
    let core_hi_y = H - ring_w;

    let checkpoints: [u32; 10] = [200, 500, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 16000];
    let mut step = 0u32;

    for &cp in &checkpoints {
        while step < cp {
            // 1. Dirichlet clamp of B at the boundary (always on
            //    in this rung -- we are studying assembly, not drain).
            clamp_border(&mut b_field, B_SUPPLY);

            // 2. Barkley wave step with uniform eps.
            sim.step();

            // 3. Schlogl reaction for X, with local B and wave drive.
            let u_snap = sim.u().to_vec();
            for k in 0..n {
                let drive_k = DRIVE * (u_snap[k] - U_THR).max(0.0);
                let b_k = b_field[k];
                let rate_x = |x: f64| {
                    K1A * x * x - K2 * x * x * x - K3 * x + K4 * b_k + drive_k
                };
                let mut one = [species_x[k]];
                react_field(&mut one, rate_x, DT).unwrap();
                species_x[k] = one[0];
            }

            // 4. Fuel consumption.
            react_field(&mut b_field, |b| -LAMBDA_B * b, DT).unwrap();

            // 5. Inward advection of B.
            advect_by(&b_field, &vx, &vy, W, H, DX, DT, &mut b_tmp).unwrap();
            std::mem::swap(&mut b_field, &mut b_tmp);

            step += 1;
        }

        let t = (step as f64) * DT;
        let excited = sim.excited_fraction();
        let x_mean: f64 = species_x.iter().sum::<f64>() / (n as f64);
        let b_mean: f64 = b_field.iter().sum::<f64>() / (n as f64);

        let mut ring_hi = 0usize;
        let mut ring_n = 0usize;
        let mut core_hi = 0usize;
        let mut core_n = 0usize;
        let mut b_ring_sum = 0.0_f64;
        let mut b_core_sum = 0.0_f64;
        for j in 0..H {
            let row = j * W;
            for i in 0..W {
                let in_core = i >= core_lo
                    && i < core_hi_x
                    && j >= core_lo
                    && j < core_hi_y;
                if in_core {
                    core_n += 1;
                    if species_x[row + i] > 2.0 {
                        core_hi += 1;
                    }
                    b_core_sum += b_field[row + i];
                } else {
                    ring_n += 1;
                    if species_x[row + i] > 2.0 {
                        ring_hi += 1;
                    }
                    b_ring_sum += b_field[row + i];
                }
            }
        }
        let ring_frac = ring_hi as f64 / ring_n as f64;
        let core_frac = core_hi as f64 / core_n as f64;
        let b_ring = b_ring_sum / ring_n as f64;
        let b_core = b_core_sum / core_n as f64;

        println!(
            "  {:>4}  {:>3.0}  {:>6.3}  {:>3.2}  {:>9.3}  {:>9.3}  {:>3.2}  {:>5.3}  {:>5.3}",
            step, t, excited, x_mean, ring_frac, core_frac, b_mean, b_ring, b_core,
        );
    }

    println!("\nReading the result:");
    println!(" - Spiral at center emits circular waves outward.");
    println!(" - As each wave passes a cell, drive briefly lifts");
    println!(" -   X above the Schlogl separatrix at X=2.");
    println!(" - In the fuel-rich ring (high B): the high fp X=3");
    println!("     exists, so the committed cell STAYS high after");
    println!("     the wave passes. The ring fills in over time.");
    println!(" - In the fuel-starved core (low B): no high fp;");
    println!("     X relaxes back to X_LOW. Core stays clear.");
    println!(" - The wall self-assembles from a single kick.");
}
