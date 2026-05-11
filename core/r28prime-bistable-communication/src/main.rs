// R28' -- Bistable communication. Phase E, second rung.
//
// Replaces R28's latch_field/advect/latch composition with the
// honest version: a single real species X obeying the Schlogl
// bistable rate law from R27', advected by a uniform velocity
// field. No comparators. The "transmitted wall" is a packet of
// real reactant moving downstream.
//
// Mechanics each tick:
//   1. Barkley.step_with_eps_field(eps)
//   2. react_field(X, schlogl_rate(...) + drive*[u - u_thr]+, dt)  [Phase E op]
//   3. advect_by(X, vx, vy, dt)  -- semi-Lagrangian transport of a real species
//   4. modulate_parameter(X - X_low, ...) -> eps
//
// The wave's footprint commits cells to X = 3 locally. Advection
// carries that high-X mass rightward. Downstream cells receive
// inflow that lifts them past the X = 2 separatrix; the Schlogl
// rate law then commits them to X = 3 on its own, with no further
// forcing. The "channel" is mass-action chemistry plus transport.
//
// The claim under test:
//   *Real reactant + bistable substrate = communication, no
//   latches anywhere.* Right-half cells should accumulate X-high
//   well outside the spiral's reach, by being downstream of the
//   committed left-half.

use flow::{advect_by, modulate_parameter, react_field, schlogl_rate, Barkley2D};

const W: usize = 128;
const H: usize = 96;

const BASE_EPS: f64 = 0.02;
const KILL_EPS: f64 = 0.05;
const DT: f64 = 0.02;
const DX: f64 = 1.0;

// Schlogl bistable params: bare rate = -(X-1)(X-2)(X-3).
const K1A: f64 = 6.0;
const K2: f64 = 1.0;
const K3: f64 = 11.0;
const K4B: f64 = 6.0;
const X_LOW: f64 = 1.0;
const X_HIGH: f64 = 3.0;

// Wave coupling.
const U_THR: f64 = 0.8;
const DRIVE: f64 = 4.0;

// Transport.
const VELOCITY: f64 = 2.5; // cells per time unit, rightward

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, BASE_EPS, DX, DT).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let mut species_x = vec![X_LOW; n];
    let mut x_tmp = vec![0.0_f64; n];
    let mut x_shifted = vec![0.0_f64; n];
    let vx = vec![VELOCITY; n];
    let vy = vec![0.0_f64; n];
    let mut eps_field = vec![BASE_EPS; n];

    let bare_rate = schlogl_rate(K1A, K2, K3, K4B);
    let mid = W / 2;
    let gain = (KILL_EPS - BASE_EPS) / (X_HIGH - X_LOW);

    println!("R28' bistable communication: real reactant carried by advection.");
    println!("Grid {}x{}, v={} cells/t.u. rightward, dt={}.", W, H, VELOCITY, DT);
    println!("Schlogl (k1A,k2,k3,k4B)=({},{},{},{})  X_low={}  X_high={}",
        K1A, K2, K3, K4B, X_LOW, X_HIGH);
    println!("  step    t   excited   <X>   X_high_L   X_high_R   <eps>");
    println!("  ----  ---   -------   ---   --------   --------   -----");

    let checkpoints: [u32; 8] = [500, 1500, 3000, 6000, 10000, 15000, 20000, 25000];
    let mut step = 0u32;
    let n_steps = *checkpoints.last().unwrap();
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            sim.step_with_eps_field(&eps_field, BASE_EPS);

            // 2. react X locally with wave drive.
            let u_snapshot = sim.u().to_vec();
            for k in 0..n {
                let drive_k = DRIVE * (u_snapshot[k] - U_THR).max(0.0);
                let mut one = [species_x[k]];
                react_field(&mut one, |x| bare_rate(x) + drive_k, DT).unwrap();
                species_x[k] = one[0];
            }

            // 3. advect X by (vx, 0).
            advect_by(&species_x, &vx, &vy, W, H, DX, DT, &mut x_tmp).unwrap();
            std::mem::swap(&mut species_x, &mut x_tmp);

            // 4. eps from X.
            for k in 0..n {
                x_shifted[k] = species_x[k] - X_LOW;
            }
            modulate_parameter(
                &x_shifted, BASE_EPS, gain,
                BASE_EPS, KILL_EPS, &mut eps_field,
            ).unwrap();

            step += 1;
        }

        // Per-half X_high metrics: fraction with X > 2 (past separatrix).
        let mut hi_l = 0usize;
        let mut hi_r = 0usize;
        let mut total_half = 0usize;
        for j in 0..H {
            let row = j * W;
            for i in 0..mid {
                if species_x[row + i] > 2.0 { hi_l += 1; }
            }
            for i in mid..W {
                if species_x[row + i] > 2.0 { hi_r += 1; }
            }
            total_half += mid;
        }
        let hi_l_frac = hi_l as f64 / total_half as f64;
        let hi_r_frac = hi_r as f64 / total_half as f64;
        let x_mean = species_x.iter().sum::<f64>() / n as f64;
        let eps_m = eps_field.iter().sum::<f64>() / n as f64;
        println!(
            "  {:>4}  {:>3.0}   {:>5.3}    {:>4.2}    {:>5.3}      {:>5.3}      {:>5.3}",
            step, sim.time(), sim.excited_fraction(), x_mean, hi_l_frac, hi_r_frac, eps_m
        );
        if step >= n_steps { break; }
    }

    println!("\nReading the result:");
    println!(" - X_high_R > 0 with no wave on the right half = chemistry");
    println!("   delivered as inflow, not as a latched bit. The");
    println!("   downstream cells then commit themselves via local");
    println!("   Schlogl dynamics (no comparator).");
    println!(" - Compare R28: same phenomenology, no latch_field anywhere.");
}
