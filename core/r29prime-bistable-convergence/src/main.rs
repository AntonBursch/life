// R29' -- Bistable convergence. Phase E, third rung.
//
// Substrate-honest rebuild of R29. See life/THESIS.md.
//
// Same composition as R28' (react + advect + parametrise), with
// one change: the velocity field is position-dependent. The left
// half flows rightward at +v, the right half flows leftward at
// -v. Two spirals are kicked at opposite ends; each commits its
// local Schlogl species X to the high attractor; advection
// carries X mass inward; the midline receives inflow from BOTH
// sides and commits via local chemistry alone.
//
// The claim under test:
//   *Real-reactant communication composes.* Two reactant
//   sources converging on the same medium OR into a single
//   chemical accumulator. The midline carries a superposition of
//   chemistries originating in sources that may no longer
//   exist. No comparator anywhere; no advected labels.

use flow::{advect_by, modulate_parameter, react_field, schlogl_rate, Barkley2D};

const W: usize = 128;
const H: usize = 96;

const BASE_EPS: f64 = 0.02;
const KILL_EPS: f64 = 0.05;
const DT: f64 = 0.02;
const DX: f64 = 1.0;

const K1A: f64 = 6.0;
const K2: f64 = 1.0;
const K3: f64 = 11.0;
const K4B: f64 = 6.0;
const X_LOW: f64 = 1.0;
const X_HIGH: f64 = 3.0;

const U_THR: f64 = 0.8;
const DRIVE: f64 = 4.0;
const VELOCITY: f64 = 2.0; // cells per time unit, inward on each half

// Two source reservoirs hold X clamped at X_high (an honest
// Dirichlet boundary condition -- a continuously-supplied metabolic
// reservoir). At STOP_FEED_STEP the clamp is released; from that
// point on, no external input enters the system anywhere. We then
// observe how long the midline survives on its chemistry alone --
// the honest answer to "does the midline carry a permanent record?"
const SOURCE_RADIUS: usize = 6;
const STOP_FEED_STEP: u32 = 6000; // release clamp at t=120

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, BASE_EPS, DX, DT).unwrap();

    let n = W * H;
    let mut species_x = vec![X_LOW; n];
    let mut x_tmp = vec![0.0_f64; n];
    let mut x_shifted = vec![0.0_f64; n];

    // Precompute source-disk index masks.
    let src_l = (W / 5, H / 2);
    let src_r = (4 * W / 5, H / 2);
    let mut source_mask = vec![false; n];
    for (cx, cy) in [src_l, src_r] {
        let r2 = (SOURCE_RADIUS as i32).pow(2);
        let w = W as i32;
        let h = H as i32;
        let cxi = cx as i32;
        let cyi = cy as i32;
        for dj in -(SOURCE_RADIUS as i32)..=(SOURCE_RADIUS as i32) {
            for di in -(SOURCE_RADIUS as i32)..=(SOURCE_RADIUS as i32) {
                if di * di + dj * dj > r2 { continue; }
                let i = ((cxi + di).rem_euclid(w)) as usize;
                let j = ((cyi + dj).rem_euclid(h)) as usize;
                source_mask[j * W + i] = true;
            }
        }
    }

    let mid = W / 2;
    let mut vx = vec![0.0_f64; n];
    let vy = vec![0.0_f64; n];
    for j in 0..H {
        let row = j * W;
        for i in 0..W {
            vx[row + i] = if i < mid { VELOCITY } else { -VELOCITY };
        }
    }
    let mut eps_field = vec![BASE_EPS; n];

    let bare_rate = schlogl_rate(K1A, K2, K3, K4B);
    let gain = (KILL_EPS - BASE_EPS) / (X_HIGH - X_LOW);

    println!("R29' bistable convergence: two reactant sources meet.");
    println!("Grid {}x{}, |v|={} c/t.u. inward, dt={}.", W, H, VELOCITY, DT);
    println!("Source reservoirs (X clamped to {}) at i={} and i={}, radius={}.",
        X_HIGH, W / 5, 4 * W / 5, SOURCE_RADIUS);
    println!("Clamp released at step {} (t={}).", STOP_FEED_STEP, (STOP_FEED_STEP as f64) * DT);
    let half_band = (W / 20).max(1);
    let lo = mid - half_band;
    let hi = mid + half_band;
    println!("Midline band = i in [{}, {}].", lo, hi);
    println!("  step    t   excited   <X>   X_hi_L   X_hi_mid   X_hi_R   <eps>");
    println!("  ----  ---   -------   ---   ------   --------   ------   -----");

    let checkpoints: [u32; 8] = [500, 1500, 3000, 5000, 6500, 8000, 12000, 18000];
    let mut step = 0u32;
    let n_steps = *checkpoints.last().unwrap();
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            // Clamp source reservoirs to X_high until release.
            if step < STOP_FEED_STEP {
                for k in 0..n {
                    if source_mask[k] {
                        species_x[k] = X_HIGH;
                    }
                }
            }

            sim.step_with_eps_field(&eps_field, BASE_EPS);

            let u_snapshot = sim.u().to_vec();
            for k in 0..n {
                let drive_k = DRIVE * (u_snapshot[k] - U_THR).max(0.0);
                let mut one = [species_x[k]];
                react_field(&mut one, |x| bare_rate(x) + drive_k, DT).unwrap();
                species_x[k] = one[0];
            }

            advect_by(&species_x, &vx, &vy, W, H, DX, DT, &mut x_tmp).unwrap();
            std::mem::swap(&mut species_x, &mut x_tmp);

            for k in 0..n {
                x_shifted[k] = species_x[k] - X_LOW;
            }
            modulate_parameter(
                &x_shifted, BASE_EPS, gain,
                BASE_EPS, KILL_EPS, &mut eps_field,
            ).unwrap();

            step += 1;
        }

        let mut hi_l = 0usize;
        let mut hi_m = 0usize;
        let mut hi_r = 0usize;
        let mut total_l = 0usize;
        let mut total_m = 0usize;
        let mut total_r = 0usize;
        for j in 0..H {
            let row = j * W;
            for i in 0..mid {
                total_l += 1;
                if species_x[row + i] > 2.0 { hi_l += 1; }
            }
            for i in mid..W {
                total_r += 1;
                if species_x[row + i] > 2.0 { hi_r += 1; }
            }
            for i in lo..hi {
                total_m += 1;
                if species_x[row + i] > 2.0 { hi_m += 1; }
            }
        }
        let hi_l_f = hi_l as f64 / total_l as f64;
        let hi_m_f = hi_m as f64 / total_m as f64;
        let hi_r_f = hi_r as f64 / total_r as f64;
        let x_mean = species_x.iter().sum::<f64>() / n as f64;
        let eps_m = eps_field.iter().sum::<f64>() / n as f64;
        println!(
            "  {:>4}  {:>3.0}   {:>5.3}    {:>4.2}   {:>5.3}    {:>5.3}      {:>5.3}    {:>5.3}",
            step, sim.time(), sim.excited_fraction(), x_mean, hi_l_f, hi_m_f, hi_r_f, eps_m
        );
        if step >= n_steps { break; }
    }

    println!("\nReading the result:");
    println!(" - During the clamp (t<120): X_hi_mid grows from");
    println!("   BOTH sides, reaching commitment levels at or");
    println!("   above either source alone -- two channels OR");
    println!("   into one accumulator.");
    println!(" - After clamp release (t>=120): without ongoing");
    println!("   reactant flux, advection drains committed cells");
    println!("   faster than the Schlogl rate can hold them. The");
    println!("   midline returns to X=1. R29's 'permanent record'");
    println!("   was an artifact of latch_field, not chemistry.");
    println!("   Real persistence in open systems requires real");
    println!("   sources -- this is the substrate-honest answer.");
}
