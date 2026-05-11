// R27' -- Bistable death.
//
// This rung replaces R27 with mechanistically honest chemistry
// (see ../../THESIS.md). R27 used a Schmitt trigger (latch_field)
// to make walls irreversible. Nature has no Schmitt triggers;
// nature has bistable reaction networks. So this rung uses one.
//
// Substrate stays Barkley (excitable medium = real cardiac/BZ
// physics). A new species X lives on the same grid and obeys the
// Schlogl model -- the canonical one-species chemical bistable:
//
//   A + 2X -> 3X            (autocatalysis)
//   X -> B                  (decay)
//
//   dX/dt = k1A * X^2 - k2 * X^3 - k3 * X + k4B  +  D * [u - u_thr]+
//
// The drive term is the only coupling to the substrate: when the
// Barkley activator u is above threshold (a wave is passing), it
// boosts the autocatalytic step. Concretely: the wave species
// reacts with the X precursor to make X faster -- a real chemical
// mechanism, not a free knob.
//
// With (k1A, k2, k3, k4B) = (6, 1, 11, 6) the bare rate law factors
// as -(X-1)(X-2)(X-3): low stable at X=1, unstable separatrix at
// X=2, high stable at X=3. A passing wave pushes a cell past the
// separatrix; the cell then falls into the high-X attractor on its
// own and stays there. No comparator, no latch, no per-operator
// state. The "memory" is the chemical attractor the cell is
// sitting in.
//
// Coupling X back to the substrate uses modulate_parameter, but
// honestly this time: the input is a real species concentration,
// not an abstract memory field. High X -> high eps -> cell is
// refractory (real analogue: calcium accumulation slows recovery
// in cardiac tissue).
//
// Operator chain:
//   1. Barkley.step_with_eps_field(eps_field)
//   2. react_field(X, schlogl + wave_drive)        -- NEW
//   3. modulate_parameter(X-X_low, ...) -> eps_field
//
// Claim under test:
//   *Same R27 phenomenology -- irreversible walls, structure
//   outlives process -- earned from real chemistry. After we kill
//   the wave, X stays at its high-attractor value in every cell
//   the wave visited, and eps stays elevated in those cells, with
//   no comparator anywhere in the chain.*

use flow::{modulate_parameter, react_field, schlogl_rate, Barkley2D};

const W: usize = 96;
const H: usize = 96;

// Schlogl parameters: low stable at 1, separatrix at 2, high stable at 3.
const K1A: f64 = 6.0;
const K2: f64 = 1.0;
const K3: f64 = 11.0;
const K4B: f64 = 6.0;
const X_LOW: f64 = 1.0;
const X_HIGH: f64 = 3.0;

// Wave-drive coupling. u_thr is high (well above the standard
// Barkley firing threshold ~0.5) so only the sharp wave front --
// where the activator is actually at peak -- drives the
// autocatalysis. The slow flanks of a spreading wave don't commit
// a cell on their own. Tuned so a single front pass deposits
// roughly the same drive a single spike does in R27's memory
// integrator -- enough to commit cells the spiral revisits, not
// enough to commit cells touched only once.
const U_THR: f64 = 0.8;
const DRIVE: f64 = 4.0;

// eps mapping.
const BASE_EPS: f64 = 0.02;
const KILL_EPS: f64 = 0.05;

fn run(kill_wave_at: u32, n_steps: u32, label: &str) {
    let dt = 0.02_f64;
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, BASE_EPS, 1.0, dt).unwrap();
    sim.seed_spiral();

    let n = W * H;
    // X starts in the low stable state everywhere -- chemically
    // resting tissue.
    let mut species_x = vec![X_LOW; n];
    let mut eps_field = vec![BASE_EPS; n];
    // Buffer of (X - X_low) for modulate_parameter input.
    let mut x_shifted = vec![0.0_f64; n];

    println!(
        "\n== {} (kill_wave_at={}) ==",
        label, kill_wave_at
    );
    println!("  step    t   excited   X_high_frac   X_mean   eps_mean   wave?");
    println!("  ----  ---   -------   -----------   ------   --------   -----");

    let bare_rate = schlogl_rate(K1A, K2, K3, K4B);

    let checkpoints: [u32; 8] = [500, 1500, 4000, 8000, 12000, 18000, 25000, n_steps];
    let mut step = 0u32;
    let mut wave_killed = false;
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            // 1. Substrate step.
            sim.step_with_eps_field(&eps_field, BASE_EPS);

            // 2. React X with substrate-coupled rate. The drive is
            //    a real chemical term: wave activator above
            //    threshold boosts the autocatalysis rate locally.
            let u = sim.u();
            for k in 0..n {
                let drive_k = DRIVE * (u[k] - U_THR).max(0.0);
                // One-cell react_field with a closure that adds
                // the local drive to the Schlogl rate.
                let mut one = [species_x[k]];
                react_field(
                    &mut one,
                    |x| bare_rate(x) + drive_k,
                    dt,
                )
                .unwrap();
                species_x[k] = one[0];
            }

            // 3. Modulate eps from real species concentration.
            //    eps = BASE_EPS + (KILL_EPS - BASE_EPS) * (X - 1) / 2,
            //    clamped to [BASE_EPS, KILL_EPS].
            let gain = (KILL_EPS - BASE_EPS) / (X_HIGH - X_LOW);
            for k in 0..n {
                x_shifted[k] = species_x[k] - X_LOW;
            }
            modulate_parameter(
                &x_shifted,
                BASE_EPS, gain,
                BASE_EPS, KILL_EPS,
                &mut eps_field,
            ).unwrap();

            step += 1;
            if kill_wave_at > 0 && step == kill_wave_at {
                sim.reset();
                wave_killed = true;
            }
        }
        let exc = sim.excited_fraction();
        let high_frac = species_x.iter().filter(|&&x| x > 2.0).count() as f64 / n as f64;
        let xm = species_x.iter().sum::<f64>() / n as f64;
        let em = eps_field.iter().sum::<f64>() / n as f64;
        println!(
            "  {:>5}  {:>3.0}   {:>5.3}     {:>5.3}        {:>5.3}    {:>6.4}    {}",
            step, sim.time(), exc, high_frac, xm, em,
            if wave_killed { "dead" } else { "live" }
        );
        if step >= n_steps { break; }
    }
}

fn main() {
    println!("R27' bistable death: irreversible walls from real chemistry.");
    println!("Grid {}x{}, base_eps={}, kill_eps={}, dt=0.02.", W, H, BASE_EPS, KILL_EPS);
    println!("Schlogl rate: -(X-1)(X-2)(X-3) + drive*[u-{}]+; drive gain {}.", U_THR, DRIVE);

    // Baseline: no killshot, just watch the spiral lay down its
    // chemical footprint. X_high_frac grows where the wave has been.
    run(0, 25000,
        "no killshot: wave pushes X past separatrix, X_high_frac grows");

    // Headline test: kill the wave at step 12000. Once a cell's X
    // has crossed the unstable separatrix at X=2 (during wave
    // passage), it falls into the high-X attractor on its own and
    // stays there with zero forcing. eps stays elevated. The walls
    // are permanent. No comparator anywhere; this is mass-action
    // chemistry running on its own.
    run(12000, 25000,
        "wave killed at step 12000: Schlogl high state is permanent");

    println!("\nReading the result:");
    println!(" - X_high_frac at the kill ~= X_high_frac at the end:");
    println!("   the bistable attractor holds the state with no input.");
    println!(" - eps_mean stays elevated after the kill:");
    println!("   refractoriness is locked in by a real species, not a label.");
    println!(" - Same R27 phenomenology, but every number is a real");
    println!("   concentration obeying mass-action. No latch.");
}
