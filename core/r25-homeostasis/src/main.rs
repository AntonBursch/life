// R25 -- Homeostasis. Phase C, second rung.
//
// The first *negative-feedback* control loop. R24 closed the loop
// locally and spatially: per-cell history -> per-cell eps. R25
// closes it globally and scalar-ly: total Barkley activity -> a
// single global eps setpoint that holds the system at a chosen
// target activity level.
//
// Control law (each step):
//
//   err        = excited_fraction(u) - target
//   eps_offset = clamp(eps_offset + k * err * dt, 0, eps_max - base)
//   eps_global = base + eps_offset
//   tissue.set_eps(eps_global) ; tissue.step()
//
// More activity than target -> err > 0 -> eps rises -> wave
// becomes harder to sustain -> activity falls back. Less activity
// than target -> eps relaxes back toward base -> wave recovers.
// A simple I-controller on a non-linear plant. No new operators
// needed; reuses the parametrise category from R24 trivially.
//
// The claim under test:
//   *The system regulates to a target.* Whatever you do to it
//   (kick it, reseed it, change parameters), excited_fraction is
//   dragged back to `target`. With control_gain = 0 the loop is
//   open and the activity drifts to its natural value.

use flow::Barkley2D;

const W: usize = 96;
const H: usize = 96;

fn run(target: f64, control_gain: f64, kick_every: u32, n_steps: u32, label: &str) {
    let base_eps = 0.02_f64;
    let dt = 0.05_f64;
    let eps_max = 0.20_f64;
    let warmup: u32 = 200; // let spiral establish before engaging controller
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, base_eps, 1.0, dt).unwrap();
    sim.seed_spiral();
    let mut eps_offset = 0.0_f64;

    println!("\n== {} (target={}, k={}, kick_every={}) ==", label, target, control_gain, kick_every);
    println!("  step    t   excited   eps_global   eps_offset   |err|");
    println!("  ----  ---   -------   ----------   ----------   -----");

    let checkpoints: [u32; 7] = [100, 400, 1000, 2000, 4000, 8000, n_steps];
    let mut step = 0u32;
    let mut next_kick = warmup + kick_every;
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            // Closed-loop step (controller engages after warmup).
            if step >= warmup {
                let activity = sim.excited_fraction();
                let err = activity - target;
                // Leaky integrator (acts proportional-ish in steady
                // state). Strong leak keeps the controller from
                // winding up past the wave-extinction threshold.
                let leak = 0.5_f64;
                let proposed = (1.0 - leak * dt) * eps_offset + control_gain * err * dt;
                eps_offset = proposed.max(0.0).min(eps_max - base_eps);
                sim.set_eps(base_eps + eps_offset);
            }
            sim.step();

            // Periodic perturbation -- can the controller recover?
            step += 1;
            if kick_every > 0 && step == next_kick {
                sim.kick(W / 3, H / 2, 8, 1.0);
                next_kick += kick_every;
            }
        }
        let activity = sim.excited_fraction();
        let err = activity - target;
        println!(
            "  {:>4}  {:>3.0}   {:>5.3}      {:>5.3}        {:>5.3}      {:>5.3}",
            step, sim.time(), activity, base_eps + eps_offset, eps_offset, err.abs()
        );
        if step >= n_steps { break; }
    }
}

fn main() {
    println!("R25 homeostasis: global negative-feedback control on Barkley eps.");
    println!("Grid {}x{}, base_eps=0.02, eps_max=0.20.", W, H);

    // Baseline: control off. Activity drifts to its natural value.
    run(0.15, 0.0, 0, 8000, "open loop (k=0): no regulation");

    // Controller on, no perturbation. Activity should settle near target.
    run(0.15, 1.0, 0, 8000, "closed loop (k=1.0): regulates to target=0.15");

    // Controller on, periodic kicks. Should still hold target on average.
    run(0.15, 1.0, 600, 8000, "closed loop + periodic kicks: rejects disturbance");

    // Different target -- lower activity.
    run(0.12, 1.0, 0, 8000, "different setpoint (target=0.12)");

    println!("\nIf 'closed loop' rows show excited_fraction converging to target");
    println!("while 'open loop' drifts to its natural value, the controller is");
    println!("doing real work. If the 'kicks' run still hits target by step 8000,");
    println!("the loop rejects disturbances. That is homeostasis.");
}
