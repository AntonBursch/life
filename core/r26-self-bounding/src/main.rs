// R26 -- Self-bounding. Phase C, third rung.
//
// R24 closed the loop with a smooth, affine map: memory -> eps.
// R25 closed it with a global setpoint. R26 closes it with a
// *sharp threshold*: where memory exceeds a cutoff, that cell is
// a wall (eps -> extinction); below the cutoff, normal tissue.
// The wave does not equilibrate with itself -- it partitions the
// medium into "lived in" and "wild", and the lived-in region
// becomes its own boundary.
//
// Chain (no new operators):
//   1. Barkley.step_with_eps_field(eps_field)
//   2. integrate_field(u, memory, dt, leak)        -- R19
//   3. bulk_gate(memory, base_eps, kill_eps,
//                threshold, sharpness, eps_field)  -- R12 op,
//                                                    sharp slope
//   4. feed eps_field back to step 1
//
// bulk_gate is the same operator used by R12 to make territory
// walls uncouple oscillators. Here it makes memory walls
// inexcitable. Reusing it shows the alphabet composing rather
// than growing.
//
// The claim under test:
//   *History becomes geometry.* Smooth feedback (R24) makes the
//   tissue dim everywhere. Threshold feedback makes the tissue
//   binary: walls and bulk. The wave is confined by a domain
//   it carved itself. Set the threshold low and the spiral
//   strangles. Set it high and the wall never builds. The
//   sharpness knob is exactly the knob between "homeostasis"
//   and "boundary-construction".

use flow::{bulk_gate, integrate_field, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn wall_fraction(memory: &[f64], threshold: f64) -> f64 {
    let n = memory.len();
    let lit = memory.iter().filter(|&&m| m > threshold).count();
    lit as f64 / n as f64
}

fn run(threshold: f64, sharpness: f64, leak: f64, kill_eps: f64, n_steps: u32, label: &str) {
    let base_eps = 0.02_f64;
    let dt = 0.05_f64;
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, base_eps, 1.0, dt).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let mut memory = vec![0.0_f64; n];
    let mut eps_field = vec![base_eps; n];

    println!(
        "\n== {} (thr={}, sharp={}, leak={} (tau={}), kill_eps={}) ==",
        label, threshold, sharpness, leak, 1.0 / leak, kill_eps
    );
    println!("  step    t   excited   wall_frac   <eps>   max(eps)");
    println!("  ----  ---   -------   ---------   -----   --------");

    let checkpoints: [u32; 7] = [100, 400, 1000, 2000, 4000, 8000, n_steps];
    let mut step = 0u32;
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            sim.step_with_eps_field(&eps_field, base_eps);
            let u = sim.u();
            integrate_field(u, &mut memory, dt, leak).unwrap();
            // bulk_gate uses |scalar|; memory is non-negative so OK.
            // k_wall = base_eps (cells with low memory, i.e. unvisited bulk)
            // k_bulk = kill_eps (cells with high |memory|, i.e. walls)
            // half_width = threshold, sharpness controls the slope.
            bulk_gate(&memory, base_eps, kill_eps, threshold, sharpness, &mut eps_field).unwrap();
            step += 1;
        }
        let exc = sim.excited_fraction();
        let wf = wall_fraction(&memory, threshold);
        let eps_mean = eps_field.iter().sum::<f64>() / n as f64;
        let eps_max = eps_field.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        println!(
            "  {:>4}  {:>3.0}   {:>5.3}     {:>5.3}      {:>5.3}    {:>5.3}",
            step, sim.time(), exc, wf, eps_mean, eps_max
        );
        if step >= n_steps { break; }
    }
}

fn main() {
    println!("R26 self-bounding: memory passes through bulk_gate (sharp).");
    println!("Grid {}x{}, base_eps=0.02.", W, H);
    println!("(Fast leak keeps memory tracking recent activity, so walls");
    println!(" form behind the wave and decay where it has not been lately.)");

    // Control: walls never reach threshold -> R7 behaviour.
    run(2.0, 0.05, 0.5, 0.10, 8000, "threshold too high: no walls (control)");

    // The sweet spot: sharp threshold around mean activity, gentle wall eps.
    run(0.3, 0.05, 0.5, 0.10, 8000, "self-bounding: sharp walls, finite domain");

    // Same threshold, soft slope: R24-style smooth scar instead of binary walls.
    run(0.3, 0.5, 0.5, 0.10, 8000, "soft slope: R24-style smooth (compare)");

    // Lower threshold + sharp: walls form too easily, spiral strangles.
    run(0.1, 0.05, 0.5, 0.10, 8000, "threshold too low: spiral strangled");

    println!("\nIf 'self-bounding' shows excited_frac persisting at reduced level");
    println!("while wall_frac stabilises below 1.0, the wave is living inside a");
    println!("domain it built. Compare 'soft slope' (smooth, no binary partition)");
    println!("with 'self-bounding': same memory, different geometry. Sharpness is");
    println!("the qualitative knob -- homeostasis becomes boundary-construction.");
}
