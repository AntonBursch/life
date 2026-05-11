// R24 -- Scar tissue. Phase-C opener.
//
// The first cybernetic rung: a derived field writes back into the
// substrate's own parameter. Past activity does not just shape what
// downstream operators see -- it shapes what the tissue itself
// will do next.
//
// Chain (closed loop):
//   1. Barkley.step_with_eps_field(eps_field)   -- substrate reads
//                                                  per-cell eps
//   2. integrate_field(u, memory, ...)          -- R19 leaky integral
//                                                  of activity
//   3. modulate_parameter(memory, ...)          -- new operator;
//                                                  eps_field =
//                                                  base + gain *
//                                                  memory, clamped
//   4. feed eps_field back into step 1
//
// New operator added: `modulate_parameter` in category "parametrise"
// (first new category since Phase A).
//
// The claim under test:
//   *History changes the substrate's responsiveness.* Where the wave
//   has been many times, the tissue becomes hard to excite ("scar
//   tissue"). The wavefront should be pushed away from its own
//   trail, and the spiral should drift / fragment / die rather than
//   sustain forever.

use flow::{integrate_field, modulate_parameter, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    // Base eps = 0.02. Scar tissue raises it; high eps = slower
    // recovery = wave can't re-enter while v lingers.
    let base_eps = 0.02_f64;
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, base_eps, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let dt = 0.05_f64;
    let leak = 0.1_f64;           // tau = 10 (memory equilibrates)
    let gain = 0.02_f64;          // very gentle: visible degradation, not extinction
    let eps_min = base_eps;
    let eps_max = 0.12_f64;       // ceiling: 6x base

    let mut memory = vec![0.0_f64; n];
    let mut eps_field = vec![base_eps; n];

    println!("R24 scar tissue: closed loop on eps.");
    println!(
        "Grid {}x{}, base_eps={}, leak={} (tau={}), gain={}, eps_min={}, eps_max={}.\n",
        W, H, base_eps, leak, 1.0 / leak, gain, eps_min, eps_max
    );
    println!("  step    t   <u>   max(mem)   <eps>   max(eps)   excited_frac");
    println!("  ----  ---   ----   --------   -----   --------   ------------");

    let checkpoints: [u32; 7] = [50, 200, 600, 1500, 3000, 6000, 12000];
    let mut step = 0u32;
    for &target in &checkpoints {
        while step < target {
            // Closed-loop step.
            sim.step_with_eps_field(&eps_field, eps_min);
            let u = sim.u();
            integrate_field(u, &mut memory, dt, leak).unwrap();
            modulate_parameter(&memory, base_eps, gain, eps_min, eps_max, &mut eps_field).unwrap();
            step += 1;
        }
        let u = sim.u();
        let u_mean = u.iter().sum::<f64>() / n as f64;
        let mem_max = memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let eps_mean = eps_field.iter().sum::<f64>() / n as f64;
        let eps_max_curr = eps_field.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exc = sim.excited_fraction();
        println!(
            "  {:>4}  {:>3.0}  {:>5.3}    {:>5.3}     {:>5.3}    {:>5.3}      {:>5.3}",
            step, sim.time(), u_mean, mem_max, eps_mean, eps_max_curr, exc
        );
    }

    println!(
        "\nIf <eps> rises and max(eps) hits the ceiling while excited_frac\n\
         eventually falls, the wave has built its own no-go region and is\n\
         losing real estate. That is the first time on this ladder where\n\
         the substrate's own future is being chosen by its past."
    );
}
