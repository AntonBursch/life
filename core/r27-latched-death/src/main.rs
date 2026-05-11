// R27 -- Latched death. Phase D opener.
//
// First Phase-D rung. R26 had walls that persisted only because
// the wave kept writing them: if the spiral dies, memory decays
// and the walls dissolve. That is not real death -- it is just
// quiescence.
//
// R27 introduces the first operator with its own persistent state:
// `latch_field`, a per-cell Schmitt trigger. A cell flips to wall
// when memory crosses a high threshold, and stays a wall until
// memory drops below a much lower threshold (which, with non-zero
// reset, never happens for cells the wave has visited even once).
// Walls become irreversible: once dead, always dead, even after
// the spiral is killed.
//
// Chain:
//   1. Barkley.step_with_eps_field(eps_field)
//   2. integrate_field(u, memory, dt, leak)        -- R19
//   3. latch_field(wall_state, memory, set, reset) -- NEW operator
//   4. modulate_parameter(wall_state, ...) -> eps_field -- R24 op
//   5. feed eps_field back to step 1
//
// New operator: `latch_field` in category "latch" (stateful
// discretiser). First operator whose output is a function of the
// input *and* its own persistent state.
//
// The claim under test:
//   *Some things are not erasable.* In R26 the wave was both
//   builder and maintainer of its domain; the structure could not
//   outlive the activity that made it. In R27 the wave is only the
//   builder -- once built, the walls persist with or without the
//   wave. This is the minimum signature of biological structure:
//   you can die and your bones remain.

use flow::{integrate_field, latch_field, modulate_parameter, Barkley2D};

const W: usize = 96;
const H: usize = 96;

/// Mode flag: latch (R27 mechanism) vs direct memory->eps (R26-equivalent control).
#[derive(Copy, Clone, PartialEq)]
enum Mode {
    /// Memory -> latch_field (with hysteresis) -> wall_state -> eps.
    Latched { set_threshold: f64, reset_threshold: f64 },
    /// Memory -> direct affine map -> eps. No latch, no per-cell state.
    /// This is the R26-style mechanism: walls live only while memory does.
    Direct,
}

fn run(
    mode: Mode,
    leak: f64,
    kill_eps: f64,
    kill_wave_at: u32,
    n_steps: u32,
    label: &str,
) {
    let base_eps = 0.02_f64;
    let dt = 0.02_f64;
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, base_eps, 1.0, dt).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let mut memory = vec![0.0_f64; n];
    let mut wall_state = vec![0.0_f64; n];
    let mut eps_field = vec![base_eps; n];

    let mode_desc = match mode {
        Mode::Latched { set_threshold, reset_threshold } => format!(
            "LATCH set={} reset={}", set_threshold, reset_threshold
        ),
        Mode::Direct => "DIRECT (memory -> eps, no latch)".to_string(),
    };
    println!(
        "\n== {} ({}, leak={} (tau={}), kill_eps={}, kill_wave_at={}) ==",
        label, mode_desc, leak, 1.0 / leak, kill_eps, kill_wave_at
    );
    println!("  step    t   excited   wall_frac   mem_mean   wave?");
    println!("  ----  ---   -------   ---------   --------   -----");

    let checkpoints: [u32; 8] = [500, 1500, 4000, 8000, 12000, 18000, 25000, n_steps];
    let mut step = 0u32;
    let mut wave_killed = false;
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            sim.step_with_eps_field(&eps_field, base_eps);
            let u = sim.u();
            integrate_field(u, &mut memory, dt, leak).unwrap();
            match mode {
                Mode::Latched { set_threshold, reset_threshold } => {
                    latch_field(&mut wall_state, &memory, set_threshold, reset_threshold).unwrap();
                    modulate_parameter(
                        &wall_state, base_eps, kill_eps - base_eps,
                        base_eps, kill_eps, &mut eps_field,
                    ).unwrap();
                }
                Mode::Direct => {
                    // Map memory directly: eps = base + (kill_eps - base) * memory,
                    // clamped. R26-equivalent: no per-cell state, walls live only
                    // while memory carries them.
                    modulate_parameter(
                        &memory, base_eps, kill_eps - base_eps,
                        base_eps, kill_eps, &mut eps_field,
                    ).unwrap();
                    // Track a comparable "wall_frac" for reporting: cells where
                    // eps is near its ceiling.
                    let near_ceiling = (kill_eps - base_eps) * 0.8 + base_eps;
                    for k in 0..n {
                        wall_state[k] = if eps_field[k] >= near_ceiling { 1.0 } else { 0.0 };
                    }
                }
            }
            step += 1;
            if kill_wave_at > 0 && step == kill_wave_at {
                sim.reset();
                wave_killed = true;
            }
        }
        let exc = sim.excited_fraction();
        let wf = wall_state.iter().sum::<f64>() / n as f64;
        let mm = memory.iter().sum::<f64>() / n as f64;
        println!(
            "  {:>5}  {:>3.0}   {:>5.3}     {:>5.3}      {:>5.3}     {}",
            step, sim.time(), exc, wf, mm,
            if wave_killed { "dead" } else { "live" }
        );
        if step >= n_steps { break; }
    }
}

fn main() {
    println!("R27 latched death: irreversible walls.");
    println!("Grid {}x{}, base_eps=0.02, dt=0.02.", W, H);

    // Baseline: one-way latch (reset = 0). Walls grow in the scar
    // and stay -- the spiral keeps adding to its footprint over time.
    run(Mode::Latched { set_threshold: 1.0, reset_threshold: 0.0 },
        0.5, 0.05, 0, 25000,
        "latched, no killshot: walls grow in the scar and plateau");

    // Headline test: kill the wave at step 12000. With reset = 0 the
    // latch is one-way: once a cell has been walled, no input field
    // can release it. Walls are permanent.
    run(Mode::Latched { set_threshold: 1.0, reset_threshold: 0.0 },
        0.5, 0.05, 12000, 25000,
        "latched, wave killed at step 12000: one-way latch -> walls permanent");

    // R26-equivalent control: same kill, but no latch. Eps is a direct
    // affine function of memory. After the kill, memory decays, eps
    // returns to base, walls vanish. Structure is not separable from
    // process.
    run(Mode::Direct, 0.5, 0.05, 12000, 25000,
        "control (R26-like, no latch): walls dissolve after kill");

    println!("\nReading the result:");
    println!(" - LATCH (reset=0): wall_frac at the kill equals wall_frac at the end.");
    println!("   The structure outlives the process that built it.");
    println!(" - DIRECT (R26-like): wall_frac falls toward 0 as memory decays.");
    println!("   Walls were only persisting because activity kept them in eps space.");
    println!(" - This is the first irreversibility on the ladder: death-as-state.");
}
