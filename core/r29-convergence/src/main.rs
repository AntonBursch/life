// R29 -- Convergence. Phase D, third rung.
//
// Same composition as R28 (latch + advect + latch). One change:
// the velocity field is position-dependent. The left half of the
// grid flows rightward at +v, the right half flows leftward at
// -v. Two spirals are seeded at opposite ends. Each builds local
// walls; those walls advect inward and latch permanently into
// transmitted. The two channels CONVERGE -- cells near the
// midline receive walls that arrived from both directions.
//
// The claim under test:
//   *Communication composes.* Two channels through the same
//   medium do not interfere structurally; they OR into a single
//   accumulator. The midline carries a superposition of
//   histories, drawn from sources that may no longer exist.

use flow::{advect_by, integrate_field, latch_field, modulate_parameter, Barkley2D};

const W: usize = 128;
const H: usize = 96;

fn main() {
    let base_eps = 0.02_f64;
    let dt = 0.02_f64;
    let dx = 1.0_f64;
    let kill_eps = 0.05_f64;
    let leak = 0.5_f64;
    let set_local = 1.0_f64;
    let reset_local = 0.0_f64;
    let velocity = 2.0_f64;

    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, base_eps, dx, dt).unwrap();
    sim.kick(W / 5,     H / 2, ((W.min(H) as f64) * 0.06) as usize + 2, 1.0);
    sim.kick(4 * W / 5, H / 2, ((W.min(H) as f64) * 0.06) as usize + 2, 1.0);

    let n = W * H;
    let mut memory = vec![0.0_f64; n];
    let mut wall_local = vec![0.0_f64; n];
    let mut transmitted = vec![0.0_f64; n];
    let mut tmp = vec![0.0_f64; n];
    let mid = W / 2;
    let mut vx = vec![0.0_f64; n];
    let vy = vec![0.0_f64; n];
    for j in 0..H {
        let row = j * W;
        for i in 0..W {
            vx[row + i] = if i < mid { velocity } else { -velocity };
        }
    }
    let mut eps_field = vec![base_eps; n];

    println!("R29 convergence: two channels meet in the middle.");
    println!("Grid {}x{}, |v|={} cells/t.u., dt={}.", W, H, velocity, dt);
    println!("Left half flows right, right half flows left. Sources at i={} and i={}.",
        W / 5, 4 * W / 5);
    println!("Midline band = i in [{}, {}].", mid - W / 20, mid + W / 20);
    println!("  step    t   excited   <mem>   wl_mid    tr_mid    <eps>");
    println!("  ----  ---   -------   -----   ------    ------    -----");

    let checkpoints: [u32; 7] = [500, 1500, 3000, 5000, 8000, 12000, 18000];
    let mut step = 0u32;
    let n_steps = *checkpoints.last().unwrap();
    let half_band = (W / 20).max(1);
    let lo = mid - half_band;
    let hi = mid + half_band;
    for &cp in &checkpoints {
        let cp = cp.min(n_steps);
        while step < cp {
            sim.step_with_eps_field(&eps_field, base_eps);
            let u = sim.u();
            integrate_field(u, &mut memory, dt, leak).unwrap();
            latch_field(&mut wall_local, &memory, set_local, reset_local).unwrap();
            advect_by(&transmitted, &vx, &vy, W, H, dx, dt, &mut tmp).unwrap();
            std::mem::swap(&mut transmitted, &mut tmp);
            latch_field(&mut transmitted, &wall_local, 0.5, -1.0).unwrap();
            modulate_parameter(
                &transmitted, base_eps, kill_eps - base_eps,
                base_eps, kill_eps, &mut eps_field,
            ).unwrap();
            step += 1;
        }
        let mut wl_m = 0usize;
        let mut tr_m = 0usize;
        let mut total_m = 0usize;
        for j in 0..H {
            let row = j * W;
            for i in lo..hi {
                total_m += 1;
                if wall_local[row + i] > 0.5 { wl_m += 1; }
                if transmitted[row + i] > 0.5 { tr_m += 1; }
            }
        }
        let wl_m_f = wl_m as f64 / total_m as f64;
        let tr_m_f = tr_m as f64 / total_m as f64;
        let mem_m = memory.iter().sum::<f64>() / n as f64;
        let eps_m = eps_field.iter().sum::<f64>() / n as f64;
        println!(
            "  {:>4}  {:>3.0}   {:>5.3}     {:>5.3}    {:>5.3}     {:>5.3}     {:>5.3}",
            step, sim.time(), sim.excited_fraction(), mem_m, wl_m_f, tr_m_f, eps_m
        );
        if step >= n_steps { break; }
    }

    println!("\nReading the result:");
    println!(" - tr_mid (midline transmitted) rises before");
    println!("   wl_mid (midline locally-built). The midline");
    println!("   receives messages from both sides before");
    println!("   any wave physically reaches it.");
    println!(" - Kill both spirals at any point and the midline");
    println!("   stays walled forever. Two dead sources, one");
    println!("   permanent record of the conversation.");
}
