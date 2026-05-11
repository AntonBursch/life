// R18 -- Waves leave marks. Barkley spirals (R7) sweep across the
// grid; the new operator threshold_event watches u at every cell
// and emits a 1 the instant u rises through a threshold. A simple
// counter accumulates those events into a per-cell firing-rate
// map; a last-fire field latches the most recent event time.
//
// First use of threshold_event -- the operator alphabet's first
// "discretise" primitive. Output is symbolic (events) instead of
// continuous (a field). Demonstrates how a reaction-diffusion
// wave train turns into a spike raster downstream.

use flow::{threshold_event, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let mut prev_u = sim.u().to_vec();
    let mut events = vec![0u8; n];
    let mut counts = vec![0u32; n];
    let mut last_time = vec![-1.0f64; n];

    let threshold = 0.5;
    let checkpoints = [200, 600, 1500, 3000, 6000];

    println!(
        "{:>6} {:>6} {:>8} {:>8} {:>8} {:>8} {:>10}",
        "step", "t", "evt", "cum", "cells_h", "max_ct", "mean_rate"
    );

    for step in 1..=6000 {
        sim.step();
        threshold_event(&prev_u, sim.u(), threshold, &mut events).unwrap();
        let mut evt_this_step = 0u32;
        let t = sim.time();
        for k in 0..n {
            if events[k] == 1 {
                counts[k] += 1;
                last_time[k] = t;
                evt_this_step += 1;
            }
        }
        prev_u.copy_from_slice(sim.u());

        if checkpoints.contains(&step) {
            let cum: u32 = counts.iter().sum();
            let hit: usize = counts.iter().filter(|c| **c > 0).count();
            let max_ct: u32 = *counts.iter().max().unwrap_or(&0);
            let mean_rate = if hit > 0 {
                cum as f64 / hit as f64 / t.max(1.0)
            } else { 0.0 };
            println!(
                "{:>6} {:>6.1} {:>8} {:>8} {:>8} {:>8} {:>10.4}",
                step, t, evt_this_step, cum, hit, max_ct, mean_rate
            );
        }
    }
}
