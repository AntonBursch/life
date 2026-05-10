//! R2 — Driven diffusion.
//!
//! A continuous source on the left and sink on the right hold a gradient
//! open. The field stops changing but flow keeps moving through. This is
//! the first appearance of "steady state ≠ equilibrium" — the category
//! life lives in.
//!
//! Headless runner. Sets up the boundary, steps the field, and prints
//! `time, flux_left, flux_right, max_change_per_step`. When
//! `max_change_per_step` falls to zero, the system has reached steady
//! state; at that point `flux_left` and `flux_right` should agree.
//!
//! Usage:
//!   r02-driven                       # default params
//!   r02-driven --left 1.0 --right 0.0
//!   r02-driven --steps 80000 --sample 1000

use flow::{BoundaryCondition, Diffusion1D};

fn main() {
    let mut steps: u64 = 40_000;
    let mut sample_every: u64 = 1000;
    let n: usize = 101;
    let d: f64 = 0.5;
    let dx: f64 = 1.0;
    let dt: f64 = 0.5;
    let mut left: f64 = 1.0;
    let mut right: f64 = 0.0;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--steps" => {
                steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(steps);
                i += 2;
            }
            "--sample" => {
                sample_every = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(sample_every);
                i += 2;
            }
            "--left" => {
                left = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(left);
                i += 2;
            }
            "--right" => {
                right = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(right);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut sim = match Diffusion1D::new(
        n,
        d,
        dx,
        dt,
        BoundaryCondition::FixedPair { left, right },
    ) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build sim: {e}");
            std::process::exit(1);
        }
    };

    println!("# R2 — driven diffusion");
    println!("# n={n} D={d} dx={dx} dt={dt} left={left} right={right} steps={steps}");
    println!("# expected steady-state flux J = D*(left - right)/L = {:.6}", d * (left - right) / (((n - 1) as f64) * dx));
    println!("# time\tflux_left\tflux_right\tmax_change_per_step");

    let mut previous: Vec<f64> = sim.phi().to_vec();
    let mut taken = 0u64;
    while taken < steps {
        let chunk = sample_every.min(steps - taken);
        sim.step_many(chunk);
        taken += chunk;

        // Average rate of change per step over the last chunk.
        let mut max_change = 0.0_f64;
        for (a, b) in sim.phi().iter().zip(previous.iter()) {
            max_change = max_change.max((a - b).abs());
        }
        let per_step = max_change / chunk as f64;

        println!(
            "{:.4}\t{:.8}\t{:.8}\t{:.3e}",
            sim.time(),
            sim.flux_left(),
            sim.flux_right(),
            per_step
        );

        previous.copy_from_slice(sim.phi());
    }
}
