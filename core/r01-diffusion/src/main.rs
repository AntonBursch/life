//! R1 — Pure 1D diffusion.
//!
//! Headless runner. Seeds a centre pulse, steps the field, and prints
//! `time, total_mass, rms_spread, peak` so we can sanity-check the math
//! and verify the sqrt(t) scaling the notes claim.
//!
//! Usage:
//!   r01-diffusion              # default params
//!   r01-diffusion --steps 5000 # override step count

use flow::{BoundaryCondition, Diffusion1D};

fn main() {
    let mut steps: u64 = 2000;
    let mut sample_every: u64 = 100;
    let n: usize = 401;
    let d: f64 = 0.5;
    let dx: f64 = 1.0;
    let dt: f64 = 0.5;

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
            _ => i += 1,
        }
    }

    let mut sim = match Diffusion1D::new(n, d, dx, dt, BoundaryCondition::ZeroFlux) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build sim: {e}");
            std::process::exit(1);
        }
    };
    sim.seed_centre_pulse();

    println!("# R1 — pure 1D diffusion");
    println!("# n={n} D={d} dx={dx} dt={dt} steps={steps}");
    println!("# time\ttotal\trms_spread\tpeak");
    print_sample(&sim);

    let mut taken = 0u64;
    while taken < steps {
        let chunk = sample_every.min(steps - taken);
        sim.step_many(chunk);
        taken += chunk;
        print_sample(&sim);
    }
}

fn print_sample(sim: &Diffusion1D) {
    let peak = sim.phi().iter().cloned().fold(f64::MIN, f64::max);
    println!(
        "{:.4}\t{:.6}\t{:.6}\t{:.6}",
        sim.time(),
        sim.total(),
        sim.rms_spread(),
        peak
    );
}
