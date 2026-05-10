//! R3 — Advection-diffusion.
//!
//! Same box as R2, but the medium itself is moving with velocity v. Run
//! from the CLI to print, for several Péclet numbers, the midpoint value
//! of the steady-state field. With v=0 you recover R2 (linear, midpoint
//! at 0.5). With v>0 the field bows toward the inflow wall.
//!
//! Usage:
//!   r03-advection
//!   r03-advection --v 0.2 --d 0.1
//!   r03-advection --steps 30000 --sample 5000

use flow::advection::AdvectionDiffusion1D;
use flow::diffusion::BoundaryCondition;

fn main() {
    let mut steps: u64 = 40_000;
    let mut sample_every: u64 = 5_000;
    let n: usize = 201;
    let dx: f64 = 1.0;
    let dt: f64 = 0.5;
    let mut diffusivity: f64 = 0.2;
    let mut velocity: f64 = 0.1;
    let left: f64 = 1.0;
    let right: f64 = 0.0;

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
            "--d" => {
                diffusivity = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(diffusivity);
                i += 2;
            }
            "--v" => {
                velocity = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(velocity);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut sim = match AdvectionDiffusion1D::new(
        n,
        diffusivity,
        velocity,
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

    println!("# R3 — advection-diffusion");
    println!(
        "# n={n} D={diffusivity} v={velocity} dx={dx} dt={dt} left={left} right={right}"
    );
    println!("# Péclet = v·L/D = {:.3}", sim.peclet());
    println!("# time\tflux_left\tflux_right\tphi[mid]\tphi[3L/4]");

    let mut taken = 0u64;
    print_sample(&sim);
    while taken < steps {
        let chunk = sample_every.min(steps - taken);
        sim.step_many(chunk);
        taken += chunk;
        print_sample(&sim);
    }
}

fn print_sample(sim: &AdvectionDiffusion1D) {
    let n = sim.len();
    let mid = sim.phi()[n / 2];
    let three_q = sim.phi()[(3 * n) / 4];
    println!(
        "{:.4}\t{:.6}\t{:.6}\t{:.6}\t{:.6}",
        sim.time(),
        sim.flux_left(),
        sim.flux_right(),
        mid,
        three_q
    );
}
