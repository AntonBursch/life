//! R6 CLI — Swift–Hohenberg.
//!
//! Sweeps `r` from below onset to well above and prints the saturated
//! variance after a fixed wall-clock of integration, so you can see the
//! sharp bifurcation at r=0.

use flow::SwiftHohenberg2D;

fn parse_flag<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    for window in args.windows(2) {
        if window[0] == name {
            if let Ok(v) = window[1].parse::<T>() {
                return v;
            }
        }
    }
    default
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let w: usize = parse_flag(&args, "--w", 64);
    let h: usize = parse_flag(&args, "--h", 64);
    let dt: f64 = parse_flag(&args, "--dt", 0.02);
    let steps: usize = parse_flag(&args, "--steps", 10000);

    println!(
        "R6 Swift-Hohenberg: {}x{}  dt={}  steps={}",
        w, h, dt, steps
    );
    println!("    r       variance   max|u|");
    println!("--------  ---------  --------");

    for &r in &[-0.5_f64, -0.1, -0.02, 0.0, 0.02, 0.1, 0.3, 0.6] {
        let mut sim = SwiftHohenberg2D::new(w, h, r, 1.0, dt).expect("valid params");
        sim.reset();
        sim.step_many(steps);
        println!(
            "{:>6.2}    {:>7.5}    {:>5.3}",
            r,
            sim.variance(),
            sim.max_abs()
        );
    }
}
