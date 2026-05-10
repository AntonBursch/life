//! R5 CLI — 2D thermal convection.
//!
//! Sweeps from pure conduction (g = 0) up through and past the convective
//! threshold, printing the Nusselt number at each setting so you can see
//! the jump where the box starts to organise.
//!
//! ```text
//! cargo run --release -p r05-convection -- --w 64 --h 24
//! ```

use flow::Convection2D;

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
    let h: usize = parse_flag(&args, "--h", 24);
    let kappa: f64 = parse_flag(&args, "--kappa", 0.1);
    let nu: f64 = parse_flag(&args, "--nu", 0.1);
    let dt: f64 = parse_flag(&args, "--dt", 0.05);
    let steps: usize = parse_flag(&args, "--steps", 6000);

    println!(
        "R5 Benard convection: {}x{}  kappa={}  nu={}  dt={}  steps={}",
        w, h, kappa, nu, dt, steps
    );
    println!("    g         Nu      <w^2>     |psi|max");
    println!("---------   ------  --------  --------");

    for &g in &[0.0_f64, 1e-4, 3e-4, 6e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1] {
        let mut sim = Convection2D::new(w, h, kappa, nu, g, 1.0, dt)
            .expect("valid params");
        sim.step_many(steps);
        println!(
            "{:>8.1e}    {:>5.3}   {:>7.4}   {:>7.4}",
            g,
            sim.nusselt(),
            sim.mean_sq_vorticity(),
            sim.max_abs_psi()
        );
    }
}
