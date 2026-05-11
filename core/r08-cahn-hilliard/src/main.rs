//! R8 CLI — Cahn–Hilliard phase separation. Watch mass stay flat while
//! variance climbs to the bulk wells and free energy drains away.

use flow::CahnHilliard2D;

fn main() {
    let w = 96;
    let h = 96;
    let mut sim = CahnHilliard2D::new(w, h, 1.0, 1.0, 1.0, 0.02)
        .expect("valid params");

    println!("R8 Cahn-Hilliard: {}x{}  M=1  kappa=1  dx=1  dt=0.02", w, h);
    println!("seeding with zero-mean noise (amp=0.05).");
    sim.seed_noise(0.05, 0.0, 42);

    let m0 = sim.mean_c();
    println!("\ninitial mean(c) = {:+.6e}", m0);

    println!("\n    time     mean(c)         var(c)    bulk_frac   free_energy");
    println!("  -------    -----------    -------     --------    -----------");
    for _ in 0..12 {
        sim.step_many(500);
        println!(
            "  {:>7.1}    {:+.6e}    {:>5.3}      {:>5.3}        {:>7.4}",
            sim.time(),
            sim.mean_c(),
            sim.variance_c(),
            sim.bulk_fraction(),
            sim.free_energy(),
        );
    }

    let m_final = sim.mean_c();
    println!(
        "\nmass drift: {:+.3e}   (Cahn-Hilliard conserves mass exactly)",
        m_final - m0
    );
}
