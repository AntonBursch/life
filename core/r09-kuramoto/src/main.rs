//! R9 CLI — Kuramoto oscillators. Sweep the coupling K and watch the
//! order parameter `r` cross from incoherence to synchrony.

use flow::Kuramoto2D;

fn main() {
    let w = 64;
    let h = 64;
    let dt = 0.05;
    let sigma = 0.3;

    println!("R9 Kuramoto: {}x{}  local 4-neighbour  sigma={}  dt={}", w, h, sigma, dt);
    println!("\nFor each K, fresh random phases. Same frozen omega distribution.\n");
    println!("     K        r(initial)    r(steady)    psi(steady)");
    println!("  -------    -----------    ---------    -----------");

    for &k in &[0.0_f64, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
        let mut sim = Kuramoto2D::new(w, h, k, dt).expect("valid params");
        sim.set_natural_frequencies(sigma, 17);
        sim.randomise_phases(42);
        let r0 = sim.order_parameter();
        sim.step_many(20_000);
        let r1 = sim.order_parameter();
        let psi = sim.mean_phase();
        println!(
            "  {:>6.2}        {:>5.3}        {:>5.3}        {:>+5.2}",
            k, r0, r1, psi
        );
    }

    println!(
        "\nLocal 2D Kuramoto has a finite critical coupling: below it `r` stays near zero,\nabove it `r` climbs toward 1. The transition is the sync onset."
    );
}
