// R11 — Phase drives reaction. A Kuramoto phase layer modulates the
// per-cell feed rate F of a Gray-Scott reaction-diffusion field.
// Reverse arrow of R10. Sweep coupling K from incoherent to globally
// synced; show how the spatial coherence of the chemistry follows.
use flow::{phase_to_scalar_field, GrayScott2D, Kuramoto2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    // Gray-Scott "coral" spot regime baseline: F=0.054, k=0.062.
    // We'll let F oscillate between F_lo and F_hi driven by phase.
    let f_lo = 0.030;
    let f_hi = 0.070;
    let kill = 0.062;

    // The chemistry runs at its own substep; phase advances slower.
    let dt = 1.0;
    let mut chem = GrayScott2D::new(w, h, 0.16, 0.08, 0.054, kill, 1.0, dt)
        .expect("valid gray-scott");
    // Seed a central blob to break the trivial state.
    chem.seed_blob(w / 2, h / 2, 6);

    // Build a frozen omega population, fix sigma; sweep K.
    let sigma = 0.02;

    println!("R11 phase-drives-reaction: Kuramoto phase modulates Gray-Scott feed.");
    println!("Grid {}x{}, F in [{}, {}], k={}, sigma={}.\n", w, h, f_lo, f_hi, kill, sigma);
    println!("    K       r(phase)     mean V       std V       v-coverage");
    println!("  ------    --------    --------    --------    ------------");

    let mut feed_field = vec![0.054_f64; n];

    for &k in &[0.0_f64, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let mut phase = Kuramoto2D::new(w, h, k, 0.5).expect("valid kuramoto");
        phase.set_natural_frequencies(sigma, 17);
        phase.randomise_phases(42);

        // Reset chemistry to the same seeded state for each K.
        chem.reset();
        chem.seed_blob(w / 2, h / 2, 6);

        // Let phase warm up first.
        phase.step_many(2000);

        // Then run coupled.
        for _ in 0..6000 {
            phase.step();
            phase_to_scalar_field(phase.theta(), f_lo, f_hi, &mut feed_field)
                .expect("ok");
            chem.step_with_feed_field(&feed_field).expect("ok");
        }

        // Diagnostics on V.
        let v = chem.v();
        let mean: f64 = v.iter().sum::<f64>() / (n as f64);
        let var: f64 = v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n as f64);
        let coverage: f64 = v.iter().filter(|x| **x > 0.2).count() as f64 / (n as f64);
        let r = phase.order_parameter();

        println!(
            "  {:5.2}      {:6.3}      {:6.4}      {:6.4}      {:8.3}",
            k, r, mean, var.sqrt(), coverage
        );
    }
    println!(
        "\nLow K: phases are incoherent, so F(x,t) is salt-and-pepper. Each cell\n\
         sees rapidly-fluctuating feed; spots don't settle. High K: phases\n\
         lock and F(x,t) becomes a single oscillating plane wave. The whole\n\
         pattern breathes in time. Coverage and variance reflect how well\n\
         the chemistry could form structure under the imposed F regime."
    );
}
