// R10 — coupled substrates. Excitable activator (Barkley) gates a
// per-cell coupling field for a Kuramoto phase layer. Show how local
// synchrony tracks the moving wavefront.
use flow::{excitable_gate, Barkley2D, Kuramoto2D};

fn main() {
    let w = 128usize;
    let h = 128usize;
    let n = w * h;

    // Barkley with a spiral seed — classic params from R7.
    let mut tissue = Barkley2D::new(w, h, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05)
        .expect("valid barkley params");
    tissue.seed_spiral();

    // Kuramoto layer: start with zero base coupling; the gate will
    // raise it to k_hi wherever the activator is high.
    let sigma = 0.15;
    let mut phase = Kuramoto2D::new(w, h, 0.0, 0.05).expect("valid kuramoto params");
    phase.set_natural_frequencies(sigma, 17);
    phase.randomise_phases(42);

    let mut k_field = vec![0.0_f64; n];
    let k_lo = 0.1;
    let k_hi = 6.0;
    let threshold = 0.4;
    let sharpness = 0.15;

    let r0 = phase.order_parameter();
    println!("R10 coupled: Barkley activator gates per-cell Kuramoto coupling.");
    println!("Grid {}x{}, k_lo={}, k_hi={}, sigma={}.\n", w, h, k_lo, k_hi, sigma);
    println!("Local correlation = mean cos(theta_i - theta_j) over 4-neighbour pairs.");
    println!("0 = independent, 1 = locked. Global r stays near zero because the");
    println!("excited zone is small; local correlation tells you neighbours march");
    println!("together in the wake of the wave.\n");
    println!("    t       excited      r(global)    local-corr");
    println!("  ------   ---------    -----------   ----------");

    let inner_steps = 50;
    for _outer in 0..40 {
        for _ in 0..inner_steps {
            tissue.step();
            excitable_gate(tissue.u(), k_lo, k_hi, threshold, sharpness, &mut k_field)
                .expect("gate ok");
            phase
                .step_with_coupling_field(&k_field)
                .expect("coupling step ok");
        }

        // Local correlation: mean cos(theta_i - theta_j) over right
        // and down neighbour pairs (periodic).
        let theta = phase.theta();
        let mut acc = 0.0;
        let mut count = 0usize;
        for j in 0..h {
            let jp = if j == h - 1 { 0 } else { j + 1 };
            for i in 0..w {
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let t0 = theta[j * w + i];
                acc += (theta[j * w + ip] - t0).cos();
                acc += (theta[jp * w + i] - t0).cos();
                count += 2;
            }
        }
        let local = acc / (count as f64);

        let t = tissue.time();
        let excited = tissue.excited_fraction();
        let r_g = phase.order_parameter();
        let _ = r0;
        println!(
            "  {:6.2}     {:7.3}      {:8.3}      {:8.3}",
            t, excited, r_g, local
        );
    }
    println!(
        "\nGlobal r ~ 0 (most cells haven't been touched yet, and the spiral\n\
         keeps shifting which patch is locked). Local correlation climbs and\n\
         stays high: the moving wave entrains neighbours into shared phase.\n\
         Run the viewer to watch the colour blobs follow the spiral arms."
    );
}
