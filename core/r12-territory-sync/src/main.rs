// R12 -- Territory shapes sync. A Cahn-Hilliard concentration field
// (R8) gates a Kuramoto coupling field (R9) through the new
// `bulk_gate` operator: walls (|phi| small) are uncoupled, bulks
// (|phi| ~ 1) get strong coupling. The two bulk domains lock
// independently, separated by uncoupled sync walls.
//
// New substrate pair *and* a new operator in the alphabet.
use flow::{bulk_gate, CahnHilliard2D, Kuramoto2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    let mut territory = CahnHilliard2D::new(w, h, 0.5, 0.5, 1.0, 0.05)
        .expect("valid cahn-hilliard");
    territory.seed_noise(0.05, 0.0, 1);

    let mut phase = Kuramoto2D::new(w, h, 0.0, 0.05).expect("valid kuramoto");
    phase.set_natural_frequencies(0.1, 17);
    phase.randomise_phases(42);

    let k_wall = 0.05_f64;
    let k_bulk = 8.0_f64;
    let half_width = 0.5_f64;
    let sharpness = 0.1_f64;

    let mut k_field = vec![0.0_f64; n];

    println!("R12 territory shapes sync: bulk_gate on Cahn-Hilliard drives Kuramoto.");
    println!(
        "Grid {}x{}, k_wall={}, k_bulk={}, half_width={}.\n",
        w, h, k_wall, k_bulk, half_width
    );
    println!("  ch_t   r(global)   r(phi>0)   r(phi<0)   cross-align");
    println!("  -----  ---------   --------   --------   -----------");

    let checkpoints: [u32; 5] = [400, 1200, 3000, 8000, 20000];
    let mut total_steps = 0_u32;
    for &target in &checkpoints {
        while total_steps < target {
            territory.step();
            let _ = bulk_gate(
                territory.c(),
                k_wall,
                k_bulk,
                half_width,
                sharpness,
                &mut k_field,
            );
            for _ in 0..5 {
                phase.step_with_coupling_field(&k_field).expect("ok");
            }
            total_steps += 1;
        }

        let r_global = phase.order_parameter();
        let phi = territory.c();
        let theta = phase.theta();
        let (mut cp, mut sp, mut np_) = (0.0_f64, 0.0_f64, 0usize);
        let (mut cn, mut sn, mut nn_) = (0.0_f64, 0.0_f64, 0usize);
        for (p, t) in phi.iter().zip(theta.iter()) {
            if p.abs() < half_width { continue; } // skip walls
            if *p > 0.0 { cp += t.cos(); sp += t.sin(); np_ += 1; }
            else        { cn += t.cos(); sn += t.sin(); nn_ += 1; }
        }
        let r_pos = if np_ == 0 { 0.0 } else { (cp * cp + sp * sp).sqrt() / np_ as f64 };
        let r_neg = if nn_ == 0 { 0.0 } else { (cn * cn + sn * sn).sqrt() / nn_ as f64 };
        let cross = if np_ == 0 || nn_ == 0 {
            0.0
        } else {
            (sp.atan2(cp) - sn.atan2(cn)).cos()
        };

        println!(
            "  {:5}    {:5.3}      {:5.3}      {:5.3}      {:+5.3}",
            target, r_global, r_pos, r_neg, cross,
        );
    }
    println!(
        "\nThe two bulk domains lock independently: r_pos and r_neg both\n\
         climb while r_global stays small. cross-align is whatever the\n\
         two domains' starting phases happened to be -- they do not\n\
         communicate across the walls. The wall is a sync barrier."
    );
}
