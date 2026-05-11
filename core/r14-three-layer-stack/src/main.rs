// R14 -- Three-layer stack. Cahn-Hilliard territory routes Kuramoto
// coupling (bulk_gate, R12 path); Kuramoto phase paces Gray-Scott
// feed (phase_to_scalar_field, R11 path). Three substrates, two
// operators, one phenomenon: territory carves chemistry's bloom
// schedule.
use flow::{bulk_gate, phase_to_scalar_field, CahnHilliard2D, GrayScott2D, Kuramoto2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    // CH territory
    let mut territory = CahnHilliard2D::new(w, h, 0.5, 0.5, 1.0, 0.05).expect("ch ok");
    territory.seed_noise(0.05, 0.0, 1);

    // Kuramoto phase
    let mut phase = Kuramoto2D::new(w, h, 0.0, 0.05).expect("k ok");
    phase.set_natural_frequencies(0.1, 17);
    phase.randomise_phases(42);

    // Gray-Scott chemistry
    let f_lo = 0.030_f64;
    let f_hi = 0.062_f64;
    let kill = 0.062_f64;
    let mut chem = GrayScott2D::new(w, h, 0.16, 0.08, 0.5 * (f_lo + f_hi), kill, 1.0, 1.0)
        .expect("gs ok");
    chem.seed_blob(w / 2, h / 2, 4);

    let k_wall = 0.05_f64;
    let k_bulk = 8.0_f64;
    let half_width = 0.5_f64;
    let sharpness = 0.1_f64;
    let phase_substeps = 20_u32;

    let mut k_field = vec![0.0_f64; n];
    let mut feed_field = vec![0.5 * (f_lo + f_hi); n];

    println!("R14 three-layer stack: CH -> Kuramoto -> GS.");
    println!(
        "Grid {}x{}; bulk_gate(k_wall={}, k_bulk={}); phase_to_scalar(f_lo={}, f_hi={}); phase_substeps={}.\n",
        w, h, k_wall, k_bulk, f_lo, f_hi, phase_substeps,
    );
    println!("  gs_t   r_pos   r_neg   meanV   coverage   bloom_split");
    println!("  -----  -----   -----   -----   --------   -----------");

    let checkpoints: [u32; 5] = [50, 200, 600, 1500, 3500];
    let mut total_steps = 0_u32;
    for &target in &checkpoints {
        while total_steps < target {
            territory.step();
            bulk_gate(territory.c(), k_wall, k_bulk, half_width, sharpness, &mut k_field).expect("ok");
            for _ in 0..phase_substeps {
                phase.step_with_coupling_field(&k_field).expect("ok");
            }
            phase_to_scalar_field(phase.theta(), f_lo, f_hi, &mut feed_field).expect("ok");
            chem.step_with_feed_field(&feed_field).expect("ok");
            total_steps += 1;
        }

        let phi = territory.c();
        let theta = phase.theta();
        let v = chem.v();

        let (mut cp, mut sp, mut np_) = (0.0_f64, 0.0_f64, 0usize);
        let (mut cn, mut sn, mut nn_) = (0.0_f64, 0.0_f64, 0usize);
        for (p, t) in phi.iter().zip(theta.iter()) {
            if p.abs() < half_width { continue; }
            if *p > 0.0 { cp += t.cos(); sp += t.sin(); np_ += 1; }
            else { cn += t.cos(); sn += t.sin(); nn_ += 1; }
        }
        let r_pos = if np_ == 0 { 0.0 } else { (cp * cp + sp * sp).sqrt() / np_ as f64 };
        let r_neg = if nn_ == 0 { 0.0 } else { (cn * cn + sn * sn).sqrt() / nn_ as f64 };

        let mean_v = v.iter().sum::<f64>() / n as f64;
        let coverage = v.iter().filter(|x| **x > 0.2).count() as f64 / n as f64;

        let (mut sp_v, mut np2, mut sn_v, mut nn2) = (0.0_f64, 0usize, 0.0_f64, 0usize);
        for (p, vi) in phi.iter().zip(v.iter()) {
            if *p > 0.0 { sp_v += vi; np2 += 1; }
            else if *p < 0.0 { sn_v += vi; nn2 += 1; }
        }
        let bloom_split = if np2 == 0 || nn2 == 0 {
            0.0
        } else {
            sp_v / np2 as f64 - sn_v / nn2 as f64
        };

        println!(
            "  {:5}   {:5.3}   {:5.3}   {:5.3}    {:4.1}%      {:+5.3}",
            target,
            r_pos, r_neg,
            mean_v,
            100.0 * coverage,
            bloom_split,
        );
    }
    println!(
        "\nThe territory splits the canvas into two phase-locked\n\
         clusters. Each cluster paces its own region of chemistry\n\
         feed. bloom_split is nonzero whenever the two halves are\n\
         out of phase: the territory has segmented the bloom\n\
         schedule. Three substrates, two operators, one signal."
    );
}
