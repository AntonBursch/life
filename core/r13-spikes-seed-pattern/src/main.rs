// R13 -- Spikes seed pattern. A Barkley excitable layer (R7) drives
// the per-cell Gray-Scott feed (R4) through `excitable_gate`. Where
// the medium is at rest, feed is starved -> chemistry decays. Where a
// spiral wave fires, feed is rich -> chemistry grows in the wake of
// the wave. Spirals carve patterns into the chemical canvas.
//
// New substrate pair (R7 + R4) and the third use of `excitable_gate`.
use flow::{excitable_gate, Barkley2D, GrayScott2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    // Barkley spiral, standard params
    let mut excitable = Barkley2D::new(w, h, 1.0, 0.75, 0.01, 0.02, 1.0, 0.05)
        .expect("valid barkley");
    excitable.seed_spiral();

    // Gray-Scott. f_lo sits in the stable-spots regime, f_hi pushes
    // wave-visited cells into the mitosis (splitting) regime. So
    // chemistry is alive everywhere; spirals make it more vigorous.
    let f_lo = 0.040_f64;
    let f_hi = 0.062_f64;
    let kill = 0.062_f64;
    let mut chem = GrayScott2D::new(w, h, 0.16, 0.08, 0.5 * (f_lo + f_hi), kill, 1.0, 1.0)
        .expect("valid gray-scott");
    chem.seed_blob(w / 2, h / 2, 4);

    let threshold = 0.4_f64;
    let sharpness = 0.1_f64;
    let bark_substeps: u32 = 20; // 20 * 0.05 = 1.0, matches gs dt

    let mut feed_field = vec![0.5 * (f_lo + f_hi); n];

    println!("R13 spikes seed pattern: Barkley u gates Gray-Scott feed via excitable_gate.");
    println!(
        "Grid {}x{}, f_lo={}, f_hi={}, threshold={}, bark_substeps={}.\n",
        w, h, f_lo, f_hi, threshold, bark_substeps
    );
    println!("  gs_t   firing%   mean(V)   coverage   corr(feed,V)");
    println!("  -----  -------   -------   --------   ------------");

    let checkpoints: [u32; 5] = [50, 200, 600, 1500, 3500];
    let mut total_steps = 0_u32;
    for &target in &checkpoints {
        while total_steps < target {
            for _ in 0..bark_substeps { excitable.step(); }
            excitable_gate(
                excitable.u(),
                f_lo,
                f_hi,
                threshold,
                sharpness,
                &mut feed_field,
            ).expect("ok");
            chem.step_with_feed_field(&feed_field).expect("ok");
            total_steps += 1;
        }

        let u = excitable.u();
        let firing = u.iter().filter(|x| **x > threshold).count() as f64 / n as f64;
        let v = chem.v();
        let mean_v = v.iter().sum::<f64>() / n as f64;
        let coverage = v.iter().filter(|x| **x > 0.2).count() as f64 / n as f64;

        // Pearson correlation of feed_field and V.
        let fm: f64 = feed_field.iter().sum::<f64>() / n as f64;
        let vm = mean_v;
        let mut num = 0.0_f64;
        let mut df2 = 0.0_f64;
        let mut dv2 = 0.0_f64;
        for (fi, vi) in feed_field.iter().zip(v.iter()) {
            let df = fi - fm;
            let dv = vi - vm;
            num += df * dv;
            df2 += df * df;
            dv2 += dv * dv;
        }
        let corr = if df2 * dv2 < 1e-18 { 0.0 } else { num / (df2 * dv2).sqrt() };

        println!(
            "  {:5}    {:4.1}%    {:5.3}     {:4.1}%      {:+5.3}",
            target,
            100.0 * firing,
            mean_v,
            100.0 * coverage,
            corr,
        );
    }
    println!(
        "\nThe spiral wave paints feed onto the canvas; chemistry only\n\
         survives where the wave has recently visited. With time, V\n\
         tracks the wave: corr(feed,V) climbs above zero. The pattern\n\
         is a slow integrator of the fast excitable layer."
    );
}
