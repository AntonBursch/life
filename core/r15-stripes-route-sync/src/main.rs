// R15 -- Stripes route sync. Swift-Hohenberg (R6) self-organises a
// striped pattern; bulk_gate turns |u| into a per-cell Kuramoto
// coupling that is high inside stripes, low in the gaps; Kuramoto
// (R9) locks along the stripes and drifts in the gaps.
use flow::{bulk_gate, Kuramoto2D, SwiftHohenberg2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    let mut pattern = SwiftHohenberg2D::new(w, h, 0.3, 1.0, 0.025).expect("sh ok");
    pattern.seed_noise(0.3);

    let mut phase = Kuramoto2D::new(w, h, 0.0, 0.05).expect("k ok");
    phase.set_natural_frequencies(0.1, 17);
    phase.randomise_phases(42);

    let k_gap = 0.05_f64;
    let k_stripe = 8.0_f64;
    let half_width = 0.3_f64;
    let sharpness = 0.05_f64;

    let mut k_field = vec![0.0_f64; n];

    println!("R15 stripes route sync: SH |u| -> bulk_gate -> Kuramoto.");
    println!(
        "Grid {}x{}, k_gap={}, k_stripe={}, half_width={}.\n",
        w, h, k_gap, k_stripe, half_width
    );
    println!("  sh_t   stripe%   r(stripes)   r(gaps)   r(global)   loc-corr");
    println!("  -----  -------   ----------   -------   ---------   --------");

    let checkpoints: [u32; 5] = [400, 1200, 3000, 8000, 20000];
    let mut total_steps = 0_u32;
    for &target in &checkpoints {
        while total_steps < target {
            pattern.step();
            bulk_gate(
                pattern.u(),
                k_gap, k_stripe, half_width, sharpness,
                &mut k_field,
            ).expect("ok");
            for _ in 0..5 {
                phase.step_with_coupling_field(&k_field).expect("ok");
            }
            total_steps += 1;
        }

        let u = pattern.u();
        let theta = phase.theta();
        let r_global = phase.order_parameter();

        let (mut sc, mut ss, mut sn_n) = (0.0_f64, 0.0_f64, 0usize);
        let (mut gc, mut gs, mut gn_n) = (0.0_f64, 0.0_f64, 0usize);
        for (ui, ti) in u.iter().zip(theta.iter()) {
            if ui.abs() > half_width {
                sc += ti.cos(); ss += ti.sin(); sn_n += 1;
            } else {
                gc += ti.cos(); gs += ti.sin(); gn_n += 1;
            }
        }
        let r_stripe = if sn_n == 0 { 0.0 } else { (sc * sc + ss * ss).sqrt() / sn_n as f64 };
        let r_gap    = if gn_n == 0 { 0.0 } else { (gc * gc + gs * gs).sqrt() / gn_n as f64 };
        let stripe_frac = sn_n as f64 / n as f64;

        let mut acc = 0.0_f64;
        let mut cnt = 0_usize;
        for j in 0..h {
            let jp = if j == h - 1 { 0 } else { j + 1 };
            for i in 0..w {
                let ip = if i == w - 1 { 0 } else { i + 1 };
                let t0 = theta[j * w + i];
                acc += (theta[j * w + ip] - t0).cos();
                acc += (theta[jp * w + i] - t0).cos();
                cnt += 2;
            }
        }
        let loc = if cnt == 0 { 0.0 } else { acc / cnt as f64 };

        println!(
            "  {:5}    {:4.1}%     {:5.3}        {:5.3}      {:5.3}      {:+5.3}",
            target,
            100.0 * stripe_frac,
            r_stripe, r_gap, r_global, loc,
        );
    }
    println!(
        "\nStripes lock (r_stripes climbs); gaps don't (r_gaps stays\n\
         small). The phase pattern inherits the spatial geometry of\n\
         the Swift-Hohenberg pattern -- sync where stripes are,\n\
         drift where they aren't."
    );
}
