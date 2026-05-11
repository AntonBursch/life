// R16 -- Walls route sync. Same substrate pair as R12 (Cahn-Hilliard
// + Kuramoto) but routed through the *new* operator
// `gradient_magnitude` instead of `bulk_gate`.
//
// R12: bulk_gate(phi) -> Kuramoto syncs INSIDE the +/-1 bulk regions,
//      walls stay incoherent.
// R16: gradient_magnitude(phi) -> Kuramoto syncs ON the walls,
//      bulk stays incoherent.
//
// Same upstream, same downstream, different operator -> opposite
// spatial signature. The "differentiate" category of the operator
// alphabet now exists and inverts the wall/bulk geometry on demand.
use flow::{gradient_magnitude, CahnHilliard2D, Kuramoto2D};

fn main() {
    let w = 96usize;
    let h = 96usize;
    let n = w * h;

    let mut territory = CahnHilliard2D::new(w, h, 0.5, 0.5, 1.0, 0.05).expect("ch ok");
    territory.seed_noise(0.05, 0.0, 12345);
    for _ in 0..400 { territory.step(); }

    let mut phase = Kuramoto2D::new(w, h, 0.0, 0.05).expect("k ok");
    phase.set_natural_frequencies(0.1, 17);
    phase.randomise_phases(42);

    let k_bulk = 0.0_f64;
    let k_wall = 10.0_f64;
    let grad_ref = 0.5_f64;
    let sharp = 0.15_f64;
    let mut grad = vec![0.0_f64; n];
    let mut k_field = vec![k_bulk; n];

    println!("R16 walls route sync: CH |grad phi| -> Kuramoto K.");
    println!(
        "Grid {}x{}, k_bulk={}, k_wall={}, grad_ref={}.\n",
        w, h, k_bulk, k_wall, grad_ref
    );
    println!("  step    t_t    wall%    r(walls)   r(bulk)   r(global)   loc-corr");
    println!("  -----   ----   ------   --------   -------   ---------   --------");

    let checkpoints: [u32; 5] = [400, 1200, 3000, 8000, 20000];
    let mut total_steps = 0_u32;
    for &target in &checkpoints {
        while total_steps < target {
            territory.step();
            gradient_magnitude(territory.c(), w, h, 1.0, &mut grad).expect("grad ok");
            let span = k_wall - k_bulk;
            let edge0 = grad_ref - sharp;
            let edge1 = grad_ref + sharp;
            let inv = if edge1 > edge0 { 1.0 / (edge1 - edge0) } else { 0.0 };
            for (g, kk) in grad.iter().zip(k_field.iter_mut()) {
                let t = ((*g - edge0) * inv).clamp(0.0, 1.0);
                let s = t * t * (3.0 - 2.0 * t);
                *kk = k_bulk + span * s;
            }
            phase.step_with_coupling_field(&k_field).expect("ok");
            total_steps += 1;
        }

        let thresh = 0.5 * grad_ref;
        let theta = phase.theta();
        let (mut wc, mut ws, mut wn) = (0.0_f64, 0.0_f64, 0_usize);
        let (mut bc, mut bs, mut bn) = (0.0_f64, 0.0_f64, 0_usize);
        for (gi, ti) in grad.iter().zip(theta.iter()) {
            if *gi > thresh { wc += ti.cos(); ws += ti.sin(); wn += 1; }
            else { bc += ti.cos(); bs += ti.sin(); bn += 1; }
        }
        let r_wall = if wn == 0 { 0.0 } else { (wc * wc + ws * ws).sqrt() / wn as f64 };
        let r_bulk = if bn == 0 { 0.0 } else { (bc * bc + bs * bs).sqrt() / bn as f64 };
        let r_global = phase.order_parameter();
        let wall_frac = wn as f64 / n as f64;

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
            "  {:5}   {:4.1}    {:4.1}%   {:5.3}      {:5.3}     {:5.3}      {:+5.3}",
            target, territory.time(), 100.0 * wall_frac, r_wall, r_bulk, r_global, loc,
        );
    }
    println!(
        "\nWall cells lock (r_walls climbs); bulk cells drift\n\
         (r_bulk stays small). The phase pattern carries the\n\
         CH wall network as a synced backbone, exactly inverting\n\
         the R12 geometry where sync lived inside the bulk."
    );
}
