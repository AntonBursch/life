// R22 -- Memory shapes flow. Phase-B composition. No new operator.
//
// Three Phase-A operators in a chain:
//   1. integrate_field(u, memory, dt, leak)   -- R19 leaky integral
//                                                of Barkley activity
//   2. gradient_field(memory, ...)            -- turn that scalar
//                                                memory map into a
//                                                velocity vector
//   3. advect_by(dye, vx, vy, ...)            -- transport a dye
//                                                field by that
//                                                velocity
//
// The velocity is alpha * grad(memory). Dye flows up the memory
// gradient toward cells where the wave has been firing often.
//
// The claim under test:
//   *Past* excitation shapes *present* transport. This is the
//   first rung where the current motion of matter depends on
//   what happened earlier -- not on the current state of any
//   substrate, but on its time-integrated history. The wave
//   leaves a memory; the memory becomes a force on the dye.

use flow::{advect_by, gradient_field, integrate_field, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let dt = 0.05_f64;
    let leak = 0.05_f64;         // tau = 20 (slower memory)
    let alpha = 4.0_f64;         // gradient -> velocity gain
    let dx = 1.0_f64;

    let mut memory = vec![0.0_f64; n];
    let mut gx = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];

    // Dye initial condition: a wide Gaussian blob to the right of
    // the spiral seed. A wide blob survives numerical diffusion;
    // a sharp stripe does not. With no memory effect, the blob
    // sits still. With memory-driven flow, it migrates toward
    // the spiral.
    let mut dye = vec![0.0_f64; n];
    let mut dye_next = vec![0.0_f64; n];
    let cx0 = (W * 3) / 4;
    let cy0 = H / 2;
    let sigma2 = 10.0_f64 * 10.0_f64;
    for j in 0..H {
        for i in 0..W {
            let dxp = i as f64 - cx0 as f64;
            let dyp = j as f64 - cy0 as f64;
            let r2 = dxp * dxp + dyp * dyp;
            dye[j * W + i] = (-r2 / (2.0 * sigma2)).exp();
        }
    }

    let dye_initial_total: f64 = dye.iter().sum();
    let dye_initial_left: f64 = (0..H).map(|j| (0..W / 2).map(|i| dye[j * W + i]).sum::<f64>()).sum();

    println!(
        "R22 memory shapes flow: integrate_field -> gradient_field -> advect_by."
    );
    println!(
        "Grid {}x{}, dt={}, tau=1/leak={:.1}, alpha={}, dx={}.\n",
        W, H, dt, 1.0 / leak, alpha, dx
    );
    println!(
        "  step    t   <u>    max(mem)   max|v|    dye_total   dye_left   centroid_x"
    );
    println!(
        "  ----  ---   ----   --------   ------    ---------   --------   ----------"
    );

    let checkpoints: [u32; 6] = [50, 200, 600, 1500, 3000, 6000];
    let mut step = 0u32;
    for &target in &checkpoints {
        while step < target {
            sim.step();
            let u = sim.u();
            integrate_field(u, &mut memory, dt, leak).unwrap();
            gradient_field(&memory, W, H, dx, &mut gx, &mut gy).unwrap();
            // v = alpha * grad(memory). Scale into vx, vy in place.
            for k in 0..n {
                gx[k] *= alpha;
                gy[k] *= alpha;
            }
            advect_by(&dye, &gx, &gy, W, H, dx, dt, &mut dye_next).unwrap();
            std::mem::swap(&mut dye, &mut dye_next);
            step += 1;
        }

        let u = sim.u();
        let u_mean = u.iter().sum::<f64>() / n as f64;
        let mem_max = memory.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let v_max = (0..n)
            .map(|k| (gx[k] * gx[k] + gy[k] * gy[k]).sqrt())
            .fold(f64::NEG_INFINITY, f64::max);
        let dye_total: f64 = dye.iter().sum();
        let dye_left: f64 = (0..H)
            .map(|j| (0..W / 2).map(|i| dye[j * W + i]).sum::<f64>())
            .sum();
        let mut cx_num = 0.0_f64;
        let mut cx_den = 0.0_f64;
        for j in 0..H {
            for i in 0..W {
                let d = dye[j * W + i];
                cx_num += i as f64 * d;
                cx_den += d;
            }
        }
        let centroid_x = if cx_den > 1e-12 { cx_num / cx_den } else { 0.0 };
        println!(
            "  {:>4}  {:>3.0}  {:>5.3}   {:>5.3}     {:>5.3}      {:>5.1}     {:>5.1}      {:>5.2}",
            step,
            sim.time(),
            u_mean,
            mem_max,
            v_max,
            dye_total,
            dye_left,
            centroid_x,
        );
    }

    println!(
        "\nInitial dye centroid_x = {:.2}; initial dye_left = {:.1} (of {:.1} total).",
        cx0 as f64,
        dye_initial_left,
        dye_initial_total
    );
    println!(
        "If centroid_x drifts to the left and dye_left grows, the dye is being\n\
         pulled across the grid by the memory-driven velocity field -- toward\n\
         the spiral's history. Past excitation is shaping present motion. No\n\
         operator in the chain saw the wave directly; the dye sees only\n\
         grad(memory), which is the wave's *integrated trace*."
    );
}
