// R23 -- Excitable courier. Phase-B composition. No new operator.
//
// Two Phase-A operators wired direct, with no integrator
// between them:
//   1. gradient_field(u, ...)              -- R17 read-vector of
//                                             the *instantaneous*
//                                             Barkley activator
//   2. advect_by(payload, alpha*grad(u))   -- R17 transport
//
// Difference from R22: there is no memory. Velocity is the
// slope of the current wave, not the slope of its integrated
// trace. The wave itself carries the payload as it passes.
//
// The claim: an excitable wave with handedness (a spiral)
// produces a coherent net drift on a passive payload even
// without any memory. Each wave passage gives the payload a
// small kick in the direction of the rotation; the spiral's
// chirality breaks left-right symmetry; the kicks accumulate.

use flow::{advect_by, gradient_field, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let dt = 0.05_f64;
    let alpha = 3.0_f64;
    let dx = 1.0_f64;

    let mut gx = vec![0.0_f64; n];
    let mut gy = vec![0.0_f64; n];

    // Payload: a wide Gaussian blob in the upper-left quadrant.
    // With no wave-driven transport, it sits put. With the
    // courier wired up, it should drift in the spiral's
    // rotation direction.
    let mut pay = vec![0.0_f64; n];
    let mut pay_next = vec![0.0_f64; n];
    let cx0 = (W / 4) as f64;
    let cy0 = (H / 4) as f64;
    let sigma2 = 8.0_f64 * 8.0_f64;
    for j in 0..H {
        for i in 0..W {
            let dxp = i as f64 - cx0;
            let dyp = j as f64 - cy0;
            pay[j * W + i] = (-(dxp * dxp + dyp * dyp) / (2.0 * sigma2)).exp();
        }
    }

    let total0: f64 = pay.iter().sum();
    let cx_init = cx0;
    let cy_init = cy0;

    println!("R23 excitable courier: gradient_field(u) -> advect_by(payload).");
    println!("Grid {}x{}, dt={}, alpha={}, dx={}. No memory in the loop.\n", W, H, dt, alpha, dx);
    println!("  step    t   <u>    max|grad u|    pay_total    centroid_x   centroid_y    drift");
    println!("  ----  ---   ----   ----------    ---------    ----------   ----------   ------");

    let checkpoints: [u32; 6] = [50, 200, 600, 1500, 3000, 6000];
    let mut step = 0u32;
    for &target in &checkpoints {
        while step < target {
            sim.step();
            let u = sim.u();
            gradient_field(u, W, H, dx, &mut gx, &mut gy).unwrap();
            for k in 0..n {
                gx[k] *= alpha;
                gy[k] *= alpha;
            }
            advect_by(&pay, &gx, &gy, W, H, dx, dt, &mut pay_next).unwrap();
            std::mem::swap(&mut pay, &mut pay_next);
            step += 1;
        }
        let u = sim.u();
        let u_mean = u.iter().sum::<f64>() / n as f64;
        let g_max = (0..n).map(|k| (gx[k] * gx[k] + gy[k] * gy[k]).sqrt())
            .fold(f64::NEG_INFINITY, f64::max);
        let total: f64 = pay.iter().sum();
        let mut cx_n = 0.0_f64;
        let mut cy_n = 0.0_f64;
        let mut cd = 0.0_f64;
        for j in 0..H {
            for i in 0..W {
                let p = pay[j * W + i];
                cx_n += i as f64 * p;
                cy_n += j as f64 * p;
                cd += p;
            }
        }
        let cx = if cd > 1e-12 { cx_n / cd } else { 0.0 };
        let cy = if cd > 1e-12 { cy_n / cd } else { 0.0 };
        let drift = ((cx - cx_init).powi(2) + (cy - cy_init).powi(2)).sqrt();
        println!(
            "  {:>4}  {:>3.0}  {:>5.3}    {:>6.3}        {:>5.1}        {:>5.2}        {:>5.2}      {:>5.2}",
            step, sim.time(), u_mean, g_max, total, cx, cy, drift,
        );
    }

    println!(
        "\nInitial centroid ({:.2}, {:.2}); total {:.1}.\n\
         If centroid drifts away from start, the wave is carrying the\n\
         payload despite there being no memory in the pipeline. Each\n\
         wavefront passage gives a kick; the spiral's chirality biases\n\
         the kicks.",
        cx_init, cy_init, total0,
    );
}
