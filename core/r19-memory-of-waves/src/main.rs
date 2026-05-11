// R19 -- Memory of waves. Final Phase-A rung. Barkley spirals (R7)
// drive a per-cell leaky integrator: dy/dt = u - leak*y. With
// leak=0 the integrator is a pure dose meter (total u-exposure
// per cell). With leak>0 it is a low-pass filter -- y settles at
// y_inf = mean(u) / leak, so the field becomes a temporal average
// of the excitable activity.
//
// New operator: integrate_field. Operator alphabet category
// "integrate" -- dual to threshold_event from R18. Continuous in,
// continuous out, with memory.

use flow::{integrate_field, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let dt = 0.05;
    let mut dose = vec![0.0f64; n];      // leak = 0
    let mut avg  = vec![0.0f64; n];      // leak = 0.1 -> tau = 10
    let leak_avg = 0.1;

    let checkpoints = [200, 600, 1500, 3000, 6000];

    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "step", "t", "u_mean", "dose_mn", "dose_mx", "avg_mn", "avg_mx"
    );

    for step in 1..=6000 {
        sim.step();
        let u = sim.u();
        integrate_field(u, &mut dose, dt, 0.0).unwrap();
        integrate_field(u, &mut avg,  dt, leak_avg).unwrap();

        if checkpoints.contains(&step) {
            let t = sim.time();
            let u_mean: f64 = u.iter().sum::<f64>() / n as f64;
            let dose_mn = dose.iter().cloned().fold(f64::INFINITY, f64::min);
            let dose_mx = dose.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let avg_mn  = avg.iter().cloned().fold(f64::INFINITY, f64::min);
            let avg_mx  = avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            println!(
                "{:>6} {:>6.1} {:>10.4} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
                step, t, u_mean, dose_mn, dose_mx, avg_mn, avg_mx
            );
        }
    }

    // Sanity: leaky integrator should approach u_mean / leak.
    let u_mean_final: f64 = sim.u().iter().sum::<f64>() / n as f64;
    let avg_mean: f64 = avg.iter().sum::<f64>() / n as f64;
    println!(
        "\nsteady-state check: <u>={:.4}, expected <y>={:.4}, observed <y>={:.4}",
        u_mean_final, u_mean_final / leak_avg, avg_mean,
    );
}
