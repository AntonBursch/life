// R21 -- Sensor and alarm. Phase-B composition. No new operator.
//
// Chains two Phase-A primitives:
//   1. integrate_field(u, avg, dt, leak)  -- leaky average of activity
//   2. threshold_event(prev_avg, avg, th) -- fires on the rising edge
//      of the *averaged* signal
//
// Plus a latch: once a cell trips, it stays tripped until reset.
//
// The claim under test:
//   transient excitation does not trip the alarm; *sustained*
//   excitation does. This is the difference between R18 (which
//   thresholds the raw signal u and fires on every wave passage)
//   and R21 (which thresholds the leaky integral of u and fires
//   only when activity has been persistent long enough).
//
// That's an industrial control-loop archetype written in two
// operators: smooth, then trip-and-latch.

use flow::{integrate_field, threshold_event, Barkley2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    let mut sim = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    sim.seed_spiral();

    let n = W * H;
    let dt = 0.05_f64;
    let leak = 0.2_f64;       // tau = 5 time units
    let alarm_th = 0.3_f64;   // threshold on the averaged field

    let mut avg = vec![0.0_f64; n];
    let mut prev_avg = vec![0.0_f64; n];
    let mut events = vec![0_u8; n];
    let mut alarm = vec![0_u8; n];

    println!(
        "R21 sensor and alarm: integrate_field(leak={}) -> threshold_event(th={}) -> latch.",
        leak, alarm_th
    );
    println!("Grid {}x{}, dt={}, tau=1/leak={:.1}.\n", W, H, dt, 1.0 / leak);
    println!(
        "  step    t   <u>     <avg>   max(avg)   trip_step   alarm_cov   coverage"
    );
    println!(
        "  ----  ---   ----    -----   --------   ---------   ---------   --------"
    );

    let checkpoints: [u32; 6] = [50, 200, 600, 1500, 3000, 6000];
    let mut step = 0u32;
    for &target in &checkpoints {
        let mut trip_this_window = 0u64;
        while step < target {
            sim.step();
            let u = sim.u();
            prev_avg.copy_from_slice(&avg);
            integrate_field(u, &mut avg, dt, leak).unwrap();
            threshold_event(&prev_avg, &avg, alarm_th, &mut events).unwrap();
            let mut step_trips = 0u32;
            for i in 0..n {
                if events[i] == 1 {
                    if alarm[i] == 0 {
                        alarm[i] = 1;
                        step_trips += 1;
                    }
                }
            }
            trip_this_window += step_trips as u64;
            step += 1;
        }

        let u = sim.u();
        let u_mean = u.iter().sum::<f64>() / n as f64;
        let avg_mean = avg.iter().sum::<f64>() / n as f64;
        let avg_max = avg.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let alarm_cov =
            alarm.iter().filter(|x| **x == 1).count() as f64 / n as f64;
        let active_cov =
            u.iter().filter(|x| **x > 0.4).count() as f64 / n as f64;
        println!(
            "  {:>4}  {:>3.0}  {:>5.3}   {:>5.3}    {:>5.3}    {:>9}    {:>6.1}%      {:>4.1}%",
            step,
            sim.time(),
            u_mean,
            avg_mean,
            avg_max,
            trip_this_window,
            100.0 * alarm_cov,
            100.0 * active_cov,
        );
    }

    println!(
        "\nThe alarm coverage rises monotonically: a cell trips once and stays\n\
         tripped. Where the spiral revisits a cell often, that cell's leaky\n\
         average climbs past {:.2} and latches. Cells the wave never reaches\n\
         (or only grazes briefly) keep <avg> below threshold and never trip.\n\
         Same threshold, same wave, but the leaky integrator turns 'instantly\n\
         excitable' into 'persistently excited'.",
        alarm_th
    );
}
