// R20 -- Events seed matter. Phase-B opener. Barkley spirals (R7)
// fire rising-edge events via threshold_event (R18's operator).
// Each event deposits V into an autonomous Gray-Scott (R4) field
// running in the stable-spots regime, where seeded blobs grow and
// split rather than fizzle.
//
// The claim under test:
//   discrete events from one substrate can deposit *persistent
//   matter* in another substrate. When the wave is turned off,
//   the pattern keeps going. The wave was creative, not just
//   modulatory.
//
// Contrast with R13: R13 gates feed *continuously* via
// excitable_gate; chemistry collapses if the wave stops because
// the bias goes with it. R20 uses discrete spikes via
// threshold_event into autonomous GS chemistry -- the wave only
// ever writes, never holds.
//
// No new operators. First Phase-B composition.

use flow::{threshold_event, Barkley2D, GrayScott2D};

const W: usize = 96;
const H: usize = 96;

fn main() {
    // Excitable layer: standard Barkley spiral parameters.
    let mut wave = Barkley2D::new(W, H, 1.0, 0.75, 0.06, 0.02, 1.0, 0.05).unwrap();
    wave.seed_spiral();

    // Autonomous chemistry. Stable-spots regime: seeded blobs
    // persist, replicate, and form a dot pattern. No feed-field
    // injection -- once a cell has V, GS does the rest.
    let mut chem = GrayScott2D::new(W, H, 0.16, 0.08, 0.030, 0.062, 1.0, 1.0).unwrap();

    let n = W * H;
    let mut prev_u = wave.u().to_vec();
    let mut events = vec![0u8; n];
    let threshold = 0.4;
    let inject_v = 0.5;        // V <- inject_v on event if cell is "empty"
    let empty_v = 0.05;        // cell considered empty when V < this
    let bark_substeps = 20u32; // 20 * 0.05 = 1.0, matches gs dt

    println!(
        "R20 events seed matter: threshold_event on Barkley u writes V into autonomous Gray-Scott."
    );
    println!(
        "Grid {}x{}, f=0.030, k=0.062 (stable spots), threshold={}, inject_v={} (empty-only).\n",
        W, H, threshold, inject_v
    );
    println!("  chem_t  wave  events_step  cum_events  mean(V)  coverage(V>0.2)");
    println!("  ------  ----  -----------  ----------  -------  ----------------");

    // Phase 1: wave on, chemistry growing under spike rain.
    // Phase 2: wave OFF -- chemistry must sustain itself.
    let checkpoints: [(u32, bool); 6] =
        [(100, true), (500, true), (1500, true), (1800, false), (2400, false), (3000, false)];

    let mut t = 0u32;
    let mut wave_on = true;
    let mut cum_events: u64 = 0;
    let mut last_events_per_step: u32 = 0;
    let mut events_at_off: u64 = 0;

    for &(target, want_wave_on) in &checkpoints {
        if !want_wave_on && wave_on {
            wave_on = false;
            events_at_off = cum_events;
        }

        while t < target {
            let mut step_events = 0u32;
            // Drive the wave at substep resolution so threshold_event
            // sees a real rising edge -- each cell fires at most
            // once per wavefront passage, not once per chem step.
            if wave_on {
                for _ in 0..bark_substeps {
                    prev_u.copy_from_slice(wave.u());
                    wave.step();
                    let u = wave.u();
                    threshold_event(&prev_u, u, threshold, &mut events).unwrap();
                    // Seed only at *isolated* event cells: the cell and
                    // its 4-neighbors must all currently be empty of V.
                    // This keeps spots spaced apart so chemistry has U
                    // left to feed each spot, instead of coating the
                    // whole grid and starving itself.
                    for j in 0..H {
                        let jn = if j == 0 { H - 1 } else { j - 1 };
                        let js = if j + 1 == H { 0 } else { j + 1 };
                        for i in 0..W {
                            let idx = j * W + i;
                            if events[idx] != 1 {
                                continue;
                            }
                            let iw = if i == 0 { W - 1 } else { i - 1 };
                            let ie = if i + 1 == W { 0 } else { i + 1 };
                            let v = chem.v();
                            if v[idx] < empty_v
                                && v[j * W + iw] < empty_v
                                && v[j * W + ie] < empty_v
                                && v[jn * W + i] < empty_v
                                && v[js * W + i] < empty_v
                            {
                                chem.v_mut()[idx] = inject_v;
                                step_events += 1;
                            }
                        }
                    }
                }
            }
            last_events_per_step = step_events;
            cum_events += step_events as u64;
            chem.step();
            t += 1;
        }

        let v = chem.v();
        let mean_v = v.iter().sum::<f64>() / n as f64;
        let coverage = v.iter().filter(|x| **x > 0.2).count() as f64 / n as f64;
        println!(
            "  {:>6}   {}    {:>11}  {:>10}   {:>5.3}     {:>5.1}%",
            t,
            if wave_on { "ON " } else { "OFF" },
            last_events_per_step,
            cum_events,
            mean_v,
            100.0 * coverage,
        );
    }

    let post_off_events = cum_events - events_at_off;
    println!(
        "\nEvents fired after wave turned off: {} (seeded {} before)",
        post_off_events, events_at_off
    );
    println!(
        "\nIf coverage holds or grows in the OFF rows, the chemistry sustained\n\
         itself on what the wave deposited. The wave was creative, not just\n\
         modulatory. Compare R13: continuous feed gating collapses without\n\
         the wave; R20's discrete spike deposit does not."
    );
}
