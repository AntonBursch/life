//! R4 — Gray-Scott reaction-diffusion.
//!
//! Run from the CLI to seed a small patch and let the box organise. Prints
//! a one-line summary every `--sample` steps so you can watch variance of
//! V climb out of the soup state.
//!
//! Usage:
//!   r04-grayscott
//!   r04-grayscott --F 0.0367 --k 0.0649 --steps 12000
//!   r04-grayscott --w 96 --h 96 --steps 8000 --sample 1000

use flow::gray_scott::GrayScott2D;

fn main() {
    let mut width: usize = 96;
    let mut height: usize = 96;
    let mut du: f64 = 0.16;
    let mut dv: f64 = 0.08;
    let mut feed: f64 = 0.0545;
    let mut kill: f64 = 0.062;
    let dx: f64 = 1.0;
    let dt: f64 = 1.0;
    let mut steps: u64 = 8_000;
    let mut sample_every: u64 = 1_000;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--w" => {
                width = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(width);
                i += 2;
            }
            "--h" => {
                height = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(height);
                i += 2;
            }
            "--du" => {
                du = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(du);
                i += 2;
            }
            "--dv" => {
                dv = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(dv);
                i += 2;
            }
            "--F" => {
                feed = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(feed);
                i += 2;
            }
            "--k" => {
                kill = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(kill);
                i += 2;
            }
            "--steps" => {
                steps = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(steps);
                i += 2;
            }
            "--sample" => {
                sample_every = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(sample_every);
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut sim = match GrayScott2D::new(width, height, du, dv, feed, kill, dx, dt) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("could not build sim: {e}");
            std::process::exit(1);
        }
    };
    sim.seed_blob(width / 2, height / 2, 8);

    println!(
        "# R4 Gray-Scott  w={width} h={height} du={du} dv={dv} F={feed} k={kill} dt={dt}"
    );
    println!("# step, time, mean_v, max_v, var_v");
    println!("{:>6}, {:>8.2}, {:>9.6}, {:>9.6}, {:>10.6}",
        0, sim.time(), sim.mean_v(), sim.max_v(), sim.var_v());

    let mut done: u64 = 0;
    while done < steps {
        let chunk = sample_every.min(steps - done);
        sim.step_many(chunk);
        done += chunk;
        println!(
            "{:>6}, {:>8.2}, {:>9.6}, {:>9.6}, {:>10.6}",
            done,
            sim.time(),
            sim.mean_v(),
            sim.max_v(),
            sim.var_v()
        );
    }
}
