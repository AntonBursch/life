//! R7 CLI — Barkley excitable medium. Seed a broken wavefront and watch
//! the medium sustain a spiral wave that keeps spinning.

use flow::Barkley2D;

fn main() {
    let w = 80;
    let h = 80;
    let mut sim = Barkley2D::new(w, h, 1.0, 0.75, 0.01, 0.02, 0.4, 0.01)
        .expect("valid params");

    println!("R7 Barkley excitable: {}x{}  D=1  a=0.75  b=0.01  eps=0.02  dt=0.01", w, h);

    println!("\nrest medium, no kick:");
    sim.reset();
    sim.step_many(5000);
    println!("  excited fraction = {:.3}   (should be 0)", sim.excited_fraction());

    println!("\nsubthreshold kick (amp=0.005, below b/a=0.0133):");
    sim.reset();
    sim.kick(40, 40, 3, 0.005);
    sim.step_many(5000);
    println!("  excited fraction = {:.3}   (should be 0)", sim.excited_fraction());

    println!("\nsuprathreshold kick (amp=0.8): expect a ring wave that crosses the box and self-annihilates through the periodic boundary.");
    sim.reset();
    sim.kick(40, 40, 6, 0.8);
    println!("    time     mean(u)    max|u|   var(u)   excited");
    println!("  -------    -------    ------   ------   -------");
    for _ in 0..10 {
        sim.step_many(100);
        println!(
            "  {:>7.1}    {:>+.4}    {:>5.3}    {:>5.3}    {:>5.3}",
            sim.time(), sim.mean_u(), sim.max_abs_u(),
            sim.variance_u(), sim.excited_fraction()
        );
    }

    println!("\nspiral seed: expect sustained activity, not decay.");
    sim.reset();
    sim.seed_spiral();
    println!("    time     mean(u)    max|u|   var(u)   excited");
    println!("  -------    -------    ------   ------   -------");
    for _ in 0..10 {
        sim.step_many(2000);
        println!(
            "  {:>7.1}    {:>+.4}    {:>5.3}    {:>5.3}    {:>5.3}",
            sim.time(), sim.mean_u(), sim.max_abs_u(),
            sim.variance_u(), sim.excited_fraction()
        );
    }
}
