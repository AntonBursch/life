// R17 -- Territory carries dye. Same CH (R8) territory as R12/R16,
// but treated as a *stream function*: gradient_field reads
// (dphi/dx, dphi/dy), and we rotate by 90 degrees to make
// velocity v = (dphi/dy, -dphi/dx). v is divergence-free, so
// advect_by transports the dye without sources or sinks. Dye
// follows level sets of phi -- i.e. it streams along the walls
// of the territory. The wall network becomes a network of
// rivers.
//
// First use of advect_by (transport primitive). First use of
// gradient_field (vector read). Operator alphabet category:
// "transport".

use flow::{advect_by, gradient_field, CahnHilliard2D};

const W: usize = 96;
const H: usize = 96;
const DX: f64 = 1.0;
const DT_CH: f64 = 0.05;
const DT_ADV: f64 = 0.1;
const VSCALE: f64 = 4.0;

fn main() {
    let mut territory = CahnHilliard2D::new(W, H, 0.5, 0.5, DX, DT_CH).unwrap();
    territory.seed_noise(0.05, 0.0, 12345);
    for _ in 0..400 {
        territory.step();
    }

    // Dye: horizontal stripes.
    let mut dye = vec![0.0_f64; W * H];
    for j in 0..H {
        let s = (j as f64 / H as f64 * std::f64::consts::TAU * 3.0).sin();
        let v = (s + 1.0) * 0.5;
        for i in 0..W {
            dye[j * W + i] = v;
        }
    }
    let dye_initial_mass: f64 = dye.iter().sum();
    let var_initial = variance(&dye);

    let mut gx = vec![0.0_f64; W * H];
    let mut gy = vec![0.0_f64; W * H];
    let mut vx = vec![0.0_f64; W * H];
    let mut vy = vec![0.0_f64; W * H];
    let mut next_dye = vec![0.0_f64; W * H];

    let checkpoints = [20, 60, 150, 400, 1200];
    println!(
        "{:>6} {:>6} {:>10} {:>10} {:>10}",
        "step", "t_ch", "mass", "var(dye)", "var0"
    );

    for step in 1..=1200 {
        territory.step();
        gradient_field(territory.c(), W, H, DX, &mut gx, &mut gy).unwrap();
        // Stream-function velocity: rotate gradient 90 degrees.
        for k in 0..gx.len() {
            vx[k] =  VSCALE * gy[k];
            vy[k] = -VSCALE * gx[k];
        }
        advect_by(&dye, &vx, &vy, W, H, DX, DT_ADV, &mut next_dye).unwrap();
        std::mem::swap(&mut dye, &mut next_dye);

        if checkpoints.contains(&step) {
            let mass: f64 = dye.iter().sum();
            let var = variance(&dye);
            println!(
                "{:>6} {:>6.1} {:>10.4} {:>10.4} {:>10.4}",
                step,
                territory.time(),
                mass / dye_initial_mass,
                var,
                var_initial,
            );
        }
    }
}

fn variance(field: &[f64]) -> f64 {
    let mean: f64 = field.iter().sum::<f64>() / field.len() as f64;
    field.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / field.len() as f64
}
