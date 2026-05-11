//! Pure-math primitives for life/ rungs.
//!
//! Zero dependencies. Every function here is a faithful translation of the
//! math in `notes/`. The crate is the thing that gets reused across native
//! binaries, the WebSocket server, and the wasm build for the public site.
//!
//! See `notes/01-flow.md` for the math behind R1.

pub mod advection;
pub mod barkley;
pub mod cahn_hilliard;
pub mod convection;
pub mod coupling;
pub mod diffusion;
pub mod gray_scott;
pub mod kuramoto;
pub mod reaction;
pub mod swift_hohenberg;

pub use advection::AdvectionDiffusion1D;
pub use barkley::Barkley2D;
pub use cahn_hilliard::CahnHilliard2D;
pub use convection::Convection2D;
pub use coupling::{excitable_gate, phase_to_scalar_field, bulk_gate, gradient_magnitude, gradient_field, advect_by, threshold_event, integrate_field, modulate_parameter, latch_field, CouplingError};
pub use diffusion::{Diffusion1D, BoundaryCondition};
pub use gray_scott::GrayScott2D;
pub use kuramoto::Kuramoto2D;
pub use reaction::{react_field, schlogl_rate};
pub use swift_hohenberg::SwiftHohenberg2D;
