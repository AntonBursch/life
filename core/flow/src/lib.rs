//! Pure-math primitives for life/ rungs.
//!
//! Zero dependencies. Every function here is a faithful translation of the
//! math in `notes/`. The crate is the thing that gets reused across native
//! binaries, the WebSocket server, and the wasm build for the public site.
//!
//! See `notes/01-flow.md` for the math behind R1.

pub mod advection;
pub mod convection;
pub mod diffusion;
pub mod gray_scott;
pub mod swift_hohenberg;

pub use advection::AdvectionDiffusion1D;
pub use convection::Convection2D;
pub use diffusion::{Diffusion1D, BoundaryCondition};
pub use gray_scott::GrayScott2D;
pub use swift_hohenberg::SwiftHohenberg2D;
