//! Pure-math primitives for life/ rungs.
//!
//! Zero dependencies. Every function here is a faithful translation of the
//! math in `notes/`. The crate is the thing that gets reused across native
//! binaries, the WebSocket server, and the wasm build for the public site.
//!
//! See `notes/01-flow.md` for the math behind R1.

pub mod diffusion;

pub use diffusion::{Diffusion1D, BoundaryCondition};
