# life/viewer

Browser viewer for the life/ rungs. Vanilla HTML/CSS/JS. No build step,
no dependencies, no `node_modules`.

This directory contains shared visualisation pieces (canvas helpers,
sliders, layout). Each rung lives in its own top-level folder
(`r01-diffusion/`, `r02-driven/`, ...) with its own `index.html` that
imports from here.

To view a rung:

- Static (the way the public site runs it): just open the rung's
  `index.html` in a browser, or run `python -m http.server 8000` from
  the repo root and visit `http://localhost:8000/r01-diffusion/`.
- Through the server (with WebSocket support, once a rung needs it):
  run `cargo run -p server` from `core/`, then visit
  `http://localhost:8787/r01-diffusion/`. The server serves this
  directory and the rung directories as static files and also exposes
  `/ws` for live data when a rung uses it.

The wasm bundle is built into `viewer/pkg/` by
`scripts/build-wasm.ps1` (or `build-wasm.sh`). Both compile
`core/flow-wasm` to `wasm32-unknown-unknown` and run `wasm-bindgen`
to produce ES module bindings.
