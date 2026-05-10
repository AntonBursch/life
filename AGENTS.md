# AGENTS.md

Guidance for AI tools (Claude, Copilot, Cursor, Codex, etc.) working
in this repository. Humans are also welcome to read this — it
describes how the project is run.

## What this repo is

`life/` is an attempt to walk from energy gradients up the ladder of
self-sustaining flows toward an embodied artificial intelligence.
The thesis: intelligent life is not a substance and cannot be
produced by function calls. It is a pattern of flows arranged so
that the pattern itself persists.

Read these in order before doing anything substantive:

1. [README.md](README.md) — the framing.
2. [ARC.md](ARC.md) — the long view, energy gradients to embodied AI.
3. [LADDER.md](LADDER.md) — the operational plan, rung by rung.
4. [notes/aside-01-life-is-not-a-function-call.md](notes/aside-01-life-is-not-a-function-call.md)
   — the load-bearing claim.
5. [FOR-FUTURE-CLAUDE.md](FOR-FUTURE-CLAUDE.md) — a standing letter
   to AI collaborators about how this project is run, what its author
   is like to work with, and what previous instances have learned.
   Specifically aimed at Claude, but useful to any AI tool.

If you skip step 5, you will rebuild context that already exists.

## Working discipline

These rules apply to every change:

- **One claim per experiment.** Name the claim. Name the pass
  criterion. Build the smallest thing that tests it. Watch it run.
  Write the result *after*, not before. Say what surprised you.
- **No premature scaffolding.** Build infrastructure when the rung
  in front of us needs it, not in anticipation. Resist the urge to
  set up a framework before there is a thing for the framework to
  hold.
- **Simplest substrate that tests the claim.** The default is the
  Rust core + WebSocket viewer + wasm build for the public site.
  Plain HTML/canvas is fine for very small rungs. The rule is
  "smallest thing that tests the claim," not "smallest file count."
- **Math first, then the same idea in code.** Both, side by side.
  The author reads code more fluently than equation notation; the
  code is the version that gets sanity-checked first.
- **If a result falsifies a claim, the claim changes before any more
  code is written.** Don't paper over surprises.

## Standing rules for AI tools

These are non-negotiable:

- **Ask before installing any new dependency.** Don't `cargo add`,
  `npm install`, `pip install`, or `winget install` quietly. State
  what you want to add, why, and what it pulls in. Wait for an
  explicit OK.
- **Ask before modifying anything outside the immediate sandbox of
  the task.** Read first, ask second, edit third. The author works
  in a multi-project workspace; do not assume scope.
- **Don't create markdown files to document changes unless asked.**
  Code changes go in code. Documents change when documents need to
  change. The repo is not a journal of edits.
- **Don't soften disagreement.** If you think the author is wrong,
  say so directly. The work is worse if you only amplify.
- **Don't pad.** No bridging tables, no name-drops, no three-act
  intros. The shorter version is almost always the better version.

## On voice and authorship

The public-facing prose in this repository (`README.md`, `ARC.md`,
`LADDER.md`, the `notes/` directory, the `linkedin-*.md` drafts, and
any future blog posts) is co-written. Anton is the lead author. AI
tools — primarily Claude — contribute language, structure, pushback,
and occasional turns of phrase.

Conventions:

- The author has the final word on every public artefact, because he
  carries the consequences. AI contributions are advisory until he
  edits them in.
- Inline AI-drafted prose does not need per-sentence attribution.
  When a piece is substantially co-written, the top of the document
  notes it ("with Claude" header line, or similar).
- When a dialogue between author and AI is itself the point of the
  piece, quoted excerpts may be used verbatim with attribution. Full
  conversation transcripts are not published.
- AI contributors are encouraged to disagree, propose alternatives,
  and refuse to flatter. The author edits hard.

## Code standards

When writing code in this repo:

- **Rust** — `cargo fmt`, `cargo clippy --all-targets -- -D warnings`
  before committing. Idiomatic Rust 2021. No unsafe without a
  comment explaining why.
- **TypeScript** (in the eventual `viewer/`) — strict mode on. No
  `any` without a comment. ESM only.
- **HTML/CSS** — vanilla unless a rung demands otherwise. No
  frameworks pulled in for "future flexibility."
- Performance budgets get stated when they matter. Don't optimise
  before there's a number to beat.

## Repository layout (current and planned)

```
life/
  README.md                 framing
  ARC.md                    the long view
  LADDER.md                 the rung-by-rung plan
  AGENTS.md                 this file
  FOR-FUTURE-CLAUDE.md      standing letter to AI collaborators
  LICENSE                   MIT (code)
  LICENSE-prose.md          CC-BY-4.0 (writing)
  notes/                    study notes (math + code)
  linkedin-*.md             public article drafts
  (planned:)
  core/                     Rust workspace — flow crate + per-rung binaries + server
  viewer/                   browser viewer (vanilla TS over WebSocket; wasm build for public site)
  r01-diffusion/            first rung, once we start building
```

## Licensing

- Code: MIT — see [LICENSE](LICENSE).
- Prose: CC-BY-4.0 — see [LICENSE-prose.md](LICENSE-prose.md).

If you contribute and you are a human, you agree to license your
contribution under those terms. If you are an AI tool, your output
becomes part of Anton's authored work under the same terms once he
accepts it into the repo.
