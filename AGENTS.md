# AGENTS.md - gadget-ng Copilot Guide

Keep this file short: it is loaded as working context by coding agents.
For deeper project documentation, inspect the repo files referenced below instead of pasting long guides into chat.

## Project

`gadget-ng` is a Rust 1.85+ / Edition 2024 cosmological N-body + SPH + MHD simulator, inspired by GADGET-4 but with no shared code or history.

- Workspace: Cargo workspace, primary binary `gadget-ng` from crate `gadget-ng-cli`.
- License: GPL-3.0-or-later.
- Main crates live under `crates/gadget-ng-*`.

## Where To Look

- Core types, config, cosmology, ICs: `crates/gadget-ng-core/src/`
- TOML config structs: `crates/gadget-ng-core/src/config.rs`
- Gravity: `crates/gadget-ng-tree/`, `crates/gadget-ng-pm/`, `crates/gadget-ng-treepm/`
- Integrators: `crates/gadget-ng-integrators/`
- Parallel/MPI/domain decomposition: `crates/gadget-ng-parallel/`
- IO formats: `crates/gadget-ng-io/`
- GPU/CUDA/HIP: `crates/gadget-ng-gpu/`, `crates/gadget-ng-cuda/`, `crates/gadget-ng-hip/`
- SPH/MHD/RT: `crates/gadget-ng-sph/`, `crates/gadget-ng-mhd/`, `crates/gadget-ng-rt/`
- Analysis/visualization: `crates/gadget-ng-analysis/`, `crates/gadget-ng-vis/`
- CLI: `crates/gadget-ng-cli/src/main.rs`, `crates/gadget-ng-cli/src/engine.rs`
- Physics validation: `crates/gadget-ng-physics/tests/`
- Reports/runbooks/paper: `docs/reports/`, `docs/runbooks/`, `docs/paper/paper.md`
- Examples/configs: `examples/`, `configs/`

## Development Rules

- Keep changes minimal and scoped. Avoid unrelated refactors.
- Use existing crate boundaries and local patterns.
- Core/shared data types belong in `gadget-ng-core`.
- Do not make `gadget-ng-cli` depend directly on internal modules of `gadget-ng-pm`.
- `cargo fmt --all` is enforced.
- `cargo clippy --workspace -- -D warnings` is blocking.
- `cargo clippy --workspace --all-targets -- -D warnings` is advisory; see `docs/reports/2026-05-clippy-all-targets-backlog.md`.
- Workspace has `unsafe_code = "warn"`. Prefer safe Rust. Any `unsafe` block needs a `// SAFETY:` justification.

## Sacred Rules

1. Do not change existing test logic or physics assertions during refactors; fix call sites instead.
2. Do not break bit-identical reproducibility in serial mode when `deterministic = true`.
3. Do not modify `pk_correction` constants (`A_grid`, `R(N)`) without updating the corresponding report and re-running CLASS validation.
4. Do not touch `experiments/nbody/phase*/reference/` unless formally replacing a reference run and documenting it.
5. Keep crate boundaries clean.

## Testing

Pick the smallest useful tier for the change:

- Quick core: `cargo test -p gadget-ng-core`
- Affected crate: `cargo test -p <crate>`
- Physics baseline: `bash scripts/check-physics.sh`
- Workspace: `cargo test --workspace`
- Deep physics: `cargo test -p gadget-ng-physics -- --include-ignored`
- Pre-push: `bash scripts/check.sh`

Run physics baseline/deep tests when touching cosmology, ICs, gravity, SPH, MHD, or validation logic.

### Parity: serial CPU, Rayon, SIMD, and GPU

Aim for **numerical parity** (within explicit tolerances) across, at minimum:

1. **CPU without Rayon** — default build paths (serial iterators); reproducibility baseline.
2. **CPU with Rayon** — typically `cfg(feature = "simd")` branches that use `rayon`; must match (1).
3. **SIMD / intrinsics without relying on Rayon** — only where Cargo features or code paths separate true vectorization from thread parallelism; must still match (1). Today several crates alias `simd` to optional `rayon`; if you split features, document and test both axes.
4. **CUDA / HIP** — GPU solvers vs CPU reference; parity/smoke tests in the owning crate when practical. CI may skip GPUs via `CUDA_SKIP=1` / `HIP_SKIP=1`.

**x86 vector tiers:** for new `#[target_feature]` / intrinsic `f64` paths on **x86_64**, provide both **AVX2 + FMA** and **AVX512F** (`avx512f`) implementations when batching pays off, with `is_x86_feature_detected!` dispatch (typically `avx512f` → `avx2`+`fma` → scalar) and parity against (1). Other targets stay portable/scalar unless there is a dedicated port.

Track accelerator gaps in `docs/reports/2026-05-simd-cuda-coverage.md` when touching SIMD/CUDA scope.

Keep **unit tests and coverage** in step with new branches: add focused tests for new `cfg` paths; do not drop physics assertions or shrink coverage when landing optimizations unless the team explicitly replaces them with stronger checks.

## Feature / Phase Pattern

For a new phase or user-visible feature:

1. Add a brief report under `docs/reports/YYYY-MM-phaseNNN-description.md`.
2. Implement in the appropriate crate with public doc comments.
3. Add focused tests in the relevant crate or `gadget-ng-physics`.
4. Wire CLI/config support if user-facing.
5. Update examples/configs as needed.
6. Update `README.md`, `CHANGELOG.md`, and `docs/roadmap.md` when user-visible.

New TOML options belong in `crates/gadget-ng-core/src/config.rs` and should be mirrored in an example under `examples/` or `configs/`.

## Commits

Use Conventional Commits:

```text
<type>(<scope>): <description>
```

Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`.
Scopes are crate names such as `core`, `tree`, `pm`, `physics`, or areas such as `experiments`, `ci`.

## Common Commands

```bash
cargo build --release -p gadget-ng-cli
cargo build --release -p gadget-ng-cli --features mpi
cargo build --release -p gadget-ng-cli --features hdf5
cargo build --release -p gadget-ng-cli --features full

./target/release/gadget-ng stepping --config examples/plummer_sphere.toml --out runs/plummer --snapshot
./target/release/gadget-ng analyze --snapshot runs/plummer/snapshot_final --out runs/plummer/analysis

cargo bench -p gadget-ng-core --features simd
cargo bench -p gadget-ng-tree --features simd
```

## Troubleshooting

- If `cargo test --workspace` hangs, run affected crates individually; MPI tests may need `mpirun`.
- HDF5 failures usually need `libhdf5-dev` or equivalent.
- CUDA/HIP are optional; CI can use `CUDA_SKIP=1` / `HIP_SKIP=1`.
- After IC or cosmology changes, run `bash scripts/check-physics.sh` and inspect relevant reference reports.
