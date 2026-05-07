# AGENTS.md — gadget-ng Copilot Guide

> Quick orientation for AI assistants and human contributors working on `gadget-ng`.
> Read this before editing anything. When in doubt, ask.

---

## 1. What is this project?

`gadget-ng` is a cosmological N-body + SPH + MHD simulator written in Rust.
It is conceptually inspired by GADGET-4 but contains no shared code or git history.

- **Language:** Rust 1.85+, Edition 2024
- **Build tool:** Cargo workspace with 18 crates
- **License:** GPL-3.0-or-later
- **Primary binary:** `gadget-ng` (crate `gadget-ng-cli`)

---

## 2. Workspace Layout (QuickRef)

```
gadget-ng/
├── crates/
│   ├── gadget-ng-gpu-layout    # Wire types for CPU↔GPU (SoA layouts), zero deps
│   ├── gadget-ng-core          # Vec3, Particle, RunConfig, cosmology, ICs (1LPT/2LPT)
│   ├── gadget-ng-tree          # Octree, Barnes-Hut, FMM (mono/quad/octupole), LET
│   ├── gadget-ng-integrators   # Leapfrog KDK, Yoshida4, hierarchical block timesteps
│   ├── gadget-ng-parallel      # SerialRuntime / MpiRuntime, SFC Hilbert 3D, domain decomp
│   ├── gadget-ng-io            # Snapshots: JSONL, bincode, HDF5, msgpack, netcdf
│   ├── gadget-ng-pm            # PM periodic 3D (CIC + FFT Poisson); optional `fftw` feature
│   ├── gadget-ng-treepm        # TreePM: Gaussian k-space splitting, short-range tree + long-range PM
│   ├── gadget-ng-gpu           # wgpu compute shaders (Vulkan/Metal/DX12)
│   ├── gadget-ng-cuda          # NVIDIA CUDA PM solver (optional, second build chain)
│   ├── gadget-ng-hip           # AMD HIP PM solver (optional, second build chain)
│   ├── gadget-ng-analysis      # FoF, P(k), pk_correction, HMF, NFW, ξ(r), lightcone/Born
│   ├── gadget-ng-sph           # SPH: Wendland C2, entropy, Balsara, Gadget-2 formulation
│   ├── gadget-ng-mhd           # Ideal MHD, SRMHD, Dedner, Braginskii, reconnection, 2-fluid plasma
│   ├── gadget-ng-rt            # Radiative transfer M1, reionization EoR, 21cm
│   ├── gadget-ng-vis           # CPU rendering: PNG, PPM, Viridis colormap
│   ├── gadget-ng-physics       # Integration / validation tests (physics, cosmology, convergence)
│   └── gadget-ng-cli           # Binary + subcommands: config, snapshot, stepping, analyze, visualize
├── examples/                   # TOML examples (Plummer, Kepler, cosmological, etc.)
├── configs/                    # Production / validation configs (256³, 128³, EoR)
├── experiments/nbody/          # Phase experiments and benchmarks (per-phase outputs)
├── docs/
│   ├── reports/                # Technical reports per phase: 2026-04-phaseNN-*.md
│   ├── paper/                  # JOSS paper draft
│   ├── runbooks/               # Operational guides (MPI cluster, validation, etc.)
│   └── scripts/                # Python postprocessing scripts
├── notebooks/                  # Jupyter notebooks (Spanish, step-by-step tutorials)
├── scripts/
│   ├── check.sh                # Full pre-push validation (fmt, clippy, tests, MPI smoke)
│   ├── check-physics.sh        # Physics baseline: tabulated transfer + pancake + CLASS
│   ├── check_release.sh        # Release build + integration checks
│   └── ci/                     # CI helper scripts
└── .github/workflows/
    ├── ci.yml                  # Main CI: fmt, clippy, test, doc, MPI smoke
    ├── physics-validation.yml  # Nightly physics validation with artifacts
    └── release.yml             # Release builds
```

---

## 3. Development Conventions

### 3.1 Commits
Use **Conventional Commits**:

```
<type>(<scope>): <description>

[optional body]
```

Types: `feat`, `fix`, `docs`, `chore`, `test`, `refactor`, `perf`, `ci`
Scopes: crate name (e.g., `core`, `tree`, `pm`, `physics`), or area (e.g., `experiments`, `ci`)

Examples from history:
- `fix(core): adaptar ICs Zel'dovich a rand 0.10`
- `feat(experiments): unificar validation_papers en modo cpu/simd/cuda`
- `docs(validation_papers/03): README FoF, resolución b=0.2 y referencias`

### 3.2 Code Style
- `cargo fmt --all` is enforced in CI.
- `cargo clippy --workspace -- -D warnings` is **blocking** in CI.
- `cargo clippy --workspace --all-targets -- -D warnings` is **advisory** (see backlog in `docs/reports/2026-05-clippy-all-targets-backlog.md`).
- Do **not** introduce new warnings unless you also fix them.
- Keep changes **minimal**; avoid refactor creep in bug-fix PRs.

### 3.3 Unsafe Code
- `unsafe_code = "warn"` is set at workspace level.
- Unsafe is permitted only in GPU/CUDA/HIP crates with clear justification.
- Prefer safe Rust; document any `unsafe` block with a `// SAFETY:` comment.

---

## 4. Testing Strategy

### 4.1 Test Tiers

| Tier | Command | Time | Purpose |
|------|---------|------|---------|
| **Quick** | `cargo test -p gadget-ng-core` | <5s | Core types, configs, utilities |
| **Physics baseline** | `bash scripts/check-physics.sh` | ~15s | Tabulated transfer, Zel'dovich pancake, CLASS validation |
| **Full workspace** | `cargo test --workspace` | ~3–5 min | All unit + integration tests |
| **Physics deep** | `cargo test -p gadget-ng-physics -- --include-ignored` | ~3.5 min | SPH Gadget-2, Evrard collapse, MHD suites |
| **Pre-push** | `bash scripts/check.sh` | ~10–15 min | fmt, clippy, tests, docs, MPI build, MPI smoke |

### 4.2 Slow / Ignored Tests
- Tests in `gadget-ng-physics` that take >5s are marked `#[ignore]`.
- Run them with `--include-ignored` when validating physics changes.
- GPU tests requiring real hardware are also `#[ignore]`; CI skips them with `CUDA_SKIP=1` / `HIP_SKIP=1`.

### 4.3 When to run what
- **Every edit:** Quick tier (or at least the affected crate).
- **Before committing:** Physics baseline + clippy.
- **Before pushing / opening PR:** Full `scripts/check.sh`.
- **When touching cosmology / ICs / gravity:** Physics deep tier.

---

## 5. Sacred Rules (Do Not Break Without Discussion)

1. **Do not change existing test logic** when refactoring. If an interface change breaks a test, fix the call site, not the test's physics assertions. This preserves the validation baseline.
2. **Do not break bit-identical reproducibility** in serial mode (`deterministic = true`). This is a core guarantee.
3. **Do not modify `pk_correction` constants** (`A_grid`, `R(N)`) without updating the corresponding report in `docs/reports/` and re-running CLASS validation.
4. **Do not touch `experiments/nbody/phase*/reference/` files** unless you are formally replacing a reference run (documented in a report).
5. **Keep crate boundaries clean.** Core types live in `gadget-ng-core`; do not let `gadget-ng-cli` depend directly on internal modules of `gadget-ng-pm`.

---

## 6. Adding a New Phase / Feature

The project organizes major milestones as **Phases** (e.g., Phase 123 = MHD crate creation). If you are adding a feature, follow this pattern:

1. **Design:** Write a brief technical note in `docs/reports/YYYY-MM-phaseNNN-descripcion.md`.
2. **Implement:** Add code to the appropriate crate(s). Expose public APIs with doc comments.
3. **Test:** Add tests in `gadget-ng-physics` or the relevant crate. Mark slow ones `#[ignore]`.
4. **Integrate:** Wire the feature into `gadget-ng-cli` (subcommands, TOML config fields) if user-facing.
5. **Validate:** Run `scripts/check-physics.sh` and `cargo test --workspace`.
6. **Document:** Update `README.md` feature table, `CHANGELOG.md`, and `docs/roadmap.md`.

### 6.1 Config fields
New TOML options belong in `gadget-ng-core/src/config.rs` (`RunConfig` + serde structs). Mirror them in an example TOML under `examples/` or `configs/`.

---

## 7. Pre-Commit Checklist

- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --workspace -- -D warnings` passes
- [ ] `cargo test --workspace` passes (or at least affected crates)
- [ ] If touching physics: `bash scripts/check-physics.sh` passes
- [ ] New/modified code has doc comments
- [ ] If adding a phase: report exists in `docs/reports/`
- [ ] `CHANGELOG.md` updated if user-visible

---

## 8. Common Commands

```bash
# Build minimal binary
cargo build --release -p gadget-ng-cli

# Build with MPI
cargo build --release -p gadget-ng-cli --features mpi

# Build with HDF5
cargo build --release -p gadget-ng-cli --features hdf5

# Build everything
cargo build --release -p gadget-ng-cli --features full

# Run a quick simulation
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --snapshot

# Analyze snapshot
./target/release/gadget-ng analyze \
  --snapshot runs/plummer/snapshot_final \
  --out runs/plummer/analysis

# Run benchmarks
cargo bench -p gadget-ng-core --features simd
cargo bench -p gadget-ng-tree --features simd
```

---

## 9. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `cargo test --workspace` hangs | Some MPI tests may hang without `mpirun`. Run `cargo test -p <crate>` individually. |
| Clippy warnings in benches/tests | See `docs/reports/2026-05-clippy-all-targets-backlog.md`. Fix them if you can; do not introduce new ones. |
| HDF5 build fails | Install `libhdf5-dev` (Debian/Ubuntu) or equivalent. |
| CUDA/HIP build fails | These are optional. Set `CUDA_SKIP=1` or `HIP_SKIP=1` to skip in CI/tests. |
| Physics test fails after IC change | Re-run `scripts/check-physics.sh` and check `experiments/nbody/phase38_class_validation/reference/` if CLASS comparison is involved. |

---

## 10. Where to look for things

| I need… | Look in… |
|---------|----------|
| Particle struct, Vec3, cosmology | `crates/gadget-ng-core/src/` |
| TOML config definition | `crates/gadget-ng-core/src/config.rs` |
| Gravity solvers (BH, PM, TreePM) | `crates/gadget-ng-tree/`, `gadget-ng-pm/`, `gadget-ng-treepm/` |
| SPH equations | `crates/gadget-ng-sph/src/` |
| MHD equations | `crates/gadget-ng-mhd/src/` |
| Analysis (FoF, P(k), HMF) | `crates/gadget-ng-analysis/src/` |
| CLI subcommands | `crates/gadget-ng-cli/src/main.rs`, `engine.rs` |
| Physics validation tests | `crates/gadget-ng-physics/tests/` |
| Report for Phase NNN | `docs/reports/2026-04-phaseNNN-*.md` or `2026-05-phaseNNN-*.md` |
| How to validate against GADGET-4 | `docs/runbooks/validation-vs-gadget4-reference.md` |
| JOSS paper | `docs/paper/paper.md` |

---

> **Last updated:** 2026-05-06
> **Maintainer:** cristian (project owner)
> **Copilots:** Read this file first. When unsure, run `bash scripts/check-physics.sh` before proposing changes.
