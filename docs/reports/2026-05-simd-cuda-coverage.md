# SIMD/CUDA coverage and parity

Date: 2026-05-13

This report records the current accelerator coverage after the CUDA availability
work, the SIMD/Rayon expansion across CPU physics crates, and the Phase 200
AVX-512/CUDA smoke-surface closure.

## Summary

There is not yet production 1:1 parity between CUDA and SIMD/Rayon.

CUDA currently covers the two gravity paths with native CUDA kernels and has
SPH/MHD/RT smoke/parity kernel sets:

- Direct `O(N^2)` gravity.
- Particle-mesh gravity.
- SPH density, Balsara viscosity limiter, classical SPH forces, Gadget-2 SPH
  forces, cooling, dust and H2 via CUDA solvers — **all with persistent device
  buffers** (`CudaPool`, AP-02 complete).
- MHD flux-freeze, mean gas density, magnetic-field statistics, induction,
  resistivity, magnetic forces, Dedner cleaning (`CudaMhdSolver` smoke surface;
  CPU `not(rayon)` + `simd`: density + pairwise div-B/∇ψ AVX2/AVX-512 inner batches
  + SIMD final ψ/B update), scalar diffusion, Braginskii viscosity, reconnection, CR
  streaming and dynamo smoke surfaces via
  `CudaMhdSolver` — **persistent buffers (AP-02)**; CR streaming
  (`mhd_cr_streaming_o2_kernel`) and CR backreaction
  (`mhd_cr_backreaction_kernel`) hardware-validated 2026-05-16.
- RT field diagnostics/photoionization and gas photoheating via `CudaRtSolver`
  — **persistent buffers (AP-02)**.
- Tree/SIDM smoke surfaces via `CudaTreeSolver` — **persistent buffers (AP-02)**.
- Cooling (AtomicHHe/Metal/MetalTabular/UV) via `CudaCoolingSolver` — **persistent buffers (AP-02)**.
- Dust (D/G accretion/sputtering + radiation pressure) via `CudaDustSolver` — **persistent buffers (AP-02)**.
- Molecular gas (HI→H2 with dust shielding) via `CudaMolecularSolver` — **persistent buffers (AP-02)**.

All CUDA solvers retain a `CudaPool` across simulation timesteps. Device buffers
are reused without `cudaMalloc`/`cudaFree` per call; the pool doubles capacity
automatically when particle count exceeds the current capacity.

SIMD/Rayon covers those CPU-side equivalents plus a wider set of CPU physics
kernels in tree gravity, SPH, MHD, radiative transfer, and analysis.

Therefore, every CUDA-covered function has a CPU/SIMD or CPU/Rayon validation
path, but some CPU/Rayon-covered functions still do not have production CUDA
implementations.

## Coverage matrix

| Area | SIMD/Rayon CPU | CUDA | Parity status |
| --- | --- | --- | --- |
| Direct gravity | Yes, `SimdDirectGravity` | Yes, `CudaDirectGravity` | Real parity |
| PM gravity | Yes, CPU PM/Rayon path | Yes, `CudaPmSolver` | Real parity |
| Tree/Barnes-Hut/LET/SIDM | Yes, local walk + LET/RMN; SIDM AVX2/AVX-512 density and pair prefilter | SIDM: `cuda_tree_sidm_scatter` wired. LET: `try_tree_walk_let` wired en flat LET path (`cuda_tree = true`, AP-18). Monopole walk: kernel existe, no wired (requiere árbol en device). | Wired opt-in (AP-18) |
| SPH density | Yes, Wendland AVX2/AVX-512 batch kernels | Yes, `try_sph_density_and_forces_core` sobre `&[Particle]` — pipeline densidad+Balsara+fuerzas clásicas; wired en `step_sph` (AP-18) | Wired opt-in (AP-18; HW validated 2026-05-16) |
| SPH forces/Gadget2 viscosity | Yes, Wendland AVX2/AVX-512 batch kernels | Yes, `CudaSphSolver` (`SphParticle` path) + nuevo `try_sph_density_and_forces_core` (`Particle` path, AP-18) | Wired opt-in (AP-18) |
| SPH cooling/dust/H2 | Cooling per-particle batches, dust growth/sputtering/radiation kick, and H2+dust shielding AVX2/AVX-512 | Yes, CUDA solvers | Experimental parity |
| MHD flux-freezing/statistics | Flux-freeze scaling + mean density AVX2/AVX-512; statistics AVX2/AVX-512 with real AVX-512 8-lane path | Yes, `CudaMhdSolver` | Experimental parity |
| MHD induction/forces/cleaning/transport | Induction / resistivity / magnetic forces: AVX2/AVX-512 pair batches.<br>Dedner (`not(rayon)` + `simd`): density + div-B/∇ψ pairwise AVX2/AVX-512 inner batches + SIMD final ψ/B update.<br>Dedner (`rayon` + `simd`, x86/x86_64): when AVX-512F or AVX2+FMA is detected, `dedner_cleaning_step_par_simd` — Rayon outer loop over `i` reusing the same per-particle SIMD kernels + SIMD final update; otherwise `dedner_cleaning_step_par` (outer `par_iter`, scalar inner `j` loop).<br>**Braginskii viscosity (`not(rayon)` + `simd`):** AVX2+FMA and AVX-512F per-particle batches without Rayon — `apply_braginskii_viscosity_avx2/avx512`.**Reconnection (`not(rayon)` + `simd`):** AVX2+FMA and AVX-512F per-particle batches without Rayon — `apply_magnetic_reconnection_avx2/avx512`.<br>Other MHD transport: see [MHD Dedner cleaning](#mhd-dedner-cleaning-cpu-detail). | Yes, smoke surfaces + Dedner hybrid CUDA (AP-17) | Experimental parity |
| RT M1 reductions/photoheating | Yes, AVX2/AVX-512 diagnostics and photoheating arithmetic | Yes, `CudaRtSolver` | Experimental parity |
| RT M1 full advection substep | Yes, CPU Rayon advection/update + SIMD final update | Yes, `CudaRtSolver::try_m1_advection` | Experimental parity |
| RT chemistry rates/cooling | AVX2/AVX-512 particle photoionization-rate batches and cooling update | Yes, `CudaRtSolver::try_chemistry_rates` + `try_apply_cooling` | Experimental parity (HW validated 2026-05-16) |
| RT chemistry stiff solver | AVX2/AVX-512 masked-lane dispatch; stiff chemistry scalar-per-lane; chunk/tail parity tests | Yes, `CudaRtSolver::try_apply_chemistry` — `rt_chemistry_stiff_kernel` f32 subcyclic implicit integrator | Experimental parity (HW validated 2026-05-16) |
| RT IGM temperature profile | AVX-512F 8-wide + AVX2+FMA 4-wide `ChemState`→T + SIMD density gate per lane; mean/variance/sort/percentiles scalar | Yes, `CudaRtSolver::try_igm_temp_profile` — `cuda_rt_igm_temp_full`: mean+sigma+compact temps; sort+percentiles in Rust host (AP-17) | Experimental parity (HW validated 2026-05-16; percentiles AP-17) |
| RT reionization state | AVX2/AVX-512 reductions | Yes, `CudaRtSolver::try_reionization_stats` | Experimental parity (HW validated 2026-05-16) |
| RT 21cm | AVX2/AVX-512 mass/volume reductions and brightness field | Yes, `CudaRtSolver::try_cm21_field` — wired in `step_reionization` | Experimental parity (HW validated 2026-05-16; wired AP-16) |
| MHD ambipolar diffusion | AVX2/AVX-512 B-field damping + ionization proxy + heating | Yes, `CudaMhdSolver::try_ambipolar_diffusion` — `mhd_ambipolar_kernel`; wired in `step_mhd` | Experimental parity (HW validated 2026-05-16; AP-16) |
| MHD two-fluid (e-i coupling) | AVX2/AVX-512 Coulomb coupling + T_e/T_i reduction | Yes, `CudaMhdSolver::try_electron_ion_coupling` — `mhd_two_fluid_kernel`; wired in `step_sph` | Experimental parity (HW validated 2026-05-16; AP-16) |
| MHD anisotropic conduction / CR diffusion | AVX2/AVX-512 pair accumulation | Yes, `mhd_anisotropic_pair_kernel` O(N²) — `try_anisotropic_conduction` + `try_cr_diffusion_anisotropic`; replaces mean-field approx (AP-17) | Experimental parity (AP-17) |
| Analysis spin/luminosity/SED/X-ray | AVX2/AVX-512 reductions for spin, luminosity, SED, and X-ray contributions | Yes, `CudaAnalysisSolver::try_galaxy_luminosity` + `try_halo_spin` + `try_xray_luminosity`; wired in `analyze_cmd.rs` via `cuda_analysis` flag (AP-17) | Experimental parity (AP-16/AP-17) |

### MHD Dedner cleaning (CPU detail)

- **`not(feature = "rayon")` + `feature = "simd"` (x86/x86_64):** density and the
  O(N²) pair accumulation for `div_B` and `∇ψ` use AVX-512F when available, else
  AVX2+FMA, with the same Wendland kernel structure as MHD induction batching; the
  final `ψ`/`B` update is also SIMD-batched. Batches that include self-index `i`,
  non-gas neighbors, or tail lengths fall back to the scalar pair kernel for
  parity with the reference double loop.
- **`feature = "rayon"` + `feature = "simd"` (x86/x86_64):** when AVX-512F or
  AVX2+FMA is detected at runtime, `dedner_cleaning_step` calls
  `dedner_cleaning_step_par_simd`: `par_iter_mut` over particles for `div_B` /
  `∇ψ` accumulation, each task dispatching the same AVX2/AVX-512 per-particle
  kernels as the serial SIMD path, then a SIMD-batched final `ψ`/`B` update.
  Otherwise (missing ISA or non-x86), `dedner_cleaning_step_par` applies: outer
  `par_iter` over gas particles with a scalar inner `j` loop.
- **`feature = "rayon"` without usable SIMD ISA on the build target:** same as
  the scalar-inner path above (`dedner_cleaning_step_par`).

## Validated commands

The SIMD/Rayon side was validated with:

```bash
cargo test -p gadget-ng-sph --features simd
cargo clippy -p gadget-ng-sph --features simd --all-targets -- -D warnings
cargo test -p gadget-ng-mhd --features simd
cargo test -p gadget-ng-mhd --features "rayon,simd"
cargo clippy -p gadget-ng-mhd --features simd --all-targets -- -D warnings
cargo clippy -p gadget-ng-mhd --features "rayon,simd" -- -D warnings
cargo test -p gadget-ng-rt --features simd
cargo clippy -p gadget-ng-rt --features simd --all-targets -- -D warnings
cargo test -p gadget-ng-analysis --features simd
cargo clippy -p gadget-ng-analysis --features simd --all-targets -- -D warnings
cargo clippy -p gadget-ng-cli --features simd,rayon,pm-rayon -- -D warnings
cargo fmt --all --check
cargo test -p gadget-ng-pm --features rayon
cargo test -p gadget-ng-rt --features rayon
cargo test -p gadget-ng-tree --features rayon
cargo test -p gadget-ng-tree --features simd
cargo test -p gadget-ng-treepm --features simd
cargo test -p gadget-ng-rt --features simd
cargo clippy -p gadget-ng-cli --features simd -- -D warnings
```

The CUDA side has been validated on NVIDIA GTX 1060 (`sm_61`) with:

```bash
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_direct_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_sph_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_mhd_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_rt_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo build -p gadget-ng-cli --features cuda
CUDA_ARCH=sm_61 cargo clippy -p gadget-ng-cli --features cuda -- -D warnings
```

RT chemistry, reionization, and MHD CR kernels were hardware-validated on
2026-05-16 (37 tests, 0 failures). Full record:
`docs/reports/2026-05-cuda-cr-hw-validation.md`.

**Known issue — `grad_w_approx` discontinuity at q = 1.0** (MHD CR backreaction):
`grad_w_approx` in `crates/gadget-ng-mhd/src/streaming.rs` has inconsistent
factors of `h` at the `q = 1.0` branch boundary, creating a discontinuity.
CPU (f64) and GPU (f32) take different branches when `q` is very close to 1.0,
producing relative errors up to ~70% per particle even with correct physics.
The CUDA smoke test therefore validates physical invariants (Newton's 3rd Law,
finiteness, boundedness) rather than direct numerical comparison. Fixing
`grad_w_approx` itself requires coordinated regression tests and is tracked
as a future clean-up item.

After Phase 200 Fases 8-9, the CUDA stub path was validated with:

```bash
CUDA_SKIP=1 cargo check -p gadget-ng-cuda
CUDA_SKIP=1 cargo test -p gadget-ng-cuda
CUDA_SKIP=1 cargo clippy -p gadget-ng-cuda -- -D warnings
cargo test -p gadget-ng-tree --lib
cargo clippy -p gadget-ng-tree -- -D warnings
cargo fmt --all --check
```

## Minimum CUDA version and hardware compatibility

The minimum CUDA architecture is **sm_60** (Pascal, GTX 10xx, 2017+). This is
set as the default in `build.rs` when auto-detection via `nvidia-smi` fails.

| CUDA Architecture | GPUs | Minimum CUDA Toolkit |
|---|---|---|
| sm_60 | Pascal (GTX 10xx, Quadro P) | CUDA 8.0 |
| sm_61 | Pascal (GTX 1060 validated) | CUDA 8.0 |
| sm_70 | Volta (V100, Titan V) | CUDA 9.0 |
| sm_75 | Turing (RTX 20xx, GTX 16xx) | CUDA 10.0 |
| sm_80 | Ampere (A100, RTX 30xx) | CUDA 11.0 |
| sm_86 | Ampere (RTX 30xx mobile) | CUDA 11.1 |
| sm_89 | Ada Lovelace (RTX 40xx) | CUDA 11.8 |
| sm_90 | Hopper (H100) | CUDA 12.0 |

The build does **not** use any CUDA 11+ APIs (no `cudaMemPool`, no cooperative
groups, no graph APIs), ensuring maximum backward compatibility with CUDA 8.0+
toolkits. All kernels compile with `-std=c++14` and use only basic CUDA Runtime
(`/cuda_runtime.h`) and cuFFT APIs.

## Persistent device buffers (AP-02)

As of AP-02, all CUDA solvers use `CudaPool` for persistent device memory:

- **Stateless kernels** (SPH, MHD, Tree, RT, Cooling, Dust, Molecular): Each
  solver struct now holds a `CudaPool` that manages device memory slots. On each
  call, `ensure_capacity(n)` guarantees the pool can hold `n` particles, and
  `reset()` marks all slots for reuse. Uploads use `upload_f32`/`upload_u8`
  (host→device copies into pool slots), and outputs use `alloc_f32` (zeroed
  device buffers) + `download_f32` (device→host copies). No `cudaMalloc` or
  `cudaFree` calls happen per kernel invocation when `n` stays constant.

- **Handle-based solvers** (PM, Direct): `CudaDirectGravity` now retains its
  CUDA handle across calls instead of creating/destroying per invocation, and
  also uses `CudaPool` for device staging buffers. `CudaPmSolver` already
  retained its handle and only needed minor cleanup.

The pool doubles capacity automatically when `n` exceeds the current capacity,
matching the standard amortized-growth pattern used by `Vec`.

The CUDA side has been validated on NVIDIA GTX 1060 (`sm_61`) with:

```bash
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_direct_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_sph_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_mhd_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo test -p gadget-ng-cuda --test cuda_rt_smoke -- --ignored --nocapture
CUDA_ARCH=sm_61 cargo build -p gadget-ng-cli --features cuda
CUDA_ARCH=sm_61 cargo clippy -p gadget-ng-cli --features cuda -- -D warnings
```

After Phase 200 Fases 8-9, the CUDA stub path was validated with:

```bash
CUDA_SKIP=1 cargo check -p gadget-ng-cuda
CUDA_SKIP=1 cargo test -p gadget-ng-cuda
CUDA_SKIP=1 cargo clippy -p gadget-ng-cuda -- -D warnings
cargo test -p gadget-ng-tree --lib
cargo clippy -p gadget-ng-tree -- -D warnings
cargo fmt --all --check
```

## Benchmark

For the current real parity area, run:

```bash
CUDA_ARCH=sm_61 bash scripts/bench_cuda_vs_simd.sh
```

The script runs the `gadget-ng-cuda` Criterion benchmark and writes:

- `runs/benchmarks/cuda-vs-simd/cuda_vs_simd_direct.csv`
- `runs/benchmarks/cuda-vs-simd/cuda_vs_simd_direct.png`

The first benchmark target compares direct gravity across CPU serial,
CPU SIMD/Rayon, and CUDA. PM parity can be added as a second benchmark target
once we decide the grid sizes and whether to include FFT setup time or only
steady-state solves.

## Roadmap to 1:1 accelerator parity

1. Validate Phase 200 MHD/Tree CUDA kernels on real NVIDIA hardware.
2. Optimize SPH/MHD/RT CUDA with persistent device buffers after smoke validation.
3. Port the full RT M1 advection substep to CUDA with a stencil-safe design.
4. Wire new CUDA MHD/Tree kernels into runtime config only after hardware parity.
5. Keep full LET traversal on CPU until there is a concrete compact-tree GPU design.
6. Extend benchmarks only where both accelerators implement the same operation.

Detailed pending work is tracked in
[`2026-05-accelerator-parity-pending.md`](2026-05-accelerator-parity-pending.md).
