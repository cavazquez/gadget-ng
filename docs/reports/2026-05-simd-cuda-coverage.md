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
  forces, cooling, dust and H2 via CUDA solvers.
- MHD flux-freeze, mean gas density, magnetic-field statistics, induction,
  resistivity, magnetic forces, Dedner cleaning (`CudaMhdSolver` smoke surface;
  CPU `not(rayon)` + `simd`: density + pairwise div-B/∇ψ AVX2/AVX-512 inner batches
  + SIMD final ψ/B update), scalar diffusion, Braginskii viscosity, reconnection, CR
  streaming and dynamo smoke surfaces via
  `CudaMhdSolver`.
- RT field diagnostics/photoionization and gas photoheating via `CudaRtSolver`.
- Tree/SIDM smoke surfaces via `CudaTreeSolver`.

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
| Tree/Barnes-Hut/LET/SIDM | Yes, local walk + LET/RMN; SIDM AVX2/AVX-512 density and pair prefilter | Monopole/SIDM smoke surface | Partial CUDA |
| SPH density | Yes, Wendland AVX2/AVX-512 batch kernels | Yes, `CudaSphSolver` | Experimental parity |
| SPH forces/Gadget2 viscosity | Yes, Wendland AVX2/AVX-512 batch kernels | Yes, `CudaSphSolver` | Experimental parity |
| SPH cooling/dust/H2 | Cooling per-particle batches, dust growth/sputtering/radiation kick, and H2+dust shielding AVX2/AVX-512 | Yes, CUDA solvers | Experimental parity |
| MHD flux-freezing/statistics | Flux-freeze scaling + mean density AVX2/AVX-512; statistics AVX2/AVX-512 with real AVX-512 8-lane path | Yes, `CudaMhdSolver` | Experimental parity |
| MHD induction/forces/cleaning/transport | Induction / resistivity / magnetic forces: AVX2/AVX-512 pair batches.<br>Dedner (`not(rayon)` + `simd`): density + div-B/∇ψ pairwise AVX2/AVX-512 inner batches + SIMD final ψ/B update.<br>Dedner (`rayon` + `simd`, x86/x86_64): when AVX-512F or AVX2+FMA is detected, `dedner_cleaning_step_par_simd` — Rayon outer loop over `i` reusing the same per-particle SIMD kernels + SIMD final update; otherwise `dedner_cleaning_step_par` (outer `par_iter`, scalar inner `j` loop).<br>Other MHD transport: mixed coverage — see [MHD Dedner cleaning](#mhd-dedner-cleaning-cpu-detail). | Yes, smoke surfaces | Experimental parity |
| RT M1 reductions/photoheating | Yes, AVX2/AVX-512 diagnostics and photoheating arithmetic | Yes, `CudaRtSolver` | Experimental parity |
| RT M1 full advection substep | Yes, CPU Rayon advection/update + SIMD final update | No | CPU-only |
| RT chemistry rates/cooling | AVX2/AVX-512 particle photoionization-rate batches and cooling update | No | SIMD-only |
| RT chemistry stiff solver | AVX2/AVX-512 masked-lane dispatch; stiff chemistry scalar-per-lane; chunk/tail parity tests | No | SIMD-only (CUDA pending) |
| RT IGM temperature profile | AVX-512F 8-wide + AVX2+FMA 4-wide `ChemState`→T + SIMD density gate per lane; mean/variance/sort/percentiles scalar | No | SIMD-only |
| RT reionization state | AVX2/AVX-512 reductions | No | SIMD-only |
| RT 21cm | AVX2/AVX-512 mass/volume reductions and brightness field | No | SIMD-only |
| Analysis spin/luminosity/SED | AVX2/AVX-512 reductions for spin, luminosity and SED contributions | No | SIMD-only |

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
