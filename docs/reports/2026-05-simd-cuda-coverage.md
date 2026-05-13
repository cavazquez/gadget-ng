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
  resistivity, magnetic forces, Dedner cleaning, scalar diffusion, Braginskii
  viscosity, reconnection, CR streaming and dynamo smoke surfaces via
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
| Tree/Barnes-Hut/LET | Yes | Monopole/SIDM smoke surface | Partial CUDA |
| SPH density | Yes | Yes, `CudaSphSolver` | Experimental parity |
| SPH forces/Gadget2 viscosity | Yes | Yes, `CudaSphSolver` | Experimental parity |
| SPH cooling/dust/H2 | Yes | Yes, CUDA solvers | Experimental parity |
| MHD flux-freezing/statistics | Yes | Yes, `CudaMhdSolver` | Experimental parity |
| MHD induction/forces/cleaning/transport | Yes | Yes, smoke surfaces | Experimental parity |
| RT M1 reductions/photoheating | Yes | Yes, `CudaRtSolver` | Experimental parity |
| RT M1 full advection substep | Partial CPU SIMD final update | No | CPU-only |
| Analysis spin/luminosity/SED | Yes | No | SIMD-only |

## Validated commands

The SIMD/Rayon side was validated with:

```bash
cargo test -p gadget-ng-sph --features simd
cargo clippy -p gadget-ng-sph --features simd --all-targets -- -D warnings
cargo test -p gadget-ng-mhd --features simd
cargo clippy -p gadget-ng-mhd --features simd --all-targets -- -D warnings
cargo test -p gadget-ng-rt --features simd
cargo clippy -p gadget-ng-rt --features simd --all-targets -- -D warnings
cargo test -p gadget-ng-analysis --features simd
cargo clippy -p gadget-ng-analysis --features simd --all-targets -- -D warnings
cargo clippy -p gadget-ng-cli --features simd,pm-rayon -- -D warnings
cargo fmt --all --check
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
3. Port the full RT M1 advection substep with a stencil-safe design.
4. Wire new CUDA MHD/Tree kernels into runtime config only after hardware parity.
5. Keep full LET traversal on CPU until there is a concrete compact-tree GPU design.
6. Extend benchmarks only where both accelerators implement the same operation.

Detailed pending work is tracked in
[`2026-05-accelerator-parity-pending.md`](2026-05-accelerator-parity-pending.md).
