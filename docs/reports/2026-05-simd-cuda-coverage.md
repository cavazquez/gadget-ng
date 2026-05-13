# SIMD/CUDA coverage and parity

Date: 2026-05-13

This report records the current accelerator coverage after the CUDA availability
work and the SIMD/Rayon expansion across CPU physics crates.

## Summary

There is not yet 1:1 parity between CUDA and SIMD/Rayon.

CUDA currently covers the two gravity paths with native CUDA kernels and has an
initial SPH O(N²) kernel set:

- Direct `O(N^2)` gravity.
- Particle-mesh gravity.
- SPH density, Balsara viscosity limiter, classical SPH forces, and Gadget-2
  SPH forces via `CudaSphSolver`.
- MHD flux-freeze, mean gas density, and magnetic-field statistics via
  `CudaMhdSolver`.
- RT field diagnostics/photoionization and gas photoheating via `CudaRtSolver`.

SIMD/Rayon covers those CPU-side equivalents plus a wider set of CPU physics
kernels in tree gravity, SPH, MHD, radiative transfer, and analysis.

Therefore, every CUDA-covered function has a CPU/SIMD or CPU/Rayon validation
path, but many SIMD/Rayon-covered functions do not yet have CUDA kernels.

## Coverage matrix

| Area | SIMD/Rayon CPU | CUDA | Parity status |
| --- | --- | --- | --- |
| Direct gravity | Yes, `SimdDirectGravity` | Yes, `CudaDirectGravity` | Real parity |
| PM gravity | Yes, CPU PM/Rayon path | Yes, `CudaPmSolver` | Real parity |
| Tree/Barnes-Hut/LET | Yes | No | SIMD-only |
| SPH density | Yes | Yes, `CudaSphSolver` | Experimental parity |
| SPH forces/Gadget2 viscosity | Yes | Yes, `CudaSphSolver` | Experimental parity |
| SPH cooling/dust/H2 | Yes | No | SIMD-only |
| MHD flux-freezing/statistics | Yes | Yes, `CudaMhdSolver` | Experimental parity |
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

1. Optimize SPH/MHD/RT CUDA with persistent device buffers after smoke validation.
2. Port the full RT M1 advection substep with a stencil-safe design.
3. Wire CUDA SPH/MHD/RT into runtime config once smoke tests pass on real hardware.
4. Keep tree/LET on CPU until there is a concrete GPU tree traversal design.
5. Extend benchmarks only where both accelerators implement the same operation.
