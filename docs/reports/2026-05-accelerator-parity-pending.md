# Accelerator parity pending work

Date: 2026-05-13

This document tracks the remaining work after Phase 200 closed the immediate
AVX-512 Tree LET and CUDA MHD/Tree smoke-surface gaps, and after Phase 201 split
Rayon from explicit SIMD feature axes.

## Current backend matrix

| Area | CPU serial | CPU Rayon | SIMD without Rayon | CUDA |
| --- | --- | --- | --- | --- |
| Direct gravity | Complete | Complete | AVX2 + AVX-512 | Complete |
| PM CIC | Complete | Complete | AVX2 + AVX-512 | Complete |
| PM FFT/Poisson | Complete | Partial Rayon around PM path | AVX2 + AVX-512 spectral kernel | Complete |
| Barnes-Hut local tree | Complete | Complete | AVX2 partial | Monopole smoke/parity surface |
| Tree LET / RMN SoA | Complete | Complete | AVX2 + AVX-512 | No full LET traversal |
| TreePM hybrid | Complete | Complete | Partial CPU SIMD | Partial GPU hybrid |
| SPH density/forces/Gadget-2 | Complete | Complete | Batch/tiling kernels | Smoke/parity kernels |
| SPH cooling/dust/H2 | Complete | Complete | No dedicated AVX | Smoke/parity kernels |
| MHD induction/forces/cleaning | Complete | Complete | No dedicated AVX | Smoke/parity kernels |
| MHD anisotropic/Braginskii/reconnection/CR/dynamo | Complete | Complete | No dedicated AVX | Smoke/parity kernels |
| RT diagnostics/photoheating | Complete | Complete | Partial explicit SIMD | Smoke/parity kernels |
| RT full M1 advection | Complete | Partial | Partial final-update vectorization | Pending |
| Analysis observables | Complete | Complete | No dedicated AVX | Pending |
| SIDM | Complete | Partial | No dedicated AVX | Smoke/parity kernel |

## Remaining implementation backlog

| ID | Pending item | Why it remains | Acceptance criteria | Status |
| --- | --- | --- | --- | --- |
| AP-01 | Split `simd` and `rayon` Cargo features | Closed in Phase 201: Rayon paths now use `feature = "rayon"` and explicit SIMD uses `feature = "simd"`. | `cargo check -p gadget-ng-cli --features simd`, `--features rayon`, and `--features simd,rayon` pass. | Complete |
| AP-02 | CUDA persistent device buffers | Current CUDA parity kernels allocate/copy/free per call, good for smoke validation but not production throughput. | Solver objects retain device buffers across steps for SPH/MHD/RT/Tree surfaces and expose resize/reuse semantics. | Pending |
| AP-03 | Hardware validation for new MHD/Tree CUDA kernels | CI stub path passes with `CUDA_SKIP=1`; real GPU parity has not been rerun after Phase 200 Fase 9. | `CUDA_ARCH=<sm> cargo test -p gadget-ng-cuda -- --ignored --nocapture` passes and tolerances are recorded. | Pending |
| AP-04 | Runtime wiring for new CUDA MHD/Tree kernels | Wrappers exist, but only mature CUDA paths should replace production CPU paths automatically. | `gadget-ng-cli` routes opt-in MHD/Tree CUDA kernels behind config after AP-03 validates parity. | Pending |
| AP-05 | Full RT M1 advection CUDA kernel | Existing CUDA covers diagnostics/photoheating, not the full stencil-safe M1 update. | CUDA M1 update matches CPU M1 within f32 tolerances on fixed fields and boundary cases. | Pending |
| AP-06 | CUDA analysis kernels | Spin, luminosity, SED and related analysis paths remain CPU/Rayon only. | Analysis CUDA kernels exist only where benchmarks show material gain and match CPU outputs. | Pending |
| AP-07 | Full GPU LET/tree traversal | CUDA Tree currently provides a direct monopole parity surface, not a full remote LET traversal. | GPU traversal consumes compact tree/LET buffers and matches CPU Barnes-Hut/LET within tolerance. | Pending |
| AP-08 | Benchmarks beyond direct gravity | The direct CUDA-vs-SIMD benchmark exists; PM/SPH/MHD/RT/Tree need comparable benchmark targets. | Criterion/CSV benchmark groups cover each backend pair only for operations implemented on both sides. | Pending |

## Implementation policy

- Do not replace deterministic CPU serial paths with accelerator paths.
- Accelerator paths are opt-in and must degrade cleanly when unavailable.
- New CUDA code must keep `CUDA_SKIP=1 cargo check/test/clippy -p gadget-ng-cuda` clean.
- New x86 SIMD code must dispatch `avx512f -> avx2+fma -> scalar` when batching pays off.
- Hardware-only claims require the command, GPU model, `CUDA_ARCH`, and tolerance summary in this document or the Phase 200 roadmap.

## Next recommended phase

Phase 201 closed AP-01. The next recommended phase is AP-03: validate the new
MHD/Tree CUDA kernels on real NVIDIA hardware before wiring them into production
runtime paths.
