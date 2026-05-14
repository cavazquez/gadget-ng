# Accelerator parity pending work

Date: 2026-05-13

This document tracks the remaining work after Phase 200 closed the immediate
AVX-512 Tree LET and CUDA MHD/Tree smoke-surface gaps, after Phase 201 split
Rayon from explicit SIMD feature axes, after Phase 202 closed the remaining
CPU Rayon coverage gaps, after Phase 203 closed the prioritized SIMD-without-Rayon gaps,
and after the follow-up RT/SPH/analysis SIMD passes split green vectorized rows from
the remaining scalar/stiff solver rows.

## Current backend matrix

| Area | CPU serial | CPU Rayon | SIMD without Rayon | CUDA |
| --- | --- | --- | --- | --- |
| Direct gravity | Complete | Complete | AVX2 + AVX-512 | Complete |
| PM CIC | Complete | Complete | AVX2 + AVX-512 | Complete |
| PM FFT/Poisson | Complete | Complete k-space + PM path | AVX2 + AVX-512 spectral kernel | Complete |
| Barnes-Hut local tree | Complete | Complete | AVX2 + AVX-512 local walk | Monopole smoke/parity surface |
| Tree LET / RMN SoA | Complete | Complete | AVX2 + AVX-512 | No full LET traversal |
| TreePM hybrid | Complete | Complete | AVX2 + AVX-512 short-range CPU kernel | Partial GPU hybrid |
| SPH density/forces/Gadget-2 | Complete | Complete | **Batch kernels in all paths**; Wendland AVX2 + AVX-512 | Smoke/parity kernels |
| SPH cooling/dust/H2 | Complete | Complete | **Cooling per-particle AVX2+AVX-512 (all modes)**; dust AVX2+AVX-512; H2+dust shielding AVX2+AVX-512 | Smoke/parity kernels |
| SPH viscosity (Balsara) | Complete | Complete | **grad_w_batch in all paths** | Smoke/parity kernels |
| MHD induction/forces/cleaning | Complete | Complete | **Cleaning final-update AVX2+AVX-512** (pairwise loop scalar) | Smoke/parity kernels |
| MHD flux-freeze | Complete | Complete | **AVX2 + AVX-512 (B-field scaling + mean density)** | Smoke/parity kernels |
| MHD two-fluid (electron-ion coupling) | Complete | Complete | **AVX2+AVX-512 e-i coupling + T_e/T_i reduction** | Smoke/parity kernels |
| MHD anisotropic/Braginskii/reconnection/CR/dynamo | Complete | Complete | **Dynamo AVX2+AVX-512 (B-field update + energy ratio)**; **Ambipolar diffusion AVX2+AVX-512** | Smoke/parity kernels |
| MHD stats b_field_stats | Complete | Complete | **AVX-512 real 8-lane (was stub)** | Smoke/parity kernels |
| RT diagnostics/photoheating | Complete | Complete | AVX2 + AVX-512 diagnostics/photoheating | Smoke/parity kernels |
| RT full M1 advection | Complete | Complete advection + update | AVX2 + AVX-512 final update | Pending |
| RT chemistry rates/cooling | Complete | Complete | AVX2 + AVX-512 particle photoionization rates and cooling update | Pending |
| RT chemistry stiff solver | Complete | Complete | Scalar implicit subcycling | Pending |
| RT reionization state / 21cm | Complete | Complete | AVX2 + AVX-512 reductions and brightness field | Pending |
| Analysis observables | Complete | Complete | AVX2 + AVX-512 spin/luminosity/SED reductions | Pending |
| SIDM | Complete | Complete density + pair evaluation | AVX2 + AVX-512 density and pair prefilter | Smoke/parity kernel |

## Remaining implementation backlog

| ID | Pending item | Why it remains | Acceptance criteria | Status |
| --- | --- | --- | --- | --- |
| AP-01 | Split `simd` and `rayon` Cargo features | Closed in Phase 201: Rayon paths now use `feature = "rayon"` and explicit SIMD uses `feature = "simd"`. | `cargo check -p gadget-ng-cli --features simd`, `--features rayon`, and `--features simd,rayon` pass. | Complete |
| AP-01b | Close remaining CPU Rayon coverage gaps | Closed in Phase 202: PM FFT/Poisson k-space, RT M1 advection, and SIDM density/pair evaluation now dispatch Rayon paths without relying on SIMD. | `cargo test -p gadget-ng-pm --features rayon`, `cargo test -p gadget-ng-rt --features rayon`, and `cargo test -p gadget-ng-tree --features rayon` pass. | Complete |
| AP-01c | Close prioritized SIMD-without-Rayon gaps | Closed in Phase 203: Barnes-Hut local walk, TreePM short-range, and RT M1 final update now expose `simd` paths without requiring Rayon. | `cargo test -p gadget-ng-tree --features simd`, `cargo test -p gadget-ng-treepm --features simd`, `cargo test -p gadget-ng-rt --features simd`, and `cargo clippy -p gadget-ng-cli --features simd -- -D warnings` pass. | Complete |
| AP-02 | CUDA persistent device buffers | Current CUDA parity kernels allocate/copy/free per call, good for smoke validation but not production throughput. | Solver objects retain device buffers across steps for SPH/MHD/RT/Tree surfaces and expose resize/reuse semantics. | Pending |
| AP-03 | Hardware validation for new MHD/Tree CUDA kernels | CI stub path passes with `CUDA_SKIP=1`; real GPU parity has not been rerun after Phase 200 Fase 9. | `CUDA_ARCH=<sm> cargo test -p gadget-ng-cuda -- --ignored --nocapture` passes and tolerances are recorded. | Pending |
| AP-04 | Runtime wiring for new CUDA MHD/Tree kernels | Wrappers exist, but only mature CUDA paths should replace production CPU paths automatically. | `gadget-ng-cli` routes opt-in MHD/Tree CUDA kernels behind config after AP-03 validates parity. | Pending |
| AP-05 | Full RT M1 advection CUDA kernel | Existing CUDA covers diagnostics/photoheating, not the full stencil-safe M1 update. | CUDA M1 update matches CPU M1 within f32 tolerances on fixed fields and boundary cases. | Pending |
| AP-06 | CUDA analysis kernels | Spin, luminosity, SED and related analysis paths remain CPU/Rayon only. | Analysis CUDA kernels exist only where benchmarks show material gain and match CPU outputs. | Pending |
| AP-07 | Full GPU LET/tree traversal | CUDA Tree currently provides a direct monopole parity surface, not a full remote LET traversal. | GPU traversal consumes compact tree/LET buffers and matches CPU Barnes-Hut/LET within tolerance. | Pending |
| AP-08 | Benchmarks beyond direct gravity | The direct CUDA-vs-SIMD benchmark exists; PM/SPH/MHD/RT/Tree need comparable benchmark targets. | Criterion/CSV benchmark groups cover each backend pair only for operations implemented on both sides. | Pending |
| AP-09 | RT chemistry stiff solver SIMD design | The explicit SIMD path covers particle photoionization rates and the cooling update. The implicit chemistry solver still subcycles adaptively per particle, with divergent active lanes, molecular/deuterium branches, clamps, and convergence exits. | A masked-lane AVX2/AVX-512 solver matches scalar `solve_chemistry_implicit` within stated tolerances across neutral, ionized, molecular, HD, high-UV, and long-recombination cases. | Pending |
| AP-10 | SPH cooling SIMD without Rayon | Cooling was the only per-particle O(N) SPH loop with no SIMD path. | `apply_cooling_with_redshift` and `apply_cooling_mhd_with_redshift` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; all cooling modes (AtomicHHe, MetalCooling, MetalTabular, UvBackground) handled via scalar-per-lane fallback inside batch. | Complete |
| AP-11 | MHD flux-freeze / mean_density SIMD without Rayon | No SIMD path existed for B-field scaling or density reduction. | `apply_flux_freeze` and `mean_gas_density` dispatch AVX2/AVX-512 with `feature = "simd"`; `f64::powf(2/3)` computed scalar-per-lane; all tests pass. | Complete |
| AP-12 | MHD b_field_stats AVX-512 real 8-lane | AVX-512 path delegated to AVX-512 4-lane stub, not a true 8-lane implementation. | `b_field_stats_avx512` processes 8 particles per iteration with `_mm512_*` intrinsics and k-mask gas filtering; passes existing tests. | Complete |
| AP-13 | MHD dynamo SIMD without Rayon | Dynamo B-field update and magnetic energy ratio had no SIMD path. | `apply_turbulent_dynamo` and `magnetic_energy_ratio` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; scalar-per-lane fallback for `dynamo_growth_rate` branches and `exp()`; all tests pass. | Complete |
| AP-14 | MHD ambipolar diffusion SIMD without Rayon | Ambipolar diffusion had no SIMD path. | `apply_ambipolar_diffusion` dispatches AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; vectorized ionization fraction proxy, B-field damping, and energy deposition; scalar-per-lane for `exp()`; all tests pass. | Complete |
| AP-15 | MHD two-fluid SIMD without Rayon | Electron-ion coupling and T_e/T_i ratio had no SIMD path. | `apply_electron_ion_coupling` and `mean_te_over_ti` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; vectorized Coulomb coupling rate and T_e/T_i reduction; scalar-per-lane for `exp()`; all tests pass. | Complete |

## RT chemistry remaining scalar work

The remaining non-green RT chemistry component is the stiff implicit solver,
`solve_chemistry_implicit`. It is intentionally kept scalar for now because each
particle may take a different number of chemistry substeps and may leave the loop
through the early-convergence condition. A production SIMD implementation should
therefore be a masked-lane solver, not a simple four-particle loop.

The main pieces still required are:

- active-lane masks for particles whose `t_elapsed < dt`;
- per-lane `dt_chem` and convergence decisions;
- vectorized updates for H/He, molecular H2 intermediates, D/HD exchange, and
  charge-normalization bookkeeping;
- scalar tail handling with exact fallback;
- parity tests against scalar for neutral gas, fully ionized recombination,
  high-UV ionization, trace H2 formation, trace HD formation, and conservation
  of H/He/D nuclei;
- benchmarks proving the masked solver pays for its extra control flow.

## Implementation policy

- Do not replace deterministic CPU serial paths with accelerator paths.
- Accelerator paths are opt-in and must degrade cleanly when unavailable.
- New CUDA code must keep `CUDA_SKIP=1 cargo check/test/clippy -p gadget-ng-cuda` clean.
- New x86 SIMD code must dispatch `avx512f -> avx2+fma -> scalar` when batching pays off.
- Hardware-only claims require the command, GPU model, `CUDA_ARCH`, and tolerance summary in this document or the Phase 200 roadmap.

## Next recommended phase

Phase 203 closed AP-01c; AP-10/11/12 close additional SIMD-without-Rayon gaps.
The next recommended phase is AP-03: validate the new MHD/Tree CUDA kernels on real
NVIDIA hardware before wiring them into production runtime paths. For SIMD,
AP-13 (dynamo), AP-14 (ambipolar diffusion), and AP-15 (two-fluid) close additional
SIMD-without-Rayon gaps. RT IGM temperature remains scalar-optimal (ChemState AoS
layout prevents profitable vectorization). The next targets are SPH metal cooling
(table lookup) and MHD CR streaming.
