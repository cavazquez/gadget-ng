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
| MHD induction/forces/cleaning | Complete | Complete | **Induction pair accumulation AVX2+AVX-512**; **Dedner**: serial SIMD pairwise + final batch; **Dedner `rayon`+`simd` on x86** (`par_simd`, per-particle SIMD + final batch) | Smoke/parity kernels |
| MHD flux-freeze | Complete | Complete | **AVX2 + AVX-512 (B-field scaling + mean density)** | Smoke/parity kernels |
| MHD two-fluid (electron-ion coupling) | Complete | Complete | **AVX2+AVX-512 e-i coupling + T_e/T_i reduction** | Smoke/parity kernels |
| MHD anisotropic/Braginskii/reconnection/CR/dynamo | Complete | Complete | **Dynamo AVX2+AVX-512 (B-field update + energy ratio)**; **Ambipolar diffusion AVX2+AVX-512** | **Complete** — kernels existentes + `mhd_cr_streaming_o2_kernel` + `mhd_cr_backreaction_kernel`; `try_cr_streaming` + `try_cr_backreaction` |
| MHD stats b_field_stats | Complete | Complete | **AVX-512 real 8-lane (was stub)** | Smoke/parity kernels |
| RT diagnostics/photoheating | Complete | Complete | AVX2 + AVX-512 diagnostics/photoheating | Smoke/parity kernels |
| RT full M1 advection | Complete | Complete advection + update | AVX2 + AVX-512 final update | **Complete** — `m1_substep_kernel` |
| RT chemistry rates/cooling | Complete | Complete | AVX2 + AVX-512 particle photoionization rates and cooling update | **Complete** — `rt_chemistry_rates_kernel` + `rt_cooling_apply_kernel`; `try_chemistry_rates` + `try_apply_cooling` |
| RT chemistry stiff solver | Complete | Complete | AVX2 + AVX-512 masked-lane dispatch; stiff chemistry scalar-per-lane | **Complete** — `rt_chemistry_stiff_kernel` (subciclo implícito f32/partícula); `try_apply_chemistry` |
| RT reionization state / 21cm | Complete | Complete | AVX2 + AVX-512 reductions and brightness field | **Complete** — `rt_reionization_stats_kernel` + `rt_cm21_field_kernel`; `try_reionization_stats` + `try_cm21_field` |
| Analysis observables | Complete | Complete | AVX2 + AVX-512 spin/luminosity/SED reductions | **Complete** — `halo_spin_kernel` + `galaxy_luminosity_kernel` + `xray_luminosity_kernel`; `CudaAnalysisSolver` |
| SIDM | Complete | Complete density + pair evaluation | AVX2 + AVX-512 density and pair prefilter | Smoke/parity kernel |

## Remaining implementation backlog

| ID | Pending item | Why it remains | Acceptance criteria | Status |
| --- | --- | --- | --- | --- |
| AP-01 | Split `simd` and `rayon` Cargo features | Closed in Phase 201: Rayon paths now use `feature = "rayon"` and explicit SIMD uses `feature = "simd"`. | `cargo check -p gadget-ng-cli --features simd`, `--features rayon`, and `--features simd,rayon` pass. | Complete |
| AP-01b | Close remaining CPU Rayon coverage gaps | Closed in Phase 202: PM FFT/Poisson k-space, RT M1 advection, and SIDM density/pair evaluation now dispatch Rayon paths without relying on SIMD. | `cargo test -p gadget-ng-pm --features rayon`, `cargo test -p gadget-ng-rt --features rayon`, and `cargo test -p gadget-ng-tree --features rayon` pass. | Complete |
| AP-01c | Close prioritized SIMD-without-Rayon gaps | Closed in Phase 203: Barnes-Hut local walk, TreePM short-range, and RT M1 final update now expose `simd` paths without requiring Rayon. | `cargo test -p gadget-ng-tree --features simd`, `cargo test -p gadget-ng-treepm --features simd`, `cargo test -p gadget-ng-rt --features simd`, and `cargo clippy -p gadget-ng-cli --features simd -- -D warnings` pass. | Complete |
| AP-02 | CUDA persistent device buffers | Solver objects retain device buffers across steps for SPH/MHD/RT/Tree surfaces and expose resize/reuse semantics. `CudaPool` manages slots with automatic doubling; `CudaDirectGravity` retains CUDA handle across calls. | Complete |
| AP-03 | Hardware validation for new MHD/Tree CUDA kernels | CI stub path passes with `CUDA_SKIP=1`; real GPU parity has not been rerun after Phase 200 Fase 9. | `CUDA_ARCH=<sm> cargo test -p gadget-ng-cuda -- --ignored --nocapture` passes and tolerances are recorded. | **Complete** — GTX 1060 sm_61 CUDA 12.4; see `docs/reports/2026-05-ap03-cuda-hardware-validation.md` |
| AP-04 | Runtime wiring for new CUDA MHD/Tree kernels | Wrappers exist, but only mature CUDA paths should replace production CPU paths automatically. | `gadget-ng-cli` routes opt-in MHD/Tree CUDA kernels behind config after AP-03 validates parity. | **Complete** — Tree SIDM CUDA wiring added behind `[accelerators] cuda_tree = true`; SPH density/forces wiring deferred (needs `Particle ↔ SphParticle` adapter refactor, tracked as follow-up); cooling/dust/H2/MHD/RT already wired in previous sessions |
| AP-05 | Full RT M1 advection CUDA kernel | Existing CUDA covers diagnostics/photoheating, not the full stencil-safe M1 update. | CUDA M1 update matches CPU M1 within f32 tolerances on fixed fields and boundary cases. | **Complete** — Kernel `m1_substep_kernel` implementado en `rt_kernels.cu`; `CudaRtSolver::try_m1_advection` en Rust; test `cuda_rt_m1_advection_matches_cpu` pasa en GTX 1060 sm_61 (tol 1e-3); wiring CLI en `step_rt` bajo `[accelerators] cuda_rt = true` |
| AP-06 | CUDA analysis kernels | Spin, luminosity, SED and related analysis paths remain CPU/Rayon only. | Analysis CUDA kernels exist only where benchmarks show material gain and match CPU outputs. | **Complete** — `analysis_kernels.cu` con kernels de reducción paralela para halo spin (momento angular), luminosidad galáctica (L_total, B-V, g-r) y luminosidad X (bremsstrahlung); `CudaAnalysisSolver` en Rust; tests pasan en GTX 1060 sm_61 (tol 1e-3/1e-5) |
| AP-07 | Full GPU LET/tree traversal | CUDA Tree currently provides a direct monopole parity surface, not a full remote LET traversal. | GPU traversal consumes compact tree/LET buffers and matches CPU Barnes-Hut/LET within tolerance. | **Complete** — `tree_let_mono_quad_oct_kernel` en `tree_kernels.cu`; mono+cuadrupolo+octupolo para todos los nodos LET pre-seleccionados; `try_tree_walk_let()` en Rust; test pasa en GTX 1060 sm_61 con max_rel=9.3e-7 (tol=1e-3) |
| AP-08 | Benchmarks beyond direct gravity | The direct CUDA-vs-SIMD benchmark exists; PM/SPH/MHD/RT/Tree need comparable benchmark targets. | Criterion/CSV benchmark groups cover each backend pair only for operations implemented on both sides. | **Complete** — `cuda_vs_simd.rs` extendido con grupos PM, SPH, MHD, RT (M1), Tree LET; benchmarks ejecutados en GTX 1060 sm_61; speedups documentados en `2026-05-ap08-cuda-benchmarks.md` (Direct: 50×@N=2048, Tree LET: 378×@4096 nodos, PM: 6×@grid=64) |
| AP-09 | RT chemistry stiff solver SIMD design | Closed for CPU SIMD: the implicit chemistry path now batches lanes with AVX2/AVX-512 dispatch while keeping stiff, branch-heavy chemistry scalar-per-lane inside the active-lane loop. | Masked-lane dispatch matches scalar `solve_chemistry_implicit` within 1e-12 across neutral, ionized, He-ionized, molecular, H-/H2+, D/HD, high-UV, recombination, chunk, tail, and end-to-end `apply_chemistry` cases. | Complete |
| AP-10 | SPH cooling SIMD without Rayon | Cooling was the only per-particle O(N) SPH loop with no SIMD path. | `apply_cooling_with_redshift` and `apply_cooling_mhd_with_redshift` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; all cooling modes (AtomicHHe, MetalCooling, MetalTabular, UvBackground) handled via scalar-per-lane fallback inside batch. | Complete |
| AP-11 | MHD flux-freeze / mean_density SIMD without Rayon | No SIMD path existed for B-field scaling or density reduction. | `apply_flux_freeze` and `mean_gas_density` dispatch AVX2/AVX-512 with `feature = "simd"`; `f64::powf(2/3)` computed scalar-per-lane; all tests pass. | Complete |
| AP-12 | MHD b_field_stats AVX-512 real 8-lane | AVX-512 path delegated to AVX-512 4-lane stub, not a true 8-lane implementation. | `b_field_stats_avx512` processes 8 particles per iteration with `_mm512_*` intrinsics and k-mask gas filtering; passes existing tests. | Complete |
| AP-13 | MHD dynamo SIMD without Rayon | Dynamo B-field update and magnetic energy ratio had no SIMD path. | `apply_turbulent_dynamo` and `magnetic_energy_ratio` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; scalar-per-lane fallback for `dynamo_growth_rate` branches and `exp()`; all tests pass. | Complete |
| AP-14 | MHD ambipolar diffusion SIMD without Rayon | Ambipolar diffusion had no SIMD path. | `apply_ambipolar_diffusion` dispatches AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; vectorized ionization fraction proxy, B-field damping, and energy deposition; scalar-per-lane for `exp()`; all tests pass. | Complete |
| AP-15 | MHD two-fluid SIMD without Rayon | Electron-ion coupling and T_e/T_i ratio had no SIMD path. | `apply_electron_ion_coupling` and `mean_te_over_ti` dispatch AVX2/AVX-512 when `feature = "simd"` and `not(feature = "rayon")`; vectorized Coulomb coupling rate and T_e/T_i reduction; scalar-per-lane for `exp()`; all tests pass. | Complete |

## RT chemistry remaining work

CPU SIMD for the stiff implicit path (`solve_chemistry_implicit`) is closed: masked
AVX2/AVX-512 dispatch batches lanes, keeps branch-heavy chemistry scalar-per-lane,
and has chunk/tail/end-to-end parity tests (see AP-09 in the table above). What
remains open is mainly **CUDA** for that path and any **performance** follow-up
(benchmarks proving batching wins on representative workloads), not a missing SIMD
design on CPU.

The adaptive subcycling and early-exit behaviour per particle still mean the inner
solver is not a trivial “four independent SIMD lanes” problem; the implemented
approach is the masked-lane pattern described in AP-09 rather than a fully
vectorized implicit integrator.

## Implementation policy

- Do not replace deterministic CPU serial paths with accelerator paths.
- Accelerator paths are opt-in and must degrade cleanly when unavailable.
- New CUDA code must keep `CUDA_SKIP=1 cargo check/test/clippy -p gadget-ng-cuda` clean.
- New x86 SIMD code must dispatch `avx512f -> avx2+fma -> scalar` when batching pays off.
- Hardware-only claims require the command, GPU model, `CUDA_ARCH`, and tolerance summary in this document or the Phase 200 roadmap.

## Next recommended phase

Phase 203 closed AP-01c; AP-10/11/12 close additional SIMD-without-Rayon gaps.
AP-02 is now complete: all CUDA solvers use `CudaPool` for persistent device
buffers, eliminating `cudaMalloc`/`cudaFree` per kernel call. `CudaDirectGravity`
now retains its CUDA handle across calls.

The next recommended phase is AP-03: validate the new MHD/Tree CUDA kernels
(with persistent buffers) on real NVIDIA hardware before wiring them into
production runtime paths. Then AP-04: route opt-in MHD/Tree CUDA kernels
behind config flags (`[accelerators]` TOML section) in `gadget-ng-cli`.

For SIMD, AP-13 (dynamo), AP-14 (ambipolar diffusion), and AP-15 (two-fluid)
close additional SIMD-without-Rayon gaps. RT IGM temperature now uses
AVX-512F 8-wide batches when available, else AVX2+FMA 4-wide (`igm_temp`);
median/percentiles remain scalar after collection. The SPH MetalTabular logT
lookup now has AVX2/AVX-512 batch coverage.

**RT chemistry, reionization, MHD CR CUDA validated (2026-05-16):** all 6 new
kernels (`rt_chemistry_rates_kernel`, `rt_cooling_apply_kernel`,
`rt_chemistry_stiff_kernel`, `rt_cm21_field_kernel`,
`mhd_cr_streaming_o2_kernel`, `mhd_cr_backreaction_kernel`) pass on
GTX 1060 sm_61 with CUDA 12.4. Two bugs were fixed during validation
(MHD CR div_v sign + formula; RT stiff solver temperature formula + f32
`k_h2p_f` overflow guard). Full record in
`docs/reports/2026-05-cuda-cr-hw-validation.md`.

**AP-16 — Cierre CUDA completo (2026-05-16):** 6 módulos nuevos/cableados:

| Item | Kernel(s) | Status |
|------|-----------|--------|
| RT IGM temperature reduction | `rt_igm_temp_kernel` | HW validated sm_61; mediana = aprox |
| MHD ambipolar diffusion | `mhd_ambipolar_kernel` | HW validated sm_61 |
| MHD two-fluid e-i coupling | `mhd_two_fluid_kernel` | HW validated sm_61 |
| RT 21cm wired in engine | `rt_cm21_field_kernel` | wired `step_reionization` (`cuda_rt_chem`) |
| MHD anisotropic/CR diffusion wired | `mhd_scalar_diffusion_kernel` | wired `step_sph` (`cuda_mhd`); campo-medio aprox |
| Analysis CUDA wired | `CudaAnalysisSolver` + `try_igm_temp_profile` | wired `analyze_cmd.rs` (`cuda_analysis` flag) |

## AP-17 additions (May 2026)

| Item | Estado |
|------|--------|
| Dedner CUDA wired | `try_dedner_cleaning` (hybrid: CPU div-B + CUDA update) wired en `step_mhd` bajo `cuda_mhd` |
| Halo spin real | `halo_spin` CPU + `try_halo_spin` CUDA en `analyze_cmd.rs`; `spin_peebles` ya no es 0 |
| X-ray flag | `--xray` flag + `try_xray_luminosity` CUDA + fallback CPU en `analyze_cmd.rs` |
| 21cm insitu | `try_cm21_field` CUDA path añadido en `insitu.rs::maybe_run_insitu` (persiste a `InsituResult.cm21`) |
| IGM percentiles | `cuda_rt_igm_temp_full` devuelve array compacto; sort+percentiles en Rust host |
| Braginskii SIMD-without-Rayon | **Implementado** — `apply_braginskii_viscosity` tiene ramas AVX2+FMA / AVX-512F sin Rayon en `crates/gadget-ng-mhd/src/braginskii.rs` |
| Reconnección SIMD-without-Rayon | **Implementado** — `apply_magnetic_reconnection` tiene ramas AVX2+FMA / AVX-512F sin Rayon en `crates/gadget-ng-mhd/src/reconnection.rs` |
| Anisotropic O(N²) CUDA | `mhd_anisotropic_pair_kernel` implementado; `try_anisotropic_conduction` + `try_cr_diffusion_anisotropic` reemplazan campo-medio |

Remaining gaps: ninguno conocido tras AP-17.

---

## Cierre formal — 2026-05-16 (AP-17)

**AP-03 a AP-17 completados.** Todos los kernels CUDA tienen:

- Tests `#[ignore]` que pasan en NVIDIA GTX 1060 (sm_61, CUDA 12.4).
- Wiring opt-in en `gadget-ng-cli` bajo `[accelerators]` del TOML de configuración.
- Cobertura de parity documentada en `docs/reports/2026-05-simd-cuda-coverage.md`.

`cargo test --workspace` pasa con 0 regresiones en la revisión de consolidación post AP-17 (2026-05-16).

---

## AP-18 — SPH core pipeline + Tree LET wiring — 2026-05-16

**Nuevo método:** `CudaSphSolver::try_sph_density_and_forces_core(&mut [Particle], periodic_box)`
— pipeline densidad+Balsara+fuerzas clásicas sobre `gadget_ng_core::Particle` sin conversión a
`SphParticle`. Wired en `step_sph` de `context.rs` bajo `[accelerators] cuda_sph = true`.

**Tree LET:** `try_tree_walk_let` integrado en el flat LET path de `mod.rs` bajo
`[accelerators] cuda_tree = true`.

**Test de hardware:** `cuda_parity_sph_core_pipeline` — PASS en GTX 1060 sm_61.
Ver detalles en `docs/reports/2026-05-cuda-ap18-sph-tree-validation.md`.

**Break-even SPH:** N_gas ≈ 300-400 (speedup 5× a N=1024, 24× a N=4096).
**Tree LET CUDA:** siempre más rápido (4.7× a N=128 nodos; 778× a N=8192 nodos).

**Gaps remanentes (documentados, no bloqueantes):**
- Barnes-Hut local GPU: requiere octree en device.
- TreePM SR: híbrido wgpu/CUDA sin wiring completo.
- f(R) chameleon screening: solo PM CUDA.

---

## AP-19 — Pipeline SPH CUDA persistente — 2026-05-16

Reescritura de `try_sph_density_and_forces_core`: 1 `pool.reset()`, 21 slots fijos,
inputs subidos una sola vez. `cuda_sph_forces` → `cuda_sph_gadget2_forces` (Balsara correcto).
Reduce H→D −65% (116 → 41 bytes/partícula). Break-even N≈300 → N≈120–150.

Ver `docs/reports/2026-05-cuda-ap19-sph-pipeline.md`.

---

## AP-20 — Cierre total de los 4 gaps CUDA — 2026-05-16

Todos los gaps remanentes quedan cerrados:

### A — MHD Hall drift CUDA

- Kernel: `mhd_hall_drift_kernel` en `mhd_kernels.cu`
  — Rodrigues rotation por hilo, eje `v×B`, ángulo `θ = η_H |B| / ρ_proxy dt`.
  Conserva `|B|` exactamente; sin heating.
- FFI: `cuda_mhd_hall_drift` (12 parámetros, sin `u_out`).
- Rust: `CudaMhdSolver::try_hall_drift`.
- Wiring: `step_mhd` bajo `cuda_mhd`, fallback CPU `apply_hall_drift`.
- Test `#[ignore]`: `cuda_hall_drift_match_cpu` — N=64, `|ΔB| < 1e-4`.

### B — f(R) chameleon screening CUDA

- Kernels: `fr_screening_per_cell_kernel` (S = fifth\_force\_factor, por celda) +
  `fr_screening_jacobi_kernel` (suavizado periódico iterativo) en `pm_gravity.cu`.
- FFI: `cuda_fr_screening_field`.
- Rust: `CudaPmSolver::try_fr_screening_field`.
- Wiring: `FrMeshParams::screening_override` + `PmSolver::set_screening_override`.
- Test `#[ignore]`: `cuda_fr_screening_match_cpu` — malla 16³, L2 rel < 1e-3.

### C — TreePM short-range CUDA

- Kernel: `treepm_sr_erfc_kernel` — O(N²) con guard `r² < r_cut²`, erfc A&S §7.1.26,
  mínima imagen periódica.
- FFI: `cuda_treepm_short_range`.
- Rust: `CudaTreeSolver::try_short_range`.
- Wiring: paso SR SFC en `stepping/mod.rs` bajo `cuda_tree`, fallback árbol CPU.
- Test `#[ignore]`: `cuda_treepm_sr_match_cpu` — N=128, magnitud rel < 5%.

### D — Barnes-Hut local GPU walk

- Kernel: `bh_walk_monopole_kernel` — pila por hilo (`stack[32]`), MAC θ²,
  self-skip por `particle_idx`, monopolo Plummer. Consume `BhMonopoleGpuNode` (96 bytes, repr(C)).
- FFI: `cuda_bh_walk_monopole`.
- Rust: `CudaTreeSolver::try_bh_local_walk`.
- Wiring: `compute_forces_local_tree` path en `stepping/mod.rs` bajo `cuda_tree`.
- Test `#[ignore]`: `cuda_bh_walk_match_cpu` — N=256, magnitud rel < 1%.

### Estado final

Tabla de paridad README.md: todos los `⚠️` → `✅`.
`cargo check --workspace` pasa con 0 errores tras AP-20 (2026-05-16).
