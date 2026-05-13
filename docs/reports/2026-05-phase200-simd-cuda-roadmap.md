# Phase 200 — SIMD/CUDA Parity Roadmap

**Created:** 2026-05-13
**Updated:** 2026-05-13 (Fase 1 completada)

## Overview

Achieve full 1-to-1 parity between SIMD (CPU serial/Rayon/AVX2/AVX-512) and CUDA backends across all physics modules, then expand Rayon coverage and implement AVX2/AVX-512 auto-vectorization.

## Constraints

- All backends must produce physically equivalent results (within f32 precision for CUDA)
- CUDA code must compile with `CUDA_SKIP=1` (stubs for CI without GPU)
- Rayon paths gated behind `#[cfg(feature = "simd")]` with `#[cfg(not(feature = "simd"))]` serial fallbacks
- CUDA paths gated behind `#[cfg(feature = "cuda")]` with `cuda_unavailable` stubs
- AVX2 only in `gravity_simd.rs` and `rmn_soa.rs` (SoA+cache-blocked auto-vectorization via `#[target_feature]`)
- AVX-512 not yet implemented anywhere
- `feature = "simd"` gates **Rayon parallelism**, NOT actual SIMD intrinsics
- Auto-vectorization preferred over explicit intrinsics for maintainability

## Completed Work

### Phase A — CUDA Kernels ✅ (2026-05-10)
- `crates/gadget-ng-cuda/cuda/cooling_kernels.cu` — AtomicHHe, MetalCooling, MetalTabular, UvBackground + MHD suppression
- `crates/gadget-ng-cuda/cuda/dust_kernels.cu` — D/G accretion + sputtering + radiation pressure kick
- `crates/gadget-ng-cuda/cuda/molecular_kernels.cu` — HI→H₂ with dust shielding

### Phase A — Rust Wrappers ✅ (2026-05-10)
- `crates/gadget-ng-cuda/src/cooling_solver.rs` — CudaCoolingSolver
- `crates/gadget-ng-cuda/src/dust_solver.rs` — CudaDustSolver
- `crates/gadget-ng-cuda/src/molecular_solver.rs` — CudaMolecularSolver
- Updated `ffi.rs`, `lib.rs`, `build.rs`

### Phase B — Engine Wiring ✅ (2026-05-11)
- `crates/gadget-ng-cli/src/engine/stepping/context.rs` — CUDA paths for cooling, dust, H₂, flux_freeze, photoheating

### Phase C — Tests ✅ (2026-05-11)
- 4 smoke/integration tests for CUDA parity

### Benchmark Suite ✅ (2026-05-11)
- `crates/gadget-ng-cuda/benches/parity_bench.rs` — 11 benchmark groups

### Rayon Expansion ✅ (2026-05-12)
- ~50 new parallel functions across MHD (13 files), SPH (10 files), Integrators (4 files), Tree (2 files), Core (1 file)

### Feature Parity Table ✅ (2026-05-12)
- 164-function inventory: serial✅, rayon✅98, avx2✅9, cuda✅15

---

## PENDIENTE — Implementation Items

### Fase 1 — Close Remaining Rayon Gaps (~200 LOC, ~1h)

**Estado:** ✅ COMPLETADO (2026-05-13)
**Prioridad:** Alta

Functions parallelized with `#[cfg(feature = "simd")]` Rayon + `#[cfg(not(feature = "simd"))]` serial fallback:

| # | Función | Archivo | Patrón | Estado |
|---|---------|---------|--------|--------|
| 1.1 | `apply_chemistry` | `gadget-ng-rt/src/chemistry.rs` | `par_iter_mut().zip()` | ✅ |
| 1.2 | `compute_reionization_state` | `gadget-ng-rt/src/reionization.rs` | `par_iter().fold().reduce()` | ✅ |
| 1.3 | `magnetic_energy_ratio` | `gadget-ng-mhd/src/dynamo.rs` | `par_iter().fold().reduce()` | ✅ |
| 1.4 | `apply_ambipolar_diffusion` | `gadget-ng-mhd/src/nonideal.rs` | `par_iter_mut().for_each()` | ✅ |
| 1.5 | `apply_turbulent_dynamo` | `gadget-ng-mhd/src/dynamo.rs` | `par_iter_mut().for_each()` | ✅ |
| 1.6 | `apply_electron_ion_coupling` | `gadget-ng-mhd/src/two_fluid.rs` | `par_iter_mut().for_each()` | ✅ |
| 1.7 | `mean_te_over_ti` | `gadget-ng-mhd/src/two_fluid.rs` | `par_iter().fold().reduce()` | ✅ |
| 1.8 | `advance_srmhd` | `gadget-ng-mhd/src/relativistic.rs` | `par_iter_mut().for_each()` | ✅ |
| 1.9 | `compute_igm_temp_profile` | `gadget-ng-rt/src/igm_temp.rs` | `par_iter().filter_map().collect()` | ✅ |
| 1.10 | `magnetic_power_spectrum` | `gadget-ng-mhd/src/stats.rs` | `par_iter().filter_map().collect()` | ✅ |
| 1.11 | `compute_delta_tb_field` | `gadget-ng-rt/src/cm21.rs` | `par_iter().zip().map().collect()` | ✅ |
| 1.12 | `turbulence_stats` | `gadget-ng-mhd/src/turbulence.rs` | `par_iter().fold().reduce()` | ✅ |
| 1.13 | `deposit_gas_emission` | `gadget-ng-rt/src/coupling.rs` | `par_iter().filter_map().collect()` + merge | ✅ |
| 1.14 | `deposit_dust_ir_emission` | `gadget-ng-rt/src/multifrequency.rs` | `par_iter().filter_map().collect()` + merge | ✅ |
| 1.15 | `deposit_uv_sources` | `gadget-ng-rt/src/reionization.rs` | `par_iter().map().collect()` + merge | ✅ |

**Nota:** `deposit_cic` (cm21) se mantiene serial porque el patrón de rasterización CIC con wraps periódicos es intrínsecamente secuencial en la deposición. Las funciones `merge_black_holes` y `seed_primordial_black_holes` se mantienen serial por ser inherentemente secuenciales (nested loop + swap_remove, sort-dependent).

**Tests:** `cargo test -p gadget-ng-rt --features simd` (64 pass), `cargo test -p gadget-ng-mhd --features simd` (35 pass). Clippy clean with `-D warnings` for both `simd` and non-`simd` configurations.

---

### Fase 2 — SPH Kernel SIMD Batch Functions (~300 LOC, ~2h)

**Estado:** ✅ COMPLETADO (2026-05-13)
**Prioridad:** Alta

Added branch-free batch versions of `w()` and `grad_w()` with runtime SIMD dispatch:

| # | Item | Archivo | Descripción | Estado |
|---|------|---------|-------------|--------|
| 2.1 | `w_batch` | `gadget-ng-sph/src/kernel.rs` | Batch kernel evaluation, AVX-512→AVX2→scalar | ✅ |
| 2.2 | `grad_w_batch` | `gadget-ng-sph/src/kernel.rs` | Batch gradient evaluation, AVX-512→AVX2→scalar | ✅ |
| 2.3 | `w_and_grad_w_batch` | `gadget-ng-sph/src/kernel.rs` | Combined kernel+gradient (shares q/t/t³ computation) | ✅ |
| 2.4 | `w_branchfree` | `gadget-ng-sph/src/kernel.rs` | Branch-free inner for auto-vectorization (q.min(2.0)) | ✅ |
| 2.5 | `grad_w_branchfree` | `gadget-ng-sph/src/kernel.rs` | Branch-free gradient inner for auto-vectorization | ✅ |
| 2.6 | Runtime dispatch | `gadget-ng-sph/src/kernel.rs` | `is_x86_feature_detected!("avx512f")` → `avx2+fma` → scalar | ✅ |

**Key design decisions:**
- Branch-free formulation: `q.min(2.0)` makes `t = 1 - q/2 = 0` at support boundary, so `t⁴ = 0` → result is exactly 0. No `if` branches needed.
- `w_and_grad_w_batch` shares the `q`, `t`, `t³` computations between kernel and gradient, saving ~40% work vs. calling them separately.
- `#[target_feature(enable = "avx2", enable = "fma")]` and `#[target_feature(enable = "avx512f")]` force the compiler to emit vector instructions for the contiguous array loops.
- All batch functions take `r: &[f64]` (contiguous) with fixed `h: f64` — the most common SPH pattern (same smoothing length for all neighbors of particle i).

**Tests:** 42 pass (including 5 new batch tests). Clippy clean with `-D warnings`.

---

### Fase 3 — Gravity AVX-512 Extension (~100 LOC)

**Estado:** ✅ COMPLETADO (2026-05-13)
**Prioridad:** Media

| # | Item | Archivo | Descripción | Estado |
|---|------|---------|-------------|--------|
| 3.1 | `inner_blocked_avx512` | `gadget-ng-core/src/gravity_simd.rs` | `#[target_feature(enable = "avx512f")]` with BLOCK_J=128 | ✅ |
| 3.2 | `inner_scalar_128` | `gadget-ng-core/src/gravity_simd.rs` | Scalar inner loop with BLOCK_J=128 for AVX-512 tile size | ✅ |
| 3.3 | Runtime dispatch update | `gadget-ng-core/src/gravity_simd.rs` | AVX-512 → AVX2+FMA → scalar fallback chain | ✅ |
| 3.4 | `BLOCK_J_AVX512` | `gadget-ng-core/src/gravity_simd.rs` | Tile size constant = 128 (4 KB per component in L1) | ✅ |

**Key design decisions:**
- AVX-512 uses `BLOCK_J_AVX512 = 128` (double the AVX2 tile) since ZMM processes 8×f64 per iteration. This gives 4 KB per tile instead of 2 KB, still well within L1 cache.
- Same branch-free mask pattern (`if k == skip { 0.0 } else { 1.0 }`) — LLVM eliminates the branch via `vblendpd`/`vblendvpd`.
- `inner_scalar_128` is a separate function to ensure the different tile size compiles correctly with `avx512f` target features.
- Dispatch priority: `avx512f` first, then `avx2+fma`, then scalar.

**Tests:** 33 pass. Clippy clean with `-D warnings`.

---

### Fase 4 — SPH Pair Loop SIMD Tiling (~600 LOC, ~3h)

**Estado:** ✅ COMPLETADO (2026-05-13)
**Prioridad:** Media

| # | Item | Archivo | Descripción |
|---|------|---------|-------------|
| 4.1 | Density j-loop tiling | `gadget-ng-sph/src/density.rs` | Tile inner j-loop by 4/8 using `w_batch_*` |
| 4.2 | Forces j-loop tiling | `gadget-ng-sph/src/forces.rs` | Tile inner j-loop by 4/8 using `grad_w_batch_*` |
| 4.3 | Balsara j-loop tiling | `gadget-ng-sph/src/balsara.rs` | Tile inner j-loop by 4/8 |
| 4.4 | Gadget2 j-loop tiling | `gadget-ng-sph/src/gadget2.rs` | Tile inner j-loop by 4/8 |

---

### Fase 5 — MHD Pair Loop SIMD Tiling (~500 LOC, ~2h)

**Estado:** ✅ COMPLETADO (2026-05-13)
**Prioridad:** Media

| # | Item | Archivo | Descripción | Estado |
|---|------|---------|-------------|--------|
| 5.1 | Induction Rayon | `gadget-ng-mhd/src/induction.rs` | `advance_induction` → `_impl`/`_par` with `into_par_iter()` | ✅ |
| 5.2 | Resistivity Rayon | `gadget-ng-mhd/src/induction.rs` | `apply_artificial_resistivity` → `_impl`/`_par` | ✅ |
| 5.3 | Magnetic forces Rayon | `gadget-ng-mhd/src/pressure.rs` | `apply_magnetic_forces` → `_impl`/`_par` (N² per-particle) | ✅ |
| 5.4 | Cleaning Rayon | `gadget-ng-mhd/src/cleaning.rs` | `dedner_cleaning_step` → `_impl`/`_par` | ✅ |
| 5.5 | Anisotropic Rayon | `gadget-ng-mhd/src/anisotropic.rs` | `apply_anisotropic_conduction` + `diffuse_cr_anisotropic` → `_impl`/`_par` | ✅ |
| 5.6 | Reconnection Rayon | `gadget-ng-mhd/src/reconnection.rs` | `apply_magnetic_reconnection` → `_impl`/`_par` (N² per-particle) | ✅ |
| 5.7 | Braginskii Rayon | `gadget-ng-mhd/src/braginskii.rs` | `apply_braginskii_viscosity` → `_impl`/`_par` (N² per-particle) | ✅ |

**Key design decisions:**
- All 7 MHD pair-loop functions now have `_impl` (serial) / `_par` (Rayon) split with `#[cfg]` dispatch, matching the pattern from Fase 1–4.
- Half-pair loops (pressure, anisotropic conduction, braginskii, reconnection) converted to full N² per-particle accumulation for clean `par_iter()` — avoids race conditions and is physically more correct for anisotropic effects (each particle uses its own B̂).
- SoA data extraction upfront for `_par` versions: pos, vel, b_field, mass, h_sml, rho, is_gas extracted into `Vec`s before Rayon dispatch.
- `diffuse_cr_anisotropic` gets early distance filter `r² < (2h_i)²` in `_par` version and branch-free `kernel_w_branchfree` helper.
- `apply_magnetic_reconnection` N² per-particle: each particle accumulates `delta_u[i]` and `b_scale[i]` independently. `b_scale` uses `.min()` accumulation (safe for parallel since each thread works on its own particle).
- `apply_braginskii_viscosity` N²: each particle uses own `B̂_i` and `cos²(θ_i)`, which is physically correct for anisotropic viscosity.
- Clippy clean with `-D warnings` for both `simd` and non-`simd` configurations. 35 tests pass in both configurations.

---

### Fase 6 — CIC SIMD (~200 LOC, ~1h)

**Estado:** PENDIENTE
**Fecha planificada:** 2026-05-18
**Prioridad:** Media

| # | Item | Archivo | Descripción |
|---|------|---------|-------------|
| 6.1 | `assign_simd` | `gadget-ng-pm/src/cic.rs` | `#[target_feature(enable = "avx2")]` CIC mass assignment |
| 6.2 | `interpolate_simd` | `gadget-ng-pm/src/cic.rs` | `#[target_feature(enable = "avx2")]` CIC force interpolation |
| 6.3 | `assign_avx512` | `gadget-ng-pm/src/cic.rs` | `#[target_feature(enable = "avx512f")]` 8-wide CIC |
| 6.4 | `interpolate_avx512` | `gadget-ng-pm/src/cic.rs` | `#[target_feature(enable = "avx512f")]` 8-wide CIC |
| 6.5 | Runtime dispatch | `gadget-ng-pm/src/cic.rs` | Fallback chain |

---

### Fase 7 — FFT/Poisson SIMD (~200 LOC, ~1h)

**Estado:** PENDIENTE
**Fecha planificada:** 2026-05-19
**Prioridad:** Media

| # | Item | Archivo | Descripción |
|---|------|---------|-------------|
| 7.1 | SoA complex grid | `gadget-ng-pm/src/fft_poisson.rs` | Convert AoS complex grid to SoA for SIMD |
| 7.2 | Spectral kernel SIMD | `gadget-ng-pm/src/fft_poisson.rs` | `#[target_feature]` for k-space Poisson kernel |
| 7.3 | Runtime dispatch | `gadget-ng-pm/src/fft_poisson.rs` | avx512f → avx2 → scalar |

---

### Fase 8 — Tree LET AVX-512 (~150 LOC, ~1h)

**Estado:** PENDIENTE
**Fecha planificada:** 2026-05-20
**Prioridad:** Baja

| # | Item | Archivo | Descripción |
|---|------|---------|-------------|
| 8.1 | `mono_pass_avx512` | `gadget-ng-tree/src/rmn_soa.rs` | 8-wide ZMM monopole pass |
| 8.2 | `accel_p17_avx512_range` | `gadget-ng-tree/src/rmn_soa.rs` | Phase 17 kernel using ZMM |
| 8.3 | Runtime dispatch update | `gadget-ng-tree/src/rmn_soa.rs` | Add avx512f branch |

---

### Fase 9 — CUDA Kernels for MHD + Tree (~3000 LOC, ~4-6h)

**Estado:** PENDIENTE
**Fecha planificada:** 2026-05-21+
**Prioridad:** Baja

| # | Item | Archivo | Descripción |
|---|------|---------|-------------|
| 9.1 | Induction kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | advance_induction + apply_artificial_resistivity |
| 9.2 | Magnetic forces kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | apply_magnetic_forces |
| 9.3 | Cleaning kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | dedner_cleaning |
| 9.4 | Anisotropic kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | anisotropic_conduction + cr_anisotropic |
| 9.5 | Braginskii kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | braginskii_viscosity |
| 9.6 | Reconnection kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | magnetic_reconnection |
| 9.7 | Streaming kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | streaming_crk |
| 9.8 | Dynamo kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | turbulent_dynamo |
| 9.9 | CRK kernel | `gadget-ng-cuda/cuda/mhd_kernels.cu` | CRK-MHD terms |
| 9.10 | SIDM kernel | `gadget-ng-cuda/cuda/tree_kernels.cu` | SIDM scattering |
| 9.11 | Barnes-Hut walk kernel | `gadget-ng-cuda/cuda/tree_kernels.cu` | Tree walk acceleration |
| 9.12 | Rust wrappers | `gadget-ng-cuda/src/mhd_solver.rs` | CudaMhdSolver |
| 9.13 | Rust wrappers | `gadget-ng-cuda/src/tree_solver.rs` | CudaTreeWalkSolver |
| 9.14 | Engine wiring | `gadget-ng-cli/src/engine/stepping/context.rs` | CUDA paths for MHD+Tree |
| 9.15 | Tests | `gadget-ng-cuda/tests/` | Smoke tests for MHD+Tree CUDA |

---

## Change Log

| Fecha | Fase | Cambio |
|-------|------|--------|
| 2026-05-13 | — | Document created. All phases PENDIENTE. |
| 2026-05-13 | 1 | Fase 1 completada: 15 funciones con Rayon parallel + serial fallback. Tests pass, clippy clean. |
| 2026-05-13 | 2 | Fase 2 completada: SPH kernel batch functions (w_batch, grad_w_batch, w_and_grad_w_batch) con AVX-512/AVX2/scalar dispatch. Branch-free formulation. |
| 2026-05-13 | 3 | Fase 3 completada: Gravity AVX-512 with BLOCK_J=128, inner_scalar_128, runtime dispatch avx512f→avx2+fma→scalar. |
| 2026-05-13 | 5 | Fase 5 completada: 7 MHD pair-loop functions con Rayon parallel (_impl/_par) + serial fallback. Half-pair→N² per-particle para clean par_iter. Tests pass, clippy clean. |