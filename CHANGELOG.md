# CHANGELOG

Todos los cambios notables de este proyecto están documentados aquí.
Sigue el formato [Keep a Changelog](https://keepachangelog.com/es/) y
[Semantic Versioning](https://semver.org/lang/es/).

---

## [Unreleased]

### Phase 36 — Validación práctica de `pk_correction` sobre corridas cosmológicas reales

- Nuevo reporte [`docs/reports/2026-04-phase36-pk-correction-validation.md`](docs/reports/2026-04-phase36-pk-correction-validation.md) que valida la API congelada de Phase 35 (`pk_correction`) sobre la matriz (N × seed × ic_kind) completa: N=32³/64³, 3 seeds, 1LPT y 2LPT, 3 snapshots por corrida (a ∈ {0.02, 0.05, 0.10}).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase36_pk_correction_validation.rs`](crates/gadget-ng-physics/tests/phase36_pk_correction_validation.rs): 5 tests (reducción del error absoluto, preservación de forma espectral, consistencia entre seeds en el IC, consistencia entre resoluciones N=32 vs N=64, no NaN/Inf). La matriz de 27 snapshots se ejecuta una sola vez vía `OnceLock` y serializa a `target/phase36/*.json`. Pasan en release en ~191 s.
- Nuevo experimento [`experiments/nbody/phase36_pk_correction_validation/`](experiments/nbody/phase36_pk_correction_validation/) con orquestador `run_phase36.sh` (tests Rust → pase CLI → figuras → copia a docs), config `lcdm_N32_2lpt_pm_phase36.toml`, scripts `apply_phase36_correction.py` (mirror Python de `correct_pk` + CPT92 + métricas) y `plot_phase36.py` (5 figuras obligatorias + `cli_evidence.png`).
- Pase CLI real (`gadget-ng snapshot` + `analyse` + mirror Python): `median |log₁₀(P_m/P_ref)| = 14.67` → `median |log₁₀(P_c/P_ref)| = 0.053`, `mean(P_c/P_ref) = 1.049`, `CV = 0.134`. Coincide cuantitativamente con los tests in-process.
- Hallazgo principal: la corrección reduce el error absoluto de amplitud de `median |log₁₀(P_m/P_ref)| ≈ 14–18` a `≈ 0.03` en el snapshot IC real de cualquier (N, seed, ic_kind) de la matriz — factor de mejora `~10¹⁴` reproducible end-to-end. **La amplitud absoluta queda cerrada "en la práctica"** en el régimen válido (`k ≤ k_Nyq/2`, `a = a_init`, `N ∈ {32, 64}`, CIC).
- Limitación documentada: el proyecto aplica `σ₈=0.8` en `a_init` sin escalar por `D(a_init)/D(0)`, lo que pone las corridas en régimen no-lineal desde el paso 1 y hace que los snapshots a `a > a_init` queden fuera del dominio lineal de `pk_correction` (ortogonal a la corrección).

### Phase 35 — Modelado de `R(N)` para corrección absoluta de `P(k)`

- Nuevo reporte [`docs/reports/2026-04-phase35-rn-modeling.md`](docs/reports/2026-04-phase35-rn-modeling.md) que caracteriza el factor de muestreo discreto `R(N)` (partículas + CIC) identificado en Phase 34 como función de resolución, fitea dos modelos (potencia pura y potencia + offset), selecciona ganador por AIC y documenta el rango de validez.
- Nuevo módulo público [`crates/gadget-ng-analysis/src/pk_correction.rs`](crates/gadget-ng-analysis/src/pk_correction.rs) con `RnModel`, `a_grid`, `correct_pk` y `RnModel::phase35_default` (valores congelados del fit: `C = 22.108`, `α = 1.8714`, tabla `N ∈ {8,16,32,64}`). API expuesta en `gadget_ng_analysis::{a_grid, correct_pk, RnModel}`.
- Nuevos tests de caracterización en [`crates/gadget-ng-physics/tests/phase35_rn_modeling.rs`](crates/gadget-ng-physics/tests/phase35_rn_modeling.rs): 6 tests sobre la matriz `N × seed` (4×4) que validan determinismo entre seeds (CV < 0.10 para N≥32), flatness de `R(N,k)` en k bajo (CV_k < 0.25), fit log-log (R² = 0.997), reducción del error de amplitud (mediana de `|log₁₀|` de 17.9 → 0.037, factor ×485), consistencia interpolación-modelo a N=48 (< 2.5 %) y verificación CIC vs TSC a N=32.
- 6 unit tests en `pk_correction` cubren `from_table`, interpolación log-log, preferencia tabla-sobre-fit y escalado lineal de `correct_pk`.
- Nuevo experimento [`experiments/nbody/phase35_rn_modeling/`](experiments/nbody/phase35_rn_modeling/) con orquestador `run_phase35.sh`, `scripts/fit_r_n.py` (OLS log-log + `scipy.curve_fit` + AIC), `scripts/plot_r_n.py` (5 figuras) y `scripts/apply_correction.py` (demo de postproceso). Las 5 figuras obligatorias se copian a `docs/reports/figures/phase35/`.
- Hallazgo: Modelo A (potencia pura) gana por ΔAIC = −11.45 frente a Modelo B (el offset asintótico sale *negativo*). Con `A_grid(N)` de Phase 34 + `R(N)` de Phase 35, la amplitud absoluta de `P(k)` se cierra al ~9 % en postproceso sin modificar el core.

### Phase 34 — Cierre de la normalización discreta de `P(k)`

- Nuevo reporte [`docs/reports/2026-04-phase34-discrete-normalization-closure.md`](docs/reports/2026-04-phase34-discrete-normalization-closure.md) que decompone el pipeline `P_cont → δ̂(k) → IFFT → δ(x) → FFT → P(k)` (con y sin partículas) en etapas independientes y aísla dónde nace el offset de amplitud absoluta reportado en Phase 30–33.
- Nuevos tests de caracterización en [`crates/gadget-ng-physics/tests/phase34_discrete_normalization.rs`](crates/gadget-ng-physics/tests/phase34_discrete_normalization.rs): 8 tests que verifican roundtrip DFT (8.9e-16), modo único, ruido blanco (ratio 0.996), offset partícula/grilla (CV 0.6 %), efecto CIC, escalado con N y determinismo entre seeds.
- Nuevo módulo `gadget_ng_core::ic_zeldovich::internals` (re-exportado como `ic_zeldovich_internals`) que expone `generate_delta_kspace`, `fft3d`, `delta_to_displacement`, `build_spectrum_fn` y `mode_int` como API testing-only documentada. Sin cambios de comportamiento en el core.
- Nuevo experimento [`experiments/nbody/phase34_discrete_normalization/`](experiments/nbody/phase34_discrete_normalization/) con orquestador `run_phase34.sh`, `scripts/stage_table.py`, `scripts/plot_stages.py` y las 5 figuras obligatorias (`grid_ratio`, `particle_ratio`, `stage_breakdown`, `cic_effect`, `single_mode_amplitude`).
- Hallazgo: el offset se descompone limpiamente en (i) un **factor de grilla cerrado** `A_grid = 2·V²/N⁹` verificado al 3 % (cierra el residuo de 17× de Phase 33) y (ii) un **factor partículas-CIC** `R(N)` determinista por resolución (CV < 1 %) pero dependiente de N.
- Decisión: se mantiene la convención interna actual (Opción B). El factor de grilla queda documentado cerrado; `R(N)` queda congelado como regresión en los tests. Sin parches al core.

### Fase 2

#### [Hito 15] — Sistema de unidades físicas
- Nueva sección `[units]` en el TOML de configuración: `enabled`, `length_in_kpc`, `mass_in_msun`, `velocity_in_km_s`.
- `RunConfig::effective_g()` calcula G en unidades internas a partir de `G = 4.3009×10⁻⁶ kpc Msun⁻¹ (km/s)²`.
- Método auxiliar `UnitsSection::time_unit_in_gyr()` y `hubble_time(h0)`.
- `SnapshotEnv` y `meta.json` incluyen bloque `units` cuando está habilitado (`length_in_kpc`, `mass_in_msun`, `velocity_in_km_s`, `time_in_gyr`, `g_internal`).
- Retrocompatible: `enabled = false` (default) deja `gravitational_constant` sin cambios.

#### [Hito 12] — Restart / Checkpointing
- Nueva opción `[output] checkpoint_interval = N`: guarda checkpoint cada N pasos en `<out>/checkpoint/`.
- Checkpoint incluye: `checkpoint.json` (paso completado, factor de escala `a`, hash de config), `particles.jsonl` y (si aplica) `hierarchical_state.json`.
- `gadget-ng stepping --resume <out_anterior>` reanuda desde el último checkpoint sin pérdida de precisión.
- Advertencia si el hash del config cambió desde que se guardó el checkpoint.
- Compatible con todos los modos de integración: leapfrog clásico, cosmológico, jerárquico y árbol distribuido.

#### [Hito 10] — Pulir
- `CHANGELOG.md` con historial semántico completo (este archivo).
- `docs/user-guide.md`: guía de usuario con ejemplos TOML comentados para cada solver y opción.
- `.github/workflows/ci.yml`: CI con `fmt`, `clippy -D warnings`, `cargo test --workspace`, benchmarks en dry-run.
- Nuevos benchmarks Criterion en `gadget-ng-pm` (`pm_gravity_128`) y `gadget-ng-treepm` (`treepm_gravity_128`).

---

## Fase 1

### [Hito 9] — MPI árbol distribuido
- `SlabDecomposition` (dominio x en slabs uniformes) en `gadget-ng-parallel::domain`.
- `allreduce_min/max_f64` en `ParallelRuntime`.
- `exchange_domain_by_x` (migración de partículas entre rangos) y `exchange_halos_by_x` (halos punto-a-punto, patrón odd-even anti-deadlock).
- `compute_forces_local_tree` en engine: árbol local de (partículas + halos).
- Activado con `[performance] use_distributed_tree = true` y `solver = "barnes_hut"`.
- Comunicación O(N_halo × 2) en lugar de Allgather O(N).

### [Hito 8] — GPU kernels reales (wgpu portátil)
- `GpuDirectGravity` real con wgpu 29 (WGSL compute shader, Vulkan/Metal/DX12/WebGPU).
- Kernel O(N²) de gravedad Plummer suavizada en f32 (error relativo O(1e-7)).
- `GpuContext` con `Arc<>` + `Send + Sync`; readback síncrono.
- Activado con `[performance] use_gpu = true`; fallback automático a CPU si no hay GPU.

### [Hito 7] — FMM (Fast Multipole Method) — cuadrupolo
- Tensor de cuadrupolo sin traza `[Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]` en `OctNode`.
- Calculado en `aggregate` vía teorema del eje paralelo.
- Corrección de aceleración cuadrupolar en `walk_inner`.
- Error relativo medio con θ=0.5 < 0.5% (vs >1% solo monopolo).

### [Hito 6] — Cosmología básica
- Formulación de momentum canónico estilo GADGET-4: `p = a²·dx_c/dt`.
- `CosmologySection` en config: `omega_m`, `omega_lambda`, `h0`, `a_init`.
- `advance_a` (RK4 Friedmann) y `drift_kick_factors` (Simpson N_SUB=16).
- `leapfrog_cosmo_kdk_step` + `CosmoFactors`; integrador jerárquico extendido.
- `redshift = 1/a − 1` en `SnapshotEnv`.

### [Hito 5] — TreePM (árbol + malla)
- Solver `TreePmSolver`: Barnes-Hut (corto alcance, kernel erfc) + PM (largo alcance, kernel erf).
- `r_split` configurable (default: `2.5 × cell_size`).

### [Hito 4] — Particle-Mesh (PM) FFT periódico
- Solver `PmSolver`: FFT 3D periódica, resolución `pm_grid_size³`.
- Estimación de densidad CIC (Cloud-In-Cell) y derivada del potencial.

### [Hito 3] — Barnes-Hut tree
- `Octree` con agregación recursiva de centros de masa.
- Criterio MAC `s/d < θ` (default θ=0.5).
- Suavizado Plummer; soporte Rayon con `RayonBarnesHutGravity`.

### [Hito 2] — Integrador jerárquico (block timesteps)
- `HierarchicalState` con niveles de potencia de 2.
- Criterio de Aarseth: `dt_i = η × sqrt(ε / |a_i|)`.
- `hierarchical_kdk_step`; guardado/carga de estado (`hierarchical_state.json`).

### [Hito 1] — MVP N-body
- Integrador Leapfrog KDK global.
- Condiciones iniciales: lattice cúbico perturbado, dos cuerpos circulares.
- Snapshots JSONL/HDF5/Bincode/MessagePack/NetCDF.
- Paralelismo MPI (`rsmpi`): `allgatherv_state`, distribución por GID.
- Diagnósticos por paso: `diagnostics.jsonl`.
- CLI: `gadget-ng config`, `gadget-ng stepping`, `gadget-ng snapshot`.
