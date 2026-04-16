# CHANGELOG

Todos los cambios notables de este proyecto están documentados aquí.
Sigue el formato [Keep a Changelog](https://keepachangelog.com/es/) y
[Semantic Versioning](https://semver.org/lang/es/).

---

## [Unreleased]

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
