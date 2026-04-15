# Roadmap

## Fase actual (MVP cerrado)

- Workspace Rust, CLI (`config`, `stepping`, `snapshot`), TOML + env.
- Gravedad directa suavizada, leapfrog KDK, MPI con `ParallelRuntime`.
- Tests: momento lineal (red cúbica), energía acotada (oscilador armónico en tests de integrador), paridad sub-bloque vs global, script serial/MPI.
- CI: `fmt`, `clippy -D warnings`, `test`, smoke MPI + paridad.

## Siguientes hitos

1. **Solvers de gravedad**: **Barnes–Hut monopolar** (octree, `gadget-ng-tree`) ✓; pendientes FMM / TreePM / PM y tests ampliados.
2. **I/O**: **HDF5** (GADGET-like) y **bincode** opcionales en `gadget-ng-io` ✓; **lectura unificada** (`SnapshotReader` + `reader_for` + `read_snapshot_formatted` para JSONL/bincode/HDF5) ✓; pendientes NetCDF, snapshots binarios avanzados.
3. **Rendimiento**: **Rayon** en bucle externo de partículas para `DirectGravity` y `BarnesHutGravity` (`RayonDirectGravity`, `RayonBarnesHutGravity`), activado con `[performance] deterministic = false` y `--features simd`; benchmarks Criterion en `gadget-ng-core` y `gadget-ng-tree` ✓; **SIMD a nivel de instrucción + caché-blocking** (`SimdDirectGravity` con SoA, `BLOCK_J=64`, `#[target_feature(avx2,fma)]`) ✓.
4. **GPU**: crate `gadget-ng-gpu` (placeholder) ✓ — layout SoA (`GpuParticlesSoA`, 8 arrays planos), `GpuDirectGravity` stub con `GravitySolver` impl en `gadget_ng_core::gpu_bridge`; pendientes kernels reales (wgpu / CUDA / HIP) y `cfg.performance.use_gpu`.
5. **Pasos temporales**: **block timesteps al estilo GADGET-4** ✓ — `TimestepSection` en `config.rs` (`hierarchical`, `eta`, `max_level`); módulo `gadget-ng-integrators::hierarchical` con `HierarchicalState`, `aarseth_bin` y `hierarchical_kdk_step` (algoritmo start-kick / drift-all / end-kick); integración en `engine.rs` vía `cfg.timestep.hierarchical`; tests de conservación de energía en dos cuerpos kepleriano. Pendientes: acoplamiento a árbol distribuido, corrección "predictive drift" para partículas inactivas, snapshot de `HierarchicalState`.
