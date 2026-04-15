# Roadmap

## Fase actual (MVP cerrado)

- Workspace Rust, CLI (`config`, `stepping`, `snapshot`), TOML + env.
- Gravedad directa suavizada, leapfrog KDK, MPI con `ParallelRuntime`.
- Tests: momento lineal (red cúbica), energía acotada (oscilador armónico en tests de integrador), paridad sub-bloque vs global, script serial/MPI.
- CI: `fmt`, `clippy -D warnings`, `test`, smoke MPI + paridad.

## Siguientes hitos

1. **Solvers de gravedad**: **Barnes–Hut monopolar** (octree, `gadget-ng-tree`) ✓; pendientes FMM / TreePM / PM y tests ampliados.
2. **I/O**: **HDF5** (GADGET-like) y **bincode** opcionales en `gadget-ng-io` ✓; pendientes lectura unificada, NetCDF, snapshots binarios avanzados.
3. **Rendimiento**: **Rayon** en bucle externo de partículas para `DirectGravity` y `BarnesHutGravity` (`RayonDirectGravity`, `RayonBarnesHutGravity`), activado con `[performance] deterministic = false` y `--features simd`; benchmarks Criterion en `gadget-ng-core` y `gadget-ng-tree` ✓; pendientes optimizaciones SIMD a nivel de instrucción y caché-blocking.
4. **GPU**: kernels y política de datos (SoA), separados del core CPU.
5. **Pasos temporales**: esquemas jerárquicos inspirados en GADGET-4 con pruebas de conservación ampliadas.
