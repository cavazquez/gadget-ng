# Roadmap

## Fase actual (MVP cerrado)

- Workspace Rust, CLI (`config`, `stepping`, `snapshot`), TOML + env.
- Gravedad directa suavizada, leapfrog KDK, MPI con `ParallelRuntime`.
- Tests: momento lineal (red cúbica), energía acotada (oscilador armónico en tests de integrador), paridad sub-bloque vs global, script serial/MPI.
- CI: `fmt`, `clippy -D warnings`, `test`, smoke MPI + paridad.

## Siguientes hitos

1. **Solvers de gravedad**: Barnes–Hut / FMM / PM con interfaces estables y tests de regresión.
2. **I/O**: NetCDF/HDF5 reales; formatos de snapshot binarios opcionales.
3. **Rendimiento**: `simd` / `rayon` con garantías de determinismo opcionales por feature.
4. **GPU**: kernels y política de datos (SoA), separados del core CPU.
5. **Pasos temporales**: esquemas jerárquicos inspirados en GADGET-4 con pruebas de conservación ampliadas.
