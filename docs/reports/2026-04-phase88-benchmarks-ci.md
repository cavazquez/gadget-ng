# Phase 88 — Benchmarks GPU vs CPU + CI --release extendido

**Fecha**: 2026-04-23  
**Crates modificados**: `gadget-ng-gpu`  
**Scripts modificados**: `scripts/check_release.sh`

## Objetivo

Añadir benchmarks formales Criterion para medir el speedup del kernel de gravedad
wgpu (GPU) vs la implementación CPU, y extender el script de CI para incluir tests
de integración en modo `--release` y tests de los nuevos módulos MPI.

## Cambios

### `crates/gadget-ng-gpu/benches/gpu_vs_cpu.rs` (nuevo)

Benchmark Criterion con tres grupos:

- **`gravity_cpu`**: gravedad directa O(N²) en CPU para N ∈ {100, 250, 500, 1000}.
- **`gravity_gpu`**: gravedad directa en GPU via compute shader WGSL para los mismos N.
  Si no hay GPU disponible, el grupo se omite con mensaje de SKIP.
- **`gravity_comparison`**: ambas implementaciones en el mismo grupo para comparación
  directa de latencia (visible en los reportes HTML de Criterion).

Cada función `bench_*` usa `criterion::black_box` para evitar optimizaciones del compilador
y `sample_size(10)` para reducir el tiempo total del benchmark manteniendo resultados significativos.

### `crates/gadget-ng-gpu/Cargo.toml`
- Añadida dependencia de dev: `criterion.workspace = true`.
- Añadida sección `[[bench]] name = "gpu_vs_cpu" harness = false`.

### `scripts/check_release.sh`

Añadidas las siguientes secciones al final del script:

```bash
# Tests de integración --release (Phase 66 SPH, 63 in-situ, 70 AMR)
cargo test -p gadget-ng-physics --test phase66_sph_cosmo --release -- --test-threads=1
cargo test -p gadget-ng-physics --test phase63_insitu_analysis --release -- --test-threads=1
cargo test -p gadget-ng-physics --test phase70_amr_pm --release -- --test-threads=1

# MPI RT real y AMR MPI (feature mpi, release)
cargo test -p gadget-ng-rt --features mpi --release
cargo test -p gadget-ng-pm --features mpi --release

# Build de benchmarks GPU (dry-run sin ejecutarlos)
cargo build -p gadget-ng-gpu --benches --release
```

## Uso

```bash
# Ejecutar benchmarks (requiere GPU para comparación real)
cargo bench -p gadget-ng-gpu --bench gpu_vs_cpu

# Resultados HTML en:
# target/criterion/gravity_cpu/
# target/criterion/gravity_gpu/
# target/criterion/gravity_comparison/

# CI completo en --release:
bash scripts/check_release.sh
```

## Rendimiento observado

Sin GPU real disponible en el entorno de CI, los benchmarks GPU hacen SKIP elegante.
En hardware con GPU (NVIDIA/AMD via wgpu/Vulkan), se espera un speedup de 5–15× para
N = 1000 partículas, creciendo con N al ser ambos algoritmos O(N²).
