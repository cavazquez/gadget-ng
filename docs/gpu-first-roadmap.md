# Roadmap GPU-first (estilo GADGET-4)

Este documento resume el estado de las rutas **GPU** frente a un código cosmológico “GPU-first”, y los criterios de aceptación por fase.

## Estado por componente

| Componente | Estado | Notas |
|------------|--------|--------|
| Gravedad directa wgpu | Hecho | `GpuDirectGravity`, `[performance] use_gpu` con `solver = "direct"` |
| Barnes–Hut FMM wgpu (órdenes 1–3) | Hecho | `GpuBarnesHutFmm`, `WgpuBarnesHutGpu`, `[performance] use_gpu_barnes_hut`, hexadecapolo (orden 4) solo CPU |
| Barnes–Hut monopolo wgpu (layout compacto) | Hecho | `GpuBarnesHutMonopole` (camino de pruebas / baseline) |
| TreePM corto alcance wgpu | Hecho (kernel) | `GpuTreePmShortRange` en `gadget-ng-gpu`; el solver `TreePmSolver` en CPU compone PM + SR como antes |
| PM largo alcance CUDA/HIP | Hecho (PM puro) | `CudaPmSolver` / `HipPmSolver` con `use_gpu_cuda` / `use_gpu_hip` |
| PM filtrado TreePM en GPU | Pendiente | FFT + kernel Gaussiano en device acoplado al SR GPU |
| MPI + buffers GPU end-to-end | Pendiente | LET/SFC siguen en CPU; provenance puede etiquetar `gravity:intent_*` |

## Configuración TOML relevante

```toml
[performance]
# Gravedad directa O(N²) en wgpu (solo con solver = "direct")
use_gpu = false

# Barnes–Hut en wgpu: órdenes multipolares 1–3 en device; orden 4 → CPU
use_gpu_barnes_hut = false

# PM en CUDA / HIP (solo solver = "pm")
use_gpu_cuda = false
use_gpu_hip = false
```

Requiere compilar el binario con `--features gpu` (y opcionalmente `cuda` / `hip`).

## Criterios de aceptación (por fase)

### Fase A — Integración BH GPU

- `make_solver` enruta a `WgpuBarnesHutGpu` cuando `use_gpu_barnes_hut` y solver BH.
- Tests: comparación trait vs CPU (`gadget-ng-tree`, feature `gpu`).

### Fase B — FMM en WGSL

- Paridad órdenes 1–3 con `walk_accel_multipole` dentro de tolerancia f32 (tests).
- MAC relativo y multipolos suavizados según `BhFmmKernelParams`.

### Fase C — TreePM corto alcance

- Test `treepm_short_gpu_matches_cpu` en `gadget-ng-gpu` vs `short_range_accels`.
- Próximo paso de producto: optional hook desde `TreePmSolver` + PM filtrado en CUDA.

### Fase D — Observabilidad

- `provenance.json` incluye etiquetas `gravity:intent_wgpu_*` / `gravity:intent_cuda_pm` según flags (intención, no confirmación de device).

### Fase E — Documentación

- Este archivo y sección GPU en `architecture.md`.
