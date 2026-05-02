# Roadmap GPU-first (estilo GADGET-4)

Este documento resume el estado de las rutas **GPU** frente a un código cosmológico “GPU-first”, y los criterios de aceptación por fase.

## Estado por componente

| Componente | Estado | Notas |
|------------|--------|--------|
| Gravedad directa wgpu | Hecho | `GpuDirectGravity`, `[performance] use_gpu` con `solver = "direct"` |
| Barnes–Hut FMM wgpu | Hecho (órdenes 1–4) | `GpuBarnesHutFmm`, `WgpuBarnesHutGpu`, `[performance] use_gpu_barnes_hut`; hexadecapolo en WGSL + pesos STF (`gadget-ng-gpu-layout::hex_pattern_weights`) |
| Barnes–Hut monopolo wgpu (layout compacto) | Hecho | baseline / pruebas |
| TreePM corto alcance wgpu | Hecho (kernel) | `GpuTreePmShortRange` |
| TreePM híbrido (LR CUDA + SR wgpu) | Hecho (opt-in) | `[performance] use_gpu_treepm = true`, `solver = "tree_pm"`, build `--features gpu,cuda`; degradación a CPU |
| PM largo alcance CUDA/HIP | Hecho | `CudaPmSolver` / `HipPmSolver`; **filtro Gaussiano TreePM** `r_split` vía `try_new_with_r_split` / `solve` (`W(k)=exp(−k²r_s²/2)`) |
| MPI + buffers GPU end-to-end | Parcial / diseño | LET y comunicación en host; fuerzas locales pueden usar GPU donde ya exista solver; sin kernel LET remoto en GPU (v1) |

## Configuración TOML relevante

```toml
[performance]
use_gpu = false
use_gpu_barnes_hut = false
use_gpu_cuda = false
use_gpu_hip = false
# TreePM: PM CUDA filtrado + SR wgpu (requiere --features gpu,cuda)
use_gpu_treepm = false
```

Requiere compilar el binario con `--features gpu` (y opcionalmente `cuda` / `hip`).

## Criterios de aceptación (por fase)

### Fase A — Integración BH GPU

- `make_solver` enruta a `WgpuBarnesHutGpu` cuando `use_gpu_barnes_hut` y solver BH.
- Tests: comparación trait vs CPU (`gadget-ng-tree`, feature `gpu`).

### Fase B — FMM en WGSL

- Paridad órdenes 1–4 con `walk_accel_multipole` dentro de tolerancia f32 (test `bh_fmm_order4_wgpu`).
- MAC relativo y multipolos suavizados según `BhFmmKernelParams`.

### Fase C — TreePM

- Test `treepm_short_gpu_matches_cpu` (SR wgpu vs CPU).
- PM filtrado CUDA vs `fft_poisson::solve_forces_filtered` (`gadget-ng-cuda`, test ignorado sin GPU).
- TreePM híbrido: mismo `accelerations_for_indices` que CPU TreePM para N moderado (validación manual / CI con GPU).

### Fase D — Observabilidad

- `provenance.json`: `gravity:intent_wgpu_*`, `gravity:intent_cuda_pm`, `gravity:intent_gpu_treepm_hybrid`.

### Fase E — MPI + GPU

- Comunicación MPI en host; posible staging SoA (`GpuParticlesSoA`) para minimizar copias; **LET remoto no GPU** en la versión actual.

### Fase F — Documentación

- Este archivo y sección GPU en `architecture.md` si aplica.
