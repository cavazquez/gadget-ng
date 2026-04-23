# Phase 57 — CUDA/HIP PM solver: segunda cadena de compilación

**Fecha:** abril 2026  
**Crates nuevos:** `gadget-ng-cuda`, `gadget-ng-hip`  
**Crate modificado:** `gadget-ng-cli`

---

## Contexto

Phase 57 agrega soporte opcional para aceleración GPU del solver Particle-Mesh (PM)
mediante dos sub-proyectos separados con cadenas de compilación independientes:

- `gadget-ng-cuda`: NVIDIA CUDA + cuFFT (compilación con `nvcc`).
- `gadget-ng-hip`: AMD HIP/ROCm + rocFFT (compilación con `hipcc`).

Ambos siguen el principio de **degradación elegante**: si el toolchain GPU no está
disponible en el host de compilación, el crate compila con stubs que devuelven `None`,
y el motor cae automáticamente al solver PM CPU.

---

## Arquitectura

### `build.rs` (patrón común)

```
CUDA_SKIP / HIP_SKIP env var → omitir detección
→ detectar nvcc/hipcc en $PATH
→ detectar cuFFT/rocFFT en $CUDA_PATH/$ROCM_PATH
→ compilar cuda/pm_gravity.cu / hip/pm_gravity.hip
→ emitir cargo::rustc-cfg=cuda_unavailable/hip_unavailable si falla
```

### Kernels GPU (CIC + FFT Poisson)

1. **Mass assignment** (Cloud-In-Cell, trilinear): cada hilo asigna la masa de una
   partícula a los 8 vértices del grid, con atomics.
2. **Forward 3D R2C FFT** via cuFFT/rocFFT.
3. **Poisson solver en k-space**: `Φ̂(k) = −ρ̂(k) / k²` (con k²=0 → 0).
4. **3× Backward 3D C2R FFT**: una por componente de fuerza `Fx, Fy, Fz`.
5. **Force interpolation** (CIC adjoint): interpola las fuerzas en la posición de
   cada partícula desde el grid.

### `CudaPmSolver` / `HipPmSolver`

```rust
impl GravitySolver for CudaPmSolver {
    fn try_new(grid_size, box_size) -> Option<Self>  // None si no hay GPU
    fn accelerations_for_indices(...)                // f64→f32→GPU→f32→f64
}
```

La conversión f64→f32 es explícita; la pérdida de precisión es aceptable para PM
(el PM ya es un solver de campo medio).

---

## Configuración

```toml
[performance]
use_gpu_cuda = true   # activar PM CUDA (requiere --features cuda)
use_gpu_hip  = false  # activar PM HIP/ROCm (requiere --features hip)
```

---

## Tests

| Test | Descripción |
|------|-------------|
| `cuda_pm_smoke` | `#[ignore]` si no hay CUDA; smoke test aceleración ≠ 0 |
| `hip_pm_smoke`  | `#[ignore]` si no hay HIP; mismo patrón |

Los tests son `#[ignore]` por defecto y solo se ejecutan en hosts con GPU.
