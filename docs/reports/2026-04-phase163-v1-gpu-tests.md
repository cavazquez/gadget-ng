# Phase 163 — V1: GPU CUDA/HIP Direct Gravity Stubs + Tests

**Fecha:** 2026-04-23  
**Status:** ✅ Stubs implementados, 1 test CI verde, 5 `#[ignore]` listos para hardware

---

## Resumen

Se añadieron los structs `CudaDirectGravity` y `HipDirectGravity` como stubs
que estarán listos para ser reemplazados por kernels reales cuando haya hardware.
Se escribió el suite de 6 tests V1 en `crates/gadget-ng-gpu/tests/v1_gpu_tests.rs`.

---

## Nuevos structs

### `CudaDirectGravity` (`gadget-ng-cuda/src/direct_solver.rs`)

```rust
pub struct CudaDirectGravity {
    pub eps: f32,
    pub block_size: usize,
}

impl CudaDirectGravity {
    pub fn try_new(eps: f32) -> Option<Self>;   // None si sin hardware CUDA
    pub fn compute(&self, pos: &[[f32;3]], mass: &[f32]) -> Vec<[f32;3]>;  // todo!()
    pub fn recommended_max_n(&self) -> usize;
}
```

### `HipDirectGravity` (`gadget-ng-hip/src/direct_solver.rs`)

Idéntica en API, usando HIP/ROCm en lugar de CUDA.

---

## Tests V1 (1 CI + 5 `#[ignore]`)

| Test | Backend | Requiere | Criterio | Status |
|---|---|---|---|---|
| `both_backends_agree_with_cpu_n16` | wgpu | Ninguno (CI) | rel_err < 1e-3 | ✅ Pasa |
| `gpu_matches_cpu_direct_gravity_n1024` | CUDA | Hardware CUDA | rel_err < 1e-5 | 🔲 ignore |
| `gpu_speedup_over_cpu_serial_weak_scaling` | wgpu | GPU real | speedup > 5× | 🔲 ignore |
| `pm_gpu_roundtrip_fft` | CUDA/HIP | Hardware | err < 1e-8 | 🔲 ignore |
| `power_spectrum_pm_gpu_matches_pm_cpu` | CUDA/HIP | Hardware | P(k) err < 1% | 🔲 ignore |
| `energy_conservation_gpu_integrator_n256_100steps` | wgpu | GPU real | drift < 0.1% | 🔲 ignore |

---

## Para ejecutar los tests `#[ignore]`

```bash
# Tests GPU con hardware real (wgpu):
cargo test -p gadget-ng-gpu --test v1_gpu_tests -- --ignored --nocapture

# Solo el test CI-friendly:
cargo test -p gadget-ng-gpu --test v1_gpu_tests

# Con CUDA disponible (cuando los kernels estén implementados):
cargo build -p gadget-ng-cuda --features cuda
cargo test -p gadget-ng-gpu --test v1_gpu_tests -- --ignored gpu_matches_cpu_direct_gravity_n1024
```

---

## Plan de implementación del kernel CUDA

1. **Kernel de gravedad directa** (`cuda/direct_gravity.cu`):
   ```cuda
   __global__ void direct_gravity_kernel(
       float3* pos, float* mass, float3* acc, int n, float eps2, float G
   );
   ```
   Usar tiles de 256 hilos con shared memory para minimizar accesos globales.

2. **Integrar con `CudaDirectGravity::compute`**: reemplazar `todo!()` con llamada
   al kernel vía FFI.

3. **Activar los tests `#[ignore]`** una vez que el kernel esté disponible.

---

## Dependencias añadidas

`crates/gadget-ng-gpu/Cargo.toml` (dev-dependencies):
```toml
gadget-ng-cuda = { path = "../gadget-ng-cuda" }
gadget-ng-hip = { path = "../gadget-ng-hip" }
```
