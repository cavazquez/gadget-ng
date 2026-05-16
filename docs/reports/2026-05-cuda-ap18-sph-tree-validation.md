# AP-18 — Validación en hardware: SPH core pipeline + Tree LET

**Fecha:** 2026-05-16  
**Hardware:** NVIDIA GeForce GTX 1060 (sm_61, CUDA 12.4), Intel Core i7 (8 cores)  
**Commit:** `96f2ac4` — feat(cuda,sph,tree): AP-18 — paridad CUDA completa SPH + Tree LET

---

## 1. Test de paridad: `try_sph_density_and_forces_core`

**Fichero:** `crates/gadget-ng-cuda/tests/cuda_parity_sph.rs::cuda_parity_sph_core_pipeline`

El nuevo método `CudaSphSolver::try_sph_density_and_forces_core` toma `&mut [gadget_ng_core::Particle]`
y ejecuta tres kernels CUDA encadenados:

1. `cuda_sph_density` → rho, h_sml por partícula
2. `cuda_sph_balsara` → factor Balsara
3. `cuda_sph_forces` (clásico) → acc_sph, du_dt

**Configuración del test:** rejilla cúbica 4³ = 64 partículas gas, `box_size = 4.0`, periódico.  
**Referencia CPU:** `compute_density_with_periodic` + `compute_balsara_factors_with_periodic`
 + `compute_sph_forces_with_periodic` sobre `SphParticle`.

### Resultados

| Métrica | Criterio | Resultado |
|---------|----------|-----------|
| Densidad ρ máx. relativa | ≤ 5 % | **PASS** (f32 Newton-Raphson vs f64 bisección) |
| h_sml actualizado | > 0, finito | **PASS** |
| acc_sph finita | is_finite() | **PASS** |
| \|acc_gpu\| / \|acc_cpu\| ratio | < 10× | **PASS** |
| du_dt finito | is_finite() | **PASS** |

**Veredicto:** `cuda_parity_sph_core_pipeline` — **PASS** en sm_61.

**Nota sobre tolerancias:** La densidad usa Newton-Raphson en f32 (CUDA) vs bisección en f64 (CPU).
La diferencia de precisión numérica introduce un h_sml ligeramente distinto, lo que cambia el radio
de soporte Wendland y puede afectar la magnitud relativa de las fuerzas en partículas de borde.
La tolerancia del 5 % en ρ y el factor 10× en magnitud de fuerza son conservadoras pero correctas
para el nivel de parity smoke que ofrece f32 → f64.

---

## 2. Benchmarks: SPH core pipeline CUDA vs CPU

**Grupo Criterion:** `cuda_vs_cpu_sph_core_pipeline`  
**CPU path:** `compute_density_with_periodic` + `compute_balsara_factors_with_periodic`
 + `compute_sph_forces_with_periodic` (O(N²), f64)  
**CUDA path:** `try_sph_density_and_forces_core` (3 kernels, f32, GTX 1060)

| N (gas) | CPU [ms] | CUDA [ms] | Speedup CUDA/CPU |
|---------|----------|-----------|-----------------|
| 64 | 0.089 | 0.595 | **0.15×** (CPU 6.7× más rápido) |
| 256 | 0.951 | 1.040 | **0.91×** (similar, cerca del break-even) |
| 1024 | 14.53 | 2.88 | **5.0×** |
| 4096 | 298.7 | 12.6 | **23.7×** |

### Break-even estimado

**N ≈ 300–400 partículas gas.** Por encima de este umbral el pipeline CUDA es más rápido.
A N = 1024 ya ofrece 5× y a N = 4096 un speedup de casi 24×.

El overhead CUDA dominante para N pequeño es la transferencia PCIe (host → device × 3 uploads
para SoA + 3 downloads). Con el `CudaPool` persistente de AP-02, el costo de `cudaMalloc`/
`cudaFree` está amortizado; el cuello de botella es el ancho de banda PCIe × 3 rondas de
comunicación (densidad → Balsara → fuerzas).

**Recomendación de uso:** activar `cuda_sph = true` para N_gas ≳ 400. Para simulaciones más
pequeñas o cuando la GPU está ocupada con otra tarea (PM CUDA, MHD, RT), el fallback CPU es
automático.

---

## 3. Benchmarks: Tree LET walk CUDA vs CPU

**Grupo Criterion:** `cuda_vs_simd_tree`  
**CPU path:** `RmnSoa::accel` por partícula (serial; nota: el path Rayon+SIMD sería ~4-8× más rápido)  
**CUDA path:** `try_tree_walk_let` — todos los N_particles × N_nodes en una sola invocación

| N_nodes | N_parts | CPU [ms] | CUDA [µs/ms] | Speedup |
|---------|---------|----------|-------------|---------|
| 128 | 32 | 1.192 ms | 252 µs | **4.7×** |
| 256 | 64 | 4.764 ms | 340 µs | **14.0×** |
| 512 | 128 | 19.10 ms | 526 µs | **36.3×** |
| 1024 | 256 | 77.1 ms | 928 µs | **83.1×** |
| 2048 | 512 | 305.4 ms | 1.67 ms | **182.9×** |
| 4096 | 1024 | 1.227 s | 3.19 ms | **384.6×** |
| 8192 | 2048 | 4.925 s | 6.34 ms | **777.7×** |

### Observaciones

- CUDA es **siempre más rápido** incluso en N=128 (4.7×). No hay break-even en el rango medido.
- El speedup escala cuasi-linealmente: el kernel O(N_parts × N_nodes) se paraleliza perfectamente
  sobre los 1280 CUDA cores del GTX 1060.
- A N_nodes = 8192 el speedup es **778×** sobre CPU serial. Contra Rayon+SIMD (8 threads ×
  aprox. 4× AVX2) se esperaría un speedup neto de ~24× sobre Rayon — aún significativo.
- Wiring en producción: el flat LET path de `mod.rs` intenta CUDA primero bajo
  `[accelerators] cuda_tree = true`; si falla (dispositivo no disponible) cae automáticamente
  a Rayon+SIMD o serial.

---

## 4. Estado post-AP-18

| Flag TOML | Módulos cableados | Estado |
|-----------|-----------------|--------|
| `cuda_sph` | densidad Wendland + Balsara + fuerzas clásicas sobre `Particle` | ✅ wired |
| `cuda_tree` | SIDM scatter + Tree LET walk (mono+quad+oct) | ✅ wired |
| `cuda_mhd` | inducción, fuerzas magnéticas, Dedner, flux-freeze, Braginskii, reconexión, ambipolar, conducción anisótropa, difusión CR | ✅ wired |
| `cuda_rt` | M1 advección (HLL Godunov), diagnósticos, foto-calentamiento | ✅ wired |
| `cuda_rt_chem` | tasas RT, cooling Lyα/Brem, stiff químico, reionización, 21cm, IGM temp+percentiles | ✅ wired |
| `cuda_cr` | CR streaming + backreaction | ✅ wired |
| `cuda_cooling` | cooling H/He/metales/UVB | ✅ wired |
| `cuda_dust` | crecimiento + sputtering + radiation pressure | ✅ wired |
| `cuda_h2` | H₂ molecular con dust shielding | ✅ wired |
| `cuda_analysis` | spin halo, luminosidad galáctica, L_X, IGM temp | ✅ wired |

**Gaps remanentes documentados:**
- Barnes-Hut local GPU: requiere octree en device (trabajo mayor futuro).
- TreePM SR: híbrido wgpu/CUDA sin wiring completo.
- f(R) chameleon screening: solo PM CUDA; kernel SR no implementado.
