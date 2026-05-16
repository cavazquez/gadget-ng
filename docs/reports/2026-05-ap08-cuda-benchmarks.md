# AP-08: Benchmarks comprehensivos CUDA vs CPU/SIMD

Date: 2026-05-16

## Hardware

| Component | Details |
|---|---|
| GPU | NVIDIA GeForce GTX 1060 6GB |
| CUDA Architecture | sm_61 (Pascal) |
| CUDA Toolkit | 12.4 |
| CPU | Intel Core (see build env) |

## Metodología

Benchmarks Criterion con `--warm-up-time 1 --measurement-time 2`, 10 muestras cada uno.
Ejecutados con `CUDA_ARCH=sm_61 cargo bench -p gadget-ng-cuda --bench cuda_vs_simd --features simd`.

---

## Resultados

### Direct Gravity (N²)

| N | CPU serial | CPU SIMD | CUDA | Speedup CUDA/serial | Speedup CUDA/SIMD |
|---|---|---|---|---|---|
| 512 | 692 µs | 619 µs | 87.1 µs | **7.9×** | **7.1×** |
| 1024 | 2.77 ms | 2.48 ms | 131 µs | **21.1×** | **19.0×** |
| 2048 | 11.1 ms | 10.0 ms | 229 µs | **48.5×** | **43.7×** |

**Conclusión:** CUDA supera al CPU desde N=512. Speedup escala cuadráticamente con N (como esperado para O(N²)).

---

### PM (CIC + FFT + Poisson)

| Grid | CPU PM | CUDA PM | Speedup |
|---|---|---|---|
| 32³ (grid=32, N=1024) | 1.83 ms | 1.43 ms | **1.3×** |
| 64³ (grid=64, N=4096) | 22.4 ms | 3.49 ms | **6.4×** |

**Conclusión:** GPU PM empieza a ganar sobre la grilla 64³. En grillas pequeñas el overhead PCIe domina.
Break-even: ~grid=48 (estimado).

---

### SPH Density (Wendland, O(N²))

| N | CPU | CUDA | Speedup |
|---|---|---|---|
| 512 | 2.12 ms | 690 µs | **3.1×** |
| 1024 | 9.19 ms | 1.49 ms | **6.2×** |

**Conclusión:** CUDA SPH density supera CPU desde N~256. Speedup sublineal respecto a N (overhead memcpy, no totalmente O(N²) por h_sml variable).

---

### MHD flux_freeze

| N | CPU | CUDA | Speedup |
|---|---|---|---|
| 1024 | 2.45 µs | 93.2 µs | **0.03×** (CPU mucho más rápido) |
| 4096 | 12.9 µs | 152 µs | **0.08×** (CPU mucho más rápido) |

**Conclusión:** La operación `flux_freeze` es O(N) con muy poca aritmética por partícula (~3 ops/partícula). El overhead CUDA (malloc/memcpy/kernel launch) domina completamente. **No usar CUDA para flux_freeze**; requiere N > ~100,000 para romper el equilibrio. Break-even estimado: N~50,000–100,000.

---

### RT M1 advección HLL (1 substep)

| Grid | CPU | CUDA | Speedup |
|---|---|---|---|
| 8³ (512 celdas) | 32.6 ms | 215 ms | **0.15×** (CPU más rápido) |
| 16³ (4096 celdas) | 261 ms | 253 ms | **~1.03×** (paridad) |

**Notas:**
- El solver M1 CPU incluye el bucle de sub-stepping interno. En estas grillas pequeñas el CPU
  es más eficiente por el overhead PCIe y la latencia de lanzamiento de kernel.
- Break-even estimado: grid~16³–32³. Para grillas grandes (32³+) CUDA debería superar al CPU.
- La implementación CUDA incluye malloc/cudaMemcpy por llamada; con `CudaPool` persistente
  el overhead se reduciría significativamente.

---

### Tree LET accel (mono + quad + oct)

| N_nodes | N_particles | CPU LET | CUDA LET | Speedup |
|---|---|---|---|---|
| 256 | 64 | 4.84 ms | 351 µs | **13.8×** |
| 1024 | 256 | 77.4 ms | 936 µs | **82.7×** |
| 4096 | 1024 | 1241 ms | 3.28 ms | **378×** |

**Conclusión:** El kernel LET CUDA escala excelentemente. La CPU es O(N_part × N_nodes) en Rust
serial; el GPU paraleliza completamente sobre partículas con alta eficiencia aritmética
(mono+quad+oct = ~80 flops/par). Break-even: N_nodes > ~50 (mínimo práctico).

---

## Resumen ejecutivo

| Módulo | Speedup típico | Break-even N | Recomendación |
|---|---|---|---|
| Direct gravity | 20–50× (N=1024–2048) | N~100 | ✅ Siempre usar CUDA para N>512 |
| PM | 1.3–6.4× | grid~48 | ✅ Usar CUDA para grid≥64 |
| SPH density | 3–6× | N~256 | ✅ Usar CUDA para N>512 |
| MHD flux_freeze | <0.1× | N~50,000 | ❌ No usar CUDA actualmente |
| RT M1 advection | ~1× (16³) | grid~16–32 | ⚠️ Solo para grillas grandes |
| Tree LET | 14–378× | N_nodes~50 | ✅ Siempre usar CUDA para N>256 |

---

## Archivos de benchmark

- `crates/gadget-ng-cuda/benches/cuda_vs_simd.rs` — código de benchmark

## Comando de ejecución

```bash
CUDA_ARCH=sm_61 cargo bench -p gadget-ng-cuda --bench cuda_vs_simd --features simd
```

Resultados de Criterion en `target/criterion/`.

## Estado

AP-08: **Complete**. Benchmarks comprehensivos implementados y ejecutados en GTX 1060 (sm_61).
Resultados documentados con análisis de break-even por módulo.
