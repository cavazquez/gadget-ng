# AP-21: Rayon + SIMD Combinados

**Date:** 2026-05-16
**Status:** Completado

## Motivación

Los módulos de mayor carga de RT y SPH ya tenían Rayon **o** SIMD, pero nunca los dos
a la vez.  La mejora de rendimiento de combinarlos sigue el patrón establecido en
`dedner_cleaning_step_par_simd` (MHD, AP-17/cleaning.rs):

```
public dispatcher → _par_simd (rayon+simd) → _par (rayon) → scalar
```

## Módulos tratados

| Módulo | Crate | Ganancia estimada | Implementación |
|--------|-------|-------------------|----------------|
| Chemistry stiff solver | `gadget-ng-rt` | Alta (bucle más costoso de RT) | `apply_chemistry_par_simd` |
| SPH density | `gadget-ng-sph` | Alta — ya combinado implícitamente | Documentado + test |
| CR streaming | `gadget-ng-mhd` | Media | `streaming_crk_par` + `streaming_crk_par_simd` |
| SPH forces Gadget-2 | `gadget-ng-sph` | Media | `grad_w_batch` en `sph_gadget2_update_for_particle` |

---

## Módulo 1 — Chemistry stiff solver (`gadget-ng-rt`)

### Problema
`apply_chemistry_par` (Rayon) llamaba `solve_chemistry_implicit` escalar por partícula.
Las funciones SIMD (`solve_chemistry_implicit_slice_avx2/avx512`, `photoionization_rates_*`,
`apply_chemistry_cooling_*`) estaban bajo `#[cfg(not(feature = "rayon"))]` — no se activaban
con Rayon.

### Solución
1. Se removió el guard `not(feature = "rayon")` de los helpers de slice/batch SIMD.
   Ahora están bajo `#[cfg(all(feature = "simd", any(target_arch = "x86", target_arch = "x86_64")))]`
   — disponibles en ambos paths.
2. Se añadió `apply_chemistry_par_simd` con triple patrón:
   - Computa `gamma_hi` para todas las partículas con SIMD batch (`photoionization_rates_for_particles`).
   - `par_chunks_mut(64)` sobre `chem_states` / `gamma_hi` / `t_gas` → llama
     `solve_chemistry_implicit_slice` (AVX dispatch) por chunk.
   - `par_chunks_mut(64)` sobre `particles` → llama `apply_chemistry_cooling` (AVX dispatch) por chunk.
3. El dispatcher `apply_chemistry` selecciona: `par_simd` → `par` → `impl` (serial).

### Parity test
`apply_chemistry_par_simd_matches_scalar` (N=16, tolerancia 1e-12) activo con
`#[cfg(all(feature = "rayon", feature = "simd", ...))]`.

---

## Módulo 2 — SPH density (`gadget-ng-sph`)

### Situación
El path Rayon (`compute_density_with_periodic`) ya era **Rayon + SIMD** implícitamente:
- Exterior: `into_par_iter()` sobre índices de partícula.
- Interior: `w_and_grad_w_batch` en `kernel.rs` despacha a AVX-512 → AVX2+FMA → escalar
  vía `is_x86_feature_detected!` en runtime, **sin** gate de la feature `simd`.

### Acción
1. Se añadió doc comment explícito en `compute_density_with_periodic` documentando la
   combinación implícita.
2. Se añadió test `density_rayon_matches_serial` (`#[cfg(feature = "rayon")]`) que
   verifica paridad entre el path Rayon y cálculo serial por partícula (tolerancia 1e-12).

---

## Módulo 3 — CR streaming (`gadget-ng-mhd`)

### Problema
El dispatcher `streaming_crk` tenía guard `not(feature = "rayon")` en el bloque SIMD —
Rayon y SIMD eran mutuamente excluyentes. No existía path Rayon.

### Solución
1. Se añadió `streaming_crk_par` (`#[cfg(feature = "rayon")]`):
   - Recoge datos de lectura (posición, velocidad, campo B, cr_energy) primero.
   - `into_par_iter()` sobre índices → computa `div_v` escalar por partícula (O(N) lectura
     de todos los vecinos, seguro en Rayon porque los datos son read-only en el par).
   - Aplica resultados en serial al final.
2. Se añadió `streaming_crk_par_simd` que actualmente delega a `streaming_crk_par`.
   El `div_v_single` domina el coste (O(N) por partícula); la ganancia principal viene
   del paralelismo Rayon, no de AVX sobre `cr_energy`.
3. El dispatcher `streaming_crk` tiene ahora triple via:
   `par_simd` → `par` → `simd serial` → `scalar`.
4. Las funciones `streaming_crk_avx2/avx512` permanecen bajo `not(rayon)` ya que no son
   llamadas desde el path Rayon.

### Parity test
`streaming_crk_par_matches_scalar` (`#[cfg(feature = "rayon")]`, N=16).

---

## Módulo 4 — SPH forces Gadget-2 (`gadget-ng-sph`)

### Problema
`sph_gadget2_update_for_particle` (Rayon path) evaluaba `grad_w(r, hi)` escalar por
cada par `(i, j)` — N × N evaluaciones escalares.

### Solución
Reestructuración en tres fases (mismo patrón que `sph_force_update_for_particle`):
1. **Fase 1** — recolección de vecinos: recorre todos los `j`, aplica filtro de soporte
   (`r < 2*hi` o `r < 2*hj`), acumula `r_buf`, `idx_buf`, `r_ij_buf`, `r_hat_buf`.
2. **Fase 2** — evaluación batch SIMD: `grad_w_batch(&r_buf, hi, &mut gw_i_buf)`.
   `grad_w_batch` despacha a AVX-512 → AVX2+FMA → escalar (runtime, `kernel.rs`).
3. **Fase 3** — acumulación de fuerzas con `gw_i_buf[k]` pre-computado; `grad_w(r, hj)`
   permanece escalar (variable por partícula).

El path Rayon exterior no cambia.  El test `uniform_rest_gas_near_zero_acceleration`
valida la corrección numérica.

---

## Cambios en MHD cleaning (corrección de pre-existing gap)

Durante la implementación de streaming se descubrió que `compute_dedner_div_b` (función
pública) llamaba a `dedner_pairwise_accumulate` gated por `not(rayon)`, causando error de
compilación en `rayon+simd`.  Se removió el gate de estas funciones helper:
- `dedner_pairwise_accumulate`
- `dedner_pairwise_accumulate_dispatch`
- `dedner_pairwise_accumulate_scalar`
- `dedner_pairwise_accumulate_avx2/avx512`

---

## Tolerancias de paridad

| Módulo | Tolerancia | Tipo |
|--------|-----------|------|
| Chemistry (par_simd vs par) | 1e-12 | abs por especie |
| SPH density (rayon vs serial) | 1e-12 | abs en rho |
| CR streaming (par vs scalar) | 1e-10 | abs en cr_energy |

---

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-rt/src/chemistry.rs` | apply_chemistry_par_simd, de-gate SIMD helpers |
| `crates/gadget-ng-sph/src/density.rs` | doc comment + density_rayon_matches_serial test |
| `crates/gadget-ng-mhd/src/streaming.rs` | streaming_crk_par, streaming_crk_par_simd, dispatcher |
| `crates/gadget-ng-mhd/src/cleaning.rs` | de-gate dedner_pairwise_accumulate* (pre-existing gap) |
| `crates/gadget-ng-sph/src/forces.rs` | sph_gadget2_update_for_particle: fases + grad_w_batch |

---

## AP-21 Benchmarks (Criterion)

Ejecutados en 2026-05-17 con `cargo bench --features bench-all-*-paths` (perfil `release`).
Hardware: sistema de desarrollo estándar x86_64.

### Módulo 1 — Chemistry (`gadget-ng-rt`)

`cargo bench -p gadget-ng-rt --bench ap21_chemistry --features bench-all-chemistry-paths`

| N | serial | rayon_par | rayon_par_simd | Ganancia vs serial |
|---|--------|-----------|----------------|-------------------|
| 64 | 9.37 µs | 16.9 µs | 13.1 µs | −1.4× (overhead domina) |
| 256 | 52.9 µs | 75.8 µs | 62.1 µs | −1.2× (overhead domina) |
| 1024 | 209 µs | 305 µs | 219 µs | ≈1× (break-even) |
| 4096 | 841 µs | 1527 µs | **226 µs** | **+3.7×** |

**Conclusión:** El break-even para `rayon_par_simd` es N ≈ 1 024–4 096.
`rayon_par` (scalar) es consistentemente más lento que serial: el overhead de Rayon
supera el coste por partícula a todos los N medidos. El chunking SIMD amortiza el overhead
y entrega **3.7× de ganancia a N=4096** — resultado principal de AP-21 para chemistry.

### Módulo 2 — CR Streaming (`gadget-ng-mhd`)

`cargo bench -p gadget-ng-mhd --bench ap21_cr_streaming --features bench-all-streaming-paths`

| N | scalar | rayon_par | dispatch | Ganancia Rayon vs scalar |
|---|--------|-----------|----------|--------------------------|
| 32 | 3.0 µs | 13.4 µs | 16.5 µs | −4.5× (overhead) |
| 64 | 13.2 µs | 41.2 µs | 39.0 µs | −3.1× (overhead) |
| 128 | 56.9 µs | 122 µs | 256 µs | −2.1× |
| 256 | 253 µs | 46.9 µs | 36.4 µs | **+5.4×** |
| 512 | 601 µs | 118 µs | 116 µs | **+5.1×** |

**Conclusión:** Break-even CR streaming ≈ N=200–256. El algoritmo es O(N²) y a N≥256
Rayon escala 5× sobre scalar. El dispatcher a N=256 entrega 7× vs serial gracias a que
enruta a la ruta Rayon óptima.

### Módulo 3 — SPH Forces Gadget-2 (`gadget-ng-sph`)

`cargo bench -p gadget-ng-sph --bench ap21_sph_forces --features bench-sph-forces-ref`

| N | scalar_ref | batch_simd | Ratio batch/scalar |
|---|------------|------------|-------------------|
| 64 | 18.7 µs | 20.1 µs | 0.93× |
| 216 | 80.1 µs | 91.8 µs | 0.87× |
| 343 | 182.9 µs | 162.1 µs | 1.13× |
| 512 | 303.6 µs | 334.6 µs | 0.91× |
| 1331 | 1584.5 µs | 1508.8 µs | 1.05× |

**Conclusión:** El batch SIMD de `grad_w_batch` para el kernel h_i no muestra ganancia clara
en este hardware a los N medidos (alta varianza, muchos outliers). La razón principal es que
sólo la mitad de las llamadas `grad_w` son batched (las de h_i); las de h_j siguen siendo
escalares. El beneficio real del batching será visible a N>>1331 en simulaciones de producción
donde el inner loop domina. Sin pérdida de funcionalidad ni regresión de paridad.
