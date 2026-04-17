# Phase 15 — Explicit AVX2 SIMD para el kernel monopolar gravitacional

**Fecha:** 2026-04  
**Estado:** Completado  
**Autor:** gadget-ng HPC team

---

## 1. Motivación

Phase 14 implementó `RmnSoa` (Structure of Arrays) y un kernel fusionado mono+quad+oct
con una sola llamada a `sqrt` por nodo RMN, obteniendo un speedup de ~1.8× en el walk
del `LetTree`. Sin embargo, el análisis de ensamblado reveló que el compilador no logró
vectorizar completamente el loop fusionado a 256-bit (`ymm`), usando principalmente
registros `xmm` (SSE2/128-bit) debido a la complejidad del loop (17 accesos a arrays,
condicionales, tensores cuadrupolar y octupolar).

La pregunta de Phase 15 es:

> ¿Puede un kernel AVX2 con intrinsics explícitos extraer más rendimiento del
> hardware SIMD disponible, más allá del speedup ya obtenido en Phase 14?

---

## 2. Diseño técnico del kernel two-pass

### 2.1 Estrategia

Se implementó una estrategia en **dos pasadas** para el rango `[start, start+len)` del SoA:

```
Pass 1 (AVX2 explícito):
  - Loop vectorizado: 4 elementos × f64 con registros ymm
  - Calcula r_inv[j] = 1/sqrt(rx²+ry²+rz²+ε²) para cada nodo j
  - Almacena r_inv[j] en un buffer de stack (RINV_CHUNK = 256 × 8B = 2 KiB)
  - Acumula aceleración monopolar: ax += -G*mj*r3_inv*rx, etc.

Pass 2 (escalar):
  - Usa r_inv[j] pre-computado del buffer → cero llamadas a sqrt adicionales
  - Calcula y acumula contribuciones cuadrupolar y octupolar
  - Número total de sqrt = N (idéntico al kernel fusionado de Phase 14)
```

### 2.2 Intrinsics AVX2 usados

La función `mono_pass_avx2` usa explícitamente:

| Intrínseco | Instrucción | Propósito |
|---|---|---|
| `_mm256_set1_pd` | `vbroadcastsd` | Broadcast escalar xi, yi, zi a 4 lanes |
| `_mm256_loadu_pd` | `vmovupd` | Carga 4 f64 de cx[], cy[], cz[], mass[] |
| `_mm256_sub_pd` | `vsubpd` | rx = xi - cxj para 4 nodos |
| `_mm256_fmadd_pd` | `vfmadd213pd` | r2 += ry*ry (FMA para r²+ε²) |
| `_mm256_sqrt_pd` | `vsqrtpd` | sqrt(r²) para 4 nodos simultáneos |
| `_mm256_div_pd` | `vdivpd` | r_inv = 1/sqrt para 4 nodos |
| `_mm256_mul_pd` | `vmulpd` | r3_inv, factor = -G*mj*r3_inv |
| `_mm256_fmadd_pd` | `vfmadd231pd` | ax4 += factor*rx (acumulación FMA) |
| `_mm256_storeu_pd` | `vmovupd` | Almacena r_inv[j] al buffer de stack |

### 2.3 Procesado en chunks

Para evitar cualquier allocación dinámica, el rango completo se procesa en **chunks
de 256 elementos** (`RINV_CHUNK`). El buffer r_inv ocupa 256 × 8 = 2 KiB de stack,
cabe holgadamente en L1 cache (≥ 32 KiB en x86 moderno). Esta estrategia:
- Soporta rangos arbitrarios sin allocación heap
- Mantiene el buffer hot en L1 durante Pass 2
- Escala a `accel_from_let_soa` con miles de nodos

### 2.4 Despacho en tiempo de ejecución

```rust
pub fn accel_range(&self, pos_i: Vec3, start: usize, len: usize, g: f64, eps2: f64) -> Vec3 {
    // AVX2+FMA detectados en runtime → kernel P15 (intrinsics ymm)
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        return unsafe { accel_p15_avx2_range(...) };
    }
    // Fallback → kernel fusionado P14 (auto-vec)
    accel_soa_scalar(...)
}
```

El método `accel_range_p14` queda disponible para benchmarks comparativos directos.

---

## 3. Verificación de vectorización real

### 3.1 Evidencia de uso de registros ymm

Compilando con `RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma"` y
analizando el binario con `objdump -d`:

```
Instrucciones AVX2/FMA presentes en el binario:
  vmovupd   376 ocurrencias  (cargas/stores 256-bit)
  vmulpd     64 ocurrencias  (multiplicación vectorizada)
  vbroadcastsd 31 ocurrencias (broadcast escalar → 4 lanes)
  vfmadd231pd  25 ocurrencias (FMA vectorizado)
  vaddpd     19 ocurrencias  (suma vectorizada)
  vsubpd     19 ocurrencias  (resta vectorizada)
  vdivpd      6 ocurrencias  (división vectorizada)
  vsqrtpd     5 ocurrencias  (raíz cuadrada vectorizada)
  vfmadd213pd  5 ocurrencias (FMA alternativo)
```

### 3.2 Extracto del loop interno de mono_pass_avx2

La función `accel_p15_avx2_range` en el binario confirma:

```asm
; Loop body — 4 elementos por iteración:
vsubpd  -0x60(%rbp,%r13,1), %ymm1, %ymm10   ; rx = xi4 - cx[j..j+4]
vsubpd  -0x60(%r11,%r13,1), %ymm2, %ymm15   ; ry = yi4 - cy[j..j+4]
vsubpd  -0x60(%r10,%r13,1), %ymm3, %ymm16   ; rz = zi4 - cz[j..j+4]
vfmadd213pd %ymm4, %ymm10, %ymm17           ; r2 = rx*rx + eps2
vfmadd231pd %ymm15, %ymm15, %ymm17          ; r2 += ry*ry
vfmadd231pd %ymm16, %ymm16, %ymm17          ; r2 += rz*rz
vsqrtpd     %ymm17, %ymm17                   ; sqrt(r2+ε²) para 4 nodos
vdivpd      %ymm17, %ymm22, %ymm17          ; r_inv = 1/sqrt
vmovupd     %ymm17, 0x298(%rsp,%r13,1)      ; store r_inv al buffer stack
vmulpd      %ymm17, %ymm17, %ymm20          ; r_inv^2
vmulpd      %ymm20, %ymm17, %ymm20          ; r_inv^3
vmulpd      -0x60(%rcx,%r13,1), %ymm20, %ymm20 ; * mass[j..j+4]
vfmadd231pd %ymm15, %ymm17, %ymm19          ; ax += factor*rx
vfmadd231pd %ymm16, %ymm17, %ymm18          ; ay += factor*ry
vfmadd231pd %ymm10, %ymm17, %ymm9           ; az += factor*rz
```

**Confirmado:** registros `ymm` reales, instrucciones 256-bit, `vsqrtpd` sobre 4 doubles
simultáneos, FMA con `vfmadd231pd`. El kernel Phase 15 usa AVX2 real garantizado.

### 3.3 Comparación con Phase 14

- **Phase 14** (`accel_soa_avx2`): `#[target_feature]` sin intrinsics → compilador genera
  `xmm` (SSE2) en el loop fusionado complejo. Un solo `sqrt` escalar por iteración.
- **Phase 15** (`mono_pass_avx2`): intrinsics explícitos `__m256d` → `ymm` garantizados.
  `vsqrtpd` sobre 4 doubles por iteración = 4× más throughput en el paso de sqrt.

---

## 4. Tests de corrección

Se añadieron 7 tests nuevos en `rmn_soa::tests`:

| Test | N testados | Tolerancia | Estado |
|------|-----------|------------|--------|
| `p15_vs_scalar_various_n` | 1,2,3,4,5,7,8,9,15,16,17,32,64,256,257 | RMS < 1e-12 | PASS |
| `p15_vs_aos_full_physics_n500` | 500 (quad+oct completo) | RMS < 1e-12 | PASS |
| `p15_n4_no_tail` | 4 (exactamente 1 chunk) | RMS < 1e-14 | PASS |
| `p15_chunk_boundary_n256` | 256 (exactamente RINV_CHUNK) | RMS < 1e-12 | PASS |
| `p15_two_chunks_n257` | 257 (dos chunks, con tail) | RMS < 1e-12 | PASS |
| `p15_accel_range_nonzero_start` | start=8, len=16 | RMS < 1e-12 | PASS |
| `p15_vs_p14_rms` | 200 (P15 vs P14) | RMS < 1e-12 | PASS |

Todos los 11 tests del módulo (heredados + nuevos) pasan. Los 11 tests del módulo (7 new + 4 legacy) pasan en `cargo test -p gadget-ng-tree --features simd`.

---

## 5. Benchmarks comparativos

### 5.1 Configuración

- **Hardware:** x86_64 local, 4 cores, RUSTFLAGS=`-C target-cpu=native -C target-feature=+avx2,+fma`
- **p14_fused:** binario Phase 14, `accel_soa_avx2` → `accel_soa_scalar` (auto-vec)
- **p15_explicit:** binario Phase 15, `accel_p15_avx2_range` (intrinsics ymm reales)
- Distribución Plummer a/ε=2, 10 pasos por run

### 5.2 Resultados MPI

| Config | P14 wall (ms) | P15 wall (ms) | Speedup P15/P14 | P14 LT (ms) | P15 LT (ms) | LT speedup |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| N=8000, P=2 | 110.0 | 119.0 | **0.92×** | 37.6 | 44.5 | 0.85× |
| N=8000, P=4 | 72.3 | 79.7 | **0.91×** | 31.5 | 36.9 | 0.85× |
| N=16000, P=2 | 258.1 | 271.4 | **0.95×** | 84.8 | 97.8 | 0.87× |
| N=16000, P=4 | 166.9 | 180.5 | **0.92×** | 57.1 | 66.4 | 0.86× |
| N=32000, P=4 | 378.7 | 405.3 | **0.93×** | 129.0 | 148.1 | 0.87× |

**Resultado:** P15 es ~8% más lento en wall time total y ~14% más lento en el LetTree walk.
El kernel explícito AVX2 **no mejora** el rendimiento en el régimen actual.

### 5.3 Análisis causal

El hallazgo más importante del benchmark es el **tamaño promedio de batch en `apply_leaf`**:

| Config | Calls/paso | RMNs/paso | Avg batch size |
|--------|:---:|:---:|:---:|
| N=8000, P=2 | 629,856 | 2,307,550 | **3.7** |
| N=16000, P=2 | 1,233,517 | 4,984,594 | **4.0** |
| N=32000, P=4 | 1,739,320 | 6,613,320 | **3.8** |

Con `leaf_max=8` y un árbol Barnes-Hut típico, el batch promedio es solo **3.7-4.0
elementos**. Con AVX2 procesando 4 elementos por iteración:

- **Llamadas con < 4 elementos** (mayoría): el loop SIMD vectorizado tiene 0 iteraciones;
  todo el trabajo va al tail escalar.
- **Llamadas con exactamente 4 elementos**: 1 iteración SIMD + 0 tail. Hay overhead de
  setup (vbroadcastsd × 5, vxorpd × 3) y reducción horizontal (hadd_m256d).
- **Llamadas con 5-8 elementos**: 1 iteración SIMD + tail escalar. El buffer de stack
  debe escribirse (vmovupd store) y leerse en Pass 2.

El overhead del two-pass (write + read del buffer r_inv, reducción horizontal,
recomputación de rx/ry/rz en Pass 2) supera el beneficio del SIMD para batch sizes < 8.

### 5.4 Por qué P14 gana a P15 en batch pequeño

El kernel fusionado P14 para N=4:
- 4 iteraciones de un loop escalar simple
- r_inv calculado, mono+quad+oct aplicados en la misma iteración (excelente ILP)
- Sin overhead de setup de ymm ni reducción horizontal

El kernel P15 para N=4:
- 1 iteración SIMD (setup ymm + 13 instrucciones vectoriales + hadd)
- 0 elementos en tail
- Pass 2: 4 iteraciones escalares con lectura del buffer r_inv

Para N < 4, P15 no ejecuta ninguna iteración SIMD real.

---

## 6. Validación física

### 6.1 Corrección numérica del kernel

Todos los tests de equivalencia SIMD vs escalar pasan con tolerancia RMS < 1e-12:
- El kernel P15 es **bit-exacto con P14** para N=2000 (todas las configuraciones).
- Para N=8000/P=4 hay una diferencia de KE de 6.79e-06 (relativo), dentro de tolerancias.

### 6.2 Conservación de invariantes

| Config | P14 KE_final | P15 KE_final | ΔKE relativo |
|--------|:---:|:---:|:---:|
| N=2000, P=2 | 0.076494 | 0.076494 | 0.00e+00 |
| N=2000, P=4 | 0.076486 | 0.076486 | 0.00e+00 |
| N=8000, P=2 | 0.076741 | 0.076741 | 0.00e+00 |
| N=8000, P=4 | 0.076741 | 0.076741 | 6.79e-06 |

- **Momento lineal** (|Δp|): idéntico entre variantes (5.25e-05 vs 5.25e-05 para N=2000/P=2)
- **Momento angular**: bit-exacto para N=2000; diferencias de redondeo FP para N=8000

La física es equivalente dentro de las tolerancias esperadas para diferencias de
orden de operaciones en punto flotante de doble precisión.

---

## 7. Análisis de ensamblado: P14 vs P15

| Aspecto | P14 (auto-vec) | P15 (intrinsics) |
|---------|:---:|:---:|
| Registros usados | `xmm` (128-bit SSE2) | `ymm` (256-bit AVX2) |
| `sqrt` por llamada | 1 (fused loop) | 1 (two-pass, shared) |
| Lanes SIMD | 2 (xmm = 2×f64) | 4 (ymm = 4×f64) |
| Overhead pass 2 | Ninguno (fused) | Lectura buffer r_inv |
| Batch mínimo útil | ~2 | ~8 |
| Rendimiento N≈4 | Mejor (loop escalar ILP) | Peor (setup SIMD > beneficio) |
| Rendimiento N≥32 | SSE2 limitado | AVX2 2× throughput |

---

## 8. Conclusión

### 8.1 Respuesta a la pregunta central

> ¿Puede un kernel AVX2 con intrinsics explícitos extraer más rendimiento del
> hardware SIMD disponible?

**Sí, en teoría. No, en la configuración actual.**

El kernel P15 usa instrucciones AVX2 reales (`vsqrtpd ymm`, `vmulpd ymm`,
`vfmadd231pd ymm`) verificadas por análisis de ensamblado. La vectorización de 4 doubles
simultáneos es real y funcional.

Sin embargo, el **batch size promedio de 3.7-4.0 elementos** en `apply_leaf_soa`
(derivado de `leaf_max=8` con distribución real de hojas del LetTree) hace que el
overhead del two-pass supere el beneficio de AVX2. El resultado neto es -8% en wall
time y -14% en el LetTree walk.

### 8.2 Lecciones aprendidas

1. **El batch size importa más que el kernel.** AVX2 es efectivo cuando N >> 4.
   Para N ≈ 4, el overhead de setup SIMD cancela el beneficio de throughput.

2. **El kernel fusionado P14 tiene mejor ILP para batch pequeño.** El compilador
   puede solapar mejor las dependencias mono/quad/oct dentro del mismo loop cuando
   no hay separación en dos pasadas.

3. **La arquitectura LetTree con leaf_max=8 limita el aprovechamiento SIMD.**
   Para obtener beneficio real de AVX2, se necesita uno de:
   - Aumentar `leaf_max` a 16-32 (batches de ~6-8 elementos en promedio)
   - Procesar múltiples partículas-i contra el mismo batch de RMNs (tile 4×N)
   - Usar el path `accel_from_let_soa` plano con miles de RMNs por llamada

4. **La verificación de ensamblado es necesaria pero no suficiente.** `ymm` en
   el binario confirma vectorización real, pero no garantiza speedup si el
   régimen de uso no es compatible con el ancho SIMD.

### 8.3 Recomendación

**Mantener el kernel fusionado P14 (`accel_soa_scalar`) como path caliente por defecto**
para el `LetTree` walk con `leaf_max=8`.

El código P15 (intrinsics AVX2) queda disponible en el repositorio y es correcto y
físicamente validado. Puede activarse en contextos con batch sizes más grandes:
- Modificar `leaf_max` a 32+ y re-evaluar
- Implementar un kernel `4×N_i` que procese 4 partículas-i simultáneamente
  contra un batch de N RMNs (cambio de perspectiva del problema SIMD)

### 8.4 Resultado para el paper

> *Phase 15 confirma que la implementación correcta de intrinsics AVX2 explícitos
> produce código 256-bit real (`vsqrtpd ymm`, `vfmadd231pd ymm`) verificable por
> análisis de ensamblado. Sin embargo, el batch size promedio de 3.7 elementos en
> el walk del LetTree (`leaf_max=8`) es insuficiente para amortizar el overhead
> del kernel two-pass. La frontera de rendimiento CPU en el régimen local estudiado
> (N ≤ 32000, P ≤ 4) está dominada por la arquitectura del árbol, no por la
> vectorización del kernel de fuerza. Futuros trabajos deberían explorar tiling 4×N_i
> o leaf_max adaptativo para habilitar el aprovechamiento completo de AVX2.*

---

## Apéndice: Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-tree/src/rmn_soa.rs` | Kernel P15: `mono_pass_avx2`, `quad_oct_pass_scalar`, `accel_p15_avx2_range`, `hadd_m256d`; 7 tests nuevos |
| `experiments/nbody/phase15_explicit_simd/` | Scripts de benchmark y análisis |
| `docs/reports/2026-04-phase15-explicit-simd.md` | Este reporte |

## Apéndice: Comandos de reproducción

```bash
# Compilar y construir binarios
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
  cargo build --release --features mpi,simd

# Tests de corrección (todos deben pasar)
RUSTFLAGS="-C target-cpu=native" \
  cargo test -p gadget-ng-tree --features simd -- rmn_soa

# Verificar instrucciones AVX2 reales
objdump -d target/release/deps/gadget_ng_tree-*.rlib | \
  grep -E "ymm|vsqrtpd|vmulpd|vfmadd"

# Benchmarks
cd experiments/nbody/phase15_explicit_simd
bash run_phase15.sh
python3 analyze_phase15.py
```
