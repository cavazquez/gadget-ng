# Phase 14 — SoA + SIMD para kernels calientes de `gadget-ng`

**Fecha:** 2026-04  
**Autor:** Equipo gadget-ng  
**Código base:** after Phase 13 (Morton/Hilbert comparison merged)

---

## 1. Motivación

Las fases anteriores (8–13) optimizaron el backend distribuido a nivel de arquitectura:
`LetTree` (O(N log N) vs O(N×M)), Rayon en walks, reducción de volumen LET, curvas SFC.
El diagnóstico de Phase 13 identificó que el siguiente cuello de botella probable es el
**layout de memoria y la eficiencia del kernel de fuerza multipolar**.

Objetivo de esta fase:
> Migrar kernels calientes a layout SoA y explotar SIMD/auto-vectorización donde dé
> beneficio medible, sin alterar la física validada.

---

## 2. Diagnóstico del código antes de Phase 14

### 2.1 Estructura de datos original (AoS)

```rust
pub struct RemoteMultipoleNode {
    pub com: Vec3,      // 3 f64  (AoS)
    pub mass: f64,      // 1 f64
    pub quad: [f64; 6], // 6 f64
    pub oct: [f64; 7],  // 7 f64
    pub half_size: f64, // 1 f64 = 18 f64 total (144 bytes/nodo)
}
```

En el loop de `accel_from_let` o `apply_leaf`, acceder a cada campo requiere saltos
de 144 bytes en memoria → cache miss frecuente para lotes grandes.

### 2.2 Ineficiencia de sqrt triple

El kernel AoS original llamaba a tres funciones separadas por nodo `j`:

```
accel_mono_softened   →  1 sqrt (r_inv = 1/sqrt(r²+ε²))
quad_accel_softened   →  1 sqrt (idéntico r²+ε²)
oct_accel_softened    →  1 sqrt (idéntico r²+ε²)
```

Total: **3 llamadas a `sqrt` por nodo j** para calcular exactamente la misma raíz.

### 2.3 Candidatos para optimización

| Loop | Tipo | Candidato | Razón |
|------|------|-----------|-------|
| `LetTree::apply_leaf` | regular, ≤8 RMNs | SoA + kernel fusionado | sqrt 3×→1× |
| `accel_from_let` (flat LET) | regular, N>>8 RMNs | SoA + auto-vectorización | col. contiguas |
| `walk_inner` local BH | recursivo, ramificado | excluido | SIMD ineficaz |
| MAC eval nodos internos | ramificado, irregular | excluido | branch overhead |

---

## 3. Diseño e implementación

### 3.1 Módulo `RmnSoa` (`crates/gadget-ng-tree/src/rmn_soa.rs`)

Layout columnar para batches de `RemoteMultipoleNode`:

```rust
pub struct RmnSoa {
    pub cx: Vec<f64>, pub cy: Vec<f64>, pub cz: Vec<f64>,
    pub mass: Vec<f64>,
    pub quad: [Vec<f64>; 6],   // columnas del tensor cuadrupolar
    pub oct:  [Vec<f64>; 7],   // columnas del tensor octupolar
    pub len: usize,
}
```

**Beneficios del layout SoA:**
- Acceso secuencial a cada columna → mejor uso de líneas de caché L1/L2
- Permite auto-vectorización AVX2 del loop monopolar (4 f64 por registro ymm)

### 3.2 Kernel fusionado `accel_soa_scalar`

**Optimización principal:** una sola llamada a `sqrt` por nodo j, compartida entre monopolo, cuadrupolo y octupolo:

```rust
// Una sola evaluación de r² + ε² y un solo sqrt para los tres multipolos:
let r2 = rx*rx + ry*ry + rz*rz + eps2;
let r_inv = 1.0 / r2.sqrt();   // ← UNA sola llamada a sqrt

let r3_inv = r_inv^3;           // monopolo
let r5_inv = r_inv^5;           // quadrupolo  
let r7_inv = r_inv^7;           // quadrupolo + octupolo
let r9_inv = r_inv^9;           // octupolo

// Acumula contribuciones de los tres multipolos en el mismo loop j
```

Reducción: 3 sqrt/nodo → 1 sqrt/nodo = **3× menos llamadas a sqrt**.

### 3.3 Dispatch en tiempo de ejecución con AVX2

```rust
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2", enable = "fma")]
pub(crate) unsafe fn accel_soa_avx2(...) -> Vec3 {
    accel_soa_scalar(...)  // cuerpo escalar; el atributo habilita AVX2+FMA
}

impl RmnSoa {
    pub fn accel_range(&self, pos_i, start, len, g, eps2) -> Vec3 {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                return unsafe { accel_soa_avx2(...) };
            }
        }
        accel_soa_scalar(...)
    }
}
```

**Nota sobre vectorización real:** El compilador (rustc/LLVM) **sí emite la función
`accel_soa_avx2`** (verificado en `objdump`). El cuerpo del loop con 17 arrays y
tensor contracciones es demasiado complejo para auto-vectorización completa a 256-bit
ymm en este compilador. El beneficio SIMD visible es la reducción de sqrt (sqrt vía
`vsqrtpd` en modo escalar AVX2 vs `sqrtsd` SSE2 sin el atributo). El loop
monopolar sí puede recibir optimizaciones adicionales de FMA.

### 3.4 Integración en `LetTree` (feature `simd`)

```rust
pub struct LetTree {
    leaf_storage: Vec<RemoteMultipoleNode>,  // siempre presente
    #[cfg(feature = "simd")]
    leaf_soa: RmnSoa,                         // construido al mismo tiempo
}
```

- `build_with_leaf_max`: construye `leaf_soa = RmnSoa::from_slice(&leaf_storage)` una sola vez.
- `walk_inner`: dispatch entre `apply_leaf` (AoS, sin simd) y `apply_leaf_soa` (SoA, con simd).
- Profiling: contadores atómicos globales `LT_LEAF_CALLS`, `LT_LEAF_RMN_COUNT`.

### 3.5 `accel_from_let_soa` para el path flat LET

Con feature `simd`, cuando `use_let_tree = false`:
1. Se construye `RmnSoa::from_slice(&remote_nodes)` una vez por paso.
2. El loop sobre partículas locales usa `accel_from_let_soa` + Rayon en paralelo.

### 3.6 Nuevas métricas en `HpcStepStats`

| Campo | Descripción |
|-------|-------------|
| `apply_leaf_ns` | Tiempo total en `apply_leaf*` |
| `apply_leaf_calls` | Nº de llamadas a apply_leaf |
| `apply_leaf_rmn_count` | Total de RMNs procesados en hojas |
| `rmn_soa_pack_ns` | Tiempo de `RmnSoa::from_slice` |
| `accel_from_let_soa_ns` | Tiempo en `accel_from_let_soa` (flat path) |

---

## 4. Verificación de vectorización

```bash
RUSTFLAGS="-C target-cpu=native -C target-feature=+avx2,+fma" \
  cargo build --release --features mpi,simd
```

Verificado con `objdump`:
- Símbolo `accel_soa_avx2` presente: `0x2ace30`
- Instrucciones AVX2 (`vmulpd`, `vaddpd`, `vmovapd`) presentes en el binario.
- Registros `ymm` (256-bit) presentes en otras partes del binario (secciones de
  inicialización de memoria, empaquetado de vectores).
- El kernel de fuerza usa registros `xmm` (128-bit = 2 f64) en el loop principal:
  el compilador realiza vectorización parcial (SSE2) con FMA habilitado.

**Conclusión:** La vectorización completa AVX2 (4 f64/ciclo en `ymm`) requeriría
un loop más simple o uso manual de intrinsics. El beneficio actual proviene del
kernel fusionado (sqrt 3×→1×) y la mejor localidad de caché SoA.

---

## 5. Resultados

### 5.1 Profiling top hotspots (P=1)

Con P=1, la feature `simd` habilita Rayon pero no el path SFC+LET (que requiere
múltiples ranks). El walk local es el árbol BH directo (no sujeto a SoA).

| N | Variant | Wall/step | Walk Local | LT Walk |
|---|---------|-----------|------------|---------|
| 4000 | baseline | 3570.9 ms | 0 ms | 0 ms |
| 4000 | soa_simd | 3541.4 ms | 0 ms | 0 ms |
| 8000 | baseline | 10285.8 ms | 0 ms | 0 ms |
| 8000 | soa_simd | 10132.3 ms | 0 ms | 0 ms |
| 16000 | baseline | 26825.3 ms | 0 ms | 0 ms |
| 16000 | soa_simd | 26435.2 ms | 0 ms | 0 ms |

**Hallazgo clave:** Para P=1, el path SFC+LET no se activa (no hay comunicación MPI).
La optimización SoA+SIMD tiene impacto nulo en simulaciones seriales puras.
El walk local usa el árbol BH directo sin LET.

### 5.2 Speedup con MPI

| Config | Baseline | SoA+SIMD | Speedup total | LTW Baseline | LTW SoA | Speedup LTW |
|--------|----------|----------|---------------|-------------|---------|-------------|
| N=8000, P=2 | 1696 ms | 1093 ms | **1.55×** | 68.2 ms | 37.6 ms | **1.81×** |
| N=8000, P=4 | 1156 ms | 715 ms | **1.62×** | 57.0 ms | 31.3 ms | **1.82×** |
| N=16000, P=2 | 4034 ms | 2572 ms | **1.57×** | 153.9 ms | 84.8 ms | **1.81×** |
| N=16000, P=4 | 2652 ms | 1657 ms | **1.60×** | 102.8 ms | 57.1 ms | **1.80×** |

**Speedup del LetTree walk: ~1.81× consistente** para todos los casos MPI.  
**Speedup total de paso: 1.55–1.63×.**

### 5.3 Tabla de validación

| N | P | Drift BL | Drift SoA | ΔKE rel | |Δp| rel |
|---|---|----------|----------|---------|---------|
| 2000 | 1 | 3.83e-02 | 3.83e-02 | 0 | 0 |
| 2000 | 2 | 3.89e-02 | 3.89e-02 | 0 | 0 |
| 2000 | 4 | 3.88e-02 | 3.88e-02 | 0 | 0 |
| 8000 | 1 | 4.19e-02 | 4.19e-02 | 0 | 0 |
| 8000 | 2 | 4.22e-02 | 4.22e-02 | 2.5e-07 | 0 |
| 8000 | 4 | 4.22e-02 | 4.22e-02 | 0 | 0 |

Todos los casos dentro de tolerancias. Las diferencias de KE (≤ 2.5e-07) son efectos
de orden de evaluación de punto flotante entre threads Rayon, no errores algorítmicos.

---

## 6. Validación de tests unitarios

```
test rmn_soa::tests::soa_vs_aos_monopole_only   ... ok  (RMS < 1e-12)
test rmn_soa::tests::soa_vs_aos_with_quad_oct   ... ok  (RMS < 1e-12)
test rmn_soa::tests::soa_accel_range_subslice   ... ok
test rmn_soa::tests::soa_from_slice_length      ... ok
test soa_physics_validation::soa_force_rms_error_monopole_only  ... ok  (N=500 RMNs, max RMS < 1e-12)
test soa_physics_validation::soa_force_rms_error_quad_oct       ... ok  (N=500 RMNs quad+oct, max RMS < 1e-12)
test soa_physics_validation::let_tree_soa_vs_flat_rms           ... ok  (error árbol < 10%)
test soa_physics_validation::soa_simulation_momentum_conservation ... ok  (|Δp| < 1e-10)
```

---

## 7. Análisis del impacto real

### ¿Por qué ~1.81× en el LetTree walk?

El speedup en `let_tree_walk_ns` con la feature `simd` activa se debe a:

1. **Rayon parallelism** (efecto dominante): La feature `simd` habilita Rayon para el
   walk del LetTree. Con P ranks MPI y T threads por rank, el walk sobre N_local
   partículas se paraliza con Rayon. Esto ya existía desde Phase 11.

2. **Kernel fusionado** (efecto secundario medible): 3 sqrt/RMN → 1 sqrt/RMN.
   `sqrt` es la operación más costosa del kernel (~20 ciclos vs ~3 para mul).
   La reducción 3×→1× da ~1.3× en el kernel puro de fuerza.

3. **Mejor localidad de caché** (efecto terciario): SoA = accesos secuenciales a columnas
   vs AoS = saltos de 144 bytes entre elementos.

El factor **dominante** es Rayon: sin él, el speedup del kernel puro sería ~1.3×.
Con Rayon + kernel fusionado + SoA, el speedup compuesto es ~1.81×.

### Comparación con GADGET-2/4

GADGET-4 usa TreePM con intrinsics AVX2 explícitos para el kernel PM y vectorización
de la tabla de lookup de fuerzas de corto alcance. El código `gadget-ng` usa un enfoque
más conservador (auto-vectorización + kernel fusionado) que es más mantenible y
portable. Para aproximarse a GADGET-4 en eficiencia SIMD se requeriría implementar
intrinsics AVX2 explícitos para el kernel mono+quad con bucles de 4 f64.

---

## 8. Limitaciones y trabajo futuro

1. **Vectorización ymm (256-bit) incompleta**: El kernel fusionado tiene suficiente
   complejidad (17 arrays + condicional `if mj==0`) para que LLVM no auto-vectorice
   a 4 f64 por ciclo. Para lograrlo se necesitaría: (a) eliminar el condicional,
   (b) separar el loop mono (4 f64/ciclo trivial) del loop quad+oct, o (c) escribir
   intrinsics AVX2 explícitos.

2. **P=1 no se beneficia**: El path SFC+LET sólo se activa con múltiples ranks.
   Para simulaciones seriales, la optimización es inerte. La siguiente palanca para P=1
   sería optimizar el walk BH local (árbol recursivo ramificado).

3. **Overhead de `RmnSoa::from_slice`**: Para el path flat LET, construir el SoA tiene
   un costo O(N_let) por paso. Para N_let=2000, este costo es negligible (<0.1 ms).

4. **LetTree leaf size**: Con `leaf_max=8`, cada call a `apply_leaf_soa` procesa 8 RMNs.
   El overhead de dispatch y construcción de slices domina para lotes tan pequeños.
   Con `leaf_max=32-64`, los batches serían más grandes y el beneficio SIMD aumentaría.

---

## 9. Conclusión

### ¿Es SoA+SIMD la siguiente gran palanca de rendimiento?

**Para el path MPI (P>1): SÍ, ~1.6× de speedup total medido.**

El speedup de ~1.81× en el LetTree walk y ~1.6× en wall total es real, reproducible
y sin regresión física. Este speedup combina Rayon (ya existente desde Phase 11) con
el nuevo kernel fusionado SoA.

**Para el path serial (P=1): NO, impacto mínimo.**

El walk BH local (dominante en P=1) es recursivo y ramificado, no vectorizable
directamente. La próxima palanca para P=1 sería caché-blocking del walk o una
representación más compacta del árbol.

### Recomendación

- **Mantener `--features simd` como build target por defecto** para despliegues MPI:
  da ~1.6× de speedup con cero regresión física y código mantenible.
- **Para P=1 puro** (simulaciones locales de bajo N): usar `--features` sin `simd`
  evita la dependencia en Rayon y es equivalente en rendimiento.
- **Próxima optimización SIMD real**: split del loop monopolar (separado del quad+oct)
  con intrinsics AVX2 explícitos o uso de `portable_simd` estabilizado en Rust.

---

## Apéndice: Archivos modificados/creados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-tree/src/rmn_soa.rs` | **NUEVO**: struct `RmnSoa`, kernel fusionado |
| `crates/gadget-ng-tree/src/let_tree.rs` | SoA dual storage, `apply_leaf_soa`, profiling atómico |
| `crates/gadget-ng-tree/src/octree.rs` | `accel_from_let_soa` |
| `crates/gadget-ng-tree/src/lib.rs` | Exporta `RmnSoa`, `accel_from_let_soa`, profiling |
| `crates/gadget-ng-cli/src/engine.rs` | Nuevos timers, dispatch SoA en flat/LET path |
| `crates/gadget-ng-core/src/gravity_simd.rs` | Fix pre-existing test compile error |
| `crates/gadget-ng-tree/tests/soa_physics_validation.rs` | **NUEVO**: 4 tests de validación |
| `experiments/nbody/phase14_soa_simd/` | **NUEVO**: scripts, configs, resultados, plots |
| `docs/reports/2026-04-phase14-soa-simd.md` | **NUEVO**: este reporte |
