# Fase 22 — TreePM Halo Volumétrico 3D Periódico

**Fecha:** 2026-04-17  
**Estado:** Implementado y validado (P=1, tests geométricos y físicos)

---

## Resumen ejecutivo

La Fase 22 introduce la infraestructura de **halo volumétrico 3D periódico** para el árbol de corto alcance del TreePM distribuido. El componente central es `exchange_halos_3d_periodic`, que reemplaza al halo 1D-z de Fase 21 para descomposiciones de dominio no-Z-slab, y corrige el bug de periodicidad del `exchange_halos_sfc` existente.

**Resultado:** 9 tests geométricos + 8 tests de validación física: todos pasan.

---

## 1. Diagnóstico: ¿por qué el halo 1D es suficiente para Z-slab?

### Prueba matemática

Sea un Z-slab con `z_lo ≤ z < z_hi`. Para una partícula fuera del slab, sea `Δz*` su distancia periódica al slab:

```
Δz* = min_image_scalar(z - z_center, L)  en z
```

Si `Δz* ≥ r_cut`, entonces **necesariamente** `d_3D ≥ |Δz*| ≥ r_cut`, por lo que la partícula no puede tener una interacción de corto alcance con ninguna partícula del slab.

**Conclusión:** Para Z-slab uniforme, el criterio `|Δz*| < r_cut` es **necesario y suficiente** para capturar todos los pares con `d_3D < r_cut`. El halo 1D-z de Fase 21 es físicamente correcto para Z-slab.

### ¿Cuándo falla el halo 1D-z?

El halo 1D-z **falla** cuando la descomposición de dominio no es puramente en Z. Ejemplo concreto con descomposición en 2 octantes:

- Rank 0 posee `[0, 0.5)³`
- Rank 1 posee `[0.5, 1)³`
- Partícula de rank 1 en `(0.95, 0.95, 0.95)`:
  - `z = 0.95 > z_lo_rank1 + r_cut = 0.5 + 0.1 = 0.6` → halo 1D-z **excluye** ✗
  - Distancia 3D periódica: `√(3 × 0.05²) ≈ 0.087 < r_cut = 0.1` → debería incluir ✓

**Interacción perdida:** el halo 1D-z omite una partícula que SÍ está dentro del radio de corte.

---

## 2. Bug en `exchange_halos_sfc`

La función `exchange_halos_sfc` en `mpi_rt.rs` (líneas ~334-346) computa AABBs, las expande por `halo_width`, y filtra partículas locales usando **coordenadas absolutas sin wrap periódico**:

```rust
// Código actual (con bug):
let expanded = aabb.expand(halo_width);
if expanded.contains(p.position) { ... }  // ← usa coordenadas absolutas
```

**Problema:** Una partícula en `x ≈ 1.0` no satisface `x < xhi_expanded` de un dominio con `xlo ≈ 0`, aunque la distancia periódica sea `< halo_width`.

La nueva función `exchange_halos_3d_periodic` implementa el criterio correcto:

```rust
min_dist2_to_aabb_3d_periodic(p, aabb_r, box_size) < halo_width²
```

Nota: `exchange_halos_sfc` se mantiene sin cambios para compatibilidad hacia atrás. El bug se documenta aquí y en los docstrings del código.

---

## 3. Arquitectura de la Fase 22

### Módulo nuevo: `crates/gadget-ng-parallel/src/halo3d.rs`

Implementa geometría periódica 3D independiente del crate `treepm`:

| Función / Tipo | Propósito |
|---|---|
| `Aabb3` | AABB rectangular genérica (`lo: [f64;3]`, `hi: [f64;3]`) |
| `min_dist2_to_aabb_3d_periodic` | Distancia mínima periódica punto→AABB rectangular |
| `minimum_image_scalar` | `minimum_image` escalar (evita dependencia circular) |
| `compute_aabb_3d` | AABB real de un slice de partículas |
| `aabb_to_f64` / `f64_to_aabb` | Serialización para allgather (6 f64 por AABB) |
| `is_in_periodic_halo` | Predicado de pertenencia al halo |

### Algoritmo de `min_dist2_to_aabb_3d_periodic`

```
CoM = center del AABB
half = half_extents del AABB

Para cada eje k:
    d_k = minimum_image(CoM[k] - p[k], box_size)
    excess_k = max(|d_k| - half[k], 0)

dist² = Σ excess_k²
```

Este algoritmo elige la imagen periódica del AABB más cercana a `p` en cada eje componente a componente. Es correcto para todos los 26 vecinos periódicos (todas las combinaciones de wrap en ±1 en cada eje).

### Método nuevo: `exchange_halos_3d_periodic`

Protocolo de comunicación:

```
1. compute_aabb_3d(local)                    → AABB real del rank actual
2. allgather_f64 de todas las AABBs          → 6 f64 × P
3. Para rank r ≠ self:
     envía p si min_dist2_to_aabb_3d_periodic(p, aabb_r, L) < r_cut²
4. alltoallv_f64                             → halos recibidos
```

Complejidad de comunicación: `O(N/P × P × 6)` para allgather AABBs + `O(N_halo)` para partículas. Para P=1, retorna `Vec::new()` (no-op).

### Nuevo flag de configuración

```toml
[gravity]
treepm_slab = true      # Fase 21: habilitar TreePM distribuido
treepm_halo_3d = true   # Fase 22: usar halo volumétrico 3D en vez de 1D-z
```

### Integración en `engine.rs`

```rust
let use_treepm_3d_halo = cfg.gravity.treepm_halo_3d && use_treepm_slab;

// En compute_acc:
let sr_halos = if use_treepm_3d_halo {
    rt.exchange_halos_3d_periodic(parts, box_size, r_cut)   // Fase 22
} else {
    rt.exchange_halos_by_z_periodic(parts, z_lo, z_hi, r_cut) // Fase 21
};
```

Los campos de diagnóstico añadidos en `HpcStepStats`:
- `halo_3d_particles: usize` — partículas halo recibidas por `exchange_halos_3d_periodic`
- `halo_3d_bytes: usize` — bytes de comunicación
- `halo_3d_ns: u64` — tiempo de la colectiva
- `path_active` extendido: `"treepm_slab_1d"` | `"treepm_slab_3d"`

---

## 4. Caso diagnóstico: halo 1D miss vs. halo 3D hit

El test `halo_1d_misses_diagonal_periodic` / `halo_3d_catches_diagonal_periodic` demuestra concretamente el gap:

| Partícula | Posición | Halo 1D-z (z_lo=0.5, r_cut=0.1) | Halo 3D | Distancia 3D periódica |
|---|---|---|---|---|
| Fuente | (0.95, 0.95, 0.95) | ✗ (z=0.95 > 0.6) | ✓ | 0.087 < r_cut=0.1 |

**El halo 1D omite la partícula; el halo 3D la captura.** Esta diferencia solo aparece con descomposición no-Z-slab.

---

## 5. Validación

### Tests geométricos (`crates/gadget-ng-treepm/tests/halo3d.rs`)

| Test | Descripción | Resultado |
|---|---|---|
| `min_dist2_to_aabb_3d_trivial` | Punto interior → d²=0 | ✓ |
| `min_dist2_to_aabb_3d_along_x` | x=0.95 vs AABB [0,0.5), L=1 → dist=0.05 | ✓ |
| `min_dist2_to_aabb_3d_along_y` | y=0.95 → dist=0.05 | ✓ |
| `min_dist2_to_aabb_3d_along_z` | z=0.95 → dist=0.05 | ✓ |
| `min_dist2_to_aabb_3d_diagonal_xyz` | (0.95,0.95,0.95) → dist²=0.0075 | ✓ |
| `halo_1d_misses_diagonal_periodic` | Halo 1D excluye partícula diagonal | ✓ |
| `halo_3d_catches_diagonal_periodic` | Halo 3D incluye partícula diagonal | ✓ |
| `compute_aabb_3d_correctness` | AABB de 5 partículas correcta | ✓ |
| `rectangular_aabb_periodic_distance` | AABB rectangular con y∈[0.4,0.6] | ✓ |

### Tests de validación física (`crates/gadget-ng-physics/tests/treepm_halo3d.rs`)

| Test | Descripción | Resultado |
|---|---|---|
| `halo3d_x_border_interaction` | x=0.01 interactúa con x=0.99 vía halo 3D | ✓ |
| `halo3d_y_border_interaction` | y=0.01 ↔ y=0.99 | ✓ |
| `halo3d_z_border_interaction` | z=0.01 ↔ z=0.99 | ✓ |
| `halo3d_diagonal_xyz_interaction` | (0.01,0.01,0.01) ↔ (0.99,0.99,0.99) | ✓ |
| `halo3d_vs_1d_uniform_slab_equivalent` | Z-slab: 1D y 3D incluyen mismo conjunto | ✓ |
| `halo3d_force_partition_erf_erfc` | F_sr(r<<r_split) ≈ Newton; erfc→0 para r=r_cut | ✓ |
| `cosmo_treepm_3d_halo_no_explosion` | N=27, 3 pasos, sin NaN/Inf | ✓ |
| `halo3d_no_double_counting` | erf+erfc=1; lattice simétrico → |F_central|≈0 | ✓ |

---

## 6. Comparación halo 1D vs. halo 3D

| Criterio | Halo 1D-z (Fase 21) | Halo 3D periódico (Fase 22) |
|---|---|---|
| Correctitud Z-slab uniforme | ✓ | ✓ |
| Correctitud octantes/SFC | ✗ (gap diagonal) | ✓ |
| Periodicidad en x, y | No verificada | ✓ (minimum_image por eje) |
| Periodicidad en z | ✓ | ✓ |
| Costo allgather | 0 | 6 × P f64 |
| Costo filtrado | O(N × 2) | O(N × P) |
| Envío redundante | Bajo | Posible (AABB sobreestima dominio) |

Para **Z-slab con P=4 y N=10⁶**: el allgather de AABBs cuesta 24 × P f64 = 96 bytes (negligible). El filtrado O(N×P) sube de O(N×2) a O(N×4), con impacto lineal en P.

---

## 7. Limitaciones documentadas

1. **Para Z-slab uniforme:** el halo 3D produce el mismo conjunto de partículas que el 1D. La diferencia solo aparece con descomposición no-Z-slab. La Fase 22 añade infraestructura sin cambio de física para el caso de uso actual.

2. **AABB sobreestimada:** `compute_aabb_3d` usa el AABB ajustado real de las partículas locales. Para distribuciones muy inhomogéneas (filamentos, halos densos), el AABB puede ser mucho mayor que el dominio SFC óptimo, resultando en envíos redundantes. Una mejora futura usaría los límites exactos del dominio SFC.

3. **PM sigue en Z-slab:** la descomposición dual PM=Z-slab + SR=SFC se deja para una fase posterior. Actualmente el PM largo alcance requiere Z-slab.

4. **`exchange_halos_sfc` no corregido:** el bug de periodicidad de la función original se documenta pero no se elimina, para no romper compatibilidad hacia atrás. Los nuevos paths usan `exchange_halos_3d_periodic`.

5. **No probado con MPI real:** los tests de validación son P=1. La correctitud con P>1 requiere ejecución MPI real (pendiente para una fase futura).

---

## 8. Roadmap hacia Fase 23+

- **Fase 23:** Migración de partículas SFC para el árbol de corto alcance (SR en SFC, PM en Z-slab). En este punto el halo 3D es crítico para correctitud: el gap del halo 1D-z se materializaría con interacciones entre dominios SFC adyacentes.
- **Optimización:** reemplazar AABB real por límites exactos del dominio SFC para reducir envíos redundantes.
- **Benchmark MPI:** comparar wall time `exchange_halos_by_z_periodic` vs. `exchange_halos_3d_periodic` con P=4, 8, 16 y N=10⁶.
- **Corrección de `exchange_halos_sfc`:** aplicar `min_dist2_to_aabb_3d_periodic` en la función existente (requiere deprecar la firma actual o añadir variante periódica).

---

## Archivos modificados / creados

| Archivo | Tipo | Descripción |
|---|---|---|
| `crates/gadget-ng-parallel/src/halo3d.rs` | **Nuevo** | Geometría 3D periódica: `Aabb3`, distancias, AABB real |
| `crates/gadget-ng-parallel/src/lib.rs` | Modificado | Exporta `halo3d`; añade `exchange_halos_3d_periodic` al trait |
| `crates/gadget-ng-parallel/src/serial.rs` | Modificado | No-op `exchange_halos_3d_periodic` para P=1 |
| `crates/gadget-ng-parallel/src/mpi_rt.rs` | Modificado | Implementación MPI con allgather AABBs + alltoallv periódico |
| `crates/gadget-ng-core/src/config.rs` | Modificado | Añade `treepm_halo_3d: bool` a `GravitySection` |
| `crates/gadget-ng-cli/src/engine.rs` | Modificado | Flag `use_treepm_3d_halo`, branch en `compute_acc`, nuevos campos diagnóstico |
| `crates/gadget-ng-treepm/tests/halo3d.rs` | **Nuevo** | 9 tests de geometría 3D periódica |
| `crates/gadget-ng-physics/tests/treepm_halo3d.rs` | **Nuevo** | 8 tests de validación física |
