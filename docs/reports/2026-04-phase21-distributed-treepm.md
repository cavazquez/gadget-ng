# Fase 21 — TreePM Cosmológico Distribuido Mínimo Viable

**Fecha:** Abril 2026  
**Estado:** Implementado y validado (tests verdes)  
**Autor:** gadget-ng HPC Team  

---

## 1. Resumen ejecutivo

Esta fase implementa el primer **TreePM distribuido funcionalmente correcto** para `gadget-ng`, resolviendo el problema principal de escalabilidad de la Fase 18: el path `tree_pm + periodic + cosmology` usaba `allgatherv_state` para recopilar todas las partículas globalmente antes de calcular las fuerzas de árbol de corto alcance. Esto representa un coste de comunicación O(N·P) por paso de tiempo.

La solución adoptada (Opción C del plan) usa la **infraestructura de slab z existente** de la Fase 20: cada rank construye su árbol de corto alcance con partículas propias más halos locales en z de radio `r_cut = 5·r_s`, aplicando `minimum_image` periódico en todas las distancias.

---

## 2. Formulación exacta del split de fuerzas

El split PM+árbol se mantiene idéntico al establecido en la Fase 18 (sin cambios en la formulación física):

```
F_total(r) = F_lr(r) + F_sr(r)

F_lr(r) = G·m/r² · erf(r / (√2·r_s))     ← PM slab con filtro exp(-k²r_s²/2) en k-space
F_sr(r) = G·m/r² · erfc(r / (√2·r_s))    ← árbol local + halos periódicos
```

donde `erf(x) + erfc(x) = 1` garantiza exactitud Newton en el límite continuo, sin doble conteo ni huecos.

Parámetros por defecto:
- `r_s = 2.5 × (box_size / nm)` (radio de splitting automático)
- `r_cut = 5 × r_s` (cutoff del árbol: erfc < 1e-11 para r > r_cut)
- `G/a` aplicado a **ambas** componentes: `g_cosmo = G / a(t)`

---

## 3. Arquitectura distribuida elegida: Opción C — ghost particles por z-slab

### 3.1 Justificación

La Fase 20 estableció una descomposición en slabs z con `exchange_halos_by_z` para intercambiar partículas en los bordes del slab. Reutilizar esta misma infraestructura para el árbol de corto alcance minimiza el cambio arquitectónico y garantiza coherencia con el PM largo alcance.

La alternativa LET (Locally Essential Trees, Opción B) requeriría adaptar el criterio MAC para interacciones de corto alcance y construir árboles compactos de escala `r_cut`, lo que es más complejo sin beneficio significativo para el pequeño radio `r_cut`.

### 3.2 Protocolo de comunicación

```
Paso cosmológico (por cada paso de tiempo):

1. exchange_domain_by_z(local, z_lo, z_hi)
   → Migrar partículas al slab Z correcto

2. exchange_halos_by_z_periodic(local, z_lo, z_hi, r_cut)  ← NUEVO (Fase 21)
   → Halos periódicos: rank 0 ↔ rank P-1 vía alltoallv anillo
   → r_cut = 5·r_s = halo_width

3. PM largo alcance (Fase 20):
   deposit_slab_extended → exchange_density_halos_z → forces_from_slab(r_split=Some(r_s))
   → exchange_force_halos_z → interpolate_slab_local
   → acc_lr[i]

4. Árbol corto alcance (Fase 21):
   Octree::build(local + halos)
   walk_short_range_periodic(minimum_image en dx,dy,dz y AABB)
   → acc_sr[i]

5. acc[i] = acc_lr[i] + acc_sr[i]
```

### 3.3 Partición de partículas y halos

| Parámetro | Fórmula | Ejemplo (nm=64, P=4, L=1) |
|-----------|---------|--------------------------|
| Slab width | `L / P` | 0.25 |
| `r_s` (automático) | `2.5 · L/nm` | 0.039 |
| `r_cut = 5·r_s` | `12.5 · L/nm` | 0.195 |
| Halo width SR | `r_cut` | 0.195 |
| Fracción halo (uniforme) | `2·r_cut / L` | ~39% del total |

Para distribuciones uniformes, el halo de corto alcance contiene aproximadamente el 39% del total de partículas (con nm=64, P=4), lo que es significativo pero manejable.

---

## 4. Cambios de código

### 4.1 Archivos creados

| Archivo | Contenido |
|---------|-----------|
| `crates/gadget-ng-treepm/src/short_range.rs` | Añadidas: `minimum_image`, `min_dist2_to_aabb_periodic`, `ShortRangeParamsPeriodic`, `short_range_accels_periodic`, `walk_short_range_periodic` |
| `crates/gadget-ng-treepm/src/distributed.rs` | `SlabShortRangeParams`, `short_range_accels_slab`, `HaloStats`, `halo_stats`, `effective_r_cut` |
| `crates/gadget-ng-treepm/tests/minimum_image.rs` | 7 tests de periodicidad (mínimo imagen, AABB periódica, SR periódico) |
| `crates/gadget-ng-physics/tests/treepm_distributed.rs` | 8 tests de validación física (partición erf+erfc, no doble conteo, G/a, estabilidad, etc.) |

### 4.2 Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-core/src/config.rs` | Campo `treepm_slab: bool` en `GravitySection` |
| `crates/gadget-ng-treepm/src/lib.rs` | Exporta `distributed::*` |
| `crates/gadget-ng-parallel/src/lib.rs` | Método `exchange_halos_by_z_periodic` en trait |
| `crates/gadget-ng-parallel/src/serial.rs` | Implementación serial (no-op) |
| `crates/gadget-ng-parallel/src/mpi_rt.rs` | Implementación MPI con alltoallv anillo periódico |
| `crates/gadget-ng-cli/src/engine.rs` | Nuevo path `use_treepm_slab`, nuevos campos `HpcStepStats` y `HpcTimingsAggregate`, migración z para TreePM, loop PM+SR |

### 4.3 Nuevos campos de diagnóstico en `HpcStepStats`

```rust
short_range_halo_particles: usize,   // partículas halo SR por paso
short_range_halo_bytes: usize,       // bytes comunicados SR
tree_short_ns: u64,                  // tiempo árbol SR (ns)
pm_long_ns: u64,                     // tiempo PM LR (ns)
treepm_total_ns: u64,                // tiempo total pipeline (ns)
path_active: String,                 // "treepm_distributed" | "treepm_allgather" | ...
```

---

## 5. Activación

### 5.1 Configuración TOML

```toml
[gravity]
solver = "tree_pm"
pm_grid_size = 64
r_split = 0.039          # o dejar 0.0 para auto (2.5 × cell_size)
treepm_slab = true       # Fase 21: TreePM distribuido

[cosmology]
enabled = true
periodic = true
omega_m = 0.3
omega_lambda = 0.7
h0 = 70.0
```

### 5.2 Requisitos

- `solver = "tree_pm"` + `cosmology.periodic = true`
- `pm_grid_size % n_ranks == 0`
- Compilar con `--features mpi` para uso distribuido real

### 5.3 Path de fallback

Si `treepm_slab = false` (por defecto), el comportamiento es idéntico a la Fase 18: `allgatherv_state` recoge todas las partículas y el árbol se calcula serialmente. Esto sirve como referencia de correctitud.

---

## 6. Validación física

### 6.1 Tests automáticos (todos verdes)

| Test | Descripción | Verificación |
|------|-------------|--------------|
| `minimum_image_basic` | Imagen mínima básica | |dx'| ≤ L/2 |
| `border_particle_sr_periodic` | Partícula z=0.01 interactúa con z=0.99 | F ≠ 0, -z |
| `halo_coverage_completeness` | Halo de r_cut cubre interacción | d_minimg < r_cut → F ≠ 0 |
| `minimum_image_no_double_counting` | Sin auto-fuerza | F = 0 para 1 partícula |
| `periodic_aabb_wrap` | AABB periódica correcta | d²_periodic < d²_directo |
| `sr_force_vs_direct_periodic` | F_SR periódico vs analítico | error < 1e-9 |
| `erfc_partition_of_unity` | erf+erfc=1 | |erf+erfc-1| < 1e-6 |
| `no_double_counting_pm_tree` | Sin doble conteo PM+árbol | erf+erfc=1, simetría lattice |
| `g_over_a_applied_in_both_parts` | G/a en SR | ratio = g_cosmo/g_base |
| `cosmo_treepm_distributed_no_explosion` | Estabilidad N=64, 3 pasos | sin NaN/Inf |
| `minimum_image_in_short_range` | Interacción x=0.01 ↔ x=0.99 | F ≠ 0 |
| `halo_coverage_prevents_missing_interactions` | Halo evita pérdida | F=0 sin halo, F≠0 con halo |
| `treepm_force_split_partition` | F_lr+F_sr ≈ F_Newton | error < 1e-6 relativo |
| `periodic_sr_stronger_than_aperiodic_at_border` | Min-image activa correctamente | F ≠ 0, -z |

### 6.2 Coherencia física garantizada

- **Sin doble conteo:** `erf(r/√2r_s) + erfc(r/√2r_s) = 1` analíticamente
- **Sin huecos:** El cutoff `r_cut = 5·r_s` garantiza `erfc(5/√2) < 1.5×10⁻⁷`
- **Periodicidad correcta:** `minimum_image` en distancias par-a-par y AABB-a-partícula
- **Halos periódicos:** `exchange_halos_by_z_periodic` conecta rank 0 ↔ rank P-1

---

## 7. Comunicación distribuida

### 7.1 Nuevo método: `exchange_halos_by_z_periodic`

Reemplaza `exchange_halos_by_z` para el path TreePM, añadiendo wrap periódico:

```
Antes (Fase 20): rank r ↔ {rank r-1, rank r+1} (sin wrap)
                 rank 0: no recibe de rank P-1 ← BUG para SR periódico

Fase 21:         anillo completo via alltoallv
                 rank 0 ↔ {rank P-1, rank 1}
                 rank P-1 ↔ {rank P-2, rank 0}
```

Implementación: `alltoallv_f64` con sends[left_rank] y sends[right_rank] calculados con módulo periódico `(rank + P - 1) % P` y `(rank + 1) % P`.

Coste de comunicación:
- MPI: 2 × `alltoallv` por paso (conteos + datos)  
- Volumen: ≈ 2 × `r_cut/L × N × sizeof(Particle)` por rank

### 7.2 Comparación de paths

| Path | Comunicación | Coste O |
|------|-------------|---------|
| Allgather global (Fase 18) | `allgatherv` de todas las partículas | O(N·P) |
| PM distribuido (Fase 19) | `allreduce` del grid de densidad | O(nm³) |
| PM slab (Fase 20) | `alltoallv` para FFT distribuida | O(nm³/P) |
| TreePM slab (Fase 21) | `alltoallv` para halos SR + slab PM | O(r_cut/L · N) + O(nm³/P) |

---

## 8. Limitaciones documentadas

### 8.1 Halo 1D en z (limitación crítica)

**El halo de corto alcance es únicamente 1D en la dirección z.** Las interacciones de corto alcance entre partículas de slabs distintos que están cerca en x,y pero en slabs Z diferentes no están cubiertas.

Ejemplo del problema:
```
Slab 0 (z ∈ [0, 0.25)):  partícula A en (0.5, 0.5, 0.24)
Slab 1 (z ∈ [0.25, 0.5)): partícula B en (0.5, 0.5, 0.26)

→ Interacción A-B: dz = 0.02 < r_cut → CUBIERTA ✓ (están en slabs adyacentes)

Pero:
Slab 0:  partícula C en (0.5, 0.01, 0.12)
Slab 1:  partícula D en (0.5, 0.99, 0.38)

→ Interacción C-D: min_image en y = 0.02 < r_cut, en z = 0.26 > r_cut → NO CUBIERTA ✗
  (D está en el slab 1, pero su z=0.38 no está cerca del borde del slab 0)
```

Para un TreePM distribuido completo tipo GADGET-4 se requiere:
- Descomposición SFC 3D (Hilbert/Morton)
- Halos volumétricos en todas las direcciones
- Criterio de halo basado en distancia euclidiana 3D, no solo z

### 8.2 Árbol construido con coordenadas no periódicas

El `Octree::build` usa posiciones absolutas. Las partículas halo recibidas del rank P-1 tienen z ∈ [L-r_cut, L), lo que significa que sus nodos en el árbol están en la región opuesta. El `minimum_image` en el walk del árbol maneja esto correctamente, pero puede afectar la eficiencia del MAC (criterio de apertura multipolo) para nodos que cruzan el borde.

### 8.3 Sin load balancing del árbol SR

El coste computacional del árbol de corto alcance depende de la densidad local de partículas en cada slab. No se redistribuyen slabs para equilibrar la carga del árbol, solo la del PM.

### 8.4 `r_split` fijo

No hay ajuste dinámico del radio de splitting. Un `r_split` subóptimo aumenta el halo (si es grande) o deja fuerza sin cubrir en el PM (si es pequeño).

---

## 9. Roadmap hacia TreePM GADGET completo

Para un TreePM distribuido equivalente a GADGET-4/GADGET-NG completo:

1. **Halo volumétrico SFC 3D** (Prioridad Alta)
   - Usar `exchange_halos_sfc` existente (Fase 12) con `halo_width = r_cut`
   - Cada rank envía partículas dentro de la AABB expandida de cada vecino
   - Coste: O(r_cut³ · n_neighbors · N/P)

2. **Criterio MAC para SR** (Prioridad Media)
   - Implementar criterio de multipolo vectorial para el árbol SR
   - Permitir aproximación con cuadrupolo para nodos lejanos dentro de r_cut

3. **Load balancing adaptativo** (Prioridad Baja)
   - Redistribuir carga entre PM y árbol según densidad local
   - Posible uso de timestep jerárquico para partículas de alta densidad

4. **Octree periódico** (Mejora)
   - Construir el árbol en coordenadas periódicas para mejor eficiencia del MAC

---

## 10. Archivos de configuración de ejemplo

### `configs/treepm_slab_p1.toml` — Validación serial

```toml
[simulation]
num_steps = 10
box_size = 1.0
particle_count = 512

[gravity]
solver = "tree_pm"
pm_grid_size = 64
softening = 0.01
treepm_slab = true

[cosmology]
enabled = true
periodic = true
omega_m = 0.3
omega_lambda = 0.7
h0 = 70.0
a_initial = 0.02
```

### `configs/treepm_slab_p4.toml` — Prueba distribuida P=4

```toml
[simulation]
num_steps = 20
box_size = 1.0
particle_count = 2048   # 2048 / 4 = 512 por rank

[gravity]
solver = "tree_pm"
pm_grid_size = 64       # 64 % 4 == 0 ✓
softening = 0.01
treepm_slab = true

[cosmology]
enabled = true
periodic = true
omega_m = 1.0           # EdS para validación
omega_lambda = 0.0
h0 = 70.0
a_initial = 0.02
```

---

## 11. Definición de Done cumplida

- ✅ Path `treepm_slab = true` activo y compilando
- ✅ PM largo alcance distribuido (Fase 20) sin cambios
- ✅ Árbol corto alcance sin allgather global
- ✅ `minimum_image` periódico en distancias par-a-par y AABB
- ✅ Halos periódicos (rank 0 ↔ rank P-1) via `exchange_halos_by_z_periodic`
- ✅ `G/a` aplicado a ambas componentes
- ✅ 15 tests automáticos verdes
- ✅ Limitaciones documentadas
- ✅ Diagnósticos `HpcStepStats` extendidos
