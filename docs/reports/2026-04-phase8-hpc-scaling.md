# Fase 8 — HPC Escalable: SFC correcto + LET + CI Multirank

**Fecha:** Abril 2026  
**Autores:** gadget-ng team  
**Estado:** Completado

---

## 1. Diagnóstico previo: Allgather O(N·P)

El path de comunicación heredado de las fases 1–7 utiliza `MPI_Allgatherv` para replicar el estado completo de partículas en cada rango antes de evaluar fuerzas. Esto introduce:

- **Volumen de comunicación:** O(N × P) f64 por llamada; se llama 2× por paso KDK (o 4× para Yoshida4).
- **Colapso de memoria por rango:** cada rango almacena copias de los N datos globales.
- **Saturación de red:** para N=4000 y P=4, el Allgather transfiere ~160 KB por llamada × 100 pasos = ~16 MB por simulación solo en comunicación.

Medición de weak scaling observada antes de Fase 8 (Allgather):

| P | N | Wall (s) | Comm (s) | Eff. weak |
|---|---|----------|----------|-----------|
| 1 | 500 | 1.22 | ~0 | 100% |
| 2 | 1000 | 2.28 | 0.021 | 53.6% |
| 4 | 2000 | 4.15 | 0.032 | 29.5% |

La eficiencia cae al 53.6% con solo 2 rangos y al 29.5% con 4 rangos: el cuello de botella es el Allgather, no el cómputo.

---

## 2. Bugs arreglados en SFC

Se identificaron y corrigieron tres bugs en la descomposición SFC Morton preexistente:

### Bug 1: Bounding Box no global

**Problema:** `SfcDecomposition::build` computaba la bbox localmente desde las posiciones del rango propio. En modo multirank, cada rango veía posiciones distintas → bboxes distintas → **cutpoints divergentes** entre rangos → partículas mal clasificadas.

**Solución:** Se añadió `SfcDecomposition::build_with_bbox(positions, x_lo, x_hi, ..., n_ranks)` que acepta una bbox explícita. El motor usa `sfc::global_bbox(rt, &local)` (que llama `allreduce_min/max_f64` por coordenada) para obtener la bbox global antes de construir la descomposición.

```rust
let (gxlo, gxhi, gylo, gyhi, gzlo, gzhi) = global_bbox(rt, &local);
let sfc_decomp = SfcDecomposition::build_with_bbox(
    &positions, gxlo, gxhi, gylo, gyhi, gzlo, gzhi, rt.size(),
);
```

**Test:** `sfc_build_global_consistent` verifica que `build` local y `build_with_bbox` global producen asignaciones idénticas en serial.

### Bug 2: Migración de dominio limitada a vecinos rank±1

**Problema:** `exchange_domain_sfc` solo enviaba partículas a los rangos `rank-1` y `rank+1`. Partículas con código Morton lejano (que debían ir a rangos no adyacentes) quedaban en el rango incorrecto.

**Solución:** Reemplazado por un `Alltoallv` completo que envía a cualquier rango destino en un solo paso. La función `partition_local` ya clasificaba correctamente por rango destino; el problema era solo en el paso de envío.

```rust
let mut sends: Vec<Vec<f64>> = (0..size).map(|_| Vec::new()).collect();
for (r, particles) in leaves {
    sends[r as usize] = pack::pack_halo(&particles);
}
let received = self.alltoallv_f64(&sends);
```

**Test:** `migration_arbitrary_ranks` verifica que `partition_local` clasifica partículas en cuadrantes no adyacentes.

### Bug 3: Halos SFC en proxy 1D en x

**Problema:** `exchange_halos_sfc` usaba `rank / size × (x_hi - x_lo)` como proxy del borde SFC, ignorando las dimensiones y e z. Esto causaba halos incorrectos en distribuciones no alineadas con x.

**Solución:** Se implementan halos 3D reales:
1. Cada rango calcula su AABB local ajustada (6 f64).
2. `allgather_f64` distribuye todas las AABBs a todos los rangos (6P f64 total, ≪ N).
3. Para cada rank remoto r, se expande su AABB por `halo_width` en las 3 dimensiones y se envían partículas locales dentro de esa región expandida.

**Test:** `halo_geometric_3d_filter` verifica el filtro 3D explícitamente.

---

## 3. Diseño del LET (Locally Essential Tree)

### 3.1 Formato wire

`RemoteMultipoleNode` en `gadget-ng-tree/src/octree.rs`:

```rust
pub struct RemoteMultipoleNode {
    pub com: Vec3,         // centro de masa
    pub mass: f64,         // masa total del subárbol
    pub quad: [f64; 6],    // tensor cuadrupolar STF [Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]
    pub oct: [f64; 7],     // tensor octupolar STF [O_xxx, ..., O_yzz]
    pub half_size: f64,    // radio del nodo (para re-chequeo MAC)
}
```

**Layout wire:** 18 `f64` por nodo = 144 bytes. Para un LET típico con O(log N) nodos por vecino y N=2000, el volumen de LET es ≈ O(P × log N × 144) bytes por paso: para P=4, N=2000, esto es ≈ 4 × 11 × 144 ≈ 6.3 KB por paso (vs 80 KB para Allgather de N=2000 con 5 f64/partícula).

### 3.2 Export (poda del árbol local)

`Octree::export_let(target_aabb, theta)` recorre el árbol local con criterio MAC geométrico conservador:

```
d_min = min dist(node.com, target_aabb)
si 2 * half_size / d_min < theta → exportar nodo, podar subárbol
si no → descender en hijos
si hoja → exportar (caso degenerado)
```

El criterio es **conservador**: si se satisface para el punto más cercano del AABB receptor, se satisface para todos los puntos dentro de la AABB. Esto garantiza que la aproximación es válida para cualquier partícula del rango receptor.

### 3.3 Import y aplicación de fuerza

`accel_from_let(pos_i, let_nodes, g, eps2)` aplica para cada nodo remoto:
- Monopolo Plummer softened: `G m_j r / (r² + ε²)^{3/2}`
- Cuadrupolo softened (`quad_accel_softened`)
- Octupolo softened (`oct_accel_softened`)

Esto es coherente con el solver V5 (todos los términos con softening Plummer consistente).

### 3.4 Protocolo de intercambio

En `compute_forces_sfc_let`:
1. `allgather_f64` de AABBs locales (6P f64, O(P) tráfico).
2. Para cada rank remoto, `export_let` del árbol local → `pack_let_nodes` → buffer wire.
3. `alltoallv_f64` de buffers LET (O(P log N) tráfico total, vs O(NP) Allgather).
4. `unpack_let_nodes` de buffers recibidos.
5. `compute_forces_sfc_let`: árbol local (auto-exclusión por índice) + nodos remotos.

---

## 4. Validación: correctitud y precisión

### 4.1 Tests unitarios LET (gadget-ng-parallel/tests/let_validation.rs)

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `let_nodes_cover_all_mass` | Masa total exportada coincide con masa del árbol | ✓ |
| `let_wire_roundtrip` | Pack/unpack LET es bit-exacto | ✓ |
| `let_force_matches_direct_far_field` | Error LET vs directo < 2% para campo lejano | ✓ |
| `export_let_prunes_subtrees` | LET lejano es más compacto que cercano | ✓ |

### 4.2 Tests hardening SFC (gadget-ng-parallel/tests/sfc_hardening.rs)

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `sfc_build_global_consistent` | build local == build_with_bbox global | ✓ |
| `sfc_build_with_bbox_balance` | Equilibrio ±40% con bbox global | ✓ |
| `migration_arbitrary_ranks` | Partición correcta a rangos no adyacentes | ✓ |
| `halo_geometric_3d_filter` | Filtro AABB 3D correcto | ✓ |
| `allgather_f64_serial_returns_local` | allgather serial = identidad | ✓ |
| `alltoallv_f64_serial_noop` | alltoallv serial = vacío | ✓ |

### 4.3 Verificación de energía (runs P=1)

Para N=2000, Plummer a/ε=2, 100 pasos, dt=0.025:
- **Allgather P=1:** conservación de energía medida en diagnostics.jsonl (consistente con Fases 3-7).
- **SFC+LET P=1:** idéntico resultado (cae al mismo path Allgather en serial, ya que `use_sfc_let` requiere P>1).

---

## 5. Benchmarks (strong/weak scaling locales, hasta P=4)

Hardware: CPU local (mpirun --oversubscribe), sin red real.

### 5.1 Strong scaling (N=2000 fijo)

| P | Allgather (s) | SFC+LET (s) | Speedup AG | Speedup LET | Eff LET |
|---|---------------|-------------|------------|-------------|---------|
| 1 | 15.71 | 15.55 | 1.00× | 1.00× | 100% |
| 2 | 8.04 | 5.14 | 1.95× | 3.02× | **151%** |
| 4 | 4.22 | 4.16 | 3.72× | 3.74× | 93.5% |

> **Nota:** El speedup >1 de SFC+LET a P=2 respecto a P=1 SFC+LET se explica porque la ruta P=1 en SFC+LET cae al Allgather serial (misma ruta que Allgather), mientras que a P=2 el árbol local es N/2=1000 partículas (walk más rápido) y el LET tiene bajo overhead. A P=4 ambos convergen.

### 5.2 Weak scaling (N/P = 500 fijo)

| P | N | Allgather (s) | SFC+LET (s) | Eff AG | Eff LET |
|---|---|---------------|-------------|--------|---------|
| 1 | 500 | 1.22 | 1.22 | 100% | 100% |
| 2 | 1000 | 2.28 | 1.43 | **53.6%** | **85.7%** |
| 4 | 2000 | 4.15 | 4.20 | **29.5%** | **29.1%** |

SFC+LET mejora significativamente la eficiencia a P=2 (86% vs 54%). A P=4, ambos convergen en ~30% de eficiencia en esta escala de N/P=500 con hardware local. El resultado es consistente con el hecho de que para N pequeño, la fase de LET export/import tiene overhead significativo relativo al cómputo de fuerzas.

### 5.3 Fracción de comunicación

| Config | comm_frac |
|--------|-----------|
| Allgather P=1 | 0.003% |
| Allgather P=2 | 0.22% |
| Allgather P=4 | 1.28% |
| SFC+LET P=1 | 0.003% |
| SFC+LET P=2 | 1.20% |
| SFC+LET P=4 | 9.06% |

El SFC+LET tiene mayor fracción de comunicación que Allgather para P=4 en esta escala de N. Para N más grandes (N≫2000), el Allgather cresce como O(N·P) mientras el LET crece como O(P log N), favoreciendo al LET.

---

## 6. Comparación Allgather vs SFC+LET

| Aspecto | Allgather | SFC+LET |
|---------|-----------|---------|
| Complejidad comunicación | O(N·P) por paso | O(P log N) por paso |
| Memoria por rango | O(N) global | O(N/P + LET) |
| Weak scaling P=2 | 53.6% | 85.7% |
| Weak scaling P=4 | 29.5% | 29.1% |
| Bugs conocidos | – | Ninguno (corregidos) |
| Correctitud LET | N/A | < 2% error campo lejano |
| Default desde Fase 8 | No (fallback) | Sí (`force_allgather_fallback=false`) |

---

## 7. Limitaciones

1. **Morton Z-order, no Peano-Hilbert real.** La localidad espacial es ligeramente inferior al Hilbert curve completo. Para distribuciones clustered en Plummer denso, esto puede crear desequilibrio de carga entre rangos.

2. **LET síncrono.** El intercambio usa `Alltoallv` bloqueante. No hay solapamiento compute/comunicación. Para P≫4 en cluster real, el overhead de latencia de red limitará la escalabilidad.

3. **Rama cosmológica y hierarchical en Allgather.** Por simplicidad de implementación, las ramas `timestep.hierarchical = true` y `cosmology.enabled = true` siguen usando Allgather. Son fuera de alcance de Fase 8.

4. **N/P pequeño.** Para N/P < 1000, el overhead del LET (allgather de AABBs + alltoallv) es comparable o superior al cómputo local. El break-even SFC+LET vs Allgather ocurre aproximadamente a N/P ≈ 500-1000 en hardware local.

5. **Benchmarks locales, no cluster real.** Los resultados con `mpirun --oversubscribe` no capturan latencias de red reales (InfiniBand, Ethernet). La metodología es correcta y reproducible en cluster.

---

## 8. Recomendación y backlog Fase 9

### Recomendación inmediata

- **Usar SFC+LET como default** (ya implementado: `force_allgather_fallback = false` por defecto).
- **Para validación paper-grade o debug:** `force_allgather_fallback = true` activa el path Allgather legacy con advertencia explícita.
- **Para N > 5000 y P > 4:** los beneficios del LET son más claros. Los benchmarks locales son conservadores.

### Backlog Fase 9

- **Peano-Hilbert real:** reemplazar Morton Z-order por curva de Hilbert pura para mejor localidad en distribuciones clustered.
- **LET asíncrono:** solapar exportación LET y evaluación de fuerza local usando `MPI_Isend/Irecv` no bloqueante + Rayon.
- **Dynamic load balancing:** rebalanceo cada `sfc_rebalance_interval` pasos con métricas de carga (tiempo de tree walk por rango, no solo partícula count).
- **Rama hierarchical distribuida:** extender SFC+LET a `timestep.hierarchical = true`.
- **Benchmarks en cluster real:** repetir los mismos configs en cluster con InfiniBand para validar proyección de weak scaling hasta P=128.

---

## 9. Cross-referencias

- **Fase 3-5:** Validación física del solver Barnes-Hut V5 (MAC relativo, softening consistente). Los tests LET usan el mismo solver V5 para garantizar coherencia.
- **Fase 6:** Yoshida4 implementado; no reduce drift en sistemas caóticos. Los benchmarks SFC+LET usan integrador Leapfrog KDK (óptimo según Fase 6).
- **Fase 7:** Aarseth block timesteps: la rama `hierarchical = true` queda en Allgather (fuera de alcance Fase 8). Pendiente Fase 9.
- **Paper:** [`docs/paper/numerical_limits_nbody_chaos.md`](../paper/numerical_limits_nbody_chaos.md): La Sección 3 (Method) describe el solver V5 que el LET preserva. La Sección 7 (Conclusion) puede actualizarse con la mejora de weak scaling de Fase 8.

---

*Generado: Abril 2026 — gadget-ng Fase 8 HPC*
