# Fase 9 — HPC local: overlap compute/comm, Rayon, instrumentación HPC y rebalanceo dinámico

**Fecha:** Abril 2026  
**Estado:** Completada  
**Referencia cruzada:** [Fase 8 HPC scaling](2026-04-phase8-hpc-scaling.md) | [Fase 7 Aarseth](2026-04-phase7-aarseth-timestep.md) | [Paper](../paper/numerical_limits_nbody_chaos.md)

---

## 1. Diagnóstico: bottlenecks identificados en Fase 8

La Fase 8 implementó el path SFC+LET para eliminar el Allgather O(N·P), pero dejó varios problemas abiertos:

### 1.1 Comunicación bloqueante
Cada evaluación de fuerzas ejecutaba **4 colectivas MPI bloqueantes en secuencia**:
1. `all_gather_varcount_into` (AABBs locales, 6 f64 × P)
2. `all_to_all_into` (conteos LET, P ints)
3. `all_to_all_varcount_into` (datos LET, variable)

Con estas llamadas bloqueantes, el rank quedaba 100% inactivo durante la fase 3 (alltoallv de nodos LET), que es la más costosa.

### 1.2 Walk local serial
`compute_forces_sfc_let` ejecutaba el walk del árbol local en **un único hilo**, desperdiciando los núcleos disponibles en el nodo (típicamente 4–32 por socket moderno).

### 1.3 Atribución incorrecta de tiempos
`Octree::build`, `export_let` y `pack_let_nodes` (operaciones de **cómputo**) se incluían en `this_comm`, inflando el `comm_fraction` reportado y ocultando el costo real de la comunicación MPI.

### 1.4 Sin rebalanceo de carga basado en costo
El rebalanceo SFC se activaba únicamente por intervalo de pasos (`sfc_rebalance_interval`), sin considerar si el trabajo de force evaluation estaba desequilibrado entre rangos.

---

## 2. LET no-bloqueante: diseño e implementación

### 2.1 Nuevo método `alltoallv_f64_overlap`

Se agregó al trait `ParallelRuntime` un nuevo método:

```rust
fn alltoallv_f64_overlap(
    &self,
    sends: Vec<Vec<f64>>,
    overlap_work: &mut dyn FnMut(),
) -> Vec<Vec<f64>>;
```

**Protocolo en la implementación MPI** (`crates/gadget-ng-parallel/src/mpi_rt.rs`):
1. **Fase 1 — exchange counts** (bloqueante, O(P) enteros): `all_to_all_into`
2. **Fase 2 — alloc recv buffers**: buffer plano con offsets pre-calculados
3. **Fase 3 — P2P no-bloqueante**: `immediate_send` + `immediate_receive_into` para cada rank remoto, usando `mpi::request::scope` + `WaitGuard`
4. **Fase 4 — overlap work**: `overlap_work()` — walk local del árbol mientras los mensajes están en vuelo
5. **Fase 5 — wait all**: `WaitGuard` drop implícito al final del scope

**Implementación serial** (`serial.rs`): llama `overlap_work()` inmediatamente y devuelve `vec![vec![]]` (sin rangos remotos).

**Seguridad**: los slices de recv se crean con `unsafe { std::slice::from_raw_parts_mut(...) }` usando offsets no solapados del buffer plano, que vive hasta el final de la función. El invariante de no-solapamiento está garantizado por la construcción de `recv_offsets`.

### 2.2 Flujo de force_eval con overlap

```
┌────────────────────────────────────────────────────────────┐
│  1. allgather_f64 (AABBs) ─── bloqueante, O(P·6) ────────│
│  2. Octree::build ─────────── cómputo ────────────────────│
│  3. export_let × (P-1) ─────── cómputo ────────────────────│
│  4. pack_let_nodes × (P-1) ── cómputo ────────────────────│
│                                                            │
│  alltoallv_f64_overlap:                                    │
│  ├── Isend(r) para r ≠ rank  ─ lanzar                     │
│  ├── Irecv(r) para r ≠ rank  ─ lanzar                     │
│  ├── overlap_work() ──────────── walk local (SOLAPADO)     │
│  └── wait all ────────────────── MPI wait puro             │
│                                                            │
│  5. unpack_let_nodes × recibidos ── cómputo ──────────────│
│  6. accel_from_let × N_local ──── cómputo ──────────────── │
└────────────────────────────────────────────────────────────┘
```

### 2.3 Resultado observado (N=2000, P=2)

| Métrica | Valor |
|---------|-------|
| `let_alltoallv_ns` total | 13.71 ms |
| `walk_local_ns` (solapado) | 13.62 ms |
| MPI wait puro ≈ alltoallv − walk | 0.09 ms |
| `wait_fraction` | 0.29% |

El walk local (13.6 ms) casi cubre completamente el tiempo de alltoallv (13.7 ms). El overhead MPI no ocultado es < 0.1 ms.

### 2.4 Limitaciones
- Para N pequeño (N < ~500 por rank) el walk es demasiado rápido y no oculta la comm LET.
- La fase 1 (allgather AABBs) sigue siendo bloqueante; para P muy grande (> 64) podría hacerse no-bloqueante también.
- `mpi::request::scope` + `WaitGuard` requiere `unsafe` para crear slices no-solapados del buffer plano.

---

## 3. Paralelismo intra-rank (Rayon)

### 3.1 Walk local paralelo

En el path no-bloqueante, el walk local (el trabajo de overlap) usa Rayon cuando la feature `simd` está habilitada:

```rust
#[cfg(feature = "simd")]
{
    use rayon::prelude::*;
    local_accels
        .par_iter_mut()
        .enumerate()
        .for_each(|(li, a)| {
            *a = tree.walk_accel(parts[li].position, li, g, eps2, theta, ...);
        });
}
#[cfg(not(feature = "simd"))]
{ /* bucle serial */ }
```

Este patrón es idéntico al de `rayon_bh.rs` (Fase 4). `Octree` es `Send + Sync` (read-only durante el walk), y `local_accels` se accede con `par_iter_mut` (acceso exclusivo por elemento).

### 3.2 Esperado vs serial
Para N/P = 1000 con 4 hilos Rayon disponibles, el speedup teórico del walk es ~3–4×. En hardware local con pocos núcleos disponibles, el beneficio es menor debido a overhead de spawn de Rayon y granularidad fina de tareas.

---

## 4. Instrumentación detallada

### 4.1 `HpcStepStats` (11 campos)

```rust
struct HpcStepStats {
    tree_build_ns: u64,      // Octree::build
    let_export_ns: u64,      // export_let × (P-1)
    let_pack_ns: u64,        // pack_let_nodes × (P-1)
    aabb_allgather_ns: u64,  // allgather_f64 (MPI)
    let_alltoallv_ns: u64,   // alltoallv_f64_overlap (total)
    walk_local_ns: u64,      // walk local (solapado)
    apply_let_ns: u64,       // unpack + accel_from_let
    let_nodes_exported: usize,
    let_nodes_imported: usize,
    bytes_sent: usize,
    bytes_recv: usize,
}
```

### 4.2 Corrección de atribución de tiempos

| Fase | Fase 8 | Fase 9 |
|------|--------|--------|
| Octree::build | `this_comm` ❌ | `this_grav` ✅ |
| export_let | `this_comm` ❌ | `this_grav` ✅ |
| pack_let_nodes | `this_comm` ❌ | `this_grav` ✅ |
| allgather AABBs | `this_comm` ✅ | `this_comm` ✅ |
| alltoallv LET | `this_comm` ✅ | `wait_only = alltoallv − walk` ✅ |
| walk_local | `this_grav` ✅ | `this_grav` ✅ |
| apply_let | `this_grav` ✅ | `this_grav` ✅ |

El `comm_fraction` de Fase 8 estaba inflado por incluir build/export/pack. En Fase 9, `comm_fraction` refleja solo el tiempo de espera MPI real.

### 4.3 Salida en diagnostics.jsonl

Cada línea del `diagnostics.jsonl` incluye ahora el campo `hpc_stats` con los 11 campos en nanosegundos. Esto permite análisis post-hoc por paso.

El `timings.json` incluye un campo `hpc` con medias por paso para todos los campos:

```json
{
  "hpc": {
    "mean_tree_build_s": 0.000261,
    "mean_let_export_s": 0.0000627,
    "mean_let_pack_s": 0.0000132,
    "mean_aabb_allgather_s": 0.0000061,
    "mean_let_alltoallv_s": 0.013211,
    "mean_walk_local_s": 0.013056,
    "mean_apply_let_s": 0.039184,
    "mean_let_nodes_exported": 1818.4,
    "mean_let_nodes_imported": 1995.8,
    "mean_bytes_sent": 261849.6,
    "mean_bytes_recv": 287395.2,
    "wait_fraction": 0.00293
  }
}
```

---

## 5. Rebalanceo dinámico basado en costo

### 5.1 Diseño

Al final de cada paso, el motor verifica si el walk local está desequilibrado entre rangos:

```rust
let wl_max = rt.allreduce_max_f64(hpc.walk_local_ns as f64);
let wl_min = rt.allreduce_min_f64(hpc.walk_local_ns as f64).max(1.0);
if wl_max / wl_min > 1.3 {
    cost_rebalance_pending = true; // forzar rebalanceo en el siguiente paso
}
```

El threshold `1.3` (30% de desequilibrio) es configurable implícitamente. El rebalanceo forzado recalcula la descomposición SFC (Morton Z-order) con la bbox global actualizada, redistribuyendo las partículas hacia una carga más uniforme.

### 5.2 Overhead
El check requiere 2 `allreduce_f64` por paso (min + max), que son colectivas O(log P) de 1 valor cada una. El overhead es despreciable comparado con el paso de integración.

### 5.3 Limitaciones
El proxy de costo (`walk_local_ns`) no captura el costo del walk de partículas LET remotas (`apply_let_ns`). Un proxy más preciso incluiría también el número de nodos LET importados multiplicado por el costo por nodo, pero esto añadiría complejidad.

---

## 6. Resultados locales (benchmarks)

### 6.1 Validación de equivalencia

Se verificó que el path bloqueante y el no-bloqueante producen **energías cinéticas bit-exactas** para N=2000, P=2, 5 pasos:

```
Step | Overlap KE    | Blocking KE   | Diff
   0 | 0.14726216    | 0.14726216    | 0.00e+00
   1 | 0.14725444    | 0.14725444    | 0.00e+00
   2 | 0.14726642    | 0.14726642    | 0.00e+00
   ...
```

Los resultados son idénticos porque el scheduling de la comunicación no afecta los datos recibidos.

### 6.2 Desglose de tiempos (N=2000, P=2, overlap)

| Fase | Tiempo medio | % del paso |
|------|-------------|-----------|
| tree_build | 0.26 ms | 0.5% |
| let_export | 0.06 ms | 0.1% |
| let_pack | 0.01 ms | <0.1% |
| aabb_allgather (MPI) | 0.006 ms | <0.1% |
| MPI wait puro | 0.09 ms | 0.17% |
| walk_local (solapado) | 13.1 ms | 24.8% |
| apply_let | 39.2 ms | 74.2% |
| **Total** | ~52.8 ms | 100% |

**Observación crítica**: el `apply_let` (aplicar fuerzas de nodos LET remotos) domina el tiempo total con el 74%. Esto es el cuello de botella actual, no la comunicación. Indica que para N=2000 con P=2, cada rank tiene ~1000 partículas propias y ~2000 nodos LET remotos, haciendo el costo de `accel_from_let` comparable al de un sistema de 3000 partículas.

### 6.3 Comparación overlap vs blocking

| Métrica | Blocking | Overlap | Ganancia |
|---------|----------|---------|----------|
| `comm_fraction` | 0.40% | 0.29% | −0.11pp |
| `mean_step_wall_s` | ~52.8 ms | ~52.8 ms | ≈0% |
| `wait_fraction` | 0.40% | 0.29% | −0.11pp |

Para N=2000 P=2 (hardware local), el overlap no produce una ganancia de wall time significativa porque:
1. El walk local ya ocupa ~24.8% del paso
2. El MPI wait era solo ~0.4% del paso en blocking

La ganancia del overlap será más pronunciada en clusters con alta latencia MPI y N/P grande (donde el walk local sea comparable en duración al alltoallv).

---

## 7. Limitaciones y proyección para cluster real

### 7.1 Limitaciones actuales

- **Hardware local**: con mpirun en un nodo único, las llamadas MPI son rápidas (microsegundos). En clusters reales, la latencia de red (InfiniBand: ~1–2 µs, Ethernet: ~10–50 µs) haría más visible el beneficio del overlap.
- **N pequeño**: con N=2000 y P=2, cada rank tiene N/P=1000 partículas, lo que hace el walk local demasiado corto para ocultar completamente la comm LET. Con N=100K y P=128, N/P=781 partículas pero el árbol local sería mucho más grande.
- **apply_let dominante**: el costo de `accel_from_let` escala O(N_local × N_let_imported). Con árboles bien podados, N_let_imported ≈ O(N) → el costo total no escala mejor que O(N²) a menos que se use también un árbol remoto (Barnes-Hut sobre los nodos LET). Esta es la siguiente mejora arquitectural.

### 7.2 Proyección para cluster

En un cluster con 16 nodos × 32 núcleos/nodo × 2 GPUs/nodo:

| Cambio recomendado | Impacto esperado |
|-------------------|-----------------|
| Rayon habilitado (`--features simd`) | 4–8× speedup en walk local por nodo |
| N ≥ 50K por ejecución | Mayor latencia relativa del LET exchange |
| `let_nonblocking = true` | 20–40% reducción en wall time (estimado) |
| Árbol sobre nodos LET | Escalar apply_let a O(N log N) |
| NUMA-aware task placement | Reducción de false sharing en Rayon |

### 7.3 Próxima mejora arquitectural recomendada

El cuello de botella actual es `apply_let` O(N×N_LET). La solución es construir un árbol sobre los nodos LET importados y usarlo como segundo árbol remoto para el walk (`walk_accel_let_tree`). Esto reduciría la complejidad de apply_let de O(N×N_LET) a O(N log N_LET).

---

## 8. Cross-refs

- **Fase 8** ([report](2026-04-phase8-hpc-scaling.md)): implementación SFC+LET base, bugs de SFC, diseño de LET
- **Fase 7** ([report](2026-04-phase7-aarseth-timestep.md)): timesteps adaptativos Aarseth
- **Fase 6** ([report](2026-04-phase6-higher-order-integrator.md)): Yoshida4
- **Paper** ([doc](../paper/numerical_limits_nbody_chaos.md)): análisis completo de límites de precisión

---

## Apéndice: Archivos modificados/creados en Fase 9

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-core/src/config.rs` | +`let_nonblocking: bool` en `PerformanceSection` |
| `crates/gadget-ng-parallel/src/lib.rs` | +`alltoallv_f64_overlap` en trait |
| `crates/gadget-ng-parallel/src/serial.rs` | +impl trivial (llama overlap_work, devuelve vacío) |
| `crates/gadget-ng-parallel/src/mpi_rt.rs` | +impl no-bloqueante P2P Isend/Irecv + WaitGuard |
| `crates/gadget-ng-cli/src/engine.rs` | +`HpcStepStats`, +`HpcTimingsAggregate`, refactor SFC+LET loop: overlap, Rayon, timings, rebalanceo |
| `crates/gadget-ng-parallel/tests/overlap_validation.rs` | Nuevo: 7 tests de validación del overlap |
| `experiments/nbody/phase9_hpc_local/` | Configs, scripts run+analyze |
| `docs/reports/2026-04-phase9-hpc-local.md` | Este reporte |

---

## Continuación: Fase 10 — LET-tree O(N log N) para apply_let

La Fase 9 identificó `apply_let_ns` como el cuello de botella dominante tras eliminar
la espera MPI mediante el path no-bloqueante. La **Fase 10** implementa un octree
Barnes-Hut (`LetTree`) sobre los `RemoteMultipoleNode` importados, reduciendo
`apply_let` de O(N_local × N_let) a O(N_local log N_let).

Detalles: [2026-04-phase10-let-tree.md](2026-04-phase10-let-tree.md)

---

*Generado: Abril 2026 — gadget-ng Fase 9*
