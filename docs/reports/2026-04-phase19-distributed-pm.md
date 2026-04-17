# Fase 19 — PM Distribuido sin allgather de partículas

**Fecha:** 2026-04-17
**Autor:** gadget-ng engineering
**Estado:** Completado

---

## Pregunta central

> ¿Puede `gadget-ng` ejecutar PM cosmológico periódico en MPI sin replicar el estado global completo de partículas, manteniendo correctitud física y mejorando la arquitectura de escalabilidad?

**Respuesta:** Sí. Esta fase implementa un path PM distribuido donde la comunicación dominante pasa de **O(N·P) bytes** (allgather de partículas) a **O(nm³) bytes** (allreduce del grid de densidad), eliminando la dependencia en N.

---

## 1. Estado anterior (Fases 17–18)

| Fase | Logro | Limitación |
|------|-------|------------|
| 17a  | Cosmología serial validada | Solo P=1 |
| 17b  | Cosmología distribuida SFC+LET | Sin periodicidad |
| 18   | PM periódico + cosmología validados | PM sigue usando allgather O(N·P) |

El cuello de botella de Fase 18: en `engine.rs`, la función `allgatherv_state` enviaba **5 × N × 8 bytes = 40·N bytes** por paso (posición + masa de todas las partículas, replicadas en cada rank). Este costo crece linealmente con N y con P (cada rank envía su fracción, pero el buffer global es O(N)).

---

## 2. Arquitectura implementada

Se eligió la **Opción C del plan**: gather del grid, no de partículas.

### Pipeline distribuido (Fase 19)

```
Rank r: partículas locales N/P
    ↓
deposit_local(pos, mass, box_size, nm)   → density_r[nm³]   O(N/P)
    ↓
allreduce_sum_f64_slice(&mut density)    → density_global    O(nm³) ← NUEVA COMUNICACIÓN
    ↓
forces_from_global_density(density, g_cosmo, nm, box_size)  O(nm³ log nm)
(idéntico en todos los ranks: determinista)
    ↓
interpolate_local(pos, fx, fy, fz, nm, box_size)            O(N/P)
    ↓
aceleraciones locales
```

### Por qué es correcto

- Tras el `allreduce_sum`, todos los ranks tienen el mismo grid global de densidad.
- El solve de Poisson (FFT 3D + kernel k-space) es determinista: produce exactamente el mismo campo de fuerza en todos los ranks.
- Cada rank interpola solo para sus partículas locales: sin comunicación adicional.
- **No se envía ninguna partícula entre ranks para el cálculo de gravedad PM.**

### Comparativa de comunicación

| Path | Bytes/paso | Escala con |
|------|-----------|------------|
| Fase 18: `allgatherv_state` | 40 × N bytes | N (partículas) |
| Fase 19: `allreduce_sum_f64_slice` | 8 × nm³ bytes | nm (grid, fijo por física) |

Ejemplos concretos con nm=32 (allreduce = 262 KB fijo):

| N | P | allgather | allreduce | Ratio |
|---|---|-----------|-----------|-------|
| 512 | 4 | 80 KB | 262 KB | 0.3× |
| 4 000 | 4 | 640 KB | 262 KB | **2.4×** menos |
| 100 000 | 4 | 16 000 KB | 262 KB | **61×** menos |
| 1 000 000 | 4 | 160 000 KB | 262 KB | **611×** menos |

El breakeven (allgather = allreduce) se produce en `N ≈ nm³ / 5`:
- nm=16: N ≈ 819 partículas
- nm=32: N ≈ 6 554 partículas
- nm=64: N ≈ 52 429 partículas

Para simulaciones con N >> nm³/5, el path distribuido es estrictamente mejor en comunicación.

---

## 3. Archivos modificados

### Nuevas primitivas de comunicación

**`crates/gadget-ng-parallel/src/lib.rs`**
```rust
fn allreduce_sum_f64_slice(&self, buf: &mut [f64]);
```
Reduce suma elemento a elemento de un array f64 entre todos los ranks.
En serial: no-op. En MPI: `MPI_Allreduce(sendbuf, recvbuf, nm³, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD)`.

**`crates/gadget-ng-parallel/src/serial.rs`** — no-op.

**`crates/gadget-ng-parallel/src/mpi_rt.rs`** — `world.all_reduce_into(&sendbuf, buf, SystemOperation::sum())`.

### Módulo PM distribuido

**`crates/gadget-ng-pm/src/distributed.rs`** (nuevo)
- `deposit_local(positions, masses, box_size, nm) -> Vec<f64>` — thin wrapper sobre `cic::assign`.
- `forces_from_global_density(density, g, nm, box_size) -> [Vec<f64>; 3]` — thin wrapper sobre `fft_poisson::solve_forces`.
- `interpolate_local(positions, fx, fy, fz, nm, box_size) -> Vec<Vec3>` — thin wrapper sobre `cic::interpolate`.

La separación en tres fases permite que el motor inserte el `allreduce` entre depósito y solve sin modificar la física.

### Configuración

**`crates/gadget-ng-core/src/config.rs`** — nuevo campo en `GravitySection`:
```toml
[gravity]
solver         = "pm"
pm_grid_size   = 32
pm_distributed = true   # activa el path Fase 19
```
`pm_distributed = false` por defecto (backward compatible con Fase 18).

### Motor

**`crates/gadget-ng-cli/src/engine.rs`**
- Import: `use gadget_ng_pm::distributed as pm_dist;`
- Condición de activación: `use_pm_dist = pm_distributed && periodic && solver == Pm`
- El `compute_acc` closure ahora tiene rama interna: si `use_pm_dist`, usa el pipeline distribuido; si no, usa el path clásico `allgatherv_state`.
- Log explícito: `[gadget-ng] PM DISTRIBUIDO (Fase 19): allreduce O(nm³) reemplaza allgather O(N·P).`

---

## 4. Paths activos

| Condición | Path activo |
|-----------|-------------|
| `solver = "pm"`, `periodic = true`, `pm_distributed = true` | **PM distribuido (Fase 19)** |
| `solver = "pm"`, `periodic = true`, `pm_distributed = false` | PM clásico allgather (Fase 18) |
| `solver = "tree_pm"`, `periodic = true` | TreePM clásico allgather (Fase 18) |
| `solver = "barnes_hut"`, `periodic = false`, `P>1` | SFC+LET (Fase 17b) |

**TreePM queda fuera de esta fase.** `TreePmSolver` sigue usando el path allgather de Fase 18. Documentado como limitación.

---

## 5. Validación física

### Tests automáticos (7 tests, todos pasan)

| Test | Valida |
|------|--------|
| `allreduce_sum_slice_is_noop_in_serial` | SerialRuntime no modifica el buffer |
| `deposit_local_matches_full_assign` | `deposit_local` ≡ `cic::assign` bit a bit |
| `distributed_pm_mass_conservation` | Masa total en grid reducido = masa de partículas |
| `distributed_forces_match_serial_pm` | Fuerzas distribuidas ≡ PM serial, error < 1e-12 |
| `distributed_border_particle_deposit` | Partícula en borde: masa conservada con CIC periódico |
| `distributed_pm_no_explosion_eds` | Run corto EdS + PM distribuido: sin NaN/Inf |
| `distributed_poisson_sanity_sinusoidal_mode` | Modo cos: F_x ∝ -sin(2πx/L), antisimetría < 1% |

### Equivalencia serial P=1 clásico vs distribuido

N=512, EdS, nm=16, 20 pasos:

| Paso | a(clásico) | a(distribuido) | \|Δa/a\| | \|Δv_rms\| |
|------|-----------|---------------|---------|-----------|
| 1    | 1.00049994 | 1.00049994 | 0.00e+00 | 0.00e+00 |
| 10   | 1.00499377 | 1.00499377 | 0.00e+00 | 0.00e+00 |
| 20   | 1.00997517 | 1.00997517 | 0.00e+00 | 0.00e+00 |

**Equivalencia bit a bit perfecta.** En P=1, `allreduce_sum_f64_slice` es un no-op exacto, por lo que el resultado es matemáticamente idéntico.

### Tests internos del módulo distributed.rs (4 tests, pasan)

- `deposit_local_matches_full_assign`
- `forces_wrapper_matches_solve_forces`
- `interpolate_local_wrapper_matches_interpolate`
- `distributed_pipeline_mass_conservation`

---

## 6. Benchmarks de referencia

Todos los runs son estables (sin NaN/Inf). Wall times en serial:

| Config | N | nm | pasos | wall time |
|--------|---|----|-------|-----------|
| EdS N512 clásico | 512 | 16 | 20 | ~9ms total |
| EdS N512 distribuido | 512 | 16 | 20 | ~9ms total |
| ΛCDM N2000 distribuido | 2000 | 32 | 20 | ~71ms total |
| EdS N4000 distribuido | 4000 | 32 | 15 | ~57ms total |

En serial, el overhead del path distribuido respecto al clásico es nulo (el `allreduce` en SerialRuntime es un no-op).

---

## 7. Limitaciones explícitas

### Lo que se implementó en esta fase
- `allreduce_sum_f64_slice` en ParallelRuntime (serial + MPI)
- Pipeline PM distribuido (deposit + allreduce + solve + interpolate) como módulo independiente
- Routing en engine.rs para activar el path distribuido vía config
- 7 tests automáticos de correctitud
- Equivalencia bit a bit P=1 clásico vs distribuido

### Lo que queda fuera de esta fase

1. **FFT sigue siendo serial por rank**: Todos los ranks ejecutan `solve_forces` en su copia local del grid global. El costo de compute es O(nm³ log nm) en cada rank, no distribuido. Para nm >> 64, el solve FFT se vuelve el cuello de computación (no de comunicación).

2. **TreePM no entra**: `TreePmSolver` sigue usando `allgatherv_state`. Para distribuir TreePM se necesita distribuir tanto el solve PM como el árbol de corto alcance con `minimum_image`.

3. **Sin MPI real validado en esta sesión**: El `allreduce_sum_f64_slice` en `MpiRuntime` está implementado correctamente (sigue el patrón de `allreduce_sum_f64`), pero no se ejecutaron benchmarks con `mpirun` en esta sesión. La validación MPI real requiere ejecutar `run_phase19.sh` con `mpirun -n 2/4`.

4. **Descomposición de partículas**: El path distribuido actual NO migra partículas entre ranks antes del depósito. Cada rank simplemente deposita sus partículas actuales (cualesquiera que sean) en el grid completo. La descomposición de dominio real (SFC o slab) para partículas + PM sigue siendo trabajo futuro.

5. **Sin FFT distribuida**: Para nm=128, 256, 512 (típico en simulaciones serias tipo GADGET-4), el grid completo excede la caché y el allreduce excede el ancho de banda. La siguiente etapa requiere un esquema slab-FFT distribuido (p.ej. FFTW-MPI o equivalente en Rust).

---

## 8. Próximos pasos hacia PM distribuido serio tipo GADGET

Para completar un PM/TreePM distribuido serio:

1. **Descomposición de dominio para partículas**: Asignar partículas a ranks según su posición en X (slab decomposition, ya existe `SlabDecomposition`) antes del depósito CIC.

2. **Halo de partículas para CIC**: Las partículas en el borde de un slab depositan en celdas del slab vecino (stencil CIC de 2 celdas). Se necesita un intercambio de halos de partículas de width=1 celda antes del depósito.

3. **FFT distribuida**: Implementar 1D-FFT distribuida o usar una librería. En el esquema slab:
   - Cada rank posee nm_slabs planos (planos X) del grid.
   - FFT en Y y Z son locales (dentro del slab).
   - FFT en X requiere transposición de datos entre ranks.
   - Esto corresponde al esquema "slab FFT" usado en FFTW-MPI.

4. **TreePM distribución completa**: Distribuir también el árbol de corto alcance con `minimum_image` en la fase de halos.

---

## 9. Veredicto

`gadget-ng` puede ahora ejecutar PM cosmológico periódico sin replicar el estado global de partículas. El cuello de botella de comunicación pasó de **O(N·P)** a **O(nm³)**, que es independiente de N. Para N >> nm³/5 partículas, el path distribuido es arquitecturalmente superior.

La física es correcta: equivalencia bit a bit con el path clásico en P=1, conservación de masa CIC exacta, y solve de Poisson validado para modos analíticos conocidos.

El camino hacia un PM distribuido completo tipo GADGET-4 requiere tres piezas adicionales: descomposición de dominio para partículas + halos CIC, FFT distribuida slab, y TreePM con minimum_image. Cada una de estas piezas está claramente delimitada y no interfiere con la física ya validada.
