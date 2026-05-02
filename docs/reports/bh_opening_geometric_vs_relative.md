# Barnes–Hut: MAC geométrico vs relativo (ErrTolForceAcc)

Los números históricos de la Fase 3 (`docs/reports/2026-04-phase3-gadget4-benchmark.md`) cuantifican el error **con criterio de apertura geométrico** (`θ`). El motor también implementa el criterio **relativo** (estilo GADGET-4 `TypeOfOpeningCriterion=1`), configurable en TOML como `opening_criterion = "relative"` y `err_tol_force_acc`.

## Regenerar tablas

Desde la raíz del repo:

```bash
bash experiments/nbody/phase3_gadget4_benchmark/bh_force_error/scripts/run_bh_accuracy.sh
```

O solo tests de integración:

```bash
cargo test -p gadget-ng-physics --test bh_force_accuracy \
  bh_force_accuracy_full_sweep bh_force_accuracy_relative_full_sweep \
  --release -- --nocapture
```

Salidas:

| Fichero | Contenido |
|---------|-----------|
| `experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy.csv` | θ barrido, **MAC geométrico** (comportamiento histórico del test) |
| `experiments/nbody/phase3_gadget4_benchmark/bh_force_error/results/bh_accuracy_relative.csv` | Mismo barrido de θ, **MAC relativo** con `err_tol_force_acc = 0.0025` (alineado con `configs/production_256.toml`) |

Parámetros comunes a ambos: `N=500`, orden multipolar 3 en BH, referencia `DirectGravity`, distribuciones `uniform_sphere` y `plummer` (mismas que en `bh_force_accuracy.rs`).

## Cómo interpretar

- Compare filas con el mismo `distribution` y `theta` entre los dos CSV: la columna `mean_err` (fracción, no %) muestra si el relativo reduce error medio frente al geométrico en ese régimen.
- Con `err_tol_force_acc = 0.0025` (valor por defecto del barrido relativo, alineado con producción), el error medio puede caer a **~10⁻⁶** frente a **~10⁻²–10⁻¹** del geométrico en Plummer con θ=0.5–0.8: confirma que el MAC relativo **ya implementado** corrige el régimen denso que motivaba el informe Fase 3.
- En **N=500**, ese mismo barrido puede hacer que `time_bh_ms` ≥ `time_direct_ms` (el árbol abre tantos nodos que el BH pierde ventaja frente al directo O(N²)). Esto es esperable en ese tamaño; en millones de partículas el BH vuelve a ser competitivo con θ/MAC adecuados.
- Para runs de producción con sistemas densos, **`opening_criterion = "relative"`** en `[gravity]` es la opción alineada con GADGET-4; los CSV permiten cuantificar el beneficio sobre el baseline geométrico del benchmark.

## Nota sobre el informe Fase 3

La recomendación antigua “implementar ErrTolForceAcc” refería al estado del código en esa fecha. En el árbol actual, el relativo **ya está implementado** (`gadget-ng-tree`, walk en `octree.rs`); falta sobre todo **elegirlo en configuración** y **medir** con los CSV anteriores cuando se discuta paridad o regresiones.

Para **dinámica energética multipolar + núcleos densos**, combina MAC relativo con `softened_multipoles = true` y `mac_softening = "consistent"` (véase [Phase 5 — energía y MAC](2026-04-phase5-energy-mac-consistency.md#7-configuración-recomendada-paper-grade)); el bloque TOML resumido está en [user-guide — Solvers de gravedad](../user-guide.md#solvers-de-gravedad) (subsección *Barnes–Hut: precisión*).
