# Phase G5 — Merger Trees: validación de Historia de Acreción de Masa (MAH)

**Fecha:** 2026-04-23  
**Crates afectados:** `gadget-ng-analysis`, `gadget-ng-cli`  
**Tests:** 6 / 6 ✅

---

## Resumen

Extensión de Phase 62 (merger trees) con extracción de la Historia de Acreción de Masa (MAH)
a lo largo de la rama principal de un halo, y comparación con el ajuste analítico de
McBride et al. (2009). Nuevo subcomando CLI `gadget-ng mah`.

---

## Nuevas funciones en `gadget-ng-analysis/src/merger_tree.rs`

### `MassAccretionHistory`

```rust
pub struct MassAccretionHistory {
    pub halo_id: u64,
    pub snapshots: Vec<usize>,   // índice de snapshot (0=más antiguo)
    pub redshifts: Vec<f64>,     // z[snap]
    pub masses: Vec<f64>,        // M[snap] [M_sun/h]
}
```

### `mah_main_branch`

```rust
pub fn mah_main_branch(
    forest: &MergerForest,
    root_halo_id: u64,
    redshifts: &[f64],
) -> MassAccretionHistory
```

**Algoritmo**: Recorre el árbol desde la raíz (snapshot más reciente) hacia atrás,
siguiendo el `prog_main_id` de cada nodo. En cada snapshot, busca el nodo del snapshot
anterior cuyo `prog_main_id` apunta al nodo actual. El recorrido termina cuando no
existe progenitor principal o se alcanza el snapshot 0.

### `mah_mcbride2009`

```rust
pub fn mah_mcbride2009(m0: f64, z: f64, alpha: f64, beta: f64) -> f64
// M(z) = M₀ · (1+z)^β · exp(−α·z)
```

Parámetros Millennium (McBride+2009):
- `α = 1.0`, `β = 0.0` → crecimiento exponencial puro
- Para halos de masa fija a z=0: `M(z=0) = M₀` exacto

---

## Subcomando CLI `gadget-ng mah`

```bash
gadget-ng mah \
  --merger-tree runs/cosmo/merger_tree.json \
  --redshifts "49,10,5,2,1,0.5,0" \
  --root-id 0 \
  --alpha 1.0 \
  --beta 0.0 \
  --out runs/cosmo/mah.json
```

Salida `mah.json`:
```json
{
  "halo_id": 0,
  "mah": [
    { "snapshot": 6, "redshift": 0.0, "mass": 1.3e14 },
    { "snapshot": 5, "redshift": 0.5, "mass": 1.1e14 },
    ...
  ],
  "mcbride_fit": [
    { "snapshot": 6, "redshift": 0.0, "mass": 1.3e14 },
    { "snapshot": 5, "redshift": 0.5, "mass": 1.15e14 },
    ...
  ]
}
```

---

## Tests

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `mah_main_branch_monotone` | MAH crece o se mantiene en sentido progenitor→descendente | ✅ |
| `mah_mcbride_z0_equals_m0` | M(z=0) == M₀ para α,β arbitrarios | ✅ |
| `mah_two_snap_trivial` | Dos snapshots → MAH con 2 puntos correctos | ✅ |
| `mah_merge_detected` | Fusión binaria → rama principal al halo más masivo | ✅ |
| `mah_single_snapshot_root` | Árbol de 1 snapshot → MAH de 1 punto | ✅ |
| `mah_mcbride_decreases_with_z` | M(z) decrece monótonamente con z (α>0, β=0) | ✅ |

---

## Dependencia con Phase 62

G5 reutiliza completamente la infraestructura de `build_merger_forest` de Phase 62.
Los `MergerTreeNode` necesitaban ya `prog_main_id: Option<u64>` que se pobla
correctamente al construir el árbol. G5 solo agrega las funciones de análisis
por encima de la estructura existente.

---

## Próximos pasos

- Validación contra simulaciones de referencia Millennium/MultiDark
- Ajuste automático de `α` y `β` por regresión no lineal
- Estadísticas de dispersión de la MAH sobre una población de halos
