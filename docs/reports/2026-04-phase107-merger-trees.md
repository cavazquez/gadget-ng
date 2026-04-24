# Phase 107 — Merger Trees con FoF Real

**Fecha**: 2026-04-23  
**Crates**: `gadget-ng-analysis`, `gadget-ng-cli`  
**Archivos clave**: `crates/gadget-ng-analysis/src/fof.rs`, `crates/gadget-ng-cli/src/merge_tree_cmd.rs`

## Problema

El módulo `merger_tree` (Phase 62) construía `ParticleSnapshot` con `halo_idx = None` para
todas las partículas. El algoritmo de particle-matching entre snapshots requiere conocer a qué
halo pertenece cada partícula; con todos los valores `None`, los merger trees resultantes eran
vacíos o incorrectos.

## Solución

### Nueva función: `find_halos_with_membership`

Agrega una variante de `find_halos` que devuelve tanto los halos como la membresía por partícula:

```rust
pub fn find_halos_with_membership(...) -> (Vec<FofHalo>, Vec<Option<usize>>)
```

Internamente reutiliza el mismo Union-Find que `find_halos`, pero captura los grupos antes de
descartar los índices de partícula.

### Nueva función: `particle_snapshots_from_catalog`

Para los casos en que solo se dispone del catálogo JSONL (sin re-ejecutar FoF), asigna membresía
por proximidad al COM del halo dentro de su radio de virial:

```rust
pub fn particle_snapshots_from_catalog(
    positions: &[Vec3],
    global_ids: &[u64],
    halos: &[FofHalo],
    box_size: f64,
) -> Vec<ParticleSnapshot>
```

### `run_merge_tree` actualizado

En `merge_tree_cmd.rs`, el loop de construcción de catalogs ahora llama a
`particle_snapshots_from_catalog` en lugar de asignar `halo_idx = None`:

```rust
let part_snapshots = particle_snapshots_from_catalog(
    &positions, &global_ids, &halos, box_size
);
```

## Tests

Archivo: `crates/gadget-ng-physics/tests/phase107_merger_trees.rs` (6 tests)

| Test | Descripción |
|------|-------------|
| `membership_two_separated_clusters` | 2 clusters separados → membresía correcta |
| `catalog_proximity_assigns_halo_idx` | Asignación por proximidad COM + r_vir |
| `merger_forest_detects_fusion` | 2 halos se fusionan → merger detectado |
| `empty_catalogs_produce_empty_forest` | Sin catálogos → forest vacío |
| `single_snapshot_no_progenitors` | 1 snapshot → sin progenitores |
| `membership_empty_particles` | 0 partículas → sin errores |

## Impacto

- El CLI `gadget-ng merge-tree` ahora produce merger trees con conexiones reales.
- Las trayectorias de AGN/BH entre snapshots son rastreables.
- `find_halos_with_membership` está disponible para análisis externos que necesiten membresía.
