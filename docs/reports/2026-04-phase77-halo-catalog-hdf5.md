# Phase 77 — Catálogo de Halos HDF5 por Snapshot

**Crate**: `gadget-ng-io/src/halo_catalog_hdf5.rs`  
**Fecha**: 2026-04

## Resumen

Escritura y lectura de catálogos de halos FoF + subhalos SUBFIND en formato
HDF5 compatible con `yt`, `h5py`, `Caesar` y `rockstar-galaxies`. Para builds
sin HDF5, se provee un formato JSONL alternativo.

## Estructura del archivo HDF5

```
halos_NNNNNN.hdf5
  /Header
    z, a, BoxSize, N_halos, N_subhalos
  /Halos/
    Mass         [N f64]
    Pos          [N×3 f64]
    Vel          [N×3 f64]
    R200         [N f64]
    Spin_Peebles [N f64]   ← Phase 72
    Npart        [N i64]
  /Subhalos/    (opcional)
    Mass         [M f64]
    Pos          [M×3 f64]
    ParentHalo   [M i64]
```

## Formato JSONL (siempre disponible)

Primera línea: header JSON. Líneas siguientes: una entrada de halo por línea.
Apto para `jq`, Python, y cualquier herramienta de texto.

## Nuevos tipos

```rust
pub struct HaloCatalogEntry { mass, pos: [f64;3], vel: [f64;3], r200, spin_peebles, npart }
pub struct SubhaloCatalogEntry { mass, pos: [f64;3], parent_halo: i64 }
pub struct HaloCatalogHeader { redshift, scale_factor, box_size, n_halos, n_subhalos }
```

## Funciones exportadas

| Función | Descripción |
|---|---|
| `write_halo_catalog_hdf5(...)` | Escribe catálogo HDF5 (requiere feature `hdf5`) |
| `read_halo_catalog_hdf5(...)` | Lee catálogo HDF5 |
| `write_halo_catalog_jsonl(...)` | Escribe catálogo JSONL (siempre disponible) |
| `read_halo_catalog_jsonl(...)` | Lee catálogo JSONL |

## Tests

4 tests siempre activos + 1 test adicional con feature `hdf5`:
- `halo_catalog_header_creation`
- `jsonl_roundtrip` (escribe y lee, verificando campos)
- `jsonl_empty_halos`
- `halo_entry_serializes`
- `hdf5_roundtrip` (con feature `hdf5`)
