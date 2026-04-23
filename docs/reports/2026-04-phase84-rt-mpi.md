# Phase 84 — RT MPI distribuida

**Fecha**: 2026-04-23  
**Crate**: `gadget-ng-rt`  
**Archivo nuevo**: `crates/gadget-ng-rt/src/mpi.rs`  
**Tipo**: Infraestructura MPI / RT

---

## Resumen

Infraestructura para distribuir el campo de radiación M1 entre múltiples ranks MPI.
Implementa la descomposición en slabs Y, intercambio de halos ghost y reducción global.

---

## Diseño

```
rank 0: iy ∈ [0,  ny/P)        + halo Y superior
rank 1: iy ∈ [ny/P, 2·ny/P)    + halos Y inferior/superior
...
rank P-1: iy ∈ [(P-1)·ny/P, ny) + halo Y inferior
```

## Estructuras implementadas

### `RadiationFieldSlab`

```rust
pub struct RadiationFieldSlab {
    pub energy: Vec<f64>,   // celdas locales + halos
    pub flux_x: Vec<f64>,
    pub flux_y: Vec<f64>,
    pub flux_z: Vec<f64>,
    pub nx: usize, pub ny_local: usize, pub nz: usize,
    pub iy_start: usize,    // índice global Y inicial
    pub dx: f64,
    pub rank: usize, pub n_ranks: usize,
}
```

- `from_global(global, rank, n_ranks)` — construye slab desde campo global
- `to_global(ny_total)` — reconstituye campo global desde slab (rank 0)
- `idx_local(ix, iy_local, iz)` — índice con offset de halo
- `idx_slab(ix, iy_slab, iz)` — índice raw en el slab con halos

### `RtRuntime`

Wrapper del communicator MPI. En modo serial: rank=0, size=1 (no-op).

## Funciones implementadas

| Función | Descripción |
|---------|-------------|
| `allreduce_radiation(rad, rt)` | Suma global de E y F (MPI_Allreduce) |
| `exchange_radiation_halos(slab, rt)` | Intercambio de capas ghost Y (MPI_Sendrecv) |
| `m1_update_slab(slab, dt, params)` | Solver M1 sobre slab con halos ghost |

## Modo serial (sin MPI)

Todas las funciones tienen implementación serial:
- `allreduce_radiation` → no-op (ya es global)
- `exchange_radiation_halos` → condición de contorno periódica (copia primera/última capa)
- `m1_update_slab` → delega a `m1_update` en modo serial

## Tests (4 tests)

| Test | Verifica |
|------|----------|
| `slab_from_global_serial_identity` | Datos copiados correctamente al slab |
| `allreduce_serial_noop` | Sin modificar el campo en modo serial |
| `exchange_halos_periodic_energy_conserved` | Halos ghost = celdas limítrofes |
| `to_global_roundtrip` | Reconstrucción fiel del campo global |
| `m1_update_slab_does_not_crash` | Energía ≥ 0 tras m1_update_slab |

---

## Referencia

Rosdahl et al. (2013), MNRAS 436, 2188 (RAMSES-RT paralelo).
