# Phase 85 — AMR MPI: comunicación de parches

**Fecha**: 2026-04-23  
**Crate**: `gadget-ng-pm`  
**Archivo nuevo**: `crates/gadget-ng-pm/src/amr_mpi.rs`  
**Tipo**: Infraestructura MPI / AMR

---

## Resumen

Infraestructura para distribuir el solver AMR-PM jerárquico entre múltiples ranks MPI.
Implementa el patrón "rank 0 coordina": identificación global de parches + broadcast de fuerzas.

---

## Diseño

**Flujo en MPI real**:
1. Cada rank calcula densidad local en grid base → `MPI_Allreduce` suma global
2. Rank 0 identifica parches (`identify_refinement_patches`)
3. Rank 0 resuelve Poisson en cada parche
4. `broadcast_patch_forces` distribuye las fuerzas a todos los ranks
5. Cada rank aplica correcciones de fuerza a sus partículas locales

## Estructuras implementadas

### `AmrPatchMessage`

```rust
pub struct AmrPatchMessage {
    pub center: Vec3,
    pub size: f64,
    pub nm: usize,
    pub forces: [Vec<f64>; 3],  // [fx, fy, fz] calculados
}
```

Representación serializable de un parche resuelto para difusión entre ranks.

### `AmrRuntime`

```rust
pub struct AmrRuntime { pub rank: usize, pub size: usize }
```

Wrapper del communicator MPI. En modo serial: rank=0, size=1.

## Funciones implementadas

| Función | Descripción |
|---------|-------------|
| `broadcast_patch_forces(patches, rt)` | Difunde fuerzas de parches: MPI_Bcast (pequeños) o MPI_Gatherv/Scatterv |
| `amr_pm_accels_multilevel_mpi(...)` | Wrapper MPI del solver multi-nivel |
| `build_amr_hierarchy_mpi(...)` | Jerarquía con reducción global de densidad |

## Modo serial

En modo serial (size=1), todas las funciones delegan directamente a:
- `amr_pm_accels_multilevel(...)` — aceleraciones seriales
- `build_amr_hierarchy(...)` — jerarquía serial

El resultado es **bit-a-bit idéntico** al path serial, verificado en tests.

## Tests (3 tests)

| Test | Verifica |
|------|----------|
| `serial_mpi_matches_direct` | Aceleraciones MPI serial == aceleraciones directas |
| `broadcast_serial_identity` | AmrPatchMessage.len() y fields == PatchGrid originales |
| `hierarchy_mpi_serial_same` | Número de parches por nivel == jerarquía serial |

---

## Referencia

Kravtsov et al. (1997), ApJS 111, 73 (AMR+MPI para N-body);
Teyssier (2002), A&A 385, 337 (RAMSES AMR distribuido).
