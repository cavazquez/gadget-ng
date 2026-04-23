# Phase 87 — MPI RT real + MPI AMR real

**Fecha**: 2026-04-23  
**Crates modificados**: `gadget-ng-rt`, `gadget-ng-pm`

## Objetivo

Reemplazar los `eprintln!` stubs de las funciones MPI de transferencia radiativa (RT) y AMR-PM
con implementaciones reales usando `rsmpi` (Rust MPI bindings). Las funciones quedan habilitadas
bajo el feature gate `mpi` para no afectar compilaciones sin MPI.

## Cambios

### `crates/gadget-ng-rt/Cargo.toml`
- Nuevo feature `mpi = ["dep:mpi"]`.
- Dependencia opcional `mpi = { workspace = true, optional = true }`.

### `crates/gadget-ng-rt/src/mpi.rs`
- **`allreduce_radiation_mpi<C: CommunicatorCollectives>`** (bajo `#[cfg(feature = "mpi")]`):  
  Suma global del campo de radiación (energía + flujos 3D) via `MPI_Allreduce`.  
  Equivalente a 4 llamadas a `world.all_reduce_into(..., SystemOperation::sum())`.

- **`exchange_radiation_halos_mpi<C: Communicator>`** (bajo `#[cfg(feature = "mpi")]`):  
  Intercambio de capas halo ghost entre ranks vecinos usando `MPI_Send`/`MPI_Recv`.  
  Patrón odd-even de dos rondas para evitar deadlock (idéntico al de `gadget-ng-parallel`).  
  Soporte para `size == 1` (fallback serial periódico).

### `crates/gadget-ng-pm/Cargo.toml`
- Nuevo feature `mpi = ["dep:mpi"]`.
- Dependencia opcional `mpi = { workspace = true, optional = true }`.

### `crates/gadget-ng-pm/src/amr_mpi.rs`
- **`broadcast_patch_forces_mpi<C: CommunicatorCollectives>`** (bajo `#[cfg(feature = "mpi")]`):  
  Serialización de parches AMR a buffer plano de f64 y difusión via `MPI_Bcast`.  
  Protocolo: (1) rank 0 serializa, (2) broadcast de longitud de buffer, (3) broadcast de datos,  
  (4) todos los ranks deserializan `Vec<AmrPatchMessage>`.

- **`amr_pm_accels_multilevel_mpi_real<C: CommunicatorCollectives>`** (bajo `#[cfg(feature = "mpi")]`):  
  Pipeline completo: densidad local → allreduce → rank 0 identifica parches → broadcast fuerzas.

- **`build_amr_hierarchy_mpi_real<C: CommunicatorCollectives>`** (bajo `#[cfg(feature = "mpi")]`):  
  Allreduce de densidad global antes de construir la jerarquía AMR.

### `crates/gadget-ng-rt/src/lib.rs` y `crates/gadget-ng-pm/src/lib.rs`
- Re-exportan las nuevas funciones MPI bajo `#[cfg(feature = "mpi")]`.

## Tests

Tests con `--features mpi` en entorno single-rank (equivalente serial):

- `allreduce_radiation_mpi_single_rank`: allreduce sobre 1 rank → campo inalterado.
- `exchange_halos_mpi_single_rank_periodic`: halos periódicos en single rank.
- `broadcast_patch_forces_mpi_single_rank`: serialización/deserialización correcta.
- `amr_pm_accels_mpi_real_single_rank_matches_serial`: coincidencia exacta con path serial.

## Verificación

```bash
cargo build -p gadget-ng-rt --features mpi
cargo build -p gadget-ng-pm --features mpi
cargo test -p gadget-ng-rt --features mpi --release
cargo test -p gadget-ng-pm --features mpi --release
```

Todos los tests pasan. El workspace completo compila sin errores.
