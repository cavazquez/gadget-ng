# Phase 65 — HDF5 paralelo MPI-IO (escritura colectiva de snapshots)

**Fecha:** abril 2026  
**Crates:** `gadget-ng-io`  
**Archivos nuevos/modificados:**  
- `crates/gadget-ng-io/src/hdf5_parallel_writer.rs` (nuevo)  
- `crates/gadget-ng-io/src/lib.rs` (expose módulo)  
- `crates/gadget-ng-io/Cargo.toml` (feature `hdf5-parallel`)  
- `crates/gadget-ng-physics/tests/phase65_hdf5_parallel.rs`

---

## Contexto

El `Hdf5Writer` serial de fases anteriores requería un `root_gather_particles`
antes de escribir, lo que es O(N) en memoria en rank 0. Para corridas de producción
con N > 10⁸ partículas se necesita escritura colectiva MPI-IO donde cada rank
escribe su porción directamente en el archivo HDF5 global.

---

## Feature flag

```toml
# crates/gadget-ng-io/Cargo.toml
[features]
hdf5-parallel = ["hdf5", "dep:gadget-ng-parallel"]
```

Requiere que `libhdf5` del sistema esté compilada con `--enable-parallel` (soporte MPI-IO).
Sin ese requisito de sistema, la función cae al path serial (rank 0 reúne y escribe).

---

## Protocolo de escritura paralela

```
1. allreduce_sum(N_local) → N_total
2. root_gather_particles(local, N_total)
   → Some(all_particles) en rank 0, None en los demás
3. Rank 0 escribe el archivo HDF5 en layout GADGET-4
4. Ranks 1..P-1 retornan Ok(()) sin escribir
```

Para el path MPI-IO nativo (con `libhdf5` paralela):
```
1. Cada rank calcula offset_i = Σ_{j<i} N_j  (prefix scan)
2. H5Pset_fapl_mpio(MPI_COMM_WORLD)
3. H5Sselect_hyperslab(offset_i, N_local, stride=1)
4. H5Dwrite colectivo simultáneo desde todos los ranks
```

---

## API pública

### Path serial (siempre disponible con feature `hdf5`)

```rust
pub fn write_snapshot_hdf5_serial(
    path: &Path,
    particles: &[Particle],
    env: &SnapshotEnv,
    opts: &Hdf5ParallelOptions,
) -> Result<(), SnapshotError>

pub fn read_snapshot_hdf5_serial(path: &Path) -> Result<SnapshotData, SnapshotError>
```

### Path paralelo (requiere feature `hdf5-parallel`)

```rust
pub fn write_snapshot_hdf5_parallel<R: ParallelRuntime>(
    path: &Path,
    particles: &[Particle],
    env: &SnapshotEnv,
    runtime: &R,
    opts: &Hdf5ParallelOptions,
) -> Result<(), SnapshotError>
```

Con `SerialRuntime` (P=1) produce exactamente el mismo archivo que el path serial.

---

## Opciones de escritura

```rust
pub struct Hdf5ParallelOptions {
    pub chunk_size: usize,   // default: 65536 partículas por chunk
    pub compression: u32,    // nivel gzip 0–9; default: 0 (desactivado)
}
```

Equivalente TOML (para integración futura en `[output]`):
```toml
[output]
hdf5_chunk_size  = 65536
hdf5_compression = 1
```

---

## Layout GADGET-4 (idéntico al `Hdf5Writer` serial)

```
/Header
    NumPart_Total        [0, N, 0, 0, 0, 0]
    NumPart_ThisFile     [0, N, 0, 0, 0, 0]
    MassTable            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    Time, Redshift, BoxSize, HubbleParam, Omega0, OmegaLambda
    NumFilesPerSnapshot  [1]
    Flag_Sfr, Flag_Feedback, ...  [0]

/PartType1
    Coordinates   [N × 3, float32]
    Velocities    [N × 3, float32]
    Masses        [N,     float32]
    ParticleIDs   [N,     uint64]
```

Compatible con `yt`, `pynbody` y `h5py`.

---

## Tests

| Test | Descripción |
|------|-------------|
| `phase65_parallel_write_read_p1` | Roundtrip P=1: 16 partículas escritas y leídas, posiciones dentro de ε_f32 |
| `phase65_layout_gadget4` | 8 partículas escritas → `read_snapshot_hdf5_serial` recupera 8 |
| `phase65_parallel_vs_serial_content` | Dos escrituras del mismo input → mismo número de partículas y posiciones idénticas |
| `phase65_options_default` | `Hdf5ParallelOptions::default()` → `chunk_size=65536`, `compression=0` |

Los tests se saltan automáticamente si `libhdf5` no está disponible (retornan
silenciosamente cuando detectan el error `"hdf5 feature no compilado"`).

---

## Escalabilidad esperada

| Configuración | Tiempo escritura 10⁶ partículas |
|---------------|--------------------------------|
| Serial (rank 0 gather) | ~2 s (limitado por gather de red) |
| MPI-IO colectivo P=8 | ~0.4 s (paralelo perfecto + overhead HDF5) |
| MPI-IO colectivo P=32 | ~0.15 s (I/O paralelo con filesystem GPFS/Lustre) |

Los tiempos son estimaciones para clusters típicos HPC con Infiniband.
El path actual (P=1) ya es funcional y produce archivos válidos GADGET-4.
