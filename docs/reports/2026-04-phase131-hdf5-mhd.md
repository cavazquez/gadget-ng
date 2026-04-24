# Phase 131 — HDF5: Campos MHD + SPH Completos

**Fecha:** 2026-04-23  
**Estado:** ✅ COMPLETADA  
**Tests:** 6/6 pasados

## Objetivo

Extender el writer HDF5 al estilo GADGET-4 para incluir todos los campos físicos implementados en las Phases 120–130. Compatibilidad total con `yt`, `h5py` y `pynbody`.

## Campos añadidos

### `PartType0` (gas SPH)

| Dataset HDF5 | Campo en `Particle` | Tipo | Descripción |
|---|---|---|---|
| `MagneticField` | `b_field` | `[f64; 3]` | Vector B en unidades internas |
| `DednerPsi` | `psi_div` | `f64` | Potencial escalar de limpieza div-B |
| `CosmicRayEnergy` | `cr_energy` | `f64` | Energía específica de CRs |
| `Metallicity` | `metallicity` | `f64` | Metalicidad Z ∈ [0, 1] |
| `H2Fraction` | `h2_fraction` | `f64` | Fracción de hidrógeno molecular |
| `DustToGas` | `dust_to_gas` | `f64` | Relación D/G |

### `PartType4` (estrellas — nuevo)

| Dataset HDF5 | Campo en `Particle` | Tipo | Descripción |
|---|---|---|---|
| `Coordinates` | `position` | `[f64; 3]` | Posición |
| `Velocities` | `velocity` | `[f64; 3]` | Velocidad |
| `Masses` | `mass` | `f64` | Masa |
| `ParticleIDs` | `global_id` | `i64` | Identificador global |
| `StellarAge` | `stellar_age` | `f64` | Edad estelar en unidades internas |
| `Metallicity` | `metallicity` | `f64` | Metalicidad Z |

## Archivos modificados

- `crates/gadget-ng-io/src/hdf5_writer.rs`: campos extendidos en `PartType0` + nuevo grupo `PartType4`
- `crates/gadget-ng-io/src/lib.rs`: re-exportación pública de `Hdf5Writer`, `Hdf5Reader`
- `crates/gadget-ng-io/src/hdf5_parallel_writer.rs`: bugfix pre-existente (`SnapshotData` campos `time` y `redshift`)
- `crates/gadget-ng-physics/Cargo.toml`: feature `hdf5` en `gadget-ng-io` + dev-dep `hdf5-metno`

## Diseño

Los campos siempre se escriben (incluso si son cero) para garantizar compatibilidad con herramientas que los esperan. Esto facilita pipelines de análisis que asumen la presencia de `MagneticField` en cualquier snapshot de un código MHD.

```python
# Ejemplo de lectura con h5py
import h5py
with h5py.File("snapshot.hdf5", "r") as f:
    B = f["PartType0/MagneticField"][:]     # shape (N, 3)
    cr = f["PartType0/CosmicRayEnergy"][:]  # shape (N,)
    ages = f["PartType4/StellarAge"][:]     # shape (N_stars,)
```

## Tests

| Test | Descripción |
|------|-------------|
| `particle_has_all_extended_fields` | Struct Particle tiene todos los campos |
| `star_particle_has_stellar_age_and_metallicity` | Partícula estelar tiene `stellar_age` y `metallicity` |
| `all_extended_fields_default_zero` | Campos nuevos son 0 por defecto |
| `hdf5_writes_magnetic_field` | `MagneticField` dataset escrito con valores correctos |
| `hdf5_writes_cr_energy_and_metallicity` | `CosmicRayEnergy` y `Metallicity` correctos |
| `hdf5_writes_parttype4_stars` | `PartType4/StellarAge` correcto |
