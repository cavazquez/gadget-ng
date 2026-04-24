# Phase 102 — HDF5 layout GADGET-4 completo

**Fecha:** 2026-04-23  
**Estado:** ✅ Completado

## Objetivo

Adaptar el writer HDF5 existente para escribir el layout estándar GADGET-4 completo:
- `PartType0` (gas SPH) con `InternalEnergy` y `SmoothingLength`.
- `PartType1` (DM colisionless) como antes.
- Header con `NumPart_ThisFile`, `Flag_Sfr`, `Flag_DoublePrecision`, etc.

## Cambios implementados

### `crates/gadget-ng-io/src/hdf5_writer.rs`

**Writer:**
- Separa partículas en gas (`internal_energy > 0`) y DM (`internal_energy == 0`).
- Si hay gas: usa `Gadget4Header::for_sph()` con `NumPart[0] = n_gas`.
- Escribe `/PartType0/` con datasets: `Coordinates`, `Velocities`, `Masses`, `ParticleIDs`, `InternalEnergy`, `SmoothingLength`.
- Escribe `/PartType1/` solo si hay DM.

**Reader:**
- Lee `/PartType0/` (opcional) con `InternalEnergy` y `SmoothingLength`.
- Lee `/PartType1/` (opcional).
- Combina todas las partículas en orden: gas primero, DM después.

### `gadget4_attrs.rs` (ya existía, ahora se usa correctamente)
- `for_sph()`: configura `NumPart[0]`, `NumPart[1]`, `Flag_Sfr = 1`, `Flag_Cooling = 1`.
- `NumFilesPerSnapshot = 1` por defecto.
- `Flag_DoublePrecision = 1` por defecto.

## Compatibilidad con yt / pynbody

Ambas herramientas leen automáticamente `PartType0` y `PartType1` si el `/Header` tiene la forma correcta. Con este cambio:

```python
import yt
ds = yt.load("snapshot.hdf5")
gas = ds.all_data()[("PartType0", "Coordinates")]
dm  = ds.all_data()[("PartType1", "Coordinates")]
```

## Tests (`crates/gadget-ng-physics/tests/phase102_hdf5_gadget4.rs`)

| Test | Descripción | Estado |
|------|-------------|--------|
| `header_for_sph_counts` | NumPart[0/1] correctos | ✅ |
| `header_nbody_no_gas` | NumPart[0] = 0 sin gas | ✅ |
| `header_mandatory_flags_present` | Flags definidos | ✅ |
| `header_cosmological_params` | Ω, H, z correctos | ✅ |
| `header_unit_constants_set` | kpc/h, M☉, km/s | ✅ |

Tests HDF5 reales (`hdf5_tests::*`) disponibles con `--features hdf5`.

**Total: 5/5 tests compilados sin HDF5**
