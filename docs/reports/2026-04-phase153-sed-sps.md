# Phase 153 — SED completa con tablas SPS BC03-lite

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-analysis`  
**Archivos nuevos/modificados:** `sps_tables.rs`, `luminosity.rs`

## Resumen

Implementación de una grilla SPS BC03-lite 6×5 (edad × metalicidad) con interpolación bilineal para calcular la distribución espectral de energía (SED) multicolor en bandas U, B, V, R, I. Extensión de `luminosity.rs` con `SedResult` y `galaxy_sed`.

## Física implementada

### Grilla SPS BC03-lite

- 6 edades: [0.01, 0.1, 0.5, 1.0, 5.0, 13.0] Gyr
- 5 metalicidades: [0.0004, 0.004, 0.008, 0.02, 0.05]
- Interpolación bilineal en (edad, Z)

### Propiedades espectrales

- Galaxias jóvenes: alta L_U, L_B (OB stars)
- Galaxias viejas: alta L_R, L_I (RGB/AGB)
- B-V se enrojece con la edad (efecto de evolución estelar)

## API pública

| Función/Struct | Descripción |
|----------------|-------------|
| `SpsGrid::bc03_lite()` | Grilla por defecto |
| `SpsGrid::interpolate(age, z, band)` | Interpolación bilineal |
| `sps_luminosity(age, z, band)` | L/M [L☉/M☉] |
| `SedResult` | SED completa con BV, VR, edad media |
| `galaxy_sed(particles)` | SED de una galaxia |

## Tests (6/6 OK)

1–6: L_B decrece, BV enrojece, Z alta → más luminoso, nodo exacto, SedResult válido, N=200 sin panics.

## Referencias

- Bruzual & Charlot (2003) MNRAS 344, 1000
- Chabrier (2003) PASP 115, 763
