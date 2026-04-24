# Phase 159 — GMC collapse + IMF Kroupa + feedback por cúmulo

**Fecha:** 2026-04-23  
**Crate afectado:** `gadget-ng-sph`  
**Archivo nuevo:** `crates/gadget-ng-sph/src/gmc.rs`

## Resumen

Implementación del módulo de formación de cúmulos estelares (GMC collapse) con muestreo de la IMF de Kroupa (2001) y retroalimentación por supernovas tipo II. Extiende el modelo de formación estelar estocástica (Phase 112) con sub-partición en cúmulos masivos.

## Física implementada

### IMF de Kroupa (2001)

```text
dN/dm ∝ m^{-α(m)}
α₁ = 1.3   para 0.1 ≤ m/M☉ < 0.5
α₂ = 2.3   para 0.5 ≤ m/M☉ ≤ 150
```

Muestreo por inversión analítica de la CDF normalizada.

### Colapso GMC

Gas con SFR alta (ρ > umbral) forma `GmcCluster` con masa `Δm = SFR × dt` y N_* estrellas de la IMF.

### Feedback SN II

Cúmulos con `age < 30 Myr` inyectan energía en el gas cercano:
- ~1 SN II por cada 100 M☉ de masa del cúmulo
- E_SN en radio de inyección R_inj = 0.5

## API pública

| Función/Struct | Descripción |
|----------------|-------------|
| `KroupaImf { m_min, m_max, alpha1, alpha2 }` | IMF de Kroupa |
| `sample_stellar_mass(imf, seed)` | Muestreo de masa |
| `GmcCluster { pos, mass_total, n_stars, age_gyr, metallicity }` | Cúmulo GMC |
| `collapse_gmc(particles, sfr_threshold, dt, seed)` | Identificar y formar cúmulos |
| `inject_sn_from_cluster(clusters, particles, dt, cfg)` | Feedback SN II |

## Tests (6/6 OK)

1–6: masa > 0, IMF en rango, masa conservada, SN II solo jóvenes, metalicidad heredada, N=200 sin panics.

## Referencias

- Kroupa (2001) MNRAS 322, 231
- Kennicutt (1998) ApJ 498, 541
