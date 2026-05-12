# Phase 180 — Pop III / First Stars

**Fecha:** 2026-05-12

## Resumen

Implementa una primera capa de formación estelar Pop III sobre la química primordial H₂/HD:

- criterio local de gas primordial frío;
- IMF top-heavy;
- formación de cúmulos Pop III;
- feedback de pair-instability SN (PISN) con calentamiento y enriquecimiento metálico.

## Implementación

- Nueva configuración `[sph.pop_iii]` (`PopIIISection`):
  - `enabled`;
  - `critical_metallicity`;
  - `density_threshold`;
  - `min_h2_fraction`, `min_hd_fraction`;
  - `max_temperature_k`;
  - parámetros de IMF top-heavy;
  - energía/yield/radio de feedback PISN.
- Nuevo módulo `gadget-ng-sph::pop_iii`:
  - `is_pop_iii_candidate`;
  - `sample_pop_iii_mass`;
  - `form_pop_iii_clusters`;
  - `apply_pop_iii_pisn_feedback`.

## Validación

- `gadget-ng-sph` unit tests:
  - criterio requiere baja Z y moléculas;
  - IMF top-heavy en rango;
  - PISN calienta y enriquece gas.
- `gadget-ng-physics --test phase180_pop_iii`:
  - criterio por H₂/HD;
  - IMF top-heavy;
  - consumo de masa de gas;
  - feedback PISN;
  - serde/default de configuración.

## Limitaciones

El módulo queda como API explícita y testeable. La integración automática en todos los caminos
del engine queda para una fase de wiring posterior junto con radiación Lyman-Werner.

## Próximo paso natural

Phase 181: RT multifrecuencia / Lyman-Werner para fotodisociación H₂/HD y feedback radiativo Pop III.
