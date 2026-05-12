# Phase 179 — Deuterium/HD Primordial Chemistry

**Fecha:** 2026-05-12

## Resumen

Extiende la red química primordial de Phase 178 de 12 especies:

- HI, HII, HeI, HeII, HeIII, e⁻
- H⁻, H₂, H₂⁺
- D, D⁺, HD

La abundancia primordial de deuterio se fija como `F_D = 2.5e-5` por número relativo a H.
Los campos `x_d`, `x_dp` y `x_hd` usan `serde(default)` para mantener compatibilidad con
checkpoints antiguos de 6 o 9 especies.

## Implementación

- `gadget-ng-rt::ChemState` agrega `x_d`, `x_dp`, `x_hd`.
- `solve_chemistry_implicit` incluye:
  - charge exchange `D + H+ <-> D+ + H`;
  - formación `D+ + H2 -> HD + H+`;
  - destrucción aproximada `HD + H+ -> H2 + D+`.
- `cooling_rate_hd(T, x_hd, n_h)` queda disponible en `gadget-ng-rt`.
- `gadget-ng-sph::cooling_rate_hd(T, n_h, x_hd)` expone el análogo en unidades internas para futuros hooks SPH con estado químico.

## Validación

- Conservación de núcleos H y D.
- Formación traza de HD desde gas con H₂ y D⁺.
- Cooling HD positivo y escalado lineal con abundancia.
- Deserialización de `ChemState` legacy sin campos deuterio.

## Próximo paso natural

Phase 180: Pop III / primeras estrellas, usando H₂/HD como criterio de gas primordial frío.
