# Phase 183 — AGN spin + mergers

## Objetivo

Extender el modelo AGN existente con spin Kerr fenomenológico, eficiencia radiativa
dependiente de spin, crecimiento de spin por acreción y fusiones de agujeros negros
con recoil gravitacional reducido.

## Implementación

- `BlackHole` suma `spin` y `velocity` con `serde(default)` para mantener checkpoints
  legacy.
- `radiative_efficiency_from_spin` aproxima la eficiencia de disco delgado Kerr.
- `spin_dependent_feedback_efficiency` escala el feedback AGN existente.
- `spin_up_by_accretion` acerca el spin al límite progrado durante acreción coherente.
- `merge_black_holes` fusiona BHs cercanos, conserva masa y momento, promedia spin por
  masa y aplica un recoil fenomenológico.
- `[sph.agn]` suma `spin_enabled`, `initial_spin`, `mergers_enabled`,
  `merger_radius` y `recoil_velocity_scale`.

## Validación

Nuevo test `phase183_agn_spin_mergers.rs`:

- eficiencia Kerr crece con spin progrado;
- feedback efectivo escala con spin;
- la acreción incrementa spin de un BH no rotante;
- BHs cercanos se fusionan conservando masa;
- serde de nuevos campos `[sph.agn]`.

## Limitaciones

El spin es escalar y supone alineación efectiva con el disco. El recoil es una ley
reducida de orden fenomenológico, no una fórmula NR calibrada completa.

