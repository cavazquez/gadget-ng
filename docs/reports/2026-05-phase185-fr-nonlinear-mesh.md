# Phase 185 — f(R) no lineal en malla

## Objetivo

Completar el candidato grande restante de la cartera de física: pasar del boost PM
homogéneo `4/3` a una aproximación de screening chameleon espacial en la malla PM.

## Implementación

- `[modified_gravity]` suma:
  - `nonlinear_mesh`
  - `mesh_iterations`
  - `screening_smoothing`
- `gadget-ng-pm::fr_screening_field` calcula `S(x) ∈ [0,1]` desde la densidad CIC
  usando `chameleon_field` y `fifth_force_factor`.
- `solve_forces_fr_screened_mesh` resuelve:
  - fuerza GR estándar desde `ρ`;
  - quinta fuerza desde `ρ × S(x)` con amplitud `G/3`;
  - suma ambas componentes.
- `PmSolver` selecciona el camino no lineal cuando `modified_gravity.enabled = true`
  y `modified_gravity.nonlinear_mesh = true`.

## Validación

Nuevo test `phase185_fr_nonlinear_mesh.rs`:

- celdas densas quedan fuertemente screened;
- la fuerza PM screened es menor que el boost homogéneo cerca de una fuente densa;
- serde de los nuevos campos `[modified_gravity]`.

## Limitaciones

Es una aproximación de campo escalar en malla, no un solve multigrid completo de la
ecuación no lineal Hu-Sawicki. Aun así preserva el límite no-screened y agrega
screening espacial estable para runs PM CPU.

