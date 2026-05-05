# Phase 160 — Validación analítica Zel'dovich pancake

## Objetivo
Agregar un test de referencia analítico pre-caústica para ICs cosmológicas tipo Zel'dovich.

## Modelo
Se usa el mapeo 1D:

- `x(q, a) = q - (delta0/k) * D(a) * sin(k q)`
- `k = 2π/L`
- `rho(q, a) = rho0 / (1 - delta0*D(a)*cos(k q))`

La caústica aparece cuando `delta0*D(a)=1`.

## Cobertura implementada
Archivo de test: `crates/gadget-ng-physics/tests/zeldovich_pancake.rs`

1. `pancake_precaustic_density_matches_analytic`
   - compara densidad numérica (derivada de espaciado local) vs analítica.
2. `pancake_convergence_vs_resolution`
   - verifica convergencia monotónica del error al aumentar `N`.
3. `pancake_convergence_vs_dt`
   - integra una versión temporal EdS simple y verifica convergencia al reducir `dt`.

## Criterio de aceptación
- Error RMS relativo pre-caústica controlado.
- Convergencia monotónica en resolución y paso temporal.

## Alcance y límites
- Valida el régimen lineal/pre-caústica (no post-shell-crossing).
- Está orientado a robustez CI y regresión numérica, no a calibración observacional.
