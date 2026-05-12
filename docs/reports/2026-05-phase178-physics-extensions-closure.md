# Phase 178 — Physics Extensions Closure

**Fecha:** 2026-05-12

## Resumen

Se cerraron las tres entradas pendientes del roadmap de extensiones de física:

- **MG solo PM:** `gadget-ng-pm::solve_forces_modified_gravity` añade el límite f(R) no-screened en espacio-k como `G_eff = 4G/3`, con `f_R0 = 0` idéntico a GR.
- **Química 9 especies:** `ChemState` pasa de 6 a 9 especies primordiales: HI, HII, HeI, HeII, HeIII, e⁻, H⁻, H₂ y H₂⁺. Los campos nuevos tienen `serde(default)` para compatibilidad de checkpoints viejos.
- **Lightcones:** el módulo `gadget_ng_analysis::lightcone` ya contiene cruces de cono de luz y acumulación Born para κ/γ; el roadmap queda actualizado para no marcarlo como pendiente.

## Validación

- `cargo test -p gadget-ng-rt`
- `cargo test -p gadget-ng-pm`
- `cargo test -p gadget-ng-physics --test phase95_eor`
- `cargo check -p gadget-ng-cli`

## Limitaciones

El PM f(R) implementa el límite homogéneo de baja densidad. El screening chameleon local existente permanece en `gadget-ng-core::apply_modified_gravity`; una solución no lineal de campo escalar en malla queda fuera de esta fase.
