# Phase 112 — Partículas Estelares Reales (Spawning)

**Fecha**: 2026-04-23
**Estado**: ✅ Completado

## Objetivo

Implementar la conversión estocástica de partículas de gas en partículas estelares
(`ParticleType::Star`) durante la simulación, integrando el proceso en el motor principal.

## Física implementada

### Probabilidad de spawning

Por partícula de gas con `sfr[i] > sfr_min`:

```
p_spawn = 1 - exp(-sfr[i] × dt / m_i)
```

### Propiedades de la estrella spawneada

- `mass = cfg.m_star_fraction × m_gas`
- `ptype = ParticleType::Star`
- `position`, `velocity` heredados del gas padre
- `metallicity` heredada del gas padre
- `stellar_age = 0`

### Gas padre

- Pierde `m_star = cfg.m_star_fraction × m_gas` de masa
- Si `m_gas < m_gas_min` → marcado para eliminación del vector de partículas

### Integración en `engine.rs`

El macro `maybe_sph!` ahora:

1. Calcula SFR
2. Aplica feedback SN (`apply_sn_feedback`)
3. **Llama `spawn_star_particles`** → obtiene `(new_stars, to_remove)`
4. Elimina partículas de gas agotadas (en orden inverso)
5. Extiende `local` con las nuevas estrellas

Las estrellas no participan en fuerzas SPH (`sph_cosmo_kdk_step` filtra por `ptype = Gas`).

### Nuevos parámetros en `FeedbackSection`

| Campo | Default | Descripción |
|-------|---------|-------------|
| `m_star_fraction` | `0.5` | Fracción de masa gas → estrella |
| `m_gas_min` | `0.01` | Masa mínima para no eliminar gas |

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `gadget-ng-core/src/config.rs` | `m_star_fraction`, `m_gas_min` en `FeedbackSection` |
| `gadget-ng-sph/src/feedback.rs` | `spawn_star_particles` |
| `gadget-ng-sph/src/lib.rs` | Re-exporta `spawn_star_particles` |
| `gadget-ng-cli/src/engine.rs` | Integración en `maybe_sph!` macro |

## Tests

`tests/phase112_stellar_spawning.rs` — 7 tests, todos ✅:

| Test | Descripción |
|------|-------------|
| `high_sfr_produces_stars_eventually` | Spawning ocurre con SFR alta |
| `mass_conserved_on_spawn` | `m_star = m_star_fraction × m_gas` |
| `star_inherits_metallicity_from_gas` | Z heredada del gas padre |
| `dm_particles_dont_spawn_stars` | DM nunca genera estrellas |
| `gas_mass_reduced_after_spawn` | Gas padre pierde masa |
| `gas_below_min_mass_marked_for_removal` | Gas con masa < m_min se elimina |
| `feedback_section_new_fields_serde` | Serde de nuevos campos |
