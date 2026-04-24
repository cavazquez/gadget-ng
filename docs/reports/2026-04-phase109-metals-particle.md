# Phase 109 — Metales en `Particle` y `ParticleType::Star`

**Fecha**: 2026-04-23
**Estado**: ✅ Completado

## Objetivo

Extender la estructura `Particle` con campos de metalicidad y edad estelar, introducir la
variante `ParticleType::Star` y agregar la sección de configuración `EnrichmentSection`.
Esto constituye la base para toda la Capa 1 de física bariónica (Phases 109-113).

## Cambios implementados

### `crates/gadget-ng-core/src/particle.rs`

- **`ParticleType::Star`**: nueva variante en el enum `ParticleType`. Las estrellas
  participan en la gravedad pero no en SPH.
- **`Particle::metallicity: f64`** (`#[serde(default)]`): fracción de masa en metales
  Z ∈ [0, 1]. Valor `0.0` para DM primordial y gas sin enriquecer.
- **`Particle::stellar_age: f64`** (`#[serde(default)]`): edad en Gyr desde la formación
  estelar. `0.0` para DM y gas.
- **`Particle::new_star(id, mass, pos, vel, metallicity) -> Self`**: constructor para
  partículas estelares.
- **`Particle::is_star() -> bool`**: predicado de tipo.

### `crates/gadget-ng-core/src/config.rs`

Nueva sección `EnrichmentSection`:

| Campo | Default | Descripción |
|-------|---------|-------------|
| `enabled` | `false` | Activa el módulo de enriquecimiento |
| `yield_snii` | `0.02` | Yield metálico SN II (fracción de masa) |
| `yield_agb` | `0.04` | Yield metálico AGB (fracción de masa) |

La sección `SphSection` ahora incluye `enrichment: EnrichmentSection`.

### `crates/gadget-ng-core/src/lib.rs`

Re-exporta `EnrichmentSection`.

### `crates/gadget-ng-parallel/src/pack.rs`

Agregados `metallicity: 0.0` y `stellar_age: 0.0` al inicializador de `Particle` en
`gather_global`.

## Compatibilidad hacia atrás

Los campos `metallicity` y `stellar_age` usan `#[serde(default)]`, por lo que cualquier
snapshot JSONL anterior deserializa correctamente con `0.0`.

## Tests

`tests/phase109_metals_particle.rs` — 9 tests, todos ✅:

| Test | Descripción |
|------|-------------|
| `new_star_sets_ptype_star` | Constructor fija `ptype = Star` |
| `new_star_stores_metallicity` | Metalicidad almacenada correctamente |
| `new_star_stellar_age_zero` | `stellar_age = 0` al nacer |
| `is_star_and_is_gas_exclusive` | Predicados mutuamente excluyentes |
| `particle_type_three_variants` | Las tres variantes existen y son distintas |
| `metallicity_serde_roundtrip` | Serialización/deserialización sin pérdida |
| `backward_compat_missing_metal_fields` | JSON sin campos → defaults a 0 |
| `enrichment_section_defaults` | Defaults correctos para `EnrichmentSection` |
| `enrichment_section_serde_roundtrip` | Serde de `EnrichmentSection` |
