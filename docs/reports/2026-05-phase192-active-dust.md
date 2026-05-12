# Phase 192 - Polvo activo silicato/grafito

## Objetivo

Agregar una primera capa COLIBRE-like de polvo activo: especies diferenciadas
silicato/grafito y acoplamiento local al shielding de H2.

## Implementación

- `DustSpeciesModel::{Single, SilicateGraphite}`.
- Nuevos campos en `[sph.dust]`:
  - `species_model`
  - `silicate_fraction`
  - `graphite_fraction`
  - `kappa_silicate_uv`
  - `kappa_graphite_uv`
  - `h2_shielding_boost`
- `effective_dust_uv_opacity` calcula la opacidad UV ponderada por especies.
- `dust_h2_shielding_factor` estima un boost suave por `tau_dust`.
- `update_h2_fraction_with_dust` usa ese boost para aumentar la fracción H2 de
  equilibrio y disminuir la fotodisociación efectiva.
- `gadget-ng stepping` expone flags `--dust-*` para barridos.

## Compatibilidad

El default `species_model = "single"` reproduce el comportamiento anterior:
usa `kappa_dust_uv` y no requiere nuevos campos en `Particle`.

## Validación

`phase192_active_dust.rs` verifica:

- normalización de fracciones silicato/grafito,
- opacidad UV efectiva ponderada,
- aumento de supervivencia H2 con shielding por polvo.
