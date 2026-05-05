# Roadmap: extensiones de física vs estado del código

Este documento implementa la **cartera** descrita en el plan interno: prioriza **Fase A**
(extensiones locales con alto valor) y enlaza cada tema con crates y flags.

## Decisión de track (Fase A cerrada en código)

| Eje | Estado | Implementación |
|-----|--------|----------------|
| CR anisótropo + pérdidas | Integrado | `[sph.cr] anisotropic_diffusion = true` usa `gadget_ng_mhd::diffuse_cr_anisotropic` si `[mhd] enabled`; `hadronic_loss_coeff` aplica `apply_cr_hadronic_losses` tras la difusión. |
| Polvo — presión de radiación | Integrado | `[sph.dust] radiation_pressure_*` + `apply_dust_radiation_pressure_kick` en `stepping` (referencia vertical `z_ref = box_size/2`). |
| Neutrinos — jerarquía + Ω_ν en run | Integrado | `CosmologyParams::from_cosmology_toml` en el motor y en `maybe_fr`; `split_m_nu_ev` + `[cosmology] neutrino_hierarchy` para reparto de masas (IC/extensiones); constante Ω_ν desde `m_nu_ev` como antes. |
| MG solo PM / química 9 especies / light cones | Pendiente (Fases B/C) | Ver tabla extendida en [architecture.md § PM en GPU](architecture.md) y secciones BH/RT. |

## Referencias rápidas por tema

- **Gravedad modificada (quinta fuerza):** [`modified_gravity.rs`](../crates/gadget-ng-core/src/modified_gravity.rs); MG en PM *k*-espacio sigue siendo diseño futuro.
- **Rayos cósmicos:** [`cosmic_rays.rs`](../crates/gadget-ng-sph/src/cosmic_rays.rs), anisótropo [`anisotropic.rs`](../crates/gadget-ng-mhd/src/anisotropic.rs).
- **Cosmología:** [`cosmology.rs`](../crates/gadget-ng-core/src/cosmology.rs) — `NeutrinoHierarchyKind`, `split_m_nu_ev`, `from_cosmology_toml`.

## Tests

- `gadget-ng-core`: invariantes de `split_m_nu_ev` y `from_cosmology_toml`.
- `gadget-ng-sph` / `gadget-ng-physics`: pérdidas CR, serde `CrSection`, polvo (los existentes usan `..Default::default()`).

## MPI y GPU

- Los hooks nuevos están en `stepping.rs` (MPI-safe por rango).
- PM CUDA/HIP no cambia en esta fase; smoke GPU sigue en `gadget-ng-cuda`.
