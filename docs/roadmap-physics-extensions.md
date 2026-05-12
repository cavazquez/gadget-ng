# Roadmap: extensiones de física vs estado del código

Este documento implementa la **cartera** descrita en el plan interno: prioriza **Fase A**
(extensiones locales con alto valor) y enlaza cada tema con crates y flags.

## Decisión de track (Fase A cerrada en código)

| Eje | Estado | Implementación |
|-----|--------|----------------|
| CR anisótropo + pérdidas | Integrado | `[sph.cr] anisotropic_diffusion = true` usa `gadget_ng_mhd::diffuse_cr_anisotropic` si `[mhd] enabled`; `hadronic_loss_coeff` aplica `apply_cr_hadronic_losses` tras la difusión. |
| Polvo — presión de radiación | Integrado | `[sph.dust] radiation_pressure_*` + `apply_dust_radiation_pressure_kick` en `stepping` (referencia vertical `z_ref = box_size/2`). |
| Neutrinos — jerarquía + Ω_ν en run | Integrado | `CosmologyParams::from_cosmology_toml` en el motor y en `maybe_fr`; `split_m_nu_ev` + `[cosmology] neutrino_hierarchy` para reparto de masas (IC/extensiones); constante Ω_ν desde `m_nu_ev` como antes. |
| MG solo PM / química 9 especies / light cones | Cerrado (Phase 178) | PM f(R) homogéneo en `gadget-ng-pm`, red primordial de 9 especies en `gadget-ng-rt`, lightcones/Born ya en `gadget-ng-analysis`. |
| Química D/HD + cooling primordial | Cerrado (Phase 179) | `ChemState` de 12 especies con `D`, `D+`, `HD`; `cooling_rate_hd` en RT/SPH. |
| Pop III / primeras estrellas | Cerrado (Phase 180) | `PopIIISection`, criterio H₂/HD + baja Z, IMF top-heavy y feedback PISN en `gadget-ng-sph`. |

## Referencias rápidas por tema

- **Gravedad modificada (quinta fuerza):** [`modified_gravity.rs`](../crates/gadget-ng-core/src/modified_gravity.rs); el límite PM homogéneo en *k*-espacio está en `gadget-ng-pm` desde Phase 178.
- **Rayos cósmicos:** [`cosmic_rays.rs`](../crates/gadget-ng-sph/src/cosmic_rays.rs), anisótropo [`anisotropic.rs`](../crates/gadget-ng-mhd/src/anisotropic.rs).
- **Cosmología:** [`cosmology.rs`](../crates/gadget-ng-core/src/cosmology.rs) — `NeutrinoHierarchyKind`, `split_m_nu_ev`, `from_cosmology_toml`.

## Candidatos Phase 179+

| Prioridad | Candidato | Qué agregaría | Crates probables | Estimación |
|-----------|-----------|---------------|------------------|------------|
| 1 | f(R) no lineal en malla | Solver iterativo del campo escalar con screening chameleon espacial, más allá del boost PM homogéneo `4/3 G`. | `gadget-ng-pm`, `gadget-ng-core`, `gadget-ng-physics` | 3–5 sesiones |
| 2 | Química D/HD + cooling primordial | **Cerrado Phase 179.** Especies `D`, `D+`, `HD`; cooling molecular HD y acoplamiento con la red H⁻/H₂/H₂⁺. | `gadget-ng-rt`, `gadget-ng-sph`, `gadget-ng-physics` | Hecho |
| 3 | Pop III / primeras estrellas | **Cerrado Phase 180.** IMF top-heavy, criterio de formación con H₂/HD, feedback PISN. | `gadget-ng-sph`, `gadget-ng-rt`, `gadget-ng-core` | Hecho |
| 4 | Polvo IR / emisión térmica | Temperatura de granos, emisión IR, opacidades dependientes de frecuencia y acoplamiento con RT. | `gadget-ng-sph`, `gadget-ng-rt`, `gadget-ng-analysis` | 2–3 sesiones |
| 5 | AGN spin + mergers | Spin de BH, eficiencia radiativa dependiente de spin, mergers de BH y recoil gravitacional. | `gadget-ng-sph`, `gadget-ng-analysis`, `gadget-ng-core` | 2–4 sesiones |
| 6 | Warm / fuzzy dark matter | Cutoff WDM en transferencia/ICs; presión cuántica efectiva para fuzzy DM como aproximación hidrodinámica. | `gadget-ng-core`, `gadget-ng-analysis`, `gadget-ng-physics` | 2–4 sesiones |
| 7 | RT multifrecuencia | Grupos HI/HeI/HeII/LW/IR para fotoquímica y reionización espectralmente resuelta. | `gadget-ng-rt`, `gadget-ng-sph`, `gadget-ng-physics` | 3–5 sesiones |

**Recomendación inmediata:** seguir con **RT multifrecuencia / Lyman-Werner**, ahora que Pop III necesita fotodisociación H₂/HD y feedback radiativo espectral.

## Tests

- `gadget-ng-core`: invariantes de `split_m_nu_ev` y `from_cosmology_toml`.
- `gadget-ng-sph` / `gadget-ng-physics`: pérdidas CR, serde `CrSection`, polvo (los existentes usan `..Default::default()`).

## MPI y GPU

- Los hooks nuevos están en `stepping.rs` (MPI-safe por rango).
- PM CUDA/HIP no cambia en esta fase; smoke GPU sigue en `gadget-ng-cuda`.
