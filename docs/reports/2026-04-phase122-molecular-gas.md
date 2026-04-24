# Phase 122 — Gas Molecular HI → H₂

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Implementar la transición HI → H₂ para gas denso y su efecto en la tasa de formación estelar.

## Modelo físico

Para gas denso (`ρ > ρ_H2`):
```
h2_fraction → h2_eq = min(1, ρ/ρ_H2)
h2_fraction += τ × (h2_eq − h2_fraction)   // relajación exponencial
```

Para gas diluido (fotodisociación UV):
```
h2_fraction × exp(−Δt / t_dissoc)
```

SFR con boost molecular:
```
SFR_eff = SFR_base × (1 + sfr_h2_boost × h2_fraction)
```

## Cambios implementados

### `crates/gadget-ng-core/src/particle.rs`

Nuevo campo:
```rust
#[serde(default)]
pub h2_fraction: f64,
```

### `crates/gadget-ng-core/src/config.rs`

Nueva struct `MolecularSection`:
```rust
pub struct MolecularSection {
    pub enabled: bool,
    pub rho_h2_threshold: f64, // default: 100.0
    pub sfr_h2_boost: f64,     // default: 2.0
}
```

### `crates/gadget-ng-sph/src/molecular_gas.rs` (nuevo)

```rust
pub fn update_h2_fraction(particles: &mut [Particle], cfg: &MolecularSection, dt: f64)
```

### `crates/gadget-ng-sph/src/feedback.rs`

Nueva función:
```rust
pub fn compute_sfr_with_h2(particles: &[Particle], cfg: &FeedbackSection, h2_boost: f64) -> Vec<f64>
```

### `crates/gadget-ng-cli/src/engine.rs`

Hook en `maybe_sph!`:
```rust
if cfg.sph.molecular.enabled {
    gadget_ng_sph::update_h2_fraction(&mut local, &cfg.sph.molecular, cfg.simulation.dt);
}
```

## Tests — `phase122_molecular_gas.rs`

| Test | Resultado |
|------|-----------|
| `disabled_no_op` | ✅ |
| `dense_gas_gains_h2` | ✅ Gas denso → h2 > 0 |
| `diffuse_gas_loses_h2` | ✅ Fotodisociación |
| `h2_fraction_bounded` | ✅ ∈ [0, 1] siempre |
| `dm_not_affected` | ✅ DM sin cambio |
| `sfr_boosted_by_h2` | ✅ ratio = 3× con boost=2, h2=1 |

**Total: 6/6**
