# Phase 121 — Conducción Térmica ICM (Spitzer)

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Implementar la conducción térmica del gas intracúmulo (ICM) siguiendo el modelo de Spitzer (1962)
con factor de supresión magnético/turbulento ψ.

## Modelo físico

La conductividad efectiva es:
```
κ_eff = κ_Spitzer × ψ × T_mean^{5/2} / log_Coulomb
```

El flujo de calor entre pares de partículas SPH (loop sobre pares únicos i < j):
```
q_ij = κ_eff × (T_j − T_i) × W(r_ij, h_ij) × Δt
```

Con conservación exacta de energía:
- `Δu_i = +q_ij` (i recibe calor de j si j está más caliente)
- `Δu_j = −q_ij` (j cede calor, simetría exacta)

## Cambios implementados

### `crates/gadget-ng-core/src/config.rs`

Nueva struct `ConductionSection`:
```rust
pub struct ConductionSection {
    pub enabled: bool,
    pub kappa_spitzer: f64,   // default: 1e-4
    pub psi_suppression: f64, // default: 0.1
}
```

Agregada como `pub conduction: ConductionSection` en `SphSection`.

### `crates/gadget-ng-sph/src/thermal_conduction.rs` (nuevo)

```rust
pub fn apply_thermal_conduction(
    particles: &mut [Particle],
    cfg: &ConductionSection,
    gamma: f64,
    t_floor_k: f64,
    dt: f64,
)
```

### `crates/gadget-ng-cli/src/engine.rs`

Hook en `maybe_sph!`:
```rust
if cfg.sph.conduction.enabled {
    gadget_ng_sph::apply_thermal_conduction(&mut local, &cfg.sph.conduction,
                                             cfg.sph.gamma, cfg.sph.t_floor_k, cfg.simulation.dt);
}
```

## Tests — `phase121_thermal_conduction.rs`

| Test | Resultado |
|------|-----------|
| `disabled_no_op` | ✅ |
| `heat_flows_hot_to_cold` | ✅ Partícula fría gana energía |
| `respects_t_floor` | ✅ u ≥ 0 siempre |
| `distant_particles_no_interaction` | ✅ Kernel cero para r >> h |
| `zero_psi_suppression_no_conduction` | ✅ ψ=0 → sin transferencia |
| `dm_particles_not_affected` | ✅ DM sin cambio |

**Total: 6/6**

## Referencia

- Spitzer (1962), Physics of Fully Ionized Gases
- Narayan & Medvedev (2001), ApJ 562, L129
- Dolag et al. (2004), ApJ 606, L97
