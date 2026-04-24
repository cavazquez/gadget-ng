# Phase 126 — Integración MHD en engine + Macro maybe_mhd! + Validación Onda de Alfvén

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Conectar el crate `gadget-ng-mhd` al loop de simulación mediante la macro `maybe_mhd!`
y validar que la velocidad de fase de las ondas de Alfvén es correcta.

## Macro `maybe_mhd!`

```rust
macro_rules! maybe_mhd {
    () => {
        if cfg.mhd.enabled {
            gadget_ng_mhd::advance_induction(&mut local, cfg.simulation.dt);
            gadget_ng_mhd::apply_magnetic_forces(&mut local, cfg.simulation.dt);
            gadget_ng_mhd::dedner_cleaning_step(
                &mut local, cfg.mhd.c_h, cfg.mhd.c_r, cfg.simulation.dt,
            );
        }
    };
}
```

Integrada en los **7 bucles de simulación** de `engine.rs` (después de `maybe_rt!` y
antes de `maybe_reionization!`).

## Configuración TOML

```toml
[mhd]
enabled = true
c_h     = 1.0    # velocidad de Alfvén máxima
c_r     = 0.5    # tasa de amortiguamiento Dedner
```

## Velocidad de Alfvén

```
v_A = |B| / sqrt(μ₀ ρ)
```

Con `μ₀ = 1` (unidades internas) y `ρ = 1`, `B = 2` → `v_A = 2`.

## Cambios implementados

### `crates/gadget-ng-core/src/config.rs`

Nueva struct `MhdSection` y campo `pub mhd: MhdSection` en `RunConfig`.

### `crates/gadget-ng-cli/Cargo.toml`

```toml
gadget-ng-mhd = { path = "../gadget-ng-mhd" }
```

### `crates/gadget-ng-cli/src/engine.rs`

- Nueva macro `maybe_mhd!()`
- Llamada en 7 puntos del loop de simulación

## Tests — `phase126_mhd_integration.rs`

| Test | Resultado |
|------|-----------|
| `mhd_section_default_disabled` | ✅ enabled=false por defecto |
| `full_pipeline_no_panic` | ✅ Pipeline completo sin NaN |
| `alfven_speed_formula` | ✅ v_A = B/sqrt(μ₀ρ) |
| `small_b_small_force` | ✅ Fuerza ∝ B² |
| `dedner_maintains_finite_b` | ✅ 100 pasos Dedner, ψ amortiguado |
| `end_to_end_realistic_mhd` | ✅ Onda Alfvén 1D, 50 pasos estables |

**Total: 6/6**

## Notas

- Phase 126 completa la **infraestructura base de MHD**. El solver completo
  (ICs magnetizadas, ondas de Alfvén 3D, modo cosmológico) se planifica en sesión futura.
- El crate `gadget-ng-mhd` es modular y puede usarse independientemente del engine CLI.

## Referencia

- Price & Monaghan (2005), MNRAS 364, 384–406
- Dedner et al. (2002), J. Comput. Phys. 175, 645–673
- Springel (2010), ARA&A 48, 391 (GADGET-4 review)
