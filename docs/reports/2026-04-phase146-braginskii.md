# Phase 146 — Viscosidad Braginskii Anisótropa

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Implementar la viscosidad anisótropa de Braginskii para plasmas magnetizados. En el ICM de cúmulos galácticos con alto β, el transporte de momento es esencialmente paralelo al campo magnético.

## Física

### Tensor de presión viscosa de Braginskii

```
π_ij = −η_visc × (b̂_i b̂_j − δ_ij/3) × (∇·v)
```

donde `b̂ = B/|B|` es el versor del campo magnético.

### Discretización SPH

El impulso viscoso en dirección ∥B entre el par `(i,j)`:

```
Δv_i ∝ η_visc × (m_j/ρ_j) × (b̂ · r̂_ij)² × (v_j − v_i) · b̂ × b̂ × W(r_ij) × dt
```

El factor `(b̂ · r̂_ij)²` produce la anisotropía: máxima difusión ∥B, nula ⊥B.

## Archivos

### `crates/gadget-ng-mhd/src/braginskii.rs` (nuevo)

- `apply_braginskii_viscosity(particles, eta_visc, dt)`: aplica viscosidad anisótropa Braginskii

### `crates/gadget-ng-core/src/config.rs`

- `MhdSection.eta_braginskii: f64` (default `0.0` — desactivado)

### `crates/gadget-ng-cli/src/engine.rs`

Hook en `maybe_mhd!`:
```rust
if cfg.mhd.eta_braginskii > 0.0 {
    gadget_ng_mhd::apply_braginskii_viscosity(&mut local, cfg.mhd.eta_braginskii, dt_mhd);
}
```

## Tests

6 tests en `phase146_braginskii.rs`:
1. `viscosity_parallel_b_transfers_momentum` — v difunde ∥B ✓
2. `eta_zero_is_noop` — eta=0 → sin cambio ✓
3. `n_zero_no_crash` — N=0 partículas → no panic ✓
4. `total_momentum_conserved` — Σ p_x conservado ✓
5. `braginskii_anisotropy_z_direction` — B∥ẑ, partículas separadas en z → vz difunde ✓
6. `zero_b_field_no_viscosity` — B=0 → sin viscosidad ✓

**Resultado:** 6/6 tests pasan ✅

## Relevancia astrofísica

- ICM de cúmulos de galaxias: `β >> 1`, viscosidad Braginskii domina sobre isótropa
- Supresión de turbulencia ICM perpendicular a B
- Amortiguamiento de ondas Alfvén: viscosidad paralela actúa sobre la compresión

## Referencias

- Braginskii (1965), Rev. Plasma Phys. 1, 205
- Kunz et al. (2011), MNRAS 410, 2446
- Schekochihin & Cowley (2006), Phys. Plasmas 13, 056501
