# Phase 123 — Crate gadget-ng-mhd + b_field + Ecuación de Inducción SPH

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1–2 sesiones

## Objetivo

Crear la infraestructura base para MHD ideal: nuevo crate `gadget-ng-mhd`,
campo `b_field: Vec3` en `Particle`, y la ecuación de inducción dB/dt = ∇×(v×B).

## Arquitectura del crate

```
crates/gadget-ng-mhd/
  Cargo.toml
  src/
    lib.rs       ← re-exports + MU0 = 1.0
    induction.rs ← dB/dt = ∇×(v×B) SPH
    pressure.rs  ← P_B + tensor Maxwell + fuerzas
    cleaning.rs  ← Dedner div-B cleaning
```

## Ecuación de inducción SPH

Formulación de Morris & Monaghan (1997):
```
dB_i/dt = Σ_j (m_j/ρ_j) [(B_ij·∇W_ij) v_ij − (v_ij·∇W_ij) B_ij]
```

donde `v_ij = v_i − v_j`, `B_ij = B_i − B_j`.

## Cambios implementados

### `crates/gadget-ng-core/src/particle.rs`

Nuevos campos:
```rust
#[serde(default)]
pub b_field: Vec3,   // Phase 123

#[serde(default)]
pub psi_div: f64,    // Phase 125
```

### `Cargo.toml` (workspace)

```toml
"crates/gadget-ng-mhd",
```

### `crates/gadget-ng-mhd/src/induction.rs` (nuevo)

```rust
pub fn advance_induction(particles: &mut [Particle], dt: f64)
```

## Tests — `phase123_mhd_induction.rs`

| Test | Resultado |
|------|-----------|
| `b_field_default_zero` | ✅ |
| `induction_changes_b_with_shear` | ✅ |
| `uniform_b_no_velocity_constant` | ✅ B uniforme + v=0 → B constante |
| `dm_not_affected_by_induction` | ✅ |
| `magnetic_pressure_correct` | ✅ P_B = 0.5 para B=(1,0,0) |
| `dedner_step_no_nan` | ✅ |

**Total: 6/6**

## Referencia

- Morris & Monaghan (1997), J. Comput. Phys. 136, 41–60
- Price & Monaghan (2005), MNRAS 364, 384–406
