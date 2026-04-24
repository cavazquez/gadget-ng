# Phase 125 — Dedner div-B Cleaning

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Implementar el esquema de Dedner et al. (2002) para eliminar errores de divergencia
del campo magnético (∇·B ≠ 0) acumulados durante la evolución SPH.

## Formulación

Sistema de ecuaciones de Dedner:
```
∂B/∂t + ∇ψ = 0
∂ψ/∂t + c_h² ∇·B = −c_r ψ
```

Integración explícita:
1. Calcular `div_B_i = Σ_j (m_j/ρ_j) (B_j − B_i)·∇W_ij`
2. Calcular `∇ψ_i = Σ_j (m_j/ρ_j) (ψ_j − ψ_i) ∇W_ij`
3. `ψ_new = ψ × exp(−c_r dt) − c_h² × div_B × dt`
4. `B_new = B − ∇ψ × dt`

## Cambios implementados

### `crates/gadget-ng-core/src/particle.rs`

Campo `psi_div: f64` (ya agregado en Phase 123).

### `crates/gadget-ng-mhd/src/cleaning.rs` (nuevo)

```rust
pub fn dedner_cleaning_step(
    particles: &mut [Particle],
    c_h: f64,
    c_r: f64,
    dt: f64,
)
```

## Tests — `phase125_dedner_cleaning.rs`

| Test | Resultado |
|------|-----------|
| `psi_div_default_zero` | ✅ |
| `psi_decays_with_damping` | ✅ ψ decrece con c_r > 0 |
| `no_damping_preserves_psi_magnitude` | ✅ c_r=0 → sin disipación |
| `b_changes_with_nonzero_psi` | ✅ Gradiente de ψ corrige B |
| `dm_not_affected` | ✅ |
| `no_nan_multi_particle` | ✅ 20 partículas, sin NaN |

**Total: 6/6**

## Referencia

- Dedner et al. (2002), J. Comput. Phys. 175, 645–673
- Tricco & Price (2012), J. Comput. Phys. 231, 7214
