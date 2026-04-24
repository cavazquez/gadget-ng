# Phase 124 — Presión Magnética + Tensor de Maxwell en Fuerzas SPH

**Fecha**: 2026-04-23  
**Estado**: ✅ Completada  
**Tiempo estimado**: 1 sesión

## Objetivo

Implementar la presión magnética `P_B = |B|²/(2μ₀)` y el tensor de Maxwell
para las fuerzas SPH magnetizadas.

## Formulación

Tensor de Maxwell:
```
M = B⊗B/μ₀ − P_B·I
```

Aceleración SPH magnética (loop sobre pares i < j):
```
a_i += m_j (M_i/ρ_i² + M_j/ρ_j²) · ∇W_ij
a_j -= m_i (M_i/ρ_i² + M_j/ρ_j²) · ∇W_ij
```

La simetría antisimétrica garantiza conservación del momento.

## Cambios implementados

### `crates/gadget-ng-mhd/src/pressure.rs` (nuevo)

```rust
pub fn magnetic_pressure(b: Vec3) -> f64
pub fn maxwell_stress(b: Vec3) -> [[f64; 3]; 3]
pub fn apply_magnetic_forces(particles: &mut [Particle], dt: f64)
```

## Tests — `phase124_magnetic_forces.rs`

| Test | Resultado |
|------|-----------|
| `pressure_scales_b_squared` | ✅ P_B ∝ B² |
| `maxwell_stress_symmetric` | ✅ M_ij = M_ji |
| `maxwell_trace_minus_p_b` | ✅ Tr(M) = −P_B |
| `magnetic_forces_newton_third_law` | ✅ Δp_0 + Δp_1 = 0 |
| `dm_not_affected_by_magnetic_forces` | ✅ |
| `perpendicular_b_creates_pressure_force` | ✅ |
| `zero_b_zero_pressure` | ✅ |

**Total: 7/7**

## Referencia

- Price & Monaghan (2005), MNRAS 364, 384–406
