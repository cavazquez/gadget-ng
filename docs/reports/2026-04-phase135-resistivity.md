# Phase 135 — Resistividad Numérica Artificial

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar resistividad numérica artificial al campo magnético para suavizar discontinuidades
y mejorar la estabilidad numérica del solver MHD (esquema de Price 2008).

## Física

La ecuación de inducción MHD tiene la forma:
```
∂B/∂t = ∇×(v×B) + η∇²B
```

donde `η` es la resistividad. En simulaciones SPH, las discontinuidades de B pueden generar
inestabilidades numéricas. La resistividad artificial de Price (2008) introduce un término
difusivo proporcional a la velocidad de señal local:

```
(∂B_i/∂t)_η = η_art × Σ_j m_j/ρ_j × (B_j − B_i) × 2|∇W_ij|/|r_ij|
```

donde `η_art = α_B × h_i × v_sig` y `v_sig = |v_j − v_i|`.

## Implementación

### Modificación: `crates/gadget-ng-mhd/src/induction.rs`

Nueva función `apply_artificial_resistivity(particles, alpha_b, dt)`:
- Calcula `η_art = alpha_b × h_i × |Δv_ij|` para cada par de partículas
- Aplica difusión SPH del campo B con peso del gradiente del kernel
- Preserva campos uniformes (sin gradiente → sin difusión)
- Con `alpha_b = 0.0` → no-op

### Modificaciones de configuración

- `MhdSection.alpha_b: f64` (default: `0.5`)
- Engine: integrado en `maybe_mhd!` como paso opcional antes de `apply_magnetic_forces`

## Tests

| Test | Descripción |
|------|-------------|
| `zero_alpha_no_op` | alpha_b=0 → sin cambio en B |
| `resistivity_smooths_b_discontinuity` | Difunde discontinuidad B |
| `uniform_b_no_change` | Campo uniforme → sin difusión |
| `b_remains_finite_after_many_steps` | Estabilidad a largo plazo |
| `resistivity_reduces_b_gradient` | Gradiente ΔB decrece con el tiempo |
| `larger_alpha_b_more_diffusion` | Difusión ∝ alpha_b |

## Referencias

- Price (2008), J. Comput. Phys. 227, 10040 — resistividad artificial en SPH-MHD
- Balsara & Kim (2004), ApJ 602, 1079 — preservación ∇·B y estabilidad
