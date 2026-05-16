# Phase 186 — MHD Hall Term

**Fecha:** 2026-05-16  
**Crates afectados:** `gadget-ng-core`, `gadget-ng-mhd`, `gadget-ng-cli`, `gadget-ng-physics`

---

## Motivación

El efecto Hall es el segundo componente de la trilogía de MHD no ideal (NIMHD):

| Término | Fase | Física | Disipa energía B |
|---------|------|--------|-----------------|
| Resistividad óhmica | 135 | Amortiguamiento de B proporcional a J² | Sí |
| Hall drift | **186** | Rotación de B por deriva electrónica | **No** |
| Difusión ambipolar | 194 | Amortiguamiento en gas poco ionizado | Sí |

El término Hall es relevante en:
- Discos protoestelares (formación de planetas, campo magnético de T Tauri)
- Galácticas tempranas con gas poco ionizado
- Corteza de estrellas de neutrones (magnetares)
- Plasma del IGM con fracción de ionización baja

---

## Física implementada

La ecuación de inducción con término Hall es:

```
∂B/∂t = ∇ × [(v × B) - η_H (J × B) / (|B| ρ)]
```

donde `J = ∇ × B / μ₀` es la densidad de corriente. El término Hall tiene velocidad de deriva:

```
v_Hall = -η_H (J × B) / (|B|² ρ)
```

### Modelo local para SPH (sin lista de vecinos)

Para una implementación local consistente con la difusión ambipolar (Phase 194), aproximamos J con la dinámica de los electrones locales y usamos `v × B` como proxy del eje de corriente. El efecto neto es una **rotación de B** alrededor del eje `k = (v × B) / |v × B|`:

```
B_new = B cos θ + (k × B) sin θ + k (k·B)(1 − cos θ)
```

con ángulo de rotación:

```
θ = η_H × |B| / ρ_proxy × dt
     ρ_proxy = mass / h³   si h > 0
               mass         si h = 0
```

Esta rotación conserva `|B|` exactamente (fórmula de Rodrigues) y no modifica la energía interna, a diferencia del término ambipolar.

---

## Implementación

### `crates/gadget-ng-core/src/vec3.rs`

Añadida función `Vec3::cross(self, other: Self) -> Self` (producto vectorial).  
Era una omisión notable; el crate ya tenía `dot()` y `norm()`.

### `crates/gadget-ng-mhd/src/nonideal.rs`

- `hall_drift_particle(p, eta_hall, dt)`: función privada por partícula.
- `apply_hall_drift(particles, eta_hall, dt)`: función pública con dispatch:
  - **Rayon** (`feature = "rayon"`): `par_iter_mut` paralelo.
  - **AVX-512F** (`feature = "simd"`, x86_64): lotes de 8 partículas; `sin/cos` en escalar por lane (igual que `exp()` en ambipolar).
  - **AVX2+FMA** (`feature = "simd"`, x86_64): lotes de 4 partículas.
  - **Serial**: fallback universal.

### `crates/gadget-ng-core/src/config/sections/mhd.rs`

Nuevos campos en `MhdSection`:

```toml
[mhd]
hall_enabled = true    # Default: false
hall_eta     = 0.1     # Default: 0.0 (desactivado)
```

### `crates/gadget-ng-cli/src/engine/stepping/context.rs`

Bloque wired en `step_mhd`, justo después de la difusión ambipolar:

```rust
if cfg.mhd.hall_enabled && cfg.mhd.hall_eta > 0.0 {
    gadget_ng_mhd::apply_hall_drift(local, cfg.mhd.hall_eta, dt_mhd);
}
```

### `configs/experiments/phase186_hall_mhd.toml`

Ejemplo de configuración con Hall habilitado (`hall_eta = 0.1`).

---

## Tests

### `crates/gadget-ng-mhd/src/nonideal.rs` (unit tests)

5 tests unitarios inline en el módulo:
- `hall_drift_conserves_b_magnitude`: |B| inalterado tras un paso.
- `hall_drift_rotates_b_direction`: B rota cuando v ⊥ B.
- `hall_drift_no_effect_on_dm`: DM ignorada.
- `hall_drift_no_effect_with_zero_eta`: η_H = 0 → sin cambio.
- `hall_drift_no_effect_with_parallel_v_b`: v ∥ B → sin rotación (eje nulo).

### `crates/gadget-ng-physics/tests/phase186_mhd_hall.rs`

7 tests de validación física (todos pasan):

| Test | Resultado |
|------|-----------|
| `hall_drift_conserves_b_magnitude` | ok |
| `hall_drift_rotates_b_direction_when_v_perp_b` | ok |
| `hall_drift_no_rotation_when_v_parallel_b` | ok |
| `hall_drift_does_not_affect_dark_matter` | ok |
| `hall_drift_zero_eta_leaves_b_unchanged` | ok |
| `hall_drift_conserves_b_over_many_steps` (100 pasos, Δ|B| < 1e-10) | ok |
| `hall_drift_does_not_change_internal_energy` | ok |

---

## Limitaciones y trabajo futuro

1. **Modelo local**: la aproximación `J ≈ ρ v / (e m)` ignora gradientes espaciales de B. Una implementación completa requeriría una suma SPH sobre vecinos para calcular `∇ × B` explícitamente (similar a Price & Wurster 2017).

2. **CUDA**: no implementado en esta fase. El kernel CUDA para Hall sería análogo a `cuda_ambipolar_diffusion_gpu` y se puede añadir en AP-20.

3. **Estabilidad CFL**: el término Hall introduce un límite de paso de tiempo proporcional a `h² / (η_H |B|)`. Para simulaciones reales con `hall_eta` grande, puede ser necesario un subciclo.

---

## Referencias

- Wardle (1999) — Hall Effect in Star Formation
- Balbus & Terquem (2001) — Linear Analysis of the Hall Effect in Protostellar Disks
- Price & Wurster (2017) — Phantom: Smoothed Particle Magnetohydrodynamics (Sec. 4.4 Hall)
