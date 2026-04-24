# Phase 127 — ICs Magnetizadas + CFL Magnético

**Fecha:** 2026-04-23  
**Estado:** ✅ COMPLETADA  
**Tests:** 6/6 pasados

## Objetivo

Agregar condiciones iniciales magnetizadas al módulo MHD y un límite CFL basado en la velocidad de Alfvén para garantizar la estabilidad numérica del integrador SPH-MHD.

## Implementación

### `BFieldKind` (config.rs)

Nuevo enum con cuatro variantes:
- `None` — sin campo inicial (default, backward-compatible)
- `Uniform` — campo uniforme con vector `b0_uniform`
- `Random` — campo aleatorio con amplitud `|b0_uniform|`
- `Spiral` — patrón helicoidal: `B = B₀ × (sin(2πy/L), cos(2πx/L), 0)`

### Campos nuevos en `MhdSection`

```toml
[mhd]
enabled     = true
b0_kind     = "uniform"        # "none" | "uniform" | "random" | "spiral"
b0_uniform  = [1.0, 0.0, 0.0] # nT o unidades internas
cfl_mhd     = 0.3              # número de Courant magnético (default 0.3)
```

### `init_b_field` (induction.rs)

```rust
pub fn init_b_field(particles: &mut [Particle], cfg: &MhdSection, box_size: f64)
```

Inicializa `b_field` en todas las partículas de gas según `b0_kind`. El modo `Random` usa un LCG con `global_id` como semilla para reproducibilidad.

### `alfven_dt` (induction.rs)

```rust
pub fn alfven_dt(particles: &[Particle], cfl: f64) -> f64
```

Calcula `dt_A = cfl × h_min / v_A_max` donde `v_A = |B| / sqrt(μ₀ ρ)`. Retorna `f64::INFINITY` si `B = 0`.

### CFL unificado en `maybe_mhd!` (engine.rs)

```rust
let dt_alfven = gadget_ng_mhd::alfven_dt(&local, cfg.mhd.cfl_mhd);
let dt_mhd = cfg.simulation.dt.min(dt_alfven);
```

El paso efectivo MHD usa `min(dt_global, dt_alfven)` para todos los integradores.

### Inicialización en engine.rs

```rust
if cfg.mhd.enabled && cfg.mhd.b0_kind != BFieldKind::None {
    gadget_ng_mhd::init_b_field(&mut local, &cfg.mhd, cfg.simulation.box_size);
}
```

## Tests

| Test | Descripción |
|------|-------------|
| `none_leaves_b_zero` | `b0_kind=None` → B permanece 0 |
| `uniform_sets_b_correctly` | `Uniform` inicializa con `b0_uniform` exacto |
| `random_preserves_b_magnitude` | `Random` tiene `\|B\| = \|b0_uniform\|` para cada partícula |
| `spiral_spatially_varying` | `Spiral` tiene estructura cos/sin correcta |
| `alfven_dt_finite_with_nonzero_b` | `alfven_dt` finito con B≠0 |
| `alfven_dt_infinite_with_zero_b` | `alfven_dt = ∞` con B=0 |

## Referencias

- Balsara & Spicer (1999), J. Comput. Phys. 149, 270 — estabilidad CFL en MHD.
- Mocz et al. (2016), MNRAS 455, 2110 — ICs magnetizadas en SPH.
