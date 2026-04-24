# Phase 145 — Reconexión Magnética Sweet-Parker

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Implementar reconexión magnética en el módulo MHD, utilizando el criterio Sweet-Parker para detectar pares de partículas con campos B antiparalelos y liberar la energía magnética correspondiente como calor.

## Física

### Modelo Sweet-Parker

La tasa de reconexión Sweet-Parker es:

```
v_rec = v_A / sqrt(Rm)    donde Rm = L × v_A / η_eff
```

### Implementación SPH

Para cada par `(i, j)` dentro de `2h` con `B_i · B_j < 0`:

```
ΔE_heat = (|B_i|² + |B_j|²) / (2 μ₀) × f_rec × dt
|B|_new = |B| × sqrt(1 − f_rec × dt)   (conservación de flujo)
```

## Archivos

### `crates/gadget-ng-mhd/src/reconnection.rs` (nuevo)

- `apply_magnetic_reconnection(particles, f_reconnection, _gamma, dt)`: aplica reconexión entre pares antiparalelos dentro de 2h
- `sweet_parker_rate(v_a, l_rec, eta_eff) -> f64`: tasa teórica Sweet-Parker

### `crates/gadget-ng-core/src/config.rs`

- `MhdSection.reconnection_enabled: bool` (default `false`)
- `MhdSection.f_reconnection: f64` (default `0.01`)

### `crates/gadget-ng-cli/src/engine.rs`

Hook en `maybe_mhd!`:
```rust
if cfg.mhd.reconnection_enabled {
    gadget_ng_mhd::apply_magnetic_reconnection(
        &mut local, cfg.mhd.f_reconnection, cfg.sph.gamma, dt_mhd,
    );
}
```

## Tests

6 tests en `phase145_reconnection.rs`:
1. `antiparallel_b_releases_heat` — B antiparalelo → calentamiento
2. `parallel_b_no_heat_release` — B paralelo → sin cambio
3. `b_magnitude_decreases_after_reconnection` — |B| se reduce
4. `f_rec_zero_no_op` — f_rec=0 → no-op
5. `far_particles_no_reconnection` — distancia > 2h → sin efecto
6. `sweet_parker_rate_formula` — v_rec = v_A/√Rm verificado

**Resultado:** 6/6 tests pasan ✅

## Referencias

- Sweet (1958), Nuovo Cim. Suppl. 8
- Parker (1957), JGR 62, 509
- Lazarian & Vishniac (1999), ApJ 517, 700
