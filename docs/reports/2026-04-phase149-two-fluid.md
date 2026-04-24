# Phase 149 — Plasma de Dos Fluidos: T_e ≠ T_i

**Fecha:** 2026-04-23  
**Autor:** gadget-ng automated development  
**Estado:** ✅ Completado

## Objetivo

Implementar el modelo de plasma de dos fluidos donde la temperatura electrónica `T_e` es independiente de la temperatura iónica `T_i`. Relevante para el ICM de cúmulos de galaxias y shocks fuertes donde el tiempo de termalización entre electrones e iones es mayor que el tiempo dinámico.

## Física

### Modelo

- `T_i`: temperatura iónica, derivada de `internal_energy` como `(γ−1) × u`
- `T_e`: temperatura electrónica, almacenada en `Particle.t_electron`

### Acoplamiento Coulomb

```
dT_e/dt = −ν_ei(T_e − T_i)
```

con frecuencia de acoplamiento:

```
ν_ei = ν_coeff × n_e / T_e^{3/2}
```

### Integración implícita exponencial

Para evitar inestabilidades numéricas, se usa la solución exacta:

```
T_e(t+dt) = T_e(t) + (T_i − T_e) × (1 − exp(−ν_ei × dt))
```

## Archivos

### `crates/gadget-ng-mhd/src/two_fluid.rs` (nuevo)

- `apply_electron_ion_coupling(particles, cfg, dt)`: acoplamiento Coulomb electrón-ión
- `mean_te_over_ti(particles) -> f64`: diagnóstico T_e/T_i promedio

### `crates/gadget-ng-core/src/particle.rs`

- Nuevo campo `pub t_electron: f64` (default `0.0`)
- Inicializado a 0.0 en `new()`, `new_gas()`, `new_star()`
- Actualizado en `pack.rs` para correcta serialización/deserialización MPI

### `crates/gadget-ng-core/src/config.rs`

Nueva sección `TwoFluidSection`:
```toml
[two_fluid]
enabled = true
nu_ei_coeff = 1.0    # coeficiente de acoplamiento Coulomb
t_e_init_k = 0.0     # T_e inicial (0 = igual a T_i)
```

### `crates/gadget-ng-cli/src/engine.rs`

Hook en `maybe_sph!`:
```rust
if cfg.two_fluid.enabled {
    gadget_ng_mhd::apply_electron_ion_coupling(&mut local, &cfg.two_fluid, cfg.simulation.dt);
}
```

## Tests

6 tests en `phase149_two_fluid.rs`:

| Test | Descripción | Estado |
|------|-------------|--------|
| `te_zero_initialized_to_ti` | T_e=0 → inicializada a T_i | ✅ |
| `coupling_reduces_temperature_gap` | \|T_e − T_i\| decrece con el tiempo | ✅ |
| `te_always_non_negative` | T_e ≥ 0 siempre | ✅ |
| `mean_te_over_ti_equilibrium` | T_e=T_i → ratio = 1 | ✅ |
| `mean_te_over_ti_below_one_out_of_equilibrium` | T_e<<T_i → ratio < 1 | ✅ |
| `non_gas_particles_ignored` | Partículas DM ignoradas | ✅ |

**Resultado:** 6/6 tests pasan ✅

## Relevancia astrofísica

### ICM de cúmulos de galaxias
- Justo detrás de shocks fuertes: `T_e/T_i ≈ m_e/m_p ≈ 1/1836`
- El acoplamiento Coulomb iguala las temperaturas en `τ_eq ≈ 10⁸ yr` a densidades del ICM
- Observaciones de X-ray miden `T_e`; temperatura iónica controla la dinámica

### Shocks de merger de cúmulos
- Los mergers crean shocks donde `T_e << T_i` durante tiempo ∼ τ_eq
- Bullet Cluster: `T_e/T_i ≈ 0.3` detrás del shock principal

## Referencias

- Spitzer (1962), Physics of Fully Ionized Gases
- Fox & Loeb (1997), ApJ 491, 460
- Rudd & Nagai (2009), ApJL 701, L16
- Markevitch & Vikhlinin (2007), Phys. Rep. 443, 1
