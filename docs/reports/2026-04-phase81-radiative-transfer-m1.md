# Phase 81 — Transferencia Radiativa M1

**Nuevo crate**: `gadget-ng-rt`  
**Fecha**: 2026-04

## Resumen

Implementación del solver de transferencia radiativa con cierre M1 (Morel 2000)
acoplado al gas SPH. Modela el transporte de fotones UV (fotoionización de HI)
e IR (calentamiento del gas), con velocidad de luz reducida para pasos de tiempo
manejables.

## Física del cierre M1

El modelo M1 traza los dos primeros momentos del campo de radiación:
- **E(r,t)**: densidad de energía radiativa [erg/cm³]
- **F(r,t)**: flujo radiativo [erg/cm²/s]

El factor de Eddington de cierre M1 es:

```
f(ξ) = (3 + 4ξ²) / (5 + 2√(4 - 3ξ²))     ξ = |F| / (c_red × E)
```

Asintótico: `f→1/3` para campo isótropo, `f→1` para streaming libre.

## Ecuaciones hiperbólicas

```
∂E/∂t + ∇·F = η - c_red × κ × E
∂F/∂t + c²_red ∇(f E) = -c_red × κ × F
```

## Implementación

- **Solver HLL**: Harten-Lax-van Leer, primer orden en espacio y tiempo.
- **Sub-stepping automático**: `n_sub = max(dt/dt_CFL, substeps)` para estabilidad.
- **Velocidad de luz reducida**: `c_red = c / c_red_factor` (default: 100).
- **Fuente implícita**: `E_new = (E_adv + η dt) × exp(-c_red κ dt)`.

## Estructura del crate

```
gadget-ng-rt/
  src/
    lib.rs      ← exports
    m1.rs       ← RadiationField, M1Params, m1_update, eddington_factor
    coupling.rs ← photoionization_rate, apply_photoheating, deposit_gas_emission
  Cargo.toml
```

## Acoplamiento con SPH

1. Gas → Radiación: `deposit_gas_emission` (bremsstrahlung/recombinación).
2. Radiación → Gas: `apply_photoheating` (energía interna += Γ × dt).
3. Splitting de operadores de primer orden.

## Nuevos tipos

```rust
pub struct RadiationField { energy_density, flux_x/y/z, nx, ny, nz, dx }
pub struct M1Params       { c_red_factor, kappa_abs, kappa_scat, substeps }
```

## Configuración TOML

```toml
[rt]
enabled      = true
c_red_factor = 100.0
kappa_abs    = 1.0
rt_mesh      = 32
substeps     = 5
```

## Tests

15 tests en `gadget-ng-rt`:

**m1::tests** (10):
- `eddington_factor_isotropic` — f(ξ=0)=1/3
- `eddington_factor_streaming` — f(ξ=1)=1
- `eddington_factor_monotone`
- `radiation_field_uniform`
- `radiation_field_total_energy`
- `m1_update_conserves_energy_vacuum` — conservación en vacío
- `m1_update_absorption_decays` — absorción decae la energía
- `m1_update_energy_positive` — E≥0 siempre
- `hll_flux_symmetric` — flujo nulo para estados iguales

**coupling::tests** (6):
- `photoionization_rate_zero_for_empty_field`
- `photoionization_rate_positive_for_uv_field`
- `photoheating_increases_internal_energy`
- `dm_particle_not_heated`
- `emission_deposits_energy`
- `coupling_step_no_crash`

## Referencia

- Morel (2000), J. Quant. Spectrosc. Radiat. Transf. 65, 769
- González et al. (2007), A&A 464, 429
- Rosdahl et al. (2013), MNRAS 436, 2188 (RAMSES-RT)
- Gnedin & Abel (2001), New Astron. 6, 437 (velocidad de luz reducida)
