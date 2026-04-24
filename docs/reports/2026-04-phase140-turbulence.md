# Phase 140 — Turbulencia MHD: Forzado Ornstein-Uhlenbeck

**Fecha:** 2026-04-23  
**Estado:** Completado ✓  
**Tests:** 6/6 passing

## Objetivo

Implementar forzado estocástico de turbulencia MHD usando el proceso Ornstein-Uhlenbeck (OU),
generando turbulencia Alfvénica con espectro de Kolmogorov (`E(k) ∝ k^{-5/3}`) o
Goldreich-Sridhar (`E(k) ∝ k^{-3/2}`) configurable.

## Física

### Proceso Ornstein-Uhlenbeck

El forzado se modela como un proceso estocástico estacionario:

```
dA/dt = −A/τ_c + σ × η(t)
```

donde `σ = √(2A²/τ_c)` y `η(t)` es ruido blanco gaussiano. En cada paso de tiempo:

```
A(t+dt) = A(t) × exp(−dt/τ_c) + σ × N(0,1) × √(dt)
```

### Espectro de Potencia

Los modos de forzado en el rango k ∈ [k_min, k_max] se pesan con:
- **Kolmogorov**: `w(k) = k^{-5/6}` → `P(k) ∝ k^{-5/3}` (turbulencia hidrodinámica)
- **Goldreich-Sridhar**: `w(k) = k^{-3/4}` → `P(k) ∝ k^{-3/2}` (turbulencia con campo B₀)

### Estadísticas de Turbulencia

- **Número de Mach sónico**: `M = v_rms / c_s`
- **Número de Mach Alfvénico**: `M_A = v_rms / v_A`

Clasificación:
- M < 1: subsónico (ISM frío)
- M > 1: supersónico (ISM denso, SNe)
- M_A < 1: sub-Alfvénico (dominado por B)
- M_A > 1: super-Alfvénico (turbulencia dinámica)

## Implementación

### Archivo nuevo: `crates/gadget-ng-mhd/src/turbulence.rs`

- **`apply_turbulent_forcing(particles, cfg, dt, step)`**: Forzado OU en el campo de
  velocidades con espectro de potencia configurable y semilla reproducible por paso.
- **`turbulence_stats(particles, gamma) → (mach_rms, alfven_mach)`**: Estadísticas de
  turbulencia calculadas sobre todas las partículas de gas.

### Nueva sección de configuración: `TurbulenceSection`

```toml
[turbulence]
enabled = true
amplitude = 1e-3          # amplitud del forzado
correlation_time = 1.0    # tiempo de correlación τ_c [unidades internas]
k_min = 1.0               # modo mínimo de forzado
k_max = 4.0               # modo máximo de forzado
spectral_index = 1.667    # 5/3 Kolmogorov, 3/2 Goldreich-Sridhar
```

## Tests

| Test | Descripción |
|------|-------------|
| `disabled_no_effect` | disabled → velocidades no cambian |
| `zero_amplitude_no_op` | amplitude=0 → sin perturbación |
| `enabled_perturbs_velocities` | forzado activo modifica velocidades |
| `stats_zero_velocity` | Mach=0 con v=0 |
| `turbulence_section_defaults` | Valores por defecto correctos |
| `stats_positive_mach_with_velocity` | Mach>0 con v>0 |

## Referencias

- Schmidt et al. (2006), A&A 450, 265
- Goldreich & Sridhar (1995), ApJ 438, 763
- Federrath et al. (2010), A&A 512, A81
