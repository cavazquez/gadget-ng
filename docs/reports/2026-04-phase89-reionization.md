# Phase 89 — Reionización del Universo: fuentes UV puntuales

**Fecha**: 2026-04-23  
**Crates modificados**: `gadget-ng-rt`, `gadget-ng-core`

## Objetivo

Implementar el módulo de reionización del Universo con fuentes UV puntuales (galaxias, cuásares),
seguimiento del frente de ionización, y cálculo de la fracción media de ionización `<x_HII>`.
El módulo se acopla al solver M1 existente y a la química no-equilibrio (Phase 86).

## Nuevo módulo: `crates/gadget-ng-rt/src/reionization.rs`

### Tipos principales

```rust
pub struct UvSource {
    pub pos: Vec3,
    pub luminosity: f64,   // Ṅ_ion [fotones/s en unidades internas]
}

pub struct ReionizationState {
    pub x_hii_mean: f64,              // <x_HII> promedio
    pub x_hii_sigma: f64,             // desviación estándar
    pub ionized_volume_fraction: f64, // fracción de partículas con x_HII > 0.5
    pub z: f64,
    pub n_sources: usize,
}
```

### Funciones principales

- **`deposit_uv_sources(rad, sources, box_size, dt)`**: depósito de energía UV en el grid M1.
  Usa indexación NGP (Nearest Grid Point) con wrapping periódico. Añade `luminosity × dt / dV`
  a la celda más cercana a cada fuente.

- **`compute_reionization_state(chem_states, z, n_sources)`**: agrega estadísticas de ionización
  desde los `ChemState` de las partículas de gas. Calcula media, sigma y fracción ionizada.

- **`reionization_step(rad, chem_states, sources, m1_params, dt, box_size, z)`**: wrapper
  conveniente que integra `deposit_uv_sources` + `m1_update` en un paso.

- **`stromgren_radius(n_ion_rate, n_h)`**: radio de Strömgren analítico para validación:
  $$R_S = \left(\frac{3\dot{N}_\mathrm{ion}}{4\pi n_H^2 \alpha_B}\right)^{1/3}$$
  con $\alpha_B = 2.6\times10^{-13}$ cm³/s (tasa de recombinación case-B a T = 10⁴ K).

## Configuración TOML (`[reionization]`)

Nuevo campo en `RunConfig` con `ReionizationSection`:

```toml
[reionization]
enabled      = true
n_sources    = 4        # fuentes homogéneamente distribuidas
uv_luminosity = 1.0    # luminosidad por fuente [unidades internas]
z_start      = 12.0
z_end        = 6.0
```

## Tests (9 tests)

| Test | Descripción |
|------|-------------|
| `deposit_uv_sources_increases_energy` | La energía total aumenta exactamente `luminosity × dt` |
| `deposit_uv_sources_periodic_wrapping` | Fuentes en posición negativa se wrapean sin panic |
| `compute_reionization_state_neutral` | Gas neutro → `x_hii_mean ≈ 0` |
| `compute_reionization_state_ionized` | Gas ionizado → `x_hii_mean > 0.7`, fracción > 0.9 |
| `compute_reionization_state_empty` | Slice vacío → sin panic, valores cero |
| `reionization_step_no_crash` | Pipeline completo no hace panic |
| `stromgren_radius_reasonable` | R_S en rango físico (1 kpc – 10 Mpc) |
| `stromgren_radius_scales_correctly` | R_S ∝ Ṅ_ion^(1/3) verificado numéricamente |
| `reionization_state_default_is_zero` | Default trait produce estado neutro |

## Referencia

Rosdahl & Teyssier (2015), MNRAS 449, 4380;  
Pawlik & Schaye (2008), MNRAS 389, 651.
