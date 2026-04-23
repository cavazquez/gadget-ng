# Phase 90 — Perfil de temperatura del IGM T(z)

**Fecha**: 2026-04-23  
**Crates modificados**: `gadget-ng-rt`, `gadget-ng-core`, `gadget-ng-cli`

## Objetivo

Implementar el cálculo del perfil de temperatura del gas intergaláctico (IGM) T(z)
a partir de los estados de energía interna y química de las partículas SPH de gas,
con filtrado de partículas en halos densos.

## Nuevo módulo: `crates/gadget-ng-rt/src/igm_temp.rs`

### Tipos principales

```rust
pub struct IgmTempBin {
    pub z: f64,
    pub t_mean: f64,
    pub t_median: f64,
    pub t_sigma: f64,
    pub t_p16: f64,    // 16° percentil (± 1σ inferior)
    pub t_p84: f64,    // 84° percentil (± 1σ superior)
    pub n_particles: usize,
}

pub struct IgmTempParams {
    pub delta_max: f64,  // umbral de densidad (× media): δ < δ_max → IGM
    pub gamma: f64,      // índice adiabático (típicamente 5/3)
}
```

### Funciones principales

- **`compute_igm_temp_profile(particles, chem_states, mean_density, z, params)`**:  
  Filtra partículas de gas con `ρ_SPH < delta_max × mean_density` (donde `ρ_SPH ≈ m/h³`)
  y calcula estadísticas de temperatura (media, mediana, sigma, percentiles 16/84).

- **`compute_igm_temp_all(particles, chem_states, z, gamma)`**:  
  Versión simplificada sin filtro de densidad (incluye todas las partículas de gas).

- **`temperature_from_particle(internal_energy, chem, gamma)`**:  
  Wrapper sobre `ChemState::temperature_from_internal_energy`. Convierte energía interna
  en unidades internas (km²/s²) a temperatura en Kelvin usando la masa molecular media
  adaptativa calculada desde las fracciones de ionización.

## Integración con in-situ analysis

### `crates/gadget-ng-core/src/config.rs`

Nuevo campo `igm_temp_enabled: bool` en `InsituAnalysisSection`:

```toml
[insitu_analysis]
enabled         = true
igm_temp_enabled = true   # activa el cálculo en cada snapshot in-situ
```

### `crates/gadget-ng-cli/src/insitu.rs`

Nuevo campo en `InsituResult`:

```rust
pub igm_temp: Option<gadget_ng_rt::IgmTempBin>,
```

El campo se calcula en el loop in-situ si `cfg.igm_temp_enabled = true`, usando las
partículas de gas locales (filtradas por `internal_energy > 0`) con estados químicos neutros
(proxy conservador cuando no hay información química in-situ).

Output en `insitu_NNNNNN.json`:

```json
{
  "igm_temp": {
    "z": 7.5,
    "t_mean": 12000.0,
    "t_median": 11500.0,
    "t_sigma": 3200.0,
    "t_p16": 8800.0,
    "t_p84": 15200.0,
    "n_particles": 1024
  }
}
```

## Tests (8 tests)

| Test | Descripción |
|------|-------------|
| `temperature_from_particle_reasonable` | T ∈ [10³, 10⁷] K para u_code = 2100 km²/s² |
| `compute_igm_temp_profile_empty_returns_default` | Slice vacío → default sin panic |
| `compute_igm_temp_profile_filters_high_density` | Solo incluye partículas IGM (δ < 10×mean) |
| `compute_igm_temp_profile_mean_and_median_reasonable` | Con energía uniforme: media ≈ mediana |
| `compute_igm_temp_all_includes_all_particles` | Sin filtro → n = 2 |
| `percentile_correct` | Percentiles sobre array ordenado |
| `igm_temp_bin_default_is_zero` | Default trait produce estado cero |

## Referencia

Lukić et al. (2015), MNRAS 446, 3697;  
Springel (2005), MNRAS 364, 1105.
