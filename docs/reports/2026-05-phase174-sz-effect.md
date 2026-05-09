# Phase 174 — Sunyaev-Zel'dovich Effect (Compton-y + kSZ)

**Fecha:** 2026-05-09

## Objetivo

Implementar mapas 2D del efecto Sunyaev-Zel'dovich térmico (Compton-y) y cinético (kSZ) a partir de partículas de gas proyectadas a lo largo de la línea de visión. Integrado como análisis in-situ configurable desde TOML.

## Modelo físico

**tSZ (térmico):** El parámetro Compton-y integra la presión electrónica:

$$y = \frac{\sigma_T}{m_e c^2} \int P_e \, dl$$

Para gas totalmente ionizado con Y = 0.24:
- $\mu_e = 2/(1+X_H) \approx 1.143$, $X_e \approx 1.16$
- $P_e = \rho u (\gamma-1) X_e / \mu_e$

**kSZ (cinético):** Modulación por velocidad peculiar:

$$\Delta T / T_{CMB} = -\sigma_T \int n_e (v \cdot \hat{n} / c) \, dl$$

Constantes combinadas: `Y_CONVERSION ≈ 2.058e-18 × 1.040e-6` y `KSZ_CONVERSION ≈ 6.652e-25 / 2.998e5 × 3.086e24`.

## API

Crate: `gadget-ng-analysis`, módulo `sz_effect`.

### Tipos principales

| Tipo | Descripción |
|------|-------------|
| `SzParams` | Parámetros de proyección (n_pixels, axis) |
| `ComptonYMap` | Mapa 2D Compton-y con mean_y, y_max |
| `KineticSzMap` | Mapa 2D kSZ con rms_ksz |

### Funciones principales

| Función | Descripción |
|---------|-------------|
| `compute_compton_y_map(particles, box_size, params, gamma)` | Mapa tSZ vía CIC |
| `compute_kinetic_sz_map(particles, box_size, params, gamma)` | Mapa kSZ vía CIC |
| `electron_pressure(p, gamma)` | Presión electrónica por partícula |
| `electron_density(p, gamma)` | Densidad electrónica por partícula |

### Config TOML

```toml
[insitu_analysis]
enabled = true
interval = 20
sz_enabled = true
sz_n_pixels = 256
```

### Output JSON (insitu)

```json
{
  "sz_compton_y": { "n_pixels": 256, "pixel_size": 0.39, "mean_y": 1.2e-6, "y_max": 3.4e-5 },
  "sz_kinetic": { "n_pixels": 256, "pixel_size": 0.39, "rms_ksz": 2.1e-7 }
}
```

## Tests (3)

1. `zero_gas_zero_y` — Sin partículas → y = 0 en todo el mapa
2. `electron_pressure_scales_with_density_and_temperature` — P_e(u=4) > P_e(u=1)
3. `map_has_correct_dimensions` — n_pixels² celdas, pixel_size ≈ box/n_pixels

## Limitaciones

- Proyección CIC simple (no SPH kernel smoothing)
- Eje de proyección fijo (default 'z')
- Factor de conversión aproximado; para publicación usar constantes NIST precisas
- No incluye CMB primario ni foregrounds

## Archivos

| Archivo | Acción |
|---------|--------|
| `crates/gadget-ng-analysis/src/sz_effect.rs` | NUEVO (~317 líneas) |
| `crates/gadget-ng-analysis/src/lib.rs` | Editar (pub mod + re-exports) |
| `crates/gadget-ng-core/src/config/sections/analysis.rs` | Editar (sz_enabled, sz_n_pixels) |
| `crates/gadget-ng-cli/src/insitu.rs` | Editar (SzComptonYOut, SzKineticOut, cálculo SZ) |