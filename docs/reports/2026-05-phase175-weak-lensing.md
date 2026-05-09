# Phase 175 — Weak Lensing: Kaiser-Squires, C_ℓ, Tomographic Binning

**Fecha:** 2026-05-09

## Objetivo

Extender el módulo de lente gravitacional débil con tres componentes clave para análisis cosmológico:

1. **Reconstrucción Kaiser-Squires** — inversión de shear γ₁,γ₂ → convergencia κ
2. **Espectro angular de potencia C_ℓ** — desde mapa de convergencia
3. **Lente débil tomográfico** — bins de redshift independientes

## API

Crate: `gadget-ng-analysis`, módulo `lightcone` (extendido).

### Tipos nuevos

| Tipo | Descripción |
|------|-------------|
| `KsParams` | Parámetros KS (n_pixels, fov_rad) |
| `KsResult` | κ reconstruido (n_pixels², fov_rad) |
| `ClBin` | Bin de C_ℓ (ell, cl) |
| `TomographyParams` | z_edges, n_pixels |
| `TomographicLensingMap` | κ, γ₁, γ₂ por bin tomográfico |

### Funciones nuevas

| Función | Descripción |
|---------|-------------|
| `kaiser_squires_reconstruct(map, fov_rad)` | Inversión FFT 2D: γ̂ → κ̂ |
| `convergence_angular_cl(map, fov_rad, n_ell_bins)` | C_ℓ via FFT de κ |
| `accumulate_tomographic_lensing(hits, masses, redshifts, observer, params)` | κ/γ por bin de z |

### Algoritmo Kaiser-Squires

Relación en espacio de Fourier: κ̂(ℓ) = −(ℓ₁² − ℓ₂² + 2iℓ₁ℓ₂)/(ℓ₁² + ℓ₂²) × γ̂(ℓ)

- FFT 2D de γ₁ y γ₂
- Aplicar kernel P_κ
- IFFT 2D inversa → κ(x,y)
- ℓ = 0: κ = 0 (modo nulo)

## Tests (5)

1. `detects_shell_crossing` — Test existente de lightcone
2. `ks_reconstruct_identity_zero_shear` — Shear cero → κ ≈ 0
3. `ks_reconstruct_preserves_dimensions` — n_pixels² celdas
4. `convergence_cl_returns_bins` — C_ℓ no vacío, ell > 0
5. `tomographic_lensing_assigns_bins` — 3 bins de redshift con z_edges

## Limitaciones

- KS usa FFT 2D plana (aproximación de campo plano, válida para FOV < ~10°)
- C_ℓ usa binning esférico sobre malla cartesiana (aproximación plana)
- Tomografía usa bins de z rígidos (no Dol-optimized)
- No incluye corrección por forma de PSF (shape noise)

## Archivos

| Archivo | Acción |
|---------|--------|
| `crates/gadget-ng-analysis/src/lightcone.rs` | EDITAR (añadir KS, C_ℓ, tomografía) |
| `crates/gadget-ng-analysis/src/lib.rs` | Editar (nuevos re-exports) |