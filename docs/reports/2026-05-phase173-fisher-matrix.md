# Phase 173 — Fisher Matrix for Cosmological Forecasting

**Fecha:** 2026-05-08

## Objetivo

Implementar la matriz de información de Fisher para forecasting cosmológico a partir del espectro de potencias de materia P(k,z). Esto permite estimar incertidumbres marginales 1σ(θ_i) y correlaciones entre parámetros cosmológicos para encuestas futuras (Euclid, DESI, Rubin/LSST).

## API

Crate: `gadget-ng-analysis`, módulo `fisher`.

### Tipos principales

| Tipo | Descripción |
|------|-------------|
| `FisherParams` | Vector canónico de parámetros (Ω_m, Ω_b, h, n_s, σ₈, w₀, wₐ, m_ν) |
| `FisherConfig` | Configuración (k-bins, redshifts, volumen, step_frac) |
| `FisherMatrix` | Matriz F_{ij} con nombres de parámetros (Serialize/Deserialize) |
| `FisherUncertainties` | Incertidumbres marginales σ(θ_i) = √(F⁻¹)_{ii} |
| `DerivativeMatrix` | Matriz ∂P(k,z)/∂θ_i para todos los (k,z) |

### Funciones principales

| Función | Descripción |
|---------|-------------|
| `pk_observable(k, z, params, use_nonlinear)` | P(k,z) desde EH+D²(z) o Halofit |
| `pk_derivatives(config, fiducial)` | ∂P/∂θ_i via diferencias finitas centrales |
| `fisher_matrix(config, fiducial)` | Ensambla F_{ij} = Σ (∂P/∂θ_i) C⁻¹ (∂P/∂θ_j) Δk |
| `fisher_uncertainties(&F)` | σ(θ_i) = √(F⁻¹)_{ii} |
| `correlation_matrix(&F)` | r_{ij} = (F⁻¹)_{ij} / √((F⁻¹)_{ii}(F⁻¹)_{jj}) |
| `gaussian_covariance(pk, k, dk, V)` | C(k,z) = 2P²/N_modes |
| `central_difference(x, step, f)` | Diferencia finita central |

### CLI

```bash
gadget-ng fisher \
  --omega-m 0.315 --omega-b 0.049 --h 0.674 \
  --n-s 0.965 --sigma8 0.8111 \
  --w0 -1.0 --wa 0.0 --m-nu-ev 0.06 \
  --survey-volume 1e9 --step-frac 0.01 \
  --use-nonlinear --out fisher_output.json
```

Sale: JSON con fiducial, config, F matrix, σ(θ_i), correlation matrix.

## Tests (4)

1. `pk_observable_positive_at_k01` — P(k=0.1, z=0) > 0
2. `fisher_diagonal_positive` — F_{ii} > 0 para Ω_m, Ω_b, n_s, σ₈
3. `fisher_uncertainties_reasonable` — σ(Ω_m) ∈ (1e-6, 10), σ(σ₈) ∈ (1e-6, 10)
4. `derivative_converges_with_step` — ∂P/∂Ω_m converge al reducir step_frac

## Limitaciones

- Solo ΛCDM plano en Halofit (w₀ = −1). Para CPL, se usa D(z) numérico pero Halofit no está calibrado para w₀ ≠ −1.
- Derivadas numéricas (no analíticas). Step_frac=0.01 es razonable; convergencia verificada en test.
- Covarianza Gaussiana únicamente (no shot-noise dominance).
- w₀, wₐ, m_ν tienen derivadas pequeñas con P(k) lineal + EH nowiggle; se recomienda usar Halofit o tablas CLASS para forecasting serio.

## Archivos

| Archivo | Acción |
|---------|--------|
| `crates/gadget-ng-analysis/src/fisher.rs` | NUEVO (~560 líneas) |
| `crates/gadget-ng-analysis/src/lib.rs` | Editar (pub mod + re-exports) |
| `crates/gadget-ng-cli/src/fisher_cmd.rs` | NUEVO (~100 líneas) |
| `crates/gadget-ng-cli/src/main.rs` | Editar (subcomando `fisher`) |
| `examples/fisher_planck.toml` | NUEVO (config de ejemplo) |