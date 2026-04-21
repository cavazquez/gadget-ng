# Phase 35 — Fit summary

| N | R_mean | CV(seeds) |
|---|-------:|---------:|
| 8 | 0.415382 | 0.229 |
| 16 | 0.139629 | 0.100 |
| 32 | 0.0337524 | 0.036 |
| 64 | 0.0088342 | 0.014 |

## Modelo A: `R(N) = C · N^(-α)`

- C = `22.1082`
- α = `1.87141`
- R² = `0.997152`
- RMS(log₁₀ residuos) = `0.0337`
- AIC = `-23.1321`

## Modelo B: `R(N) = C · N^(-α) + R_∞`

- fuente del fit: `scipy_curve_fit`
- C = `10.0672`
- α = `1.518`
- R_∞ = `-0.0127787`
- RMS(log₁₀ residuos) = `0.1097`
- AIC = `-11.6772`

**Ganador:** Modelo A — B no mejora AIC lo suficiente; gana Modelo A (Occam)
