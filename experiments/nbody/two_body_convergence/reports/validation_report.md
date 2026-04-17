# Reporte: Convergencia Kepler — Experimento two_body_convergence

## Objetivo

Verificar el orden de convergencia del integrador leapfrog KDK comparando el error relativo
de energía `|ΔE/E₀|` y de momento angular `|ΔL_z/L_{z,0}|` al cabo de exactamente 1 período
orbital para distintos valores de dt.

## Configuración

| Parámetro | Valor |
|-----------|-------|
| G | 1.0 |
| M₁ (estrella) | 1.0 |
| M₂ (planeta) | 1e-6 |
| r (separación inicial) | 1.0 |
| Suavizado ε | 1e-6 |
| T_orbit = 2π | ≈ 6.2832 |
| Integrador | Leapfrog KDK global |
| Solver de gravedad | DirectGravity (O(N²), N=2) |

## Valores de dt evaluados

| Etiqueta | dt | N_pasos | dt/T |
|---|---|---|---|
| T/20 | 0.31416 | 20 | 1/20 |
| T/50 | 0.12566 | 50 | 1/50 |
| T/100 | 0.06283 | 100 | 1/100 |
| T/200 | 0.03142 | 200 | 1/200 |
| T/500 | 0.01257 | 500 | 1/500 |

## Predicción teórica

El integrador leapfrog KDK es de **segundo orden** (Yoshida 1990, Quinn et al. 1997).
Por lo tanto se espera:

```
|ΔE/E₀| ∝ dt²
|ΔL/L₀| ∝ dt²
```

En escala log-log, la pendiente debe ser ≈ 2.0.

## Comparación con GADGET-4

GADGET-4 usa el mismo integrador leapfrog KDK (Springel 2005, Springel et al. 2021).
El orden de convergencia es idéntico. La diferencia es que GADGET-4 usa block timesteps
(pasos individuales) que reducen el error efectivo para sistemas heterogéneos. Para el
problema de dos cuerpos con paso global, ambos códigos son equivalentes.

## Reproducibilidad

```bash
cd experiments/nbody/two_body_convergence
bash scripts/run_convergence.sh --release
python scripts/analyze_convergence.py
python scripts/plot_convergence.py
```

## Resultados (rellenar tras la ejecución)

Los resultados se guardan automáticamente en:
- `results/convergence.csv`: tabla de métricas por dt
- `results/energy_timeseries.csv`: E(t) para cada dt
- `plots/convergence_loglog.png`: gráfico log-log de error vs dt
- `plots/energy_timeseries.png`: E(t) y |ΔE/E₀| para cada dt
