# Reporte: Estabilidad de Esfera de Plummer — plummer_stability

## Objetivo

Validar que gadget-ng mantiene el equilibrio dinámico de una esfera de Plummer
durante 10 tiempos de cruce, verificando:

1. Conservación de energía total `|ΔE/E₀| < 5%` (con BH θ=0.5)
2. Ratio virial `Q = -T/W` estable cerca de 0.5
3. Radio de media masa `r_hm` aproximadamente constante
4. Paridad numérica serial vs MPI `max|Δr| < 1e-10` (con `deterministic = true`)

## Configuración

| Parámetro | Valor |
|-----------|-------|
| N partículas | 200 |
| Radio escala `a` | 1.0 |
| G | 1.0 |
| M_total | 1.0 |
| Suavizado ε | 0.05 |
| dt | 0.025 = t_cross/100 |
| num_steps | 1000 |
| t_total | 25.0 ≈ 10·t_cross |
| Solver | Barnes-Hut θ=0.5 |
| snapshot_interval | 10 (100 frames) |

Donde `t_cross = √(6·a³/(G·M)) ≈ 2.449`.

## Referencia teórica

La esfera de Plummer (Plummer 1911) es una solución exacta de la ecuación de
Boltzmann sin colisiones. En equilibrio dinámico: `2T + W = 0 → Q = 0.5`.
El radio de media masa teórico es `r_hm = a·(2^(2/3) - 1)^(1/2) ≈ 1.305·a`.

## Comparación con GADGET-4

GADGET-4 usa el mismo integrador KDK y Barnes-Hut monopolar. Los resultados
de conservación de energía y estabilidad virial son cualitativamente idénticos
para este problema (sin cosmología ni SPH). La principal diferencia es que
GADGET-4 usa block timesteps que mejorarían la precisión, especialmente para
partículas de alta velocidad en el núcleo.

## Ejecución

```bash
cd experiments/nbody/plummer_stability
bash scripts/run_stability.sh --release
python scripts/analyze_stability.py
python scripts/plot_stability.py
```

## Resultados (rellenar tras la ejecución)

Los resultados se guardan en:
- `results/serial_timeseries.csv`
- `results/mpi_2rank_timeseries.csv`
- `results/mpi_4rank_timeseries.csv`
- `results/serial_mpi_2rank_comparison.csv`
- `plots/energy_evolution.png`
- `plots/virial_ratio.png`
- `plots/half_mass_radius.png`
- `plots/serial_mpi_*_parity.png`
