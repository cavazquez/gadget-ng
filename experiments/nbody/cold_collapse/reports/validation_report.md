# Reporte: Colapso Gravitacional Frío — cold_collapse

## Objetivo

Validar el comportamiento de gadget-ng ante el benchmark de colapso gravitacional frío
(Aarseth, Hénon & Wielen 1974). Una esfera uniforme en reposo colapsa bajo su propia
gravedad y el sistema virializa.

## Física del benchmark

| Cantidad | Valor |
|---|---|
| G | 1.0 |
| M_total | 1.0 |
| R (radio inicial) | 1.0 |
| Densidad ρ₀ | 3/(4π) ≈ 0.239 |
| T_ff = π·√(R³/2GM) | ≈ 2.221 [u.t.] |
| r_hm teórico (esfera uniforme) | R·(0.5)^(1/3) ≈ 0.794 R |

## Fases del colapso (predicciones)

1. **Precolapso** (0 < t < T_ff): r_hm cae desde ~0.794·R.
2. **Colapso** (t ≈ T_ff): máxima compresión; r_hm → 0 (limitado por suavizado ε).
3. **Rebote** (T_ff < t < 2·T_ff): expansión violenta, mezcla.
4. **Virialización** (t > 2–3·T_ff): Q = -T/W → 0.5.

## Criterios de validación

| Métrica | Criterio | Justificación |
|---|---|---|
| r_hm cae al 50% del inicial | t < 1.5·T_ff | Colapso efectivo |
| Q(5·T_ff) ∈ [0.2, 0.8] | post-virialización | Teorema del virial |
| |ΔE/E₀| en precolapso | < 5% (BH θ=0.5) | Conservación numérica |

## Nota sobre energía durante el colapso

Con paso fijo (dt = T_ff/100) y ε = 0.05·R, el error de energía puede llegar
a ~30–50% durante la fase violenta (t ≈ T_ff). Esto es esperado y conocido:
la solución correcta son **block timesteps** (`hierarchical_kdk_step`, ya implementado).
La energía se recobra parcialmente tras la virialización.

## Comparación con GADGET-4

GADGET-4 usa block timesteps automáticos (criterio de Aarseth), que le permiten
usar pasos muy pequeños para partículas de alta aceleración durante el colapso.
Esto resulta en:
- Mejor conservación de energía durante el colapso (~1% vs 30–50%)
- Mayor coste computacional (evaluación de fuerzas múltiples por paso global)

gadget-ng tiene implementado `hierarchical_kdk_step` con el mismo criterio de Aarseth
(`[timestep] hierarchical = true`). Para uso de producción del colapso frío, se
recomienda activarlo.

## Ejecución

```bash
cd experiments/nbody/cold_collapse
bash scripts/run_collapse.sh --release
python scripts/analyze_collapse.py
python scripts/plot_collapse.py
```

## Resultados (rellenar tras la ejecución)

Archivos generados:
- `results/collapse_timeseries.csv`: serie temporal completa
- `plots/collapse_overview.png`: 4 paneles (r_hm, Q, E, δE)
- `plots/rHm_vs_Tff.png`: r_hm vs referencia analítica
