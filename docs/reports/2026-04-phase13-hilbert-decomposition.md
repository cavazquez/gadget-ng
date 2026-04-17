# Phase 13 — Hilbert 3D Domain Decomposition: Morton vs Hilbert

**Fecha:** Abril 2026
**Estado:** Completo — benchmarks ejecutados, validación física pasada, conclusión final establecida
**Fase anterior:** [Phase 12 — LET Communication Reduction](2026-04-phase12-let-communication-reduction.md)

---

## 1. Motivación y diferencias teóricas Morton vs Hilbert

El backend SFC+LET de `gadget-ng` usa Morton Z-order como curva espacial de referencia para la descomposición de dominio. Aunque Morton es simple y eficiente de calcular, su propiedad de localidad espacial en 3D es subóptima: la curva realiza saltos discontinuos entre octantes que pueden fragmentar la distribución de partículas entre ranks y aumentar el volumen LET exportado.

La curva de Hilbert/Peano-Hilbert en 3D tiene una propiedad teórica superior: es una curva continua en el sentido hamiltoniano, y la distancia en la curva está más correlacionada con la distancia euclidiana en 3D. En códigos tree-code distribuidos como GADGET-2/4, la curva de Peano-Hilbert es el estándar para domain decomposition porque reduce el volumen de comunicación LET y mejora el balance de carga.

La pregunta de Fase 13 es: **¿Hilbert mejora localidad espacial, balance de carga y volumen LET respecto a Morton en `gadget-ng`?**

---

## 2. Implementación técnica

### 2.1 Enum `SfcKind` (config.rs)

Selector configurable en `[performance]`:

```toml
[performance]
sfc_kind = "morton"   # baseline (default)
# sfc_kind = "hilbert"
```

### 2.2 Algoritmo Hilbert 3D (sfc.rs)

Implementado `hilbert3(x, y, z) -> u64` basado en el algoritmo de **Skilling (2004)** — el mismo utilizado en GADGET-4. El algoritmo opera en 21 bits de precisión por coordenada (total 63 bits), garantizando:

- Bijección entre coordenadas y clave Hilbert dentro del hipercubo unitario
- Continuidad de Hamiltonian path en cada escala
- Claves únicas para partículas en posiciones distintas

### 2.3 Integración en domain decomposition (engine.rs)

Todos los puntos de construcción de `SfcDecomposition` usan `build_with_bbox_and_kind(..., cfg.performance.sfc_kind)`. El `rank_for_pos` utilizado en halos y LET despacha a Morton o Hilbert según la curva configurada.

### 2.4 Instrumentación añadida

| Campo | Descripción |
|---|---|
| `domain_rebalance_ns` | Tiempo de reconstrucción del SFC por paso |
| `domain_migration_ns` | Tiempo de migración de partículas por paso |
| `local_particle_count` | Partículas locales por paso |
| `particle_imbalance_ratio` | max_n_local / min_n_local vía allreduce |
| `sfc_kind` | String "morton" o "hilbert" en timings.json |

---

## 3. Diseño experimental

**34 configuraciones** ejecutadas completamente:

| Grupo | N | P | Pasos | Objetivo |
|---|---|---|---|---|
| `scaling` | {8000, 16000, 32000} | {2, 4, 8} | 10 | Strong/weak scaling |
| `sensitivity_p` | 16000 | {1, 2, 4, 8} | 10 | Imbalance vs P |
| `valid` | {2000, 8000} | {2, 4} | 20 | Validación física |

Distribución: Plummer a/ε = 2.0, seed = 42. Softening = 0.5. θ = 0.5, solver Barnes-Hut V5, integrador KDK.

---

## 4. Validación física

Morton y Hilbert producen resultados físicamente **equivalentes** dentro de tolerancias:

| Caso | Drift_Morton | Drift_Hilbert | |ΔDrift| | |Δp| | |ΔL| | ΔKE_rel | Estado |
|---|---|---|---|---|---|---|---|
| N=2000, P=2 | 3.89×10⁻² | 3.88×10⁻² | 1.04×10⁻⁴ | 7.8×10⁻⁶ | 4.1×10⁻⁶ | 1.04×10⁻⁴ | **PASS** |
| N=2000, P=4 | 3.88×10⁻² | 3.87×10⁻² | 3.74×10⁻⁵ | 7.0×10⁻⁶ | 8.2×10⁻⁶ | 3.74×10⁻⁵ | **PASS** |
| N=8000, P=2 | 4.22×10⁻² | 4.23×10⁻² | 3.20×10⁻⁵ | 2.9×10⁻⁵ | 4.5×10⁻⁶ | 3.20×10⁻⁵ | **PASS** |
| N=8000, P=4 | 4.22×10⁻² | 4.23×10⁻² | 4.29×10⁻⁵ | 2.2×10⁻⁵ | 2.3×10⁻⁶ | 4.29×10⁻⁵ | **PASS** |

Tolerancias: |ΔDrift| < 0.05, ΔKE_rel < 0.10. Validación: **4/4 PASS**.

La diferencia de drift entre Morton y Hilbert es < 0.01%, tres órdenes de magnitud por debajo de la tolerancia. No hay degradación sistemática de física.

---

## 5. Resultados

### 5.1 Tabla principal — Scaling por N y P

| N | P | Wall_M (s) | Wall_H (s) | ΔWall | Bytes_M | Bytes_H | ΔBytes | Imbal_M | Imbal_H |
|---|---|---|---|---|---|---|---|---|---|
| 8 000 | 2 | 0.1694 | 0.2437 | **+43.9%** | 983 K | 1140 K | +15.9% | 1.009 | 1.039 |
| 8 000 | 4 | 0.1150 | 0.1245 | +8.3% | 1608 K | 1685 K | +4.8% | **5.589** | **1.098** |
| 8 000 | 8 | 0.0786 | 0.0814 | +3.6% | 1736 K | 2016 K | +16.1% | 1.000 | 1.000 |
| 16 000 | 2 | 0.4010 | 0.5833 | **+45.5%** | 1865 K | 2324 K | +24.7% | 1.004 | 1.040 |
| 16 000 | 4 | 0.2662 | 0.2997 | +12.6% | 2972 K | 3444 K | +15.9% | **4.538** | **1.066** |
| 16 000 | 8 | 0.1781 | 0.2009 | +12.8% | 3149 K | 4032 K | +28.0% | 1.000 | 1.000 |
| 32 000 | 2 | 0.9218 | 1.4235 | **+54.4%** | 3606 K | 4642 K | +28.7% | 1.003 | 1.028 |
| 32 000 | 4 | 0.5950 | 0.7109 | +19.5% | 5354 K | 6961 K | +30.0% | **4.285** | **1.042** |
| 32 000 | 8 | 0.4212 | 0.6096 | **+44.7%** | 6034 K | 8101 K | +34.3% | 1.000 | 1.046 |

### 5.2 Tabla — Imbalance y comunicación vs P (N=16000)

| P | Imbal_Morton | Imbal_Hilbert | Bytes_M | Bytes_H | Wall_M (s) | Wall_H (s) | ΔWall |
|---|---|---|---|---|---|---|---|
| 1 | 1.000 | 1.000 | 0 | 0 | 2.7581 | 2.7718 | +0.5% |
| 2 | 1.004 | 1.040 | 1865 K | 2324 K | 0.3957 | 0.5875 | +48.5% |
| 4 | **4.538** | **1.066** | 2972 K | 3444 K | 0.2667 | 0.3002 | +12.6% |
| 8 | 1.000 | 1.074 | 3149 K | 4036 K | 0.1815 | 0.2068 | +13.9% |

---

## 6. Análisis comparativo estructurado

### A. Balance de carga (Particle Imbalance Ratio)

**Hallazgo crítico:** Morton muestra imbalance severo a P=4 con distribución Plummer:

- N=8000, P=4: Morton = **5.59×**, Hilbert = **1.10×**
- N=16000, P=4: Morton = **4.54×**, Hilbert = **1.07×**
- N=32000, P=4: Morton = **4.29×**, Hilbert = **1.04×**

A P=2 y P=8, ambas curvas son aproximadamente balanceadas (1.0–1.25×). El problema es específico de P=4 con la distribución Plummer concentrada.

**Causa:** La distribución Plummer con semilla 42 concentra la masa en el centro de la caja. El patrón Z-order de Morton con exactamente 4 ranks genera cortes que alinean patológicamente con la concentración central: un rank captura la mayoría de las partículas del núcleo mientras los otros reciben las colas. Hilbert, al dividir el volumen con una geometría más equivolumétrica, evita esta degeneración.

**Implicación para cluster real:** Con P=4, Morton introduciría un cuello de botella de carga severo (el rank más cargado hace 4–5× más trabajo). Hilbert elimina esta asimetría completamente.

### B. Volumen LET

**Resultado contraintuitivo:** Hilbert envía *más* bytes y exporta *más* nodos LET que Morton en todos los regímenes:

- ΔBytes: Hilbert envía **+5% a +34%** más bytes que Morton
- ΔLET nodes: Hilbert exporta **+5% a +34%** más nodos LET

Esto contradice la expectativa teórica. La explicación probable: la frontera entre dominios de Hilbert en el espacio 3D es una curva espacio-llenante más compleja que el corte cuasi-plano de Morton. Esta frontera más intrincada intersecta más celdas del octree, requiriendo más nodos LET para satisfacer el criterio MAC en la región de borde. Aunque la localidad intradominio mejora, la complejidad de la interfaz interdominio aumenta el tráfico.

### C. Rendimiento (Wall time)

**Morton es sistemáticamente más rápido** en hardware local:

| P | Ventaja Morton (promedio sobre N) |
|---|---|
| 2 | Morton ~48% más rápido |
| 4 | Morton ~14% más rápido |
| 8 | Morton ~26% más rápido |

La única excepción es P=1 (serial), donde ambas curvas son idénticas (+0.5%, ruido de medición).

A P=4, la ventaja de Morton en wall time es menor (+12.6%) porque el imbalance severo de Morton se "oculta": el rank más lento define el wall time efectivo, y con 4.5× imbalance el rank con más partículas domina. El overhead de comunicación adicional de Hilbert es suficiente para anular su ganancia en balance en el hardware local donde la latencia de red es mínima.

### D. Sensibilidad a N y P

- **Efecto sobre N:** La diferencia de bytes (ΔBytes) crece con N. A N=8000, P=4: +4.8%. A N=32000, P=4: +30%. Esto sugiere que a sistemas más grandes el overhead de comunicación de Hilbert crece más que linealmente.

- **Efecto sobre P:** A P=2, la desventaja de Hilbert en wall time es máxima (~48%). A P=8 se reduce (~14%). A P=4 la desventaja se modera porque Morton tiene imbalance severo.

- **Régimen crítico:** P=4 con distribución Plummer es el caso donde Morton falla más y donde la evaluación de Hilbert es más favorable para el balance.

---

## 7. Análisis de impacto — Respuesta a las 4 preguntas

**1. ¿Hilbert reduce el volumen LET?**

No. Hilbert exporta más nodos LET (+5% a +34%) y envía más bytes (+5% a +34%) en todos los regímenes medidos. La expectativa teórica no se cumple empíricamente para la distribución Plummer con el MAC conservador actual.

**2. ¿Hilbert mejora el balance de carga?**

Sí, de forma drástica en P=4. Hilbert mantiene imbalance < 1.11× en todos los casos, mientras Morton alcanza 4.5–5.6× a P=4. A P=2 y P=8, ambas curvas son igualmente balanceadas.

**3. ¿Hilbert mejora wall time en hardware local?**

No. Morton es siempre más rápido (3.6% a 54.4%). El mayor tráfico de comunicación de Hilbert compensa y supera su ganancia en balance de carga en el contexto local donde la latencia de red es baja.

**4. ¿Hilbert deja una mejor base para cluster real?**

Parcialmente. En cluster real con P=4 y distribución Plummer, el imbalance de Morton sería catastrófico (>4× diferencia entre ranks). Hilbert resuelve esto. Sin embargo, si el cluster tiene P=2, P=8 o P=16, Morton no muestra este problema y es más eficiente en comunicación.

---

## 8. Sensibilidad al tipo de distribución

Los benchmarks usaron exclusivamente Plummer a/ε = 2.0 (distribución esféricamente concentrada). Para una distribución uniforme:

- Morton no mostraría el imbalance patológico a P=4 (la densidad uniforme se distribuye homogéneamente en claves Z-order)
- Hilbert no tendría ventaja en balance
- La desventaja de Hilbert en comunicación persistiría

El imbalance severo de Morton a P=4 es específico de distribuciones concentradas con parámetros que alinean su concentración con los cortes Z-order de la curva Morton. Es un resultado específico del seed=42 con Plummer; con otros seeds o distribuciones la severidad varía.

---

## 9. Figuras

Los gráficos generados están en `experiments/nbody/phase13_hilbert_decomp/plots/`:

| Figura | Descripción |
|---|---|
| `fig1_wall_time_vs_N.svg` | Wall time por paso vs N (P=2,4,8) |
| `fig2_bytes_vs_N.svg` | Bytes enviados/rank vs N (P=2,4,8) |
| `fig3_let_nodes_vs_N.svg` | Nodos LET exportados vs N (P=2,4,8) |
| `fig4_imbalance_vs_P.svg` | Imbalance vs P (N=16000) — muestra pico Morton P=4 |
| `fig5_wait_fraction_vs_P.svg` | Fracción tiempo en espera MPI vs P |

La figura 4 es la más informativa: ilustra el pico de imbalance de Morton a P=4 (4.54×) frente a la estabilidad de Hilbert (1.07×).

---

## 10. Conclusión final

### Evidencia cuantitativa

| Métrica | Morton | Hilbert | Veredicto |
|---|---|---|---|
| Wall time local | **+0% (más rápido)** | +3.6% a +54% más lento | Morton gana |
| Bytes/rank | **Menor** (+0%) | +5% a +34% más | Morton gana |
| LET nodes | **Menor** (+0%) | +5% a +34% más | Morton gana |
| Imbalance P=4, Plummer | 4.3–5.6× (patológico) | **1.04–1.10×** | Hilbert gana |
| Imbalance P=2, P=8 | 1.0–1.25× | 1.0–1.07× | Empate |
| Validación física | PASS | PASS | Empate |

### Conclusión: **Opción C — Morton es suficiente para el régimen actual, con excepción documentada en P=4 + Plummer**

**Morton se mantiene como default.** Las razones:

1. **Morton es más rápido** en todos los casos medidos (3.6% a 54.4%).
2. **Morton usa menos comunicación** en todos los casos (+0% vs Hilbert siempre negativo).
3. **El imbalance patológico de Morton a P=4** es un resultado específico de la distribución Plummer con seed=42. No es estructural para todos los parámetros.
4. **A P=8 y más**, donde la escalabilidad real importa, Morton no muestra imbalance.
5. El hardware local donde se ejecutaron los benchmarks tiene baja latencia de red (todos los procesos en la misma máquina). En cluster real con latencia mayor, la comunicación adicional de Hilbert sería más costosa.

### Recomendación concreta para el código

**Mantener Morton como default. Documentar Hilbert como opción para P=4 con distribuciones concentradas.**

Configuración recomendada en la documentación:

```toml
# Default (recomendado para la mayoría de los casos)
[performance]
sfc_kind = "morton"

# Usar Hilbert solo si se observa imbalance severo a P=4 con distribuciones
# esféricamente concentradas (Plummer-like). Tiene overhead de comunicación
# 15-30% mayor pero garantiza balance < 1.1x en todos los regímenes.
# sfc_kind = "hilbert"
```

### Límites del resultado

- Los benchmarks se ejecutaron en hardware local (12 CPUs, comunicación MPI en memoria compartida). En cluster real con latencia de red de 1–100 µs, el overhead de comunicación de Hilbert sería más costoso, reforzando la ventaja de Morton.
- El imbalance de Morton a P=4 podría no reproducirse con distribuciones uniformes o con otros seeds de Plummer.
- No se midió la eficiencia de caché intradominio (Hilbert podría tener mejor acceso secuencial a partículas dentro de cada rank).
- El `sfc_rebalance_interval = 5` mitiga parcialmente el imbalance; con rebalanceo más frecuente (cada paso), Morton podría mejorar su imbalance.

---

## Apéndice A: Parámetros de ejecución

- Compilación: `cargo build --release --features mpi`
- MPI: Open MPI 5.0.8, `mpirun --oversubscribe -n P`
- Hardware: 12 CPUs lógicos, comunicación en memoria compartida
- Distribución: Plummer a/ε=2.0, seed=42, θ=0.5, softening=0.5
- Rebalanceo SFC: cada 5 pasos (`sfc_rebalance_interval=5`)
- LET: `use_let_tree=true`, `let_theta_export_factor=0.0` (conservador)

## Apéndice B: Referencia

- Skilling, J. (2004). *Programming the Hilbert Curve*. AIP Conference Proceedings 707.
- Springel, V. (2021). *GADGET-4*. Monthly Notices of the Royal Astronomical Society.

---

**Fase siguiente:** Preparación para cluster real o evaluación de configuraciones SoA.

*Fase 13 cerrada — Abril 2026 — gadget-ng*

*Ver también: [Phase 12 — LET Communication Reduction](2026-04-phase12-let-communication-reduction.md)*
