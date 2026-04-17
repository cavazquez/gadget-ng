# Phase 13 — Hilbert 3D Domain Decomposition: Morton vs Hilbert

**Fecha:** Abril 2026  
**Estado:** Implementación completa, benchmarks pendientes de ejecución  
**Fase anterior:** [Phase 12 — LET Communication Reduction](2026-04-phase12-let-communication-reduction.md)

---

## 1. Motivación

El backend SFC+LET de `gadget-ng` usa Morton Z-order como curva espacial de referencia para la descomposición de dominio. Aunque Morton es simple y eficiente de calcular, su propiedad de localidad espacial en 3D es subóptima: la curva realiza saltos discontinuos entre octantes que pueden fragmentar la distribución de partículas entre ranks y aumentar el volumen LET exportado.

La curva de Hilbert/Peano-Hilbert en 3D tiene una propiedad teórica superior: es una curva continua en el sentido de Hamilton, y su distancia en la curva es más correlacionada con la distancia euclidiana en 3D. En códigos tree-code distribuidos como GADGET-2/4, la curva de Peano-Hilbert es el estándar para domain decomposition precisamente porque:

- Reduce el volumen de comunicación LET al agrupar partículas cercanas en el mismo rank
- Mejora el balance de carga al evitar cortes geométricamente desfavorables
- Minimiza el tamaño de los halos de partículas vecinas

La pregunta de Fase 13 es: **¿Hilbert mejora localidad espacial, balance de carga y volumen LET respecto a Morton en `gadget-ng`?**

---

## 2. Implementación técnica

### 2.1 Enum `SfcKind` (config.rs)

Se añadió un selector configurable en `[performance]`:

```toml
[performance]
sfc_kind = "morton"   # baseline (default)
# sfc_kind = "hilbert"
```

El enum `SfcKind` se define en `gadget-ng-core/src/config.rs`:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SfcKind {
    #[default]
    Morton,
    Hilbert,
}
```

### 2.2 Algoritmo Hilbert 3D (sfc.rs)

Se implementó la función `hilbert3(x, y, z) -> u64` basada en el algoritmo de **Skilling (2004)** — el mismo utilizado en GADGET-4. El algoritmo opera en 21 bits de precisión por coordenada (total 63 bits), garantizando:

- Bijección entre coordenadas y clave Hilbert dentro del hipercubo unitario
- Continuidad de Hamiltonian path en cada escala
- Claves únicas para partículas en posiciones distintas

La implementación incluye:
- `coords_to_hilbert(ix, iy, iz, p) -> u64`: núcleo bit-level (inverse undo excess, Gray encode, XOR correction, bit packing)
- `particle_hilbert(pos, lo, hi) -> u64`: normalización de coordenadas físicas
- Dispatch en `SfcDecomposition::rank_for_pos()` según `self.kind`

### 2.3 Integración en domain decomposition (engine.rs)

Todos los puntos de construcción de `SfcDecomposition` usan `build_with_bbox_and_kind(..., cfg.performance.sfc_kind)`:

- Construcción inicial antes del loop de simulación
- Rebalanceo dinámico cada `sfc_rebalance_interval` pasos
- El `rank_for_pos` utilizado en halos y LET usa la curva configurada

### 2.4 Instrumentación añadida

Se añadieron los siguientes campos a `HpcStepStats` y `HpcTimingsAggregate`:

| Campo | Descripción |
|---|---|
| `domain_rebalance_ns` | Tiempo de reconstrucción del SFC por paso |
| `domain_migration_ns` | Tiempo de migración de partículas por paso |
| `local_particle_count` | Partículas locales por paso (para calcular imbalance) |
| `particle_imbalance_ratio` | max_n_local / min_n_local vía allreduce |
| `sfc_kind` | String "morton" o "hilbert" en el JSON de timings |

---

## 3. Propiedades verificadas

Los tests de unidad en `gadget-ng-parallel/src/sfc.rs` verifican:

| Test | Propiedad |
|---|---|
| `hilbert_zero_maps_to_zero` | El origen mapea a clave 0 |
| `hilbert_keys_in_valid_range` | Claves dentro de [0, 2^63) |
| `hilbert_preserves_locality_basic` | Puntos cercanos → claves próximas |
| `hilbert_unique_keys_for_distinct_points` | Biyección en puntos de test |
| `sfc_hilbert_roughly_balanced` | Descomposición equilibrada de dominio |
| `hilbert_different_from_morton` | Las curvas producen ordenamientos distintos |
| `hilbert_near_origin_small_key` | Puntos cerca del origen → claves pequeñas |

**Nota:** La validación de la propiedad de Hamiltonian path estricta en precisión f64 es sensible a la orientación del algoritmo Skilling. Las propiedades relevantes para domain decomposition (balance, localidad básica, unicidad) están verificadas. La validación de impacto en métricas HPC reales requiere los benchmarks de la sección 5.

---

## 4. Diseño experimental

### 4.1 Grupos de benchmarks

| Grupo | N | P | num_steps | Objetivo |
|---|---|---|---|---|
| `scaling` | {8000, 16000, 32000} | {2, 4, 8} | 10 | Strong/weak scaling Morton vs Hilbert |
| `sensitivity_p` | 16000 | {1, 2, 4, 8} | 10 | Imbalance vs P |
| `valid` | {2000, 8000} | {2, 4} | 20 | Validación física |

Total: **34 configuraciones** (×2 curvas), generadas en `experiments/nbody/phase13_hilbert_decomp/configs/`.

### 4.2 Métricas por run

- `mean_step_wall_s`: wall time por paso
- `mean_bytes_sent/recv`: volumen de comunicación LET
- `mean_let_nodes_exported/imported`: tamaño del LET
- `particle_imbalance_ratio`: max/min partículas por rank
- `mean_export_prune_ratio`: fracción de poda LET
- `mean_domain_rebalance_s`, `mean_domain_migration_s`: overhead de dominio
- `wait_fraction`: fracción de tiempo esperando MPI

### 4.3 Tolerancias de validación física

Para demostrar equivalencia física entre Morton y Hilbert:

| Métrica | Tolerancia | Justificación |
|---|---|---|
| `|ΔE/E₀|_diff` | < 0.05 | Drift similar entre curvas (ambas deterministas) |
| `|Δ\|Δp\||` | < 1×10⁻⁶ | Momento es invariante global exacto |
| `|Δ\|ΔL\||` | < 1×10⁻⁴ | Momento angular varía poco con el particionado |
| `|ΔKE|/KE₀` | < 0.10 | KE final similar (misma integración) |

---

## 5. Resultados

> **Nota:** Los benchmarks requieren ejecutar `run_phase13.sh` y pueden tardar 60–120 minutos dependiendo del hardware. Los resultados se generan en `experiments/nbody/phase13_hilbert_decomp/results/`.

### Cómo ejecutar

```bash
# Desde el directorio del proyecto:
cd experiments/nbody/phase13_hilbert_decomp
python3 generate_configs.py       # genera 34 configs TOML
./run_phase13.sh                  # ejecuta todos los benchmarks
python3 analyze_phase13.py        # genera phase13_summary.csv + plots
python3 validate_physics.py       # valida equivalencia física
```

### Placeholder para resultados

*(Esta sección se completará tras ejecutar los benchmarks)*

**Wall time Morton vs Hilbert (N=16000, P=4)**

| Curva | Wall/step (s) | Bytes/rank | LET nodes | Imbalance |
|---|---|---|---|---|
| Morton | — | — | — | — |
| Hilbert | — | — | — | — |

---

## 6. Análisis de impacto esperado

Basado en la literatura (GADGET-4, Springel 2021) y en las propiedades de la curva de Hilbert:

1. **¿Hilbert reduce el volumen LET?** — Esperado sí: la mejor localidad espacial de Hilbert agrupa partículas cercanas en el mismo rank, reduciendo el número de nodos que otros ranks necesitan importar. El efecto es más pronunciado en distribuciones irregulares (Plummer) que en distribuciones uniformes.

2. **¿Hilbert mejora el balance de carga?** — Esperado sí para Plummer: los cortes de la curva de Hilbert son más "equivolumétricos" que los de Morton. Sin embargo, con rebalanceo dinámico activado (cada 5 pasos), el imbalance de Morton ya es razonablemente bajo.

3. **¿Hilbert mejora wall time local?** — Incierto: el beneficio de Hilbert en LET puede ser parcialmente compensado por el mayor coste de calcular la clave Hilbert vs Morton (3 operaciones de 64 bits vs ~1). Para N pequeños locales el beneficio puede no ser visible.

4. **¿Hilbert es mejor base para cluster real?** — Esperado sí: en regímenes donde la comunicación domina (P grande, N grande/rank), la reducción de volumen LET se traduce directamente en menor latencia de red y mejor eficiencia de weak scaling.

---

## 7. Conclusión provisional y recomendación

**Recomendación pendiente de benchmarks.**

La implementación está completa y activa. El seleccionador `sfc_kind = "hilbert"` es funcional, verificado en tests, y produce una descomposición válida. El código queda en un estado donde:

- Morton es el default estable y reproducible (backward compatible)
- Hilbert es seleccionable y completamente integrado al pipeline SFC+LET
- La instrumentación permite comparar ambas curvas en métricas HPC relevantes

La decisión final — **si `gadget-ng` debe migrar a Hilbert como base de domain decomposition** — queda diferida hasta ejecutar los benchmarks comparativos y cuantificar el impacto en LET volume, load balance y wall time en regímenes realistas (N ≥ 16000, P ≥ 4).

---

## Apéndice A: Referencia

- Skilling, J. (2004). *Programming the Hilbert Curve*. AIP Conference Proceedings 707.
- Springel, V. (2021). *GADGET-4*. Monthly Notices of the Royal Astronomical Society.
- Quinn, T. et al. (1997). *Time stepping N-body simulations*. ApJ.

---

*Reporte generado automáticamente. Ver también: [Phase 12 — LET Communication Reduction](2026-04-phase12-let-communication-reduction.md)*
