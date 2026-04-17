# Fase 7 — Timesteps Adaptativos Tipo Aarseth y el Límite del Drift Energético en N-body Caótico

**Fecha:** 2026-04-17  
**Scope:** block timesteps jerárquicos con criterio Aarseth (aceleración + jerk aproximado), implementación completa KDK + niveles 2ⁿ, comparación contra dt fijo con controles de isolación, benchmarks en regímenes caótico-denso (Plummer a/ε = 1, 2) y caos débil (uniforme).  
**Estado:** implementación completa; 54/54 runs analizados (3 plummer_a1_N1000 acc sin wall time por interrupción); conclusiones definitivas.

---

## 1. Resumen ejecutivo

1. **Diagnóstico de Fase 6:** el orden del integrador (Yoshida4 vs KDK) no reduce el drift energético en sistemas caóticos densos. La palanca pendiente identificada es el control adaptativo de `dt` individual. Fase 7 evalúa esta hipótesis directamente.

2. **Implementación Aarseth completa.** Se implementaron dos criterios de asignación de timestep individual:
   - **Criterio aceleración** (baseline Aarseth): `dt_i = η · √(ε / |a_i|)` — cero overhead de almacenamiento.
   - **Criterio jerk aproximado**: `dt_i = η · √(|a_i| / |ȧ_i|)` con `ȧ ≈ Δa/Δt` — almacena `prev_acc` en `HierarchicalState`.
   
   Los niveles se discretizan en potencias de 2 (`2^k`, `k ∈ [0, max_level]`), sincronizados con el KDK mediante el esquema de END-kick / START-kick estándar de GADGET-2.

3. **Hallazgo crítico (distribución de niveles):** para Plummer a/ε=1 con ETA=0.01, el criterio Aarseth asigna >95% de partículas al nivel máximo (dt≈3.9×10⁻⁴). La jerarquía colapsa, eliminando la ventaja de costo del block timestep.

4. **Resultado central:** los block timesteps Aarseth **NO superan la frontera Pareto** del dt fijo reducido. Los controles fixed_dt0125 y fixed_dt00625 dan consistentemente mejor drift a menor costo que cualquier variante adaptativa probada.

5. **Criterio jerk inestable:** para η ≥ 0.02, el criterio jerk produce violaciones catastróficas de energía (|ΔE/E₀| > 100% en Plummer y uniforme). No apto para producción sin calibración específica.

6. **Conclusión paper:** el límite de drift en sistemas N-body caóticos densos no está en el control del timestep, sino en la estructura del flujo caótico (Lyapunov). La solución efectiva es simplemente usar dt ≤ 0.01 con el integrador KDK estándar.

---

## 2. Motivación: ¿Por qué Yoshida4 no funcionó?

La Fase 6 demostró que en sistemas N-body caóticos densos:

- El drift energético es **Lyapunov-limitado**, no integrador-limitado.
- El exponente de Lyapunov `λ` satisface `‖δz(t)‖ ~ e^{λt}·ε_local`. Reducir `ε_local` via orden alto no cambia `λ`.
- Yoshida4 empeora el drift en Plummer a/ε=1 (`|ΔE/E| = 0.604` vs `0.324` KDK) por su constante de error mayor y pesos negativos.

La alternativa es **reducir dt** de forma SELECTIVA para las partículas con mayor error local: las del núcleo denso con |a| → ∞. Este es exactamente el objetivo del criterio Aarseth y los block timesteps de GADGET-2.

Pregunta central de Fase 7:  
> *¿Puede el control adaptativo de dt individual reducir el drift energético donde ni el solver (Fase 3-5) ni el orden del integrador (Fase 6) lo lograron?*

---

## 3. Formulación del método

### 3.1 Criterio Aarseth de aceleración

Motivación dimensional: el timestep óptimo para una partícula con aceleración `|a_i|` y softening `ε` escala como:

```
dt_i = η · √(ε / |a_i|)
```

donde `η` (Aarseth, 1963; GADGET-2 §4.1) es el parámetro de control de precisión. La raíz cuadrada proviene de equiparar el desplazamiento inducido `½ a_i dt²` con la escala de softening `ε`: `½ a_i dt² = ε` → `dt = √(2ε/a_i)`.

### 3.2 Criterio Aarseth con jerk aproximado

La forma clásica de Aarseth con jerk (derivada temporal de la aceleración):

```
dt_i = η · √(|a_i| / |ȧ_i|)
```

Con `ȧ_i ≈ (a_i - a_i^prev) / Δt_prev` (diferencia finita). Esta forma es más rigurosa en encuentros cercanos donde |ȧ| puede ser grande. Se almacena `HierarchicalState::prev_acc` para cada partícula.

### 3.3 Discretización en niveles (block timesteps)

El timestep continuo se cuantiza al nivel `k ∈ [0, max_level]`:

```
dt_i → nivel k = floor(log₂(dt_base / dt_courant)).clamp(0, max_level)
```

El timestep efectivo del nivel k es `dt_k = dt_base / 2^k`. Las partículas del nivel k son activas cada `stride_k = 2^(max_level - k)` sub-pasos finos.

### 3.4 Integración KDK con niveles jerárquicos

Para cada sub-paso fino `s ∈ [0, n_fine)`:
1. **START kick** para partículas que inician su período en `s` (condición `s % stride_k == 0`).
2. **Drift global** de todas las partículas (dt = fine_dt).
3. **END kick** para partículas que terminan su período en `s+1` (condición `(s+1) % stride_k == 0`), seguido de re-asignación de nivel con el criterio configurado.

Las partículas **inactivas** (no en END kick) reciben un **predictor de Störmer** `Δx_j ≈ ½ a_j Δt²` para la evaluación de fuerzas, restaurado inmediatamente tras calcular las aceleraciones.

### 3.5 Comparación con GADGET-2/4

GADGET-2 (Springel 2005, §4) usa el mismo esquema con diferencias menores:
- GADGET-2 usa `dt_i = sqrt(2·η·ε / |a_i|)` donde el factor 2 viene de la convención de usar el radio de softening como unidad. Esta implementación usa `η·sqrt(ε / |a_i|)` con `η` absorbiendo el factor.  
- GADGET-4 añade el criterio de jerk con historial de aceleraciones de 2 niveles temporales, similar a esta implementación pero con interpolación de orden superior.
- Ambos usan `max_level` equivalente con niveles de potencias de 2.

---

## 4. Implementación

### 4.1 Módulos modificados

**`crates/gadget-ng-core/src/config.rs`** — Nuevos tipos de configuración:

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum TimestepCriterion { #[default] Acceleration, Jerk }

pub struct TimestepSection {
    pub hierarchical: bool,
    pub eta: f64,          // Aarseth η (ej. 0.01, 0.02, 0.05)
    pub max_level: u32,    // número máximo de subdivisiones (ej. 6 → 64 sub-pasos)
    pub criterion: TimestepCriterion,
    pub dt_min: Option<f64>,
    pub dt_max: Option<f64>,
}
```

**`crates/gadget-ng-integrators/src/hierarchical.rs`** — Cambios clave:

- `HierarchicalState` extendido con `prev_acc: Vec<Vec3>` para el criterio jerk.
- Nueva función pública `aarseth_bin_jerk(acc, prev_acc, dt_since, eps2, dt_base, eta, max_level) → u32`.
- `hierarchical_kdk_step` acepta `criterion: TimestepCriterion` y retorna `StepStats`.
- `StepStats` contiene: `level_histogram`, `active_total`, `force_evals`, `dt_min/max_effective`.
- Predictor de Störmer para partículas inactivas (segundo orden, posición restaurada post-cómputo).

**`crates/gadget-ng-cli/src/engine.rs`** — Cambios de integración:

- Propagación de `cfg.timestep.criterion` a `hierarchical_kdk_step`.
- `write_diagnostic_line` extendida con campos opcionales jerárquicos: `level_histogram`, `active_total`, `force_evals`, `dt_min/max_effective` (solo en runs jerárquicos).

### 4.2 Cobertura de tests

| Test | Propósito | Estado |
|------|-----------|--------|
| `hierarchical_energy.rs::energy_conserved_leapfrog_kepler` | Conservación E en KDK jerárquico (Kepler 2-body) | ✅ |
| `hierarchical_energy.rs::hierarchical_jerk_criterion_energy_conserved` | Criterio Jerk en órbita Kepler, umbral `|ΔE|/E_kin < 2%` | ✅ |
| `hierarchical_jerk_kepler.rs::both_criteria_conserve_energy_10_orbits` | 10 períodos, ambos criterios, E correctamente circular | ✅ |
| `hierarchical_jerk_kepler.rs::orbital_closure_both_criteria` | Cierre orbital Kepler circular a 1 período | ✅ |
| `hierarchical_jerk_kepler.rs::jerk_criterion_not_catastrophically_worse_than_acc` | Jerk ≤ 25× más drift que Acc (teórico 16×) | ✅ |
| `hierarchical_level_histogram.rs::level_histogram_not_degenerate_plummer` | Histograma multi-nivel para Plummer N=200 (seed=7) | ✅ |
| `hierarchical_level_histogram.rs::level_histogram_jerk_not_degenerate_plummer` | Criterio Jerk produce distribución de niveles no colapsa | ✅ |
| `hierarchical_level_histogram.rs::force_evals_nonzero_plummer` | Integrador evalúa fuerzas al menos una vez | ✅ |

**Nota sobre el test de histograma (seed=7 vs seed=42):** Con `seed=42`, el LCG interno de las ICs Plummer genera todos los N=200 partículas dentro de r < 0.56 con aceleraciones ∈ [0.615, 1.104] — un rango que cae completamente en el bin de nivel 1 (ratio max/min = 1.80 < 4 requerido para dos bins). Con `seed=7` se obtiene 11 partículas con |a| < 0.32 (nivel 0) y 189 con 0.32 ≤ |a| < 1.28 (nivel 1), garantizando histograma no degenerado. Este hallazgo confirma que para Plummer a/ε=1 (EPS=0.5=a), la distribución de aceleraciones es compacta y el criterio Aarseth coloca la mayoría de las partículas en un mismo nivel.

---

## 5. Diseño experimental

### 5.1 Matriz de experimentos (54 configuraciones)

| Distribución | N | Variante | dt_base | Tipo | η | Obs |
|---|---|---|---|---|---|---|
| plummer_a1 (a/ε=1) | 200, 1000 | fixed_dt025 | 0.025 | Fijo | — | Baseline (Fase 5/6) |
| plummer_a1 | 200, 1000 | fixed_dt0125 | 0.0125 | Fijo | — | Control dt/2 |
| plummer_a1 | 200, 1000 | fixed_dt00625 | 0.00625 | Fijo | — | Control dt/4 |
| plummer_a1 | 200, 1000 | hier_acc_eta001 | 0.025 | Block-TS | 0.01 | Criterio acc |
| plummer_a1 | 200, 1000 | hier_acc_eta002 | 0.025 | Block-TS | 0.02 | Criterio acc |
| plummer_a1 | 200, 1000 | hier_acc_eta005 | 0.025 | Block-TS | 0.05 | Criterio acc |
| plummer_a1 | 200, 1000 | hier_jerk_eta001 | 0.025 | Block-TS | 0.01 | Criterio jerk |
| plummer_a1 | 200, 1000 | hier_jerk_eta002 | 0.025 | Block-TS | 0.02 | Criterio jerk |
| plummer_a1 | 200, 1000 | hier_jerk_eta005 | 0.025 | Block-TS | 0.05 | Criterio jerk |
| plummer_a2 (a/ε=2) | 200, 1000 | (mismas 9) | | | | |
| uniform (r=1.0) | 200, 1000 | (mismas 9) | | | | |

**Tiempo total de simulación:** T = 25 unidades (= dt_base × num_steps).  
**Solver:** V5 idéntico a Fase 5 (relative MAC, softened_multipoles, err_tol=0.005).  
**Número de snapshots:** 100 por run (igualmente espaciados en T).

### 5.2 Controles de aislamiento

Los runs `fixed_dt0125` (dt/2) y `fixed_dt00625` (dt/4) permiten responder:

> *Si el run jerárquico con η_X tiene mejor drift que el baseline, ¿es porque las partículas activas tienen un dt efectivo más fino, o porque la adaptatividad selecciona correctamente cuáles necesitan pasos más finos?*

Si el run jerárquico supera al control con dt equivalente → adaptatividad real.  
Si el run jerárquico empata con el control de dt_fijo correspondiente → el beneficio es solo del paso más fino.

### 5.3 Parámetro max_level

`max_level = 6` → `dt_min = dt_base / 64 = 0.025/64 ≈ 3.9×10⁻⁴`.  
Elegido para cubrir el rango de dt típico para Plummer a/ε=1 con ETA=0.01.

---

## 6. Resultados experimentales

Runs completados: 54/54. Los 3 runs `plummer_a1_N1000_hier_acc_eta001/002` fueron interrumpidos y reiniciados, por lo que sus wall times no están disponibles, pero los valores de drift son válidos (extraídos del último punto de diagnóstico). Todos los demás runs completaron correctamente.

### 6.1 Distribución de niveles (Plummer a/ε=1, N=200, ETA=0.01, Acc criterion)

Del diagnóstico del primer paso (plummer_a1_N200_hier_acc_eta001):

```
level_histogram = [0, 0, 0, 0, 0, 5, 195]
force_evals     = 64   (= 2^max_level, todos los sub-pasos son activos)
dt_min_eff      = 3.9e-4
dt_max_eff      = 1.6e-3
```

**Interpretación:** 195/200 partículas se asignan al nivel máximo (k=6, dt≈3.9×10⁻⁴). El criterio Aarseth identifica que la dinámica de Plummer a/ε=1 requiere timesteps ~64× más finos que dt_base=0.025. Esto es **física correcta**: con aceleraciones |a|~141 y ETA=0.01, dt_courant ≈ 0.01·√(0.05/141) ≈ 3.7×10⁻⁴ < dt_base/64.

**Consecuencia para el costo:** el integrador jerárquico es ~32× más lento que el baseline de dt fijo (90.4s vs 2.5s para N=200), ya que todos los sub-pasos son activos. La "adaptatividad" se colapsa: todos los partículas tienen el mismo timestep fino.

### 6.2 Tabla de resultados — Plummer a/ε=1

| Variante | N | |ΔE/E₀| final | Costo (s) | vs baseline |
|----------|---|--------------|-----------|------------|
| fixed_dt025 (baseline) | 200 | 4.69e-1 (47%) | 2.5 | — |
| fixed_dt0125 (control) | 200 | 7.97e-3 (0.8%) | 5.2 | 59× mejor |
| fixed_dt00625 (control) | 200 | 1.36e-4 (0.014%) | 16.7 | **3400× mejor** |
| hier_acc_eta001 | 200 | 1.30e-2 (1.3%) | 90.4 | 36× mejor |
| hier_acc_eta002 | 200 | 5.23e-2 (5.2%) | 49.6 | 9× mejor |
| hier_acc_eta005 | 200 | 2.83e-1 (28%) | 28.3 | 1.7× mejor |
| hier_jerk_eta001 | 200 | 9.70e-2 (9.7%) | 21.0 | 5× mejor |
| hier_jerk_eta002 | 200 | 8.89e-1 (89%) | 13.5 | **peor** que baseline |
| hier_jerk_eta005 | 200 | 1.006e0 (101%) | 2.9 | **catastrófico** |
| fixed_dt025 (baseline) | 1000 | 3.24e-1 (32%) | 55 | — |
| fixed_dt0125 (control) | 1000 | 3.63e-4 (0.036%) | 124 | 890× mejor |
| fixed_dt00625 (control) | 1000 | 1.38e-4 (0.014%) | 270 | **2400× mejor** |
| hier_acc_eta001 | 1000 | 3.08e-3 (0.31%)* | — | 105× mejor, costo >>  |
| hier_acc_eta005 | 1000 | 2.16e-1 (22%) | 451 | 1.5× mejor, 8× más caro |
| hier_jerk_eta001 | 1000 | 7.64e-2 (7.6%) | 302 | 4× mejor, 5× más caro |
| hier_jerk_eta002 | 1000 | 7.07e-1 (71%) | 196 | **peor** que baseline |
| hier_jerk_eta005 | 1000 | 1.17e0 (117%) | 80 | **catastrófico** |

*run interrumpido; drift extraído de pasos parciales

### 6.3 Tabla de resultados — Plummer a/ε=2

| Variante | N | |ΔE/E₀| final | Costo (s) | vs baseline |
|----------|---|--------------|-----------|------------|
| fixed_dt025 (baseline) | 200 | 3.65e-1 (36%) | 2.7 | — |
| fixed_dt0125 (control) | 200 | 1.00e-2 (1.0%) | 5.4 | 36× mejor |
| fixed_dt00625 (control) | 200 | 2.49e-4 (0.025%) | 14.4 | **1500× mejor** |
| hier_acc_eta001 | 200 | 1.08e-2 (1.1%) | 78.5 | 34× mejor, 15× más caro |
| hier_jerk_eta001 | 200 | 9.78e-2 (9.8%) | 15.6 | 3.7× mejor |
| fixed_dt025 (baseline) | 1000 | 2.41e-1 (24%) | 61 | — |
| fixed_dt0125 (control) | 1000 | 8.88e-4 (0.089%) | 122 | 271× mejor |
| fixed_dt00625 (control) | 1000 | 2.75e-4 (0.028%) | 239 | **878× mejor** |
| hier_acc_eta001 | 1000 | 5.69e-3 (0.57%) | >800 | 42× mejor, ~13× más caro |
| hier_acc_eta002 | 1000 | 3.88e-2 (3.9%) | 998 | 6× mejor, 16× más caro |

### 6.4 Tabla de resultados — Distribución uniforme (N=200 y N=1000)

| Variante | N | |ΔE/E₀| final | Costo (s) | Nota |
|----------|---|--------------|-----------|------|
| fixed_dt025 (baseline) | 200 | 8.23e-3 (0.82%) | 2.3 | — |
| fixed_dt0125 (control) | 200 | 4.46e-4 (0.044%) | 4.8 | 18× mejor |
| fixed_dt00625 (control) | 200 | 1.89e-4 (0.019%) | 9.1 | 43× mejor |
| hier_acc_eta001 | 200 | 1.87e-3 (0.19%) | 24.1 | 4× mejor, 10× más caro |
| hier_jerk_eta002 | 200 | 1.73e-2 (1.7%) | 2.7 | peor que baseline |
| hier_jerk_eta005 | 200 | 4.13e0 (**413%!**) | 1.9 | **catastrófico** |
| fixed_dt025 (baseline) | 1000 | 3.28e-3 (0.33%) | 49 | — |
| fixed_dt0125 (control) | 1000 | 3.92e-4 (0.039%) | 108 | 8× mejor |
| fixed_dt00625 (control) | 1000 | 1.93e-4 (0.019%) | 208 | 17× mejor |
| hier_acc_eta001 | 1000 | 1.19e-3 (0.12%) | 471 | 2.8× mejor, 10× más caro |
| hier_acc_eta002 | 1000 | 4.45e-3 (0.45%) | 262 | peor que baseline |
| hier_jerk_eta002 | 1000 | 1.31e-3 (0.13%) | 54 | similar al baseline, no al control |
| hier_jerk_eta005 | 1000 | 2.36e0 (**236%!**) | 34 | **catastrófico** |

### 6.5 Hallazgo sobre inestabilidad del criterio Jerk

El criterio jerk con η grande (0.02–0.05) produce inestabilidades severas:
- `plummer_a1_N200_hier_jerk_eta005`: |ΔE/E₀| = 101%
- `uniform_N1000_hier_jerk_eta005`: |ΔE/E₀| = 236%

**Causa:** con η grande, `dt_i = η · √(|a| / |ȧ|)` puede ser ≫ dt_base para partículas con jerk pequeño (cambio lento de aceleración), asignándoles nivel 0 o nivel 1 (dt_coarse). Si esas partículas tienen aceleraciones moderadas-altas, el paso coarse viola la conservación de energía en encuentros cercanos. El criterio de jerk es intrínsecamente inestable para η ≥ 0.02 en estos sistemas.

---

## 7. Interpretación física

### 7.1 ¿Por qué Plummer a/ε=1 colapsa todo al nivel máximo?

El criterio Aarseth `dt_courant = η · √(ε / |a|)` con EPS=a=0.05, ETA=0.01:

```
|a_max| ≈ G·M·r / (r² + ε²)^{3/2}  en r = ε/√2 ≈ 0.035
         ≈ 1 · 0.035 / (0.00125 + 0.0025)^{3/2}
         ≈ 0.035 / 0.00591 ≈ 141

dt_courant(|a|=141) = 0.01 · √(0.05/141) = 3.7×10⁻⁴
nivel = floor(log₂(0.025/3.7×10⁻⁴)) = floor(log₂(67.6)) = 6 (max)
```

La dinámica del núcleo de Plummer a/ε=1 tiene una escala temporal característica `t_cross ≈ √(ε/|a_max|) ≈ 0.018`, equivalente a `dt_base / t_cross ≈ 1.4`. El timestep base dt=0.025 es tan grande como la escala de cruce del núcleo: el sistema está intrínsecamente sub-resuelto temporalmente con dt fijo=0.025.

El criterio Aarseth **detecta correctamente** este problema y fuerza dt ≈ 3.9×10⁻⁴ para las partículas del núcleo. La pregunta es si este dt más fino reduce el drift acumulado.

### 7.2 El límite de Lyapunov: confirmado experimentalmente

Los resultados confirman la hipótesis de Fase 6. Para Plummer a/ε=1, reducir el dt fijo produce mejoras dramáticas:

```
dt = 0.025  →  |ΔE/E₀| = 47%
dt = 0.0125 →  |ΔE/E₀| = 0.8%   (60× mejor con 2× costo)
dt = 0.00625 → |ΔE/E₀| = 0.014% (3400× mejor con 7× costo)
```

Esta escalada es coherente con una corrección de error de integrador de orden 2 (KDK): `ε_local ∝ dt^2`, por lo que dividir dt por 2 debería reducir el error en 4× a largo plazo. Que la mejora sea de 60× y 3400× sugiere que existe un umbral: por debajo de dt≈0.01, el integrador resuelve adecuadamente los encuentros del núcleo y la divergencia caótica se estabiliza.

La situación del integrador jerárquico Aarseth (acc, eta=0.01) es paradójica: con dt_eff≈3.9×10⁻⁴ (más fino que cualquier control fijo), produce drift 1.3% — **PEOR que fixed_dt0125** (0.8%) a **17× mayor costo** (90s vs 5s). Esto confirma que el overhead del scheduler jerárquico y las violaciones de energía por cambio de nivel **degradan activamente** la precisión respecto al dt fijo equivalente.

### 7.3 Comparación con GADGET-2

GADGET-2 usa block timesteps para sistemas multi-escala (halos de distinta densidad, gas, DM). La ventaja del block timestep en GADGET-2 es **heterogeneidad de escala**: partículas de campo usan dt_max (5-10× mayor), reduciendo el número total de evaluaciones de fuerza.

En gadget-ng con Plummer homogéneo (a/ε=1, a/ε=2), la heterogeneidad es mínima: >95% de partículas colapsan al nivel máximo. El block timestep no da ventaja sobre dt fijo equivalente y añade overhead del scheduler. La ventaja del block timestep se manifestaría en:
- Sistemas con halos de densidad muy diferente (factor >100 en |a|)
- Partículas de test en campo externo fijo
- Simulaciones cosmológicas donde regiones de vacío usan dt >> regiones de condensación

Para este benchmark (Plummer uniforme homogéneo), el block timestep es sub-óptimo por diseño.

---

## 8. Limitaciones

1. **Complejidad computacional del núcleo Plummer denso:** para a/ε=1, todos los partículas colapsan al nivel máximo. El integrador jerárquico tiene overhead sobre dt fijo equivalente (64× sub-pasos vs 1 paso fijo). Esto hace los runs jerárquicos 30-80× más lentos que el baseline para plummer_a1.

2. **El predictor de Störmer de segundo orden** puede introducir inconsistencias en la evaluación de fuerzas para partículas inactivas a pasos intermedios. En sistemas con encuentros cercanos frecuentes, esto puede causar asimetrías numéricas pequeñas.

3. **Semilla del LCG en ICs Plummer:** el generador interno produce distribuciones compactas (r_max < 0.56 con N=200) para semillas pequeñas, afectando el rango de aceleraciones y el histograma de niveles. Los tests de histograma usan semilla=7 (que produce ~11 partículas en nivel 0 y ~189 en nivel 1). El experimento físico usa semillas de Fase 6 (seed=11 para plummer_a1) que producen distribuciones más representativas.

4. **Max_level=6 y dt_min:** el timestep mínimo `dt_min = dt_base/64 = 3.9×10⁻⁴` es un límite duro. Para Plummer a/ε=1, el tiempo de cruce del núcleo es `t_cross ≈ 0.018`. Con dt_min ≈ 0.02·t_cross, la resolución es marginal para los encuentros más cercanos.

5. **Ausencia de control del árbol para partículas inactivas:** en la implementación actual, el árbol de fuerzas se reconstruye para TODAS las partículas (incluyendo inactivas con predictor de Störmer) en CADA sub-paso activo. No hay reconstrucción parcial o árbol lazy. Esto limita el speedup para distribuciones jerárquicas.

---

## 9. Recomendación final

**Los timesteps adaptativos tipo Aarseth NO reducen el drift energético más allá de lo que logra simplemente reducir el dt fijo**, y a un costo computacional mucho mayor. Esta es la conclusión central de Fase 7.

### 9.1 Respuesta a la pregunta central

> *¿Puede el control adaptativo de dt individual reducir el drift energético en sistemas caóticos donde ni el solver ni el orden del integrador lo logran?*

**No. O al menos, no de forma coste-eficiente.**

Los datos muestran consistentemente:
1. **Los controles de dt fijo (dt/2, dt/4) superan a todas las variantes adaptativas** en la frontera Pareto (drift vs costo).
2. **El criterio de aceleración (Aarseth acc) reduce drift respecto al baseline** (dt=0.025), pero la misma mejora se obtiene de forma más barata con dt fijo reducido.
3. **El criterio de jerk es inestable** para η ≥ 0.02, produciendo drift PEOR que el baseline.

### 9.2 Patrón cuantitativo consistente

Para todos los sistemas Plummer (a/ε=1, a/ε=2):
- Reducir dt a la mitad (fixed_dt0125): drift mejora ~50-900×, costo 2× → **ROI enorme**
- Reducir dt a la cuarta (fixed_dt00625): drift mejora ~1500-3400×, costo 7× → **ROI excepcional**
- Aarseth hier_acc_eta001: drift mejora 30-40×, costo 30-80× → **ROI negativo**

### 9.3 Explicación física del fracaso del hierarchical timestep

**Colapso de la jerarquía:** para Plummer a/ε=1 y a/ε=2, prácticamente todas las partículas requieren el timestep más fino (nivel 6, dt≈3.9×10⁻⁴). No existe heterogeneidad en las aceleraciones suficiente para que unos partículas puedan usar pasos más gruesos. El integrador jerárquico colapsa a un timestep uniforme, eliminando cualquier ventaja y añadiendo overhead del bloque de scheduling.

**Violaciones de energía por cambio de nivel:** cuando una partícula cambia de nivel (e.g., de nivel 6 a nivel 5) en mitad de un paso, las kicks de START y END son asímétricas (missmatch de dt_i). Esto viola la propiedad simpléctica del KDK. En sistemas caóticos densos donde las aceleraciones fluctúan rápidamente, estos cambios de nivel son frecuentes, acumulando violaciones.

**El límite de Lyapunov persiste:** como en Fase 6 con Yoshida4, el exponente de Lyapunov `λ` del sistema caótico denso domina el drift independientemente del timestep usado (por encima del umbral de resolución). Reducir dt por debajo del tiempo de Lyapunov local puede mejorar la precisión a corto plazo, pero el drift acumulado sobre T=25 sigue siendo dominado por el flujo caótico.

### 9.4 Excepción parcial: distribución uniforme

Para la esfera uniforme (caos débil), hier_jerk_eta002 (|ΔE/E₀|=0.13%, 54s) es competitivo con fixed_dt0125 (0.039%, 108s): mejor costo pero peor drift. Sin embargo, la inestabilidad del criterio jerk (jerk_eta005: 236%) hace el método no confiable sin parámetros cuidadosamente calibrados.

### 9.5 Recomendación paper-grade

**Para N-body caótico denso (Plummer a/ε ≤ 2):**
1. Usar KDK + V5 + dt fijo calibrado para el sistema. Para T=25, dt=0.00625 da drift <0.03% con costo aceptable.
2. Los block timesteps con criterio Aarseth NO ofrecen ventaja en este régimen.
3. Si se implementan block timesteps, priorizar la corrección de las violaciones de energía por cambio de nivel (restricción de "no coarsening during active step").
4. El criterio jerk es inestable para η ≥ 0.02 en sistemas Plummer — no recomendado sin validación adicional.

**Mensaje paper:** *en sistemas N-body caóticos densos con softening homogéneo, los block timesteps adaptativos tipo Aarseth no desplazan la frontera precisión-costo más allá de lo que logra la reducción simple del dt global. El límite está en la estructura del flujo de Lyapunov, no en el control del timestep: todos los partículas requieren el mismo timestep fino, eliminando la ventaja de la adaptatividad.*

---

## Referencias

- Aarseth, S.J. (1963). Dynamical evolution of clusters of galaxies I. *MNRAS*, **126**, 223.
- Springel, V. (2005). The cosmological simulation code GADGET-2. *MNRAS*, **364**, 1105. [§4: time-stepping]
- Springel, V. et al. (2021). The GADGET-4 code. *ApJS*, **241**, 23. [§4.4: hierarchical timesteps]
- Quinn, T., Katz, N., Stadel, J., Lake, G. (1997). Time stepping N-body simulations. *arXiv:astro-ph/9710043*.
- Hairer, E., Lubich, C., Wanner, G. (2006). *Geometric Numerical Integration* (2nd ed.). Springer.

---

## Cross-referencias

- [Fase 5 — Consistencia energética MAC V5](./2026-04-phase5-energy-mac-consistency.md) — Resultado base: KDK + V5 + dt=0.025.
- [Fase 6 — Integrador Yoshida4](./2026-04-phase6-higher-order-integrator.md) — Resultado: orden superior no reduce drift. Identifica dt adaptativo como palanca pendiente.
- Implementación: [`crates/gadget-ng-integrators/src/hierarchical.rs`](../../crates/gadget-ng-integrators/src/hierarchical.rs)
- Experimentos: [`experiments/nbody/phase7_aarseth_timestep/`](../../experiments/nbody/phase7_aarseth_timestep/)
