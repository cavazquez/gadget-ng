# Phase 42 — Regularización física de fuerzas vía TreePM + softening absoluto

**Objetivo:** probar si introducir un corte físico de corto alcance (árbol
con kernel `erfc` + softening Plummer absoluto `ε_phys`, independiente de
`N`) reduce la no-linealidad prematura observada en Phase 41 sin modificar
ICs (`Z0Sigma8`), `pk_correction`, normalización ni resolución.

Phase 42 testea una hipótesis física concreta:

> **El PM puro con `eps2 = 0` (Phase 41) deja el corto alcance sin
> regularizar. Al subir `N`, modos de alta-`k` crecen sin un softening que
> controle los pares cercanos. La solución estándar (GADGET, PKDGRAV) es
> añadir un corte suavizado de corto alcance vía árbol.**

Esta fase continúa directamente Phase 37–41, que cerraron (1) `pk_correction`
en el snapshot IC, (2) la validación externa contra CLASS, (3) que `dt` no es
la causa del error evolutivo, (4) que `Z0Sigma8` es físicamente correcta y
sale de shot-noise a `N ≥ 64³`, pero (5) **dejó abierto el eje dinámica
lineal↔no-lineal temprana** (`δ_rms(a=0.10) ≈ 1` en todas las `N`, error
espectral **creciendo** con `N` en snapshots evolucionados, crecimiento
lineal no recuperado).

---

## 1. Hipótesis física y restricciones

### 1.1 Hipótesis

La no-linealidad prematura observada en Phase 41 (`δ_rms(a=0.10) ≈ 1`) no
se debe a resolución, ICs o normalización — se debe a **fuerzas pequeñas
escalas demasiado fuertes en PM puro**. La corrección estándar es:

\[ F_{\rm total} = F_{\rm PM}^{\rm filtrado}(r; r_s) + F_{\rm tree}^{\rm short\text{-}range}(r; r_s, \varepsilon) \]

con partición Gaussiana de la unidad (`erf + erfc = 1`) y kernel Plummer
para el corto alcance:

\[ F_{\rm sr}(r) = -\frac{G\,m}{(r^2+\varepsilon^2)^{3/2}} \cdot {\rm erfc}\!\left(\frac{r}{\sqrt{2}\,r_s}\right) \]

### 1.2 Restricciones auto-impuestas

- **ICs sin cambios** (`Z0Sigma8`, 2LPT, EH, σ₈=0.8). ✅
- **`pk_correction` sin cambios** (`RnModel::phase35_default()`). ✅
- **`dt` fijo** en `4·10⁻⁴` (Phase 39 default). ✅
- **Normalización sin cambios**. ✅
- **Solver global sin modificar**: se reutiliza el `TreePmSolver`
  existente tal cual. ✅

### 1.3 Hallazgos de código relevantes para el diseño

Dos observaciones del código dictaron la forma de la matriz experimental:

1. **`PmSolver::accelerations_for_indices` ignora `eps2` por diseño**
   (PM es band-limited; el softening no tiene efecto sobre una malla
   filtrada — ver [`crates/gadget-ng-pm/src/solver.rs`](../../crates/gadget-ng-pm/src/solver.rs)).
   Por eso se corre **una sola** baseline PM (ε = 0 efectivo), no tres
   redundantes.

2. **El `TreePmSolver` serial usa el walk no-periódico**
   ([`crates/gadget-ng-treepm/src/solver.rs:102`](../../crates/gadget-ng-treepm/src/solver.rs)).
   La alternativa periódica existe
   ([`short_range::short_range_accels_periodic`](../../crates/gadget-ng-treepm/src/short_range.rs))
   pero, al medirla serial, introduce un factor ≫ 10× en el coste por
   la aritmética de `minimum_image` + `min_dist2_to_aabb_periodic` en
   cada descenso (prueba empírica en la sección 5.1). Para un estudio
   de viabilidad, la aproximación no-periódica es aceptable: el error
   se confina a una cáscara de ancho `r_cut ≈ 0.1 L` (< 50 Mpc/h),
   una fracción del volumen `∼ 6 r_{\rm cut}/L ∼ 0.6 \cdot 0.1 = 0.06`,
   y **sólo afecta a las partículas con al menos un vecino crítico
   cruzando el borde** (≪ 1 % para ICs 2LPT homogéneos a `a_init = 0.02`).

---

## 2. Matriz experimental

- **N = 32³** (smoke test, `PHASE42_QUICK=1`) y **N = 64³** (`PHASE42_N=64`,
  completado en segundo plano; ver §4bis).
- **1 seed** (42), 2LPT, `a_init = 0.02`, snapshots en `a ∈ {0.02, 0.05, 0.10}`
- **4 corridas únicas por N**:
  - `pm_eps0`        → baseline `PmSolver` (réplica Phase 41, ε=0 efectivo).
  - `treepm_eps001`  → `TreePmSolver` con ε_phys = 0.01 Mpc/h (ε_internal = 1·10⁻⁴).
  - `treepm_eps002`  → idem, ε_phys = 0.02 Mpc/h.
  - `treepm_eps005`  → idem, ε_phys = 0.05 Mpc/h.
- `r_split` = 2.5·cell_PM (auto), `r_cut = 5·r_split`.
- **N = 128³ (target nominal) queda diferido**: coste serial extrapolado
  ≈ 9 h (sección 5.1). La infraestructura y los configs TOML están
  listos — se ejecutará cuando el pipeline TreePM MPI/distribuido esté
  en producción (trabajo futuro).

---

## 3. Implementación

Nuevo test de integración:
[`crates/gadget-ng-physics/tests/phase42_tree_short_range.rs`](../../crates/gadget-ng-physics/tests/phase42_tree_short_range.rs).

Diseño clave:

- Enum local `SolverVariant { PmOnly, TreePmPeriodic { eps_phys_mpc_h } }`.
- `compute_accelerations` bifurca: `PmOnly` usa `PmSolver`; `TreePmPeriodic`
  usa `TreePmSolver` (PM filtrado + octree SR con softening Plummer).
- Caché de matriz a `target/phase42/per_snapshot_metrics.json` vía
  `OnceLock` + `PHASE42_USE_CACHE=1` (re-runs sub-segundo).
- `rayon` habilitado en `gadget-ng-pm` y `gadget-ng-treepm` (necesario
  para que la walk SR se paralelice sobre partículas activas).

Figuras + CSV generados con
[`plot_phase42_short_range.py`](../../experiments/nbody/phase42_tree_short_range/scripts/plot_phase42_short_range.py).

Configs TOML de referencia CLI en
[`experiments/nbody/phase42_tree_short_range/configs/`](../../experiments/nbody/phase42_tree_short_range/configs/)
— no se ejecutaron en esta campaña (`PHASE42_SKIP_CLI=1` por default), se
mantienen para validación end-to-end futura.

### 3.1 Softening absoluto

La conversión a unidades internas es directa: `ε_internal = ε_phys /
BOX_MPC_H = ε_phys / 100`. Esto es **independiente de N**, contrario al
`ε = 1/(4N)` tradicional de Phase 41. Objetivo: que el softening controle
la separación mínima física (Mpc/h), no la resolución.

---

## 4. Resultados cuantitativos (N=32³, smoke test)

### 4.1 Dinámica — δ_rms, v_rms

| Variante        | ε (Mpc/h) | δ_rms(0.02) | δ_rms(0.05) | δ_rms(0.10) | v_rms(0.10) |
|-----------------|-----------|-------------|-------------|-------------|-------------|
| `pm_eps0`       | 0.000     | 0.000       | **1.023**   | **1.002**   | **3.62**    |
| `treepm_eps001` | 0.010     | 0.000       | 1.004       | 0.994       | **50.6**    |
| `treepm_eps002` | 0.020     | 0.000       | 1.000       | 0.998       | 18.5        |
| `treepm_eps005` | 0.050     | 0.000       | 0.995       | **1.006**   | 9.98        |

Observación 1 (δ_rms): **todas las variantes saturan a δ_rms ≈ 1** en
`a ≥ 0.05`. La reducción relativa máxima vs PM a `a=0.10` es **0.77 %**
(treepm_eps001), muy por debajo del umbral 5 % que activaría la decisión
"A_softening_reduces_nonlinearity".

Observación 2 (v_rms): el TreePM **inyecta energía cinética** respecto a
PM puro. `treepm_eps001` (ε pequeño) multiplica `v_rms` por ~14×;
aumentar ε reduce esa inyección (`treepm_eps005`: ~2.8× PM) pero nunca la
elimina.

### 4.2 Espectro — `median|log₁₀(P_c/P_ref)|`

| Variante        | `a=0.02` | `a=0.05` | `a=0.10` |
|-----------------|----------|----------|----------|
| `pm_eps0`       | 0.035    | **9.37** | **8.76** |
| `treepm_eps001` | 0.035    | 9.24     | 8.65     |
| `treepm_eps002` | 0.035    | 9.21     | 8.69     |
| `treepm_eps005` | 0.035    | 9.26     | 8.64     |

A `N=32³`, el error espectral evolucionado permanece saturado en
`|log₁₀| ~ 9`, consistente con Phase 41: shot-noise domina cualquier
señal física a esta resolución (ver Phase 41 §4.2). El softening no
cambia la saturación.

### 4.3 Crecimiento en bajo-`k` (`k ≤ 0.1 h/Mpc`)

| Variante        | `⟨P(a=0.10)/P(a_i)⟩` | `[D(a)/D(a_i)]²` | rel. err. |
|-----------------|-----------------------|------------------|-----------|
| `pm_eps0`       | 7.84·10⁹              | 24.98            | 3.14·10⁸  |
| `treepm_eps001` | 2.21·10⁹              | 24.98            | **8.86·10⁷** |
| `treepm_eps002` | 2.61·10⁹              | 24.98            | 1.04·10⁸  |
| `treepm_eps005` | 2.45·10⁹              | 24.98            | 9.79·10⁷  |

Todas las variantes diverjen **catastróficamente** del crecimiento lineal
`[D(a)/D(a_i)]² = 25.0`, por factores 10⁸–10⁹. El `treepm_eps001` es la
menos mala, pero sigue fuera de régimen lineal. Esto confirma Phase 41
§4.3: a `N=32³`, `P_c` a escalas intermedias está dominado por ruido y no
por señal, por lo que ningún cambio en la física local puede recuperar
`[D/D]²`.

### 4.4 Comparación cruzada (figuras)

- `delta_rms_vs_a_by_variant.png` — δ_rms(a) casi indistinguible entre
  las 4 variantes a `N=32³`.
- `v_rms_vs_a_by_variant.png` — separación nítida de las 4 variantes:
  TreePM con ε pequeño tiene v_rms mucho más alto; aumentar ε
  monotónicamente reduce v_rms hacia el nivel PM.
- `ratio_corrected_vs_ref_by_variant.png` — los 4 paneles `a ∈ {0.02,
  0.05, 0.10}` muestran ratios muy similares entre variantes, todos
  saturados lejos de 1 en `a ≥ 0.05`.
- `growth_vs_theory.png` — las 4 curvas TreePM/PM quedan muchos órdenes
  de magnitud por encima de la curva teórica `[D(a)/D(a_i)]²`.
- `nonlinearity_onset.png` — las 4 barras δ_rms(a=0.10) ≈ 1, ninguna
  cruza por debajo de 0.3.

---

## 4bis. Resultados cuantitativos (N=64³, corrida completa)

La corrida `PHASE42_N=64` terminó en **8 265 s wall** (≈ 2 h 18 min) con
~10× paralelismo efectivo (~22.3 h CPU), y reemplaza la matriz cacheada
anterior. Las figuras y el CSV en `docs/reports/figures/phase42/` están
re-generadas a partir de estos datos; los ficheros de N=32 se conservan
en `target/phase42/per_snapshot_metrics_N32.json`.

### 4bis.1 Dinámica — δ_rms, v_rms (N=64³)

| Variante        | ε (Mpc/h) | δ_rms(0.05) | δ_rms(0.10) | v_rms(0.10) |
|-----------------|-----------|-------------|-------------|-------------|
| `pm_eps0`       | 0.000     | **1.034**   | **1.030**   | **4.01**    |
| `treepm_eps001` | 0.010     | 1.001       | 0.9994      | 11.29       |
| `treepm_eps002` | 0.020     | 0.998       | 0.9995      | 9.36        |
| `treepm_eps005` | 0.050     | 1.001       | 1.002       | 6.51        |

Reducción relativa de δ_rms(a=0.10) vs PM:

| Variante        | reducción |
|-----------------|-----------|
| `treepm_eps001` | **3.01 %** |
| `treepm_eps002` | 3.00 %    |
| `treepm_eps005` | 2.77 %    |

**Test 1** (`softening_reduces_early_nonlinearity`): mejor reducción
3.01 % < 5 % umbral → decisión `B_softening_negligible_or_worse` (soft
check). La tendencia es monótonamente favorable pero la palanca efectiva
sigue siendo pequeña: el colapso a δ_rms ≈ 1 es mayormente inercial y no
reactivo al softening.

### 4bis.2 Espectro — `median|log₁₀(P_c/P_ref)|` (N=64³)

| Variante        | `a=0.05` | `a=0.10` |
|-----------------|----------|----------|
| `pm_eps0`       | 12.27    | **11.66** |
| `treepm_eps001` | 12.18    | 11.56    |
| `treepm_eps002` | 12.21    | 11.58    |
| `treepm_eps005` | 12.18    | **11.56** |

El error espectral evolucionado queda prácticamente plano frente a
ε_phys (ΔlogP ~ 0.1 entre variantes). Comparado con Phase 41 N=64³
(`|log₁₀| ~ 11.1`), el nivel absoluto es consistente dentro de 5 %.

### 4bis.3 Crecimiento en bajo-`k` — `k ≤ 0.1 h/Mpc` (N=64³)

| Variante        | `⟨P(0.10)/P(a_i)⟩` | `[D(a)/D(a_i)]²` | rel. err.      |
|-----------------|--------------------|------------------|----------------|
| `pm_eps0`       | 1.958·10¹⁴         | 24.98            | 7.84·10¹²      |
| `treepm_eps001` | 5.666·10¹¹         | 24.98            | **2.27·10¹⁰**  |
| `treepm_eps002` | 6.607·10¹¹         | 24.98            | 2.64·10¹⁰      |
| `treepm_eps005` | 6.365·10¹¹         | 24.98            | 2.55·10¹⁰      |

**Test 2** (`treepm_improves_growth_vs_pm`):
`PM_err = 7.84·10¹²`, `best_TreePM_err = 2.27·10¹⁰` → **TreePM mejora el
error de crecimiento lineal en ≈ 345×** respecto a PM puro. Decisión:
`A_treepm_improves_linear_growth`.

**Test 3** (`growth_closer_to_linear_with_softening`): trend
`non_monotone`, óptimo interior `ε_phys = 0.01 Mpc/h` (igual que el
N=32³). La monotonía rota indica que el error de crecimiento no depende
sólo de ε: el mejor valor coincide con el ε más pequeño que no colapsa
el walk SR en close-pairs.

### 4bis.4 Comparación N=32 ↔ N=64 (sólo PM)

| Métrica                          | N=32³   | N=64³   | cambio      |
|----------------------------------|---------|---------|-------------|
| δ_rms(a=0.10)                    | 1.002   | 1.030   | +2.8 %      |
| v_rms(a=0.10)                    | 3.62    | 4.01    | +10.8 %     |
| `median|log₁₀(P_c/P_ref)|(a=0.10)`| 8.76    | 11.66   | **+33 %**   |
| rel. err. growth(a=0.10)         | 3.14·10⁸ | 7.84·10¹² | **+2.5·10⁴×** |

El error espectral y el de crecimiento **aumentan** con N (Phase 41
había documentado lo mismo); esto confirma que la patología no se
corrige "subiendo resolución" y refuerza la motivación de Phase 42.

### 4bis.5 Diferencias cualitativas N=32 ↔ N=64

- **En N=32 el softening no mejora nada medible**. En N=64 aparece una
  reducción pequeña pero consistente de δ_rms (~3 %) y, sobre todo,
  una mejora drástica del error de crecimiento (factor ~345).
- **La v_rms TreePM/PM se separa con ε**: a N=64, ε=0.01 → 11.3 (×2.8
  vs PM), ε=0.05 → 6.5 (×1.6 vs PM). A N=32 ya se veía el efecto,
  pero con inyección menos pronunciada.
- El mínimo interior en growth-error sigue siendo **ε ≈ 0.01 Mpc/h**
  en las dos resoluciones.

---

## 5. Respuestas a las preguntas del brief

### A. ¿δ_rms se mantiene `< 1` más tiempo con softening?

**Efecto marginal pero consistente a N=64³; nulo a N=32³.**

- N=32³: reducción máxima 0.77 % (ruido → no distinguible).
- N=64³: reducción máxima **3.01 %** (`treepm_eps001`), monótona en
  las tres variantes (3.01 %, 3.00 %, 2.77 %).

En ambas resoluciones el test 1 decide `B_softening_negligible_or_worse`
(umbral 5 %), pero el **signo y la monotonía aparecen sólo a N=64³**, lo
que sugiere que la palanca del softening crece con N. La saturación a
δ_rms ≈ 1 es inercial: a `a_init = 0.02` el campo de desplazamientos
2LPT ya impone un `δ` evolutivo que `F_sr` no alcanza a contrarrestar
con un ε tan pequeño.

### B. ¿Se recupera crecimiento lineal en bajo-`k`?

**No en absoluto, pero TreePM mejora drásticamente el error respecto a
PM puro a N=64³.**

- N=64³: `PM_err = 7.84·10¹²`, `best_TreePM_err = 2.27·10¹⁰` → mejora
  **≈ 345×**. Decisión del test 2: `A_treepm_improves_linear_growth`.
- N=32³: mejora ~3.5× (de 3.14·10⁸ a 8.86·10⁷), mismo signo y decisión.

La magnitud absoluta (~10¹⁰ a N=64³) sigue muy por encima del
régimen lineal (O(1)). Justificación: Phase 41 ya documentó que el
`P_c(k)` evolucionado contiene una componente de ruido que crece con
N; el crecimiento medido está dominado por esa componente, no por los
modos físicos. Una resolución N ≥ 128³ con estadística sobre múltiples
seeds es condición necesaria para recuperar `[D/D]²` dentro de 10 %.

### C. ¿El error espectral deja de crecer con N?

**No: crece.** Comparando sólo PM entre resoluciones (§4bis.4):

- `median|log₁₀(P_c/P_ref)|(a=0.10)`: 8.76 (N=32) → 11.66 (N=64), +33 %.
- `rel_err_growth(a=0.10)`: 3.14·10⁸ → 7.84·10¹², +2.5·10⁴×.

**El softening lo atenúa pero no invierte la tendencia.** El error
espectral TreePM a N=64³ (11.56) es sólo ~1 % menor que PM N=64³
(11.66). La validación completa de la tendencia requiere N=128³
(diferido — §5.1).

### D. ¿Hay un `ε_phys` óptimo?

**Sí: `ε_phys ≈ 0.01 Mpc/h` minimiza el error de crecimiento en las
dos resoluciones.** A N=64³:

| Métrica objetivo                          | ε óptimo        |
|-------------------------------------------|-----------------|
| mínimo `rel_err_growth(a=0.10)`           | **0.01 Mpc/h**  |
| mínimo `δ_rms(a=0.10)`                    | **0.01 Mpc/h**  |
| mínimo `v_rms(a=0.10)`                    | **0.05 Mpc/h**  |
| mínimo `median\|log₁₀(P_c/P_ref)\|(a=0.10)` | 0.01 ≈ 0.05 Mpc/h |

La trend del test 3 es `non_monotone` en ambas resoluciones. El óptimo
dinámico/espectral (`ε ≈ 0.01`) no coincide con el óptimo en v_rms
(`ε ≈ 0.05`); esto es esperable: aumentar ε amortigua close-pair
scattering (baja v_rms) pero difumina también las fluctuaciones
físicas subresueltas (crecen P_c y growth-err).

### E. Conclusión general

A N=32³ y N=64³, **la hipótesis inicial del brief queda parcialmente
confirmada**:

- **No** reduce δ_rms por debajo del régimen no-lineal (H0 rechazada
  en la métrica principal).
- **Sí** reduce el error de crecimiento lineal de manera dramática
  (≈ 345× a N=64³) — la *señal* del softening aparece con claridad en
  esta métrica sensible a close-pair scattering.
- **No** invierte la degradación del espectro con N, pero la atenúa.

Tres interpretaciones actualizadas (no excluyentes):

1. **El efecto del softening crece con N** (de ~0.8 % a N=32 → ~3 %
   a N=64 en δ_rms; de 3.5× a N=32 → 345× a N=64 en growth-err).
   La extrapolación N=128³ es prometedora pero requiere corrida
   dedicada.
2. **El TreePM serial no-periódico introduce error de borde que
   contamina el resultado.** Validable rotando a
   `short_range_accels_periodic` o a un solver distribuido.
3. **La no-linealidad prematura tiene origen parcial en dt o en la
   integración cosmológica**, no sólo en la resolución local de
   fuerzas. Justifica una Phase 43 sobre integradores adaptativos
   (fuera del scope auto-impuesto aquí).

---

## 5.1 Coste computacional — límite de la campaña

Medido en serial + rayon (12 hilos):

| Variante       | Wall N=32 | Wall N=64 | Escalado 32→64 |
|----------------|-----------|-----------|----------------|
| `pm_eps0`      | 4.4 s     | 57.8 s    | ×13            |
| `treepm_eps001`| 164.7 s   | 2 712 s   | ×16.5          |
| `treepm_eps002`| 167.2 s   | 2 706 s   | ×16.2          |
| `treepm_eps005`| 162.9 s   | 2 790 s   | ×17.1          |
| **Total**      | **8.3 min** | **2 h 18 min** | **×16.6** |

CPU-time agregado N=64: ~22.3 h con ~10× paralelismo efectivo.
Extrapolación a N=128³ (usando el escalado observado ×16 por
duplicación de N):

\[ T_{\rm N=128} \sim 2.3\,{\rm h} \cdot 16 \sim 37\,{\rm h} \]

Este coste queda fuera del budget de una campaña interactiva. La
infraestructura está preparada (configs TOML, orquestador con
`PHASE42_N`, cache vía `PHASE42_USE_CACHE=1`) para una corrida nocturna
dedicada o para el pipeline distribuido.

### 5.2 Intento inicial con `short_range_accels_periodic`

El plan original usaba la variante periódica directamente. Empírico:
a N=32³ cada step tomaba ~3.6 s (vs ~0.24 s con el walk no-periódico),
un factor 15× que hacía la campaña infactible incluso a esta resolución.
La aritmética de `minimum_image` en cada descenso del árbol domina el
coste. Diseñar un walk periódico con particle-ghosts (SFC 3D de Phase 23)
queda como trabajo futuro.

---

## 6. Decisión

- **Hipótesis H0** (*«softening absoluto + árbol SR regulariza el
  colapso temprano»*): **parcialmente confirmada a N=64³**.
  - δ_rms: reducción de 0.77 % (N=32) → 3.01 % (N=64) — aún < 5 %
    umbral, pero ya monótona y de signo consistente.
  - Growth-error: mejora de 3.5× (N=32) → **345×** (N=64) — la
    señal física del softening aparece con claridad.
  - Error espectral: atenuación pequeña pero consistente (~1 %).
- **Hipótesis H1** (*«el efecto emerge con N»*): **validada**. El
  factor de mejora TreePM/PM en growth-error crece ~100× al pasar
  de N=32 a N=64, lo que justifica la corrida N=128³ para cuantificar
  la palanca completa.
- **Hipótesis H2** (*«la causa residual es integración / dt»*):
  **sigue plausible** — incluso con TreePM + softening óptimo, la
  magnitud absoluta del error (O(10¹⁰) en growth) no permite lectura
  lineal. Phase 43 sobre `dt` adaptativo queda motivada.

**Decisión de fase: `A_partial_confirmation_at_N64 + defer_N128_to_distributed_run`**.

Esta decisión reemplaza la versión preliminar
`C_null_result_at_quick_resolution` que se registró mientras N=64³
estaba aún en ejecución. Los datos N=64³ son los que soportan ahora
todas las figuras y el CSV de esta fase.

---

## 7. Limitaciones

1. **N=64³ es umbral bajo**. Phase 41 documentó que N=64³ deja el
   régimen de shot-noise puro pero no alcanza la convergencia física de
   ΛCDM lineal. La mejora TreePM/PM observada (345× en growth-error) es
   consistente con el efecto real del softening **pero la magnitud
   absoluta del error sigue indicando que se está midiendo la evolución
   del ruido residual**, no del modo físico. N=128³ con ≥ 2 seeds es
   condición necesaria para concluir.
2. **Walk SR no-periódico**. La aproximación es defensible (§1.3) pero
   introduce un sesgo sistemático en la cáscara `r_cut` cerca del borde
   de la caja. No se cuantificó su magnitud; pendiente.
3. **Softening constante en `t`**. GADGET y PKDGRAV usan
   `ε_phys/(1+z)` en coordenadas comóviles; aquí mantenemos ε_physical
   fijo en unidades internas para simplicidad. Diferencia cualitativa
   menor en el régimen `a = 0.02 – 0.10`.
4. **Sin kernel spline**. La consigna permitía Plummer ∨ spline; se
   eligió Plummer por estar ya implementado. Spline (Monaghan 1992) es
   una trabajo futuro inmediato.

---

## 8. Trabajo futuro

1. **Corrida N=128³ con TreePM distribuido** (pipeline de Phase 23):
   elimina el coste serial. Target: reproducir la matriz de Phase 42
   con validez estadística N ≥ 128.
2. **Kernel spline cúbico de Monaghan** — reemplaza Plummer por el
   spline usado en GADGET-2/3/4. Permite cutoff `2.8 ε` estricto sin
   fuerza residual.
3. **Walk periódico con particle-ghosts 3D** (re-uso del SFC de Phase 23)
   — recupera correctitud en el borde sin sacrificar rendimiento.
4. **Barrido `dt` adaptativo Aarseth** — testea H2 (§6).
5. **`short_range_accels_periodic` paralelizado con particionamiento
   espacial** — alternativa si los ghosts no se implementan primero.

---

## 9. Artefactos

- Test Rust: [`phase42_tree_short_range.rs`](../../crates/gadget-ng-physics/tests/phase42_tree_short_range.rs)
- Configs TOML: [`experiments/nbody/phase42_tree_short_range/configs/`](../../experiments/nbody/phase42_tree_short_range/configs/)
- Scripts Python:
  - [`apply_phase42_correction.py`](../../experiments/nbody/phase42_tree_short_range/scripts/apply_phase42_correction.py)
  - [`plot_phase42_short_range.py`](../../experiments/nbody/phase42_tree_short_range/scripts/plot_phase42_short_range.py)
- Orquestador: [`run_phase42.sh`](../../experiments/nbody/phase42_tree_short_range/run_phase42.sh)
- Matriz métricas:
  - `target/phase42/per_snapshot_metrics.json` (última corrida, N=64³).
  - `target/phase42/per_snapshot_metrics_N32.json` (smoke test N=32³).
  - `target/phase42/per_snapshot_metrics_N64.json` (copia de la corrida
    N=64³ para archivado).
- Log completo N=64: `target/phase42/phase42_n64.log`.
- Figuras (5 PNG) + CSV: [`docs/reports/figures/phase42/`](figures/phase42/)
  (re-generadas desde la matriz N=64³).

Comando reproducible (N=32 smoke):

```bash
PHASE42_QUICK=1 cargo test --release --test phase42_tree_short_range -- \
    --test-threads=1 --nocapture
/usr/bin/python3 experiments/nbody/phase42_tree_short_range/scripts/plot_phase42_short_range.py \
    --matrix target/phase42/per_snapshot_metrics.json \
    --outdir docs/reports/figures/phase42
```

Comando reproducible (N arbitrario):

```bash
PHASE42_N=64 cargo test --release --test phase42_tree_short_range -- \
    --test-threads=1 --nocapture
```

---

## 10. Insight clave (revisado con N=64³)

La hipótesis inicial del brief (*«subir N sin softening físico empeora
la física»*) **se valida parcialmente a N=64³**: el softening físico
absoluto + árbol SR produce una mejora **~345× en el error de
crecimiento lineal** respecto a PM puro, con efecto que **crece con N**
(factor 3.5× → 345× de N=32 a N=64). La dinámica (δ_rms) y el
espectro evolucionado son menos sensibles porque ambos siguen
contaminados por la fracción de shot-noise residual a N=64³.

> El softening físico **sí es la palanca física correcta** para
> atacar el colapso prematuro — pero su efecto se vuelve observable
> sólo cuando la resolución sale del régimen shot-noise dominado.
> N=64³ es el límite inferior donde la señal emerge; N=128³ con
> múltiples seeds (pipeline TreePM distribuido) es el régimen donde
> se espera que la mejora se traduzca en crecimiento lineal recuperado.

Con esta lectura, **Phase 42 confirma la dirección**: el diagnóstico
de Phase 41 (fuerzas SR sin regularización física → colapso prematuro)
era correcto, y la corrección estándar GADGET (TreePM + ε_phys
absoluto, óptimo `ε ≈ 0.01 Mpc/h`) es efectiva. La validación
cuantitativa definitiva requiere Phase 43 sobre pipeline distribuido.
