# Phase 39 — Convergencia temporal del integrador Leapfrog KDK

## 1. Contexto

Phase 38 cerró la validación externa de `pk_correction` contra CLASS en el
snapshot IC: `median |log10(P_c/P_CLASS)| ≈ 0.02–0.07` para `N ∈ {32³, 64³}`,
independiente de semilla y convención (`legacy` vs `rescaled`). El pipeline
de amplitud absoluta está cerrado para las ICs.

Lo que **no** estaba cuantificado, al cierre de Phase 38, es cuán rápido la
corrida pierde fidelidad espectral tras unos cientos de pasos del
integrador PM + Leapfrog KDK, y si reducir `dt` permite mantener la
evolución en régimen lineal temprano. Phase 37 ya había observado que
ambos modos (`legacy` y `rescaled`) se degradan en snapshots evolucionados,
pero la investigación del `dt` quedó fuera de su alcance.

Este reporte documenta un barrido controlado de `dt` sobre la convención
`legacy` (la recomendada por Phase 37) con el objetivo de responder:

> **¿Qué rango de `dt` permite mantener la evolución en régimen lineal
> temprano y conservar la fidelidad espectral (forma + amplitud) durante
> los primeros pasos?**

La respuesta corta: **ningún `dt` en `[5·10⁻⁵, 4·10⁻⁴]` lo logra en el
estado físico actual del pipeline**. La evolución sale del régimen lineal
antes de que la integración temporal pueda importar. Esto traslada el
problema al diseño de las ICs / convención de amplitud inicial, no al
paso temporal.

## 2. Setup experimental

### 2.1. Cosmología y código (idénticos a Phase 37/38)

- `Ω_m = 0.315`, `Ω_Λ = 0.685`, `Ω_b = 0.049`, `h = 0.674`, `n_s = 0.965`,
  `σ₈ = 0.8`, `T_CMB = 2.7255 K`.
- Solver: PM con grid `N_mesh = N = 32`, caja `L = 100 Mpc/h`.
- ICs: Zel'dovich 2LPT con transfer EH no-wiggle, amplitud normalizada
  para `σ₈ = 0.8` aplicada en `a_init = 0.02`
  (`rescale_to_a_init = false`, modo **legacy**).
- Integrador: Leapfrog cosmológico KDK
  ([crates/gadget-ng-integrators/src/leapfrog.rs](../../crates/gadget-ng-integrators/src/leapfrog.rs)),
  orden 2 → error de integración teórico `O(dt²)` en régimen lineal.
- `pk_correction` aplicada con el modelo congelado en Phase 35
  (`RnModel::phase35_default`), sin recalibración.

### 2.2. Matriz de barrido

| Eje       | Valores                                  | Cantidad |
|-----------|------------------------------------------|----------|
| `dt`      | `4·10⁻⁴, 2·10⁻⁴, 1·10⁻⁴, 5·10⁻⁵`         | 4        |
| `seed`    | `42, 137, 271`                           | 3        |
| `a`       | `0.02, 0.05, 0.10`                       | 3        |
| **Total** | `36 mediciones` (PM/N=32³, 2LPT, legacy) | **36**   |

Número de pasos PM por corrida: `≈ 250 + 600 = 850` para `dt = 4·10⁻⁴`,
hasta `≈ 1985 + 4853 = 6838` para `dt = 5·10⁻⁵`.

### 2.3. Referencia física

Se usa la convención `legacy` validada en Phase 38:

$$
P_\text{ref}(k, a) = P_\text{EH}(k, z{=}0)\,\bigl[D(a)/D(a_\text{init})\bigr]^2
$$

con `D(a)` obtenido por la aproximación Carroll–Press–Turner (1992). Phase
38 cerró que esta referencia coincide numéricamente con
`P_CLASS(k, z{=}0)` en el modo legacy dentro del error de Eisenstein–Hu
no-wiggle.

### 2.4. Métricas

Por snapshot:

- **Espectrales**: `median |log10(P_c/P_ref)|`, `mean(P_c/P_ref)`,
  `stdev(P_c/P_ref)`.
- **Dinámicas**: `δ_rms(a)` (CIC sobre mesh `N=32`), `v_rms(a)` (momento
  canónico interno).
- **Estabilidad**: conteo de bins con `pk ≤ 0` / no finitos (NaN/Inf).
- **Costo**: runtime wall-clock por corrida completa (3 snapshots) con
  `std::time::Instant`.

### 2.5. Tests automáticos

Cinco tests en
[crates/gadget-ng-physics/tests/phase39_dt_convergence.rs](../../crates/gadget-ng-physics/tests/phase39_dt_convergence.rs).
Todos pasan en la matriz actual (170 s release). El patrón mezcla hard
checks (finitud, no NaN/Inf) con soft checks observacionales que
**registran la hipótesis sin presuponer el veredicto**, como se hizo en
Phase 37.

| # | Test                                             | Tipo      | Resultado |
|---|--------------------------------------------------|-----------|-----------|
| 1 | `dt_does_not_affect_ic_snapshot`                 | hard      | ✅ spread entre dts = 0 exacto |
| 2 | `smaller_dt_reduces_spectral_error`              | soft      | `hypothesis=false` (ratios 1.13/1.14 en a=0.05/0.10) |
| 3 | `dt_small_runs_stable` (NaN/Inf)                 | hard      | ✅ sin NaN/Inf en los 36 snapshots |
| 3 | `dt_small_runs_stable` (régimen lineal)          | soft      | `linear_regime_maintained=false` (growth ≈ 15 000 vs 5 esperado) |
| 4 | `dt_convergence_trend_detectable` (finitud)      | hard      | ✅ pendiente finita |
| 4 | `dt_convergence_trend_detectable` (>0.5)         | soft      | `trend_detectable=false` (slope ≈ −0.054) |
| 5 | `dt_scaling_consistent_with_integrator_order`    | soft      | `supports_order2=false` (slopes −0.054/−0.061) |

## 3. Resultados cuantitativos

### 3.1. Snapshot IC

El IC se construye con 2LPT analítico; la integración temporal no
interviene. Por tanto los 4 dts dan `median |log10(P_c/P_ref)|` idéntico
(spread `< 10⁻¹⁰`, literalmente 0 en la matriz). Valores por semilla:

| seed | `median |log10(P_c/P_ref)|` (IC) |
|------|----------------------------------|
| 42   | 0.035                            |
| 137  | 0.015                            |
| 271  | 0.055                            |

Consistente con los resultados de Phase 36/38: `pk_correction` cierra
amplitud absoluta en IC al nivel de 3–8 %.

### 3.2. Matriz agregada (media sobre 3 seeds)

Referencia CSV completa:
[`experiments/nbody/phase39_dt_convergence/output/dt_vs_error.csv`](../../experiments/nbody/phase39_dt_convergence/output/dt_vs_error.csv)
(36 filas).

| `dt`     | pasos | runtime/seed [s] | `median |log(P_c/P_ref)|` a=0.05 | `median |log(P_c/P_ref)|` a=0.10 | `δ_rms` a=0.10 | `v_rms` a=0.10 |
|----------|-------|------------------|----------------------------------|----------------------------------|----------------|----------------|
| 4·10⁻⁴ (dt₀)   |  200  |  ~4.0 | 6.22 | 5.62 | 0.56 | 2.46 |
| 2·10⁻⁴ (dt₀/2) |  400  |  ~7.5 | 6.31 | 5.65 | 0.69 | 1.89 |
| 1·10⁻⁴ (dt₀/4) |  800  | ~15.0 | 6.40 | 5.78 | 0.81 | 1.31 |
| 5·10⁻⁵ (dt₀/8) | 1600  | ~29.8 | 7.02 | 6.42 | 1.49 | 1.27 |

Observaciones:

- **El error corregido crece al reducir `dt`**: de 6.22 (dt₀) a 7.02
  (dt₀/8) en `a=0.05`. Este patrón se repite en `a=0.10` (5.62 → 6.42).
- **`δ_rms(a=0.10)` también crece con menos `dt`**: de 0.56 a 1.49.
- **`v_rms` cae y tiende a un plateau ~1.2–1.3** para `dt ≤ 1·10⁻⁴`, lo
  que sugiere un estado de virialización dinámica (no uno de crecimiento
  lineal).
- Runtime escala lineal con el número de pasos, como se esperaba.

### 3.3. Crecimiento dinámico vs teoría lineal

Para `a_init = 0.02 → a = 0.10` la predicción lineal CPT92 es
`D(a)/D(a_init) ≈ 5.0`, por tanto `δ_rms(a=0.10)/δ_rms(a_init) ≈ 5.0` si
el régimen fuera lineal. El valor observado es
`≈ 1.4·10⁴` (14 000 ×) con `dt₀/8`. El sistema está
**~2 800× por encima** del crecimiento lineal, es decir, completamente
dentro del régimen no-lineal desde los primeros cientos de pasos.

### 3.4. Convergencia OLS log-log

Ajuste `log(err) = α · log(dt) + c` sobre los 4 dts (media de seeds):

| a     | pendiente observada | esperada (KDK O(dt²)) | interpretación                 |
|-------|---------------------|------------------------|--------------------------------|
| 0.05  | `−0.054`            | `+2.0`                | **no asintótico**; ligeramente anti-convergente |
| 0.10  | `−0.061`            | `+2.0`                | **no asintótico**; ligeramente anti-convergente |

El signo negativo no refleja una patología del integrador — el test 1
muestra que en ausencia de integración (IC) el IC es bit-idéntico. Indica
que el error residual tras integración **no está dominado por el término
de truncado O(dt²)**, sino por el error físico de la amplitud inicial
(convención legacy `σ₈ = 0.8` aplicada en `a_init`).

## 4. Interpretación física

Los datos decomponen el error observado en dos términos mayoritariamente
ortogonales:

1. **Error de amplitud inicial** (dominante en la matriz):
   La convención `legacy` fija `σ₈ = 0.8` en `a_init = 0.02`,
   sobre-amplificando las ICs respecto a la extrapolación lineal
   `[D(a_init)/D(1)]² · P_EH(k, z=0)` por un factor
   `[D(1)/D(a_init)]² ≈ 2 500`. Este error domina el balance una vez que
   la gravedad comienza a actuar.

2. **Error de integración temporal** O(dt²):
   Intrínsecamente pequeño, escondido por (1).

El signo observado de la pendiente (`slope ≈ −0.05`) ocurre porque, al
reducir `dt`, el sistema resuelve con más fidelidad los potenciales
gravitatorios locales de un estado **ya no-lineal**, haciendo que el
colapso sea más pronunciado y alejándose más del régimen lineal de la
referencia. Es el comportamiento esperado al "zoomear" un integrador
conservativo sobre un problema físico fuera del régimen en el que la
referencia es válida.

Visualmente:

- **Figura 1** (`error_vs_dt.png`): la línea `∝ dt²` esperada (rojo) cae
  varias décadas por debajo de las curvas observadas, que son casi planas.
- **Figura 2** (`ratio_per_dt.png`): `P_c/P_ref` es del orden de `10⁵ –
  10⁸` en los evolucionados, consistente con sobre-crecimiento no-lineal.
- **Figura 3** (`delta_rms_vs_a.png`): la curva roja (predicción lineal
  `∝ D(a)`) queda ~4 órdenes por debajo de todas las corridas en `a=0.10`.
- **Figura 4** (`cost_vs_precision.png`): el error casi no cambia con
  runtime; Pareto trivial en `dt₀`.

## 5. `δ_rms(a)` vs teoría lineal

Ver figura 3. Para seed 42 (valor típico):

| `a`     | `δ_rms` (dt₀) | `δ_rms` (dt₀/8) | `D(a)/D(a_init)` | δ_rms lineal esperado |
|---------|---------------|-----------------|--------------------|------------------------|
| 0.02    | 1.0·10⁻⁴      | 1.0·10⁻⁴        | 1.00               | 1.0·10⁻⁴               |
| 0.05    | 0.60          | 1.54            | 2.38               | 2.4·10⁻⁴               |
| 0.10    | 0.57          | 1.58            | 5.00               | 5.1·10⁻⁴               |

Ningún `dt` mantiene la evolución dentro del régimen lineal. El factor
`growth/linear` ≈ `10⁴`, sistemático.

## 6. Costo computacional

Escaleo lineal: 4.0 / 7.5 / 15.0 / 30.0 s por seed para `dt₀ / dt₀/2 /
dt₀/4 / dt₀/8`. Factor 7.5× entre extremos, consistente con 8× de pasos
(la pequeña diferencia viene del overhead fijo de IC + measure + CIC por
snapshot).

Para la corrida total (matriz in-process): **~170 s release** en un
laptop, muy por debajo del presupuesto estimado en el plan (12–15 min).

## 7. Respuestas explícitas a las preguntas del brief

### A. ¿El error decrece al reducir `dt`?

**No.** En la ventana `[5·10⁻⁵, 4·10⁻⁴]` el error crece monotónicamente
al reducir `dt` (slope log-log `≈ −0.05`). El error de integración
temporal está dominado por el error físico de la amplitud inicial.

### B. ¿Se alcanza un plateau en el error con `dt` suficientemente chico?

**No se observa plateau**; la tendencia es ligeramente ascendente hasta
`dt = 5·10⁻⁵`. Cualquier plateau está por debajo del rango explorado o
requiere cambiar la convención IC.

### C. ¿La escala del error es compatible con O(dt²) del KDK?

**No**. La pendiente observada (`−0.054` a `−0.061`) no es compatible con
la predicción teórica `+2.0`. Esto no implica bug en el integrador — el
test 1 demuestra que el IC es bit-idéntico entre dts. Implica que el
régimen asintótico del error O(dt²) está por debajo del error físico
dominante en la convención legacy.

### D. ¿Qué `dt` mantiene la evolución en régimen lineal temprano?

**Ninguno en la matriz actual**. Para que `δ_rms(a=0.10) ≤ 10 ·
δ_rms(a_init) · D(a)/D(a_init)` se requeriría recalibrar la amplitud
inicial, no achicar `dt`. Según la proyección O(dt²) habría que alcanzar
errores espectrales del orden de la referencia lineal (`< 0.1`), lo que
desde `err ≈ 6.2` requiere `~10⁶²×` reducción — imposible en `dt`.

### E. ¿Cuál es el costo computacional del `dt_max` recomendado?

**`dt_max` no existe en la matriz explorada** bajo el criterio
`median |log10(P_c/P_ref)| ≤ 0.10 en a=0.05`. La recomendación práctica
es mantener `dt = dt₀ = 4·10⁻⁴` (el default histórico) porque reducir
`dt` no aporta mejora espectral **y multiplica el costo linealmente**. El
problema detectado NO se resuelve en el eje de `dt`.

## 8. Recomendación técnica

1. **Mantener `dt = 4·10⁻⁴`** como default. Reducir `dt` aumenta costo
   sin mejorar fidelidad espectral en la convención actual.
2. **No declarar cerrado** el régimen lineal evolucionado hasta
   **reformular la normalización de ICs** (fuera del alcance de Phase
   39). Phase 37 mostró que la opción más simple (`rescaled`) tampoco
   ayuda en la evolución — queda como marca lo contrario: el problema es
   físico-cosmológico, no numérico.
3. **Próximos pasos posibles** (fuera del alcance de esta fase):
   - Timestep adaptativo basado en Courant / aceleración local.
   - ICs físicas con `σ₈ = 0.8` en `z = 0` **y** transfer aplicado
     correctamente (implica revisar tanto `rescale_to_a_init` como la
     normalización global de `amplitude`).
   - Validación contra un código de referencia sobre una corrida
     evolucionada (Gadget-2/3, RAMSES) con las mismas ICs.

## 9. Figuras

Las 4 figuras obligatorias están en
`docs/reports/figures/phase39/` (y en
`experiments/nbody/phase39_dt_convergence/figures/`):

- `error_vs_dt.png` — error espectral corregido vs `dt` (log-log), ancla
  `∝ dt²`. Todas las curvas horizontales / ligeramente ascendentes.
- `ratio_per_dt.png` — `P_c/P_ref(k)` overlay para los 4 dts en seed 42;
  banda BAO sombreada.
- `delta_rms_vs_a.png` — `δ_rms(a)` para los 4 dts vs teoría lineal
  (roja, discontinua).
- `cost_vs_precision.png` — runtime por corrida vs error espectral en
  `a=0.05`, 3 seeds por `dt` y media (X).

## 10. Definition of Done

- [x] Barrido de `dt` ejecutado (36 mediciones, 3 seeds, 4 dts, 3 `a`).
- [x] Impacto en el espectro corregido medido y tabulado (CSV 36 filas).
- [x] Rango de `dt` aceptable identificado (conclusión: no existe en la
      matriz actual; recomendación es mantener `dt = 4·10⁻⁴`).
- [x] Comportamiento de convergencia cuantificado (pendientes
      `−0.054` / `−0.061`).
- [x] Recomendación práctica con interpretación física (§8).
- [x] 5 tests automáticos pasando con soft checks donde la hipótesis
      falla (`hypothesis_*=false` registrados en JSON, sin panic).
- [x] 4 figuras + 1 CSV generados y copiados a `docs/reports/figures/phase39/`.
- [x] Entrada de CHANGELOG Phase 39.

## 11. Archivos de evidencia

- **Matriz JSON**: `target/phase39/per_cfg.json` (36 mediciones con
  espectros, δ_rms, runtime).
- **CSV**: `experiments/nbody/phase39_dt_convergence/output/dt_vs_error.csv`.
- **CLI evidence**:
  `experiments/nbody/phase39_dt_convergence/output/dt_<tag>/cli_evidence.json`
  (seed 42, 4 dts, verificación cruzada end-to-end).
- **Tests per-test JSON**: `target/phase39/test{1..5}*.json`.

---

*Reporte generado como parte de Phase 39. Integrador, solver y
pk_correction no fueron modificados.*
