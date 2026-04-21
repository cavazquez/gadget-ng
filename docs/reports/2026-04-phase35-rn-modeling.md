# Phase 35 — Modelado de `R(N)` para corrección absoluta de `P(k)`

**Fecha:** 2026-04
**Estado:** cerrado
**Predecesores:** Phase 30–34
**Sucesores sugeridos:** crecimiento D(a) con CAMB/CLASS, barrido de kernels TSC/PCS

---

## 1. Contexto

Phases 30–33 validaron la **forma espectral** y el **crecimiento relativo** de `P(k)` en `gadget-ng`, pero identificaron un *offset* absoluto sistemático de amplitud frente al `P(k)` continuo teórico.

Phase 34 descompuso ese *offset* en dos factores independientes:

```text
P_measured(k) = A_grid(N) · R(N) · P_cont(k)
```

* `A_grid(N) = 2·V²/N⁹` — **cerrado analíticamente** (convención DFT + binning).
* `R(N)` — factor de muestreo partículas + CIC + deconvolución. **Determinista**, pero **resolución-dependiente**, por lo que Phase 34 no lo corrigió en el core y lo dejó caracterizar.

Phase 35 cierra esa caracterización: mide `R(N)` con baja varianza entre seeds, lo modela, lo tabula y lo expone como corrección de postproceso en un módulo nuevo `gadget_ng_analysis::pk_correction`, **sin tocar el estimador interno ni el solver**.

## 2. Definición formal

Dado el estimador interno `power_spectrum::power_spectrum` (CIC + FFT 3D) y un `P(k)` continuo de referencia (Eisenstein–Hu no-wiggle con `σ₈ = 0.8`, `n_s = 0.965`, `box_mpc_h = 100`):

$$R(N, k) \;\equiv\; \frac{P_\text{m}(k)}{A_\text{grid}(N)\,P_\text{cont}(k)}, \qquad A_\text{grid}(N)=\frac{2V^{2}}{N^{9}}$$

con `V = box_size_internal³`. Luego:

$$R_\text{mean}(N) \;=\; \langle R(N,k)\rangle_{k\le k_\text{Nyq}/2}, \qquad \text{CV}_R(N) \;=\; \sigma_R/\mu_R.$$

Las seeds se promedian *después* de calcular `R_mean` por realización.

## 3. Campaña experimental

| Variable | Valor |
|---|---|
| Resoluciones | `N ∈ {8, 16, 32, 64}` |
| Seeds por N | `[42, 137, 271, 314]` (4 seeds) |
| Caja interna | `1.0` |
| `box_mpc_h` | `100` |
| Kernel CIC | oficial (`power_spectrum.rs`, deconvolución `sinc²`) |
| Kernel TSC | local, 1 punto (`N=32, seed=42`) |
| Lattice ZA | puro (sin jittering) |
| `σ₈` | `0.8` |
| Transferencia | Eisenstein–Hu no-wiggle |

Volumen total: 16 IC + medición CIC + 1 verificación TSC. Ejecuta en <10 s en release sobre un laptop moderno.

## 4. Medidas

### 4.1 `R_mean(N)` y CV entre seeds

| N | `R_mean(N)` | CV entre seeds | Umbral test |
|---|---|---|---|
| 8 | 0.4154 | 0.229 | 0.30 |
| 16 | 0.1396 | 0.100 | 0.15 |
| 32 | 0.0338 | 0.036 | 0.10 |
| 64 | 0.00883 | 0.014 | 0.10 |

La CV baja monotónicamente con N: a N=64 el factor `R` es reproducible al 1.4 % entre seeds. A N=8 hay sólo ~2 bins con `k ≤ k_Nyq/2`, por lo que la dispersión es dominada por shot-noise intrínseco.

### 4.2 `R(N, k)` — planicidad en k

`R(N, k)` promediado sobre las 4 seeds muestra **CV_k < 0.25** para todo N en la ventana `k ≤ k_Nyq/2` (ver `rn_of_k.png`). No se observa tendencia sistemática en k — la dispersión es mayoritariamente shot-noise de bines con pocos modos, lo que justifica tratar `R` como **escalar por N**.

### 4.3 Determinismo

El test `r_n_stable_across_seeds` exige y cumple:
* N=8: CV < 0.30
* N=16: CV < 0.15
* N ≥ 32: CV < 0.10

## 5. Modelado

### 5.1 Modelo A — ley de potencia pura

$$R(N) = C \cdot N^{-\alpha}$$

Ajuste OLS log-log:

| Parámetro | Valor |
|---|---|
| `C` | **22.108** |
| `α` | **1.8714** |
| `R²` | **0.99715** |
| RMS(log₁₀ residuos) | `0.034` |
| AIC | `-23.13` |

### 5.2 Modelo B — ley de potencia + offset

$$R(N) = C \cdot N^{-\alpha} + R_\infty$$

Ajuste con `scipy.optimize.curve_fit`:

| Parámetro | Valor |
|---|---|
| `C` | 10.07 |
| `α` | 1.518 |
| `R_∞` | `−0.0128` (negativo, sin sentido físico) |
| RMS(log₁₀ residuos) | 0.110 |
| AIC | `-11.68` |

### 5.3 Selección

ΔAIC = −23.13 − (−11.68) = **−11.45** a favor de Modelo A. Criterio AIC + navaja de Occam: **gana Modelo A**. El fit de Modelo B *no* encuentra un offset asintótico positivo, lo que sugiere que en el rango medido `R(N) → 0` monotónicamente.

La pendiente `α ≈ 1.87` es compatible con un scaling dominado por el número efectivo de modos útiles por bin (∝ N²) modulado por la ventana CIC — no corresponde exactamente a un exponente entero, lo que apunta a un comportamiento mixto que no encuadra en ninguna predicción analítica trivial.

## 6. Error residual tras la corrección

Aplicando `P_corr = P_m / (A_grid(N)·R_A(N))` sobre la matriz completa:

| N | Mediana `|log₁₀(P_m/P_cont)|` | Mediana `|log₁₀(P_corr/P_cont)|` | Reducción |
|---|---|---|---|
| 8 | 8.20 | 0.086 | ×95 |
| 16 | 11.42 | 0.039 | ×293 |
| 32 | 14.71 | 0.035 | ×420 |
| 64 | 18.01 | 0.035 | ×514 |
| **Global** | **17.93** | **0.037** | **×485** |

La corrección reduce el error de amplitud **casi tres órdenes de magnitud** y lo deja en < 10 % en unidades log₁₀ (ratio ~1.09). El test `r_n_model_reduces_amplitude_error` valida esto con umbral `mediana < 0.15`.

## 7. CIC vs TSC — un punto verificativo

A `N = 32, seed = 42`:

| Kernel | `R_mean` |
|---|---|
| CIC (oficial, deconv `sinc²`) | **0.03491** |
| TSC (local, deconv `sinc⁶`) | **0.03493** |
| Ratio max/min | **1.001** |

**Observación**: tras aplicar la deconvolución adecuada a cada kernel, ambos dan `R_mean` prácticamente idénticos a este N. Esto **no invalida** la dependencia en kernel en general: simplemente indica que para ZA lattice puro las correcciones convergen muy bien en `k ≤ k_Nyq/2`. Documentar ratio > 1.1 sería arbitrario a N=32; se deja abierto a una fase futura dedicada al barrido de kernels (TSC, PCS, interlacing).

## 8. API de corrección — `pk_correction`

Crate: [`gadget-ng-analysis`](../../crates/gadget-ng-analysis/src/pk_correction.rs).

```rust
use gadget_ng_analysis::pk_correction::{a_grid, correct_pk, RnModel};

let model = RnModel::phase35_default();   // C=22.108, α=1.8714
let pk_phys = correct_pk(
    &pk_measured,
    /* box_size_internal */ 1.0,
    /* n                  */ 64,
    /* box_mpc_h          */ Some(100.0),
    &model,
);
```

* `a_grid(box_size, n)` — factor analítico `2V²/N⁹`.
* `RnModel::phase35_default()` — valores congelados (Modelo A + tabla).
* `RnModel::evaluate(n)` — usa la tabla si `N` está en ella, si no, el fit.
* `RnModel::evaluate_interpolated(n)` — interpolación log-log entre puntos de la tabla.
* `RnModel::from_table(table)` — reajusta OLS desde datos propios.

La función `correct_pk` acepta un `Option<box_mpc_h>`; si se provee, reescala la amplitud a `(Mpc/h)³`.

### Validación unitaria

6 tests en `gadget_ng_analysis::pk_correction::tests` validan:

* tabla y modelo coinciden al 20 % por punto,
* `evaluate` prioriza tabla sobre fit,
* `evaluate_interpolated` coincide con el modelo dentro del 5 % a N=48,
* `from_table` recupera `(C, α)` exactos de datos sintéticos,
* `correct_pk` escala linealmente en `P` y aplica el factor de unidades `Mpc/h`.

## 9. Rango de validez

La corrección es aplicable en:

* **Resolución**: `N ∈ [8, 64]` por extrapolación conservadora hasta `N = 128` con el fit. Fuera de ese rango se recomienda **re-fit** con nuevos puntos.
* **Condiciones iniciales**: ZA (1LPT) sobre lattice regular. 2LPT, jittered lattice o partículas estructuradas (e.g., glass) requieren nueva calibración.
* **Kernel**: CIC con la deconvolución actual del estimador interno. TSC casualmente coincide a N=32 pero no hay garantía general.
* **Espectro de referencia**: Eisenstein–Hu no-wiggle. Para `P_cont` con BAO se espera que `R(N, k)` siga plano dentro del mismo CV, pero no se validó.
* **Pipeline**: ZA → CIC → deconvolución → binning, **sin evolución dinámica**. Tras `N_steps > 0` la distribución de partículas se aleja del lattice y `R(N)` puede cambiar.

## 10. Limitaciones + referencias

* No se estudia `R(N, k)` fuera de `k ≤ k_Nyq/2`. Por encima de Nyquist/2 aparece *aliasing* cuya corrección requiere *interlacing* u otras técnicas.
* Con sólo 4 seeds por N la CV a N=8 queda en 0.23; no es un problema para el fit (Modelo A pondera log N) pero pudiera apretarse con más seeds.
* No se compara contra CAMB/CLASS: Phase 35 usa EH como referencia canónica, consistente con Phases 30–34.
* El *offset* negativo de Modelo B (`R_∞ = −0.013`) confirma que una saturación *superior* a cero no es soportada por los datos.

**Referencias internas:** [Phase 33](./2026-04-phase33-offset-analytic-derivation.md) · [Phase 34](./2026-04-phase34-discrete-normalization-closure.md).

---

## Tabla final (congelada)

```text
R_mean(N):
    N=8   →  0.4154
    N=16  →  0.1396
    N=32  →  0.03375
    N=64  →  0.008834

Modelo A (ganador):
    R(N) = 22.108 · N^(-1.8714)
    R² = 0.9972, RMS(log₁₀) = 0.034
```

## Figuras

1. `rn_vs_N.png` — `R(N)` vs N log-log, ambos modelos superpuestos.
2. `rn_of_k.png` — `R(N, k)` vs k para los 4 N.
3. `fit_residuals.png` — residuos log-log de Modelos A y B.
4. `p_corrected_vs_theory.png` — mediana de `|log₁₀(P/P_cont)|` antes/después.
5. `tsc_vs_cic.png` — comparación de kernels a `N=32, seed=42`.

## Tests

* [`phase35_rn_modeling.rs`](../../crates/gadget-ng-physics/tests/phase35_rn_modeling.rs) (6 tests, todos pasan en `release`).
* Unit tests en [`pk_correction.rs`](../../crates/gadget-ng-analysis/src/pk_correction.rs) (6 tests, todos pasan).

## Entregables

* **Código**:
  * Nuevo módulo `crates/gadget-ng-analysis/src/pk_correction.rs`.
  * Re-export en `crates/gadget-ng-analysis/src/lib.rs`.
* **Tests**: `crates/gadget-ng-physics/tests/phase35_rn_modeling.rs`.
* **Experimento**: `experiments/nbody/phase35_rn_modeling/` (orquestador, scripts, output, figuras).
* **Figuras copiadas**: `docs/reports/figures/phase35/`.

---

**Cierre:** Phase 35 deja `R(N)` modelado con precisión suficiente para corregir la amplitud absoluta de `P(k)` en postproceso al < 10 %, sin tocar el estimador interno. Combinado con el `A_grid(N)` cerrado en Phase 34, la cadena completa de normalización discreta queda **cerrada, documentada y probada**.
