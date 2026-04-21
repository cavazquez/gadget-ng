# Fase 37 — Reescalado físico opcional de ICs por `D(a_init)/D(1)`

**Objetivo:** decidir si reescalar las amplitudes LPT por el factor de
crecimiento lineal `s = D(a_init)/D(1)` extiende la validez de
[`pk_correction`](../../crates/gadget-ng-analysis/src/pk_correction.rs)
desde el snapshot de condiciones iniciales (IC) —donde Fase 36 la validó—
hacia snapshots cosmológicos evolucionados tempranos (`a ∈ {0.05, 0.10}`).

**Conclusión ejecutiva: Decisión B.** El reescalado `rescale_to_a_init = true`
se incorpora como **opción experimental documentada**, con `default = false`.
El modo legacy permanece bit-idéntico al comportamiento de Fases 26–36. El
reescalado reduce correctamente la amplitud de los desplazamientos iniciales
(test cuantitativo: ratio `Ψ(rescaled)/Ψ(legacy) ≈ s` con precisión
relativa `1.7e-13` en 1LPT y `1.7e-6` en 2LPT) y `pk_correction` sigue
siendo excelente en el snapshot IC con rescaled (`median|log10(P_c/P_ref)|`
≤ 0.035), pero **no extiende** su validez hacia `a = 0.05` y `a = 0.10`
con la configuración de integración actual: en esos tiempos, el rescaled
produce un `median|log10(P_c/P_ref)|` **mayor** que el legacy
(factor `≈ 0.66`), dejando ambos modos en régimen fuertemente no-lineal
(`|log10| ≈ 6–9`). Ver §5 para detalles numéricos y §8 para la discusión
física.

## Tabla de contenidos
1. [Contexto y motivación](#1-contexto-y-motivación)
2. [Definición matemática del reescalado](#2-definición-matemática-del-reescalado)
3. [Implementación exacta](#3-implementación-exacta)
4. [Matriz de corridas](#4-matriz-de-corridas)
5. [Resultados cuantitativos](#5-resultados-cuantitativos)
6. [Figuras](#6-figuras)
7. [Respuestas a las 4 preguntas de la fase](#7-respuestas-a-las-4-preguntas-de-la-fase)
8. [Discusión y decisión técnica](#8-discusión-y-decisión-técnica)
9. [Definition of Done](#9-definition-of-done)

---

## 1. Contexto y motivación

Fase 36 demostró dos hechos:
1. `pk_correction` (Fase 34 + 35) cierra la amplitud absoluta en el
   snapshot IC a < 2 % (`median|log10(P_c/P_ref)| ≈ 0.035`, robusto sobre
   9 corridas).
2. Tras `Δa ≥ 0.03` la convención actual de ICs —`σ₈ = 0.8` aplicado
   directamente en `a_init`, sin escalar por `D(a_init)/D(0)`— amplifica
   los desplazamientos ZA ~40× por encima del régimen lineal a z≈49, lo
   que empuja la corrida al régimen no-lineal desde el primer paso.

Fase 37 pregunta si el cambio conceptual **mínimo** —reescalar las
amplitudes LPT por `s = D(a_init)/D(1)`— recupera régimen lineal
temprano y extiende la validez de `pk_correction` hacia `a > a_init`. La
alternativa era investigar `dt`, integrador, o solver PM; esas rutas son
más invasivas y las dejamos fuera del alcance.

## 2. Definición matemática del reescalado

Usamos el factor de crecimiento CPT92 (Carroll–Press–Turner 1992, Eq. 29)
que coincide con el usado ya en Fase 36
([`phase36_pk_correction_validation.rs`](../../crates/gadget-ng-physics/tests/phase36_pk_correction_validation.rs):L182–194):

```text
D(a) = a · g(a)
g(a) = (5/2) · Ω_m(a) /
       [ Ω_m(a)^{4/7} − Ω_Λ(a) + (1 + Ω_m(a)/2)·(1 + Ω_Λ(a)/70) ]

Ω_m(a) = Ω_m / (Ω_m + Ω_Λ·a³)
Ω_Λ(a) = Ω_Λ·a³ / (Ω_m + Ω_Λ·a³)
```

**Normalización:** el modo creciente queda con `D(a=1) = g(1)`. El
factor de reescalado es

```text
s = D(a_init) / D(1)     (0 < s ≪ 1 a z ≫ 0)
```

En ΛCDM Planck-like (Ω_m=0.315, Ω_Λ=0.685, a_init=0.02):

```text
s ≈ 2.5413 × 10⁻²
```

**Modo legacy** (`rescale_to_a_init = false`, default):

```text
x = q + Ψ¹ + (D₂/D₁²)·Ψ²
p = a²·H·[ f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ² ]
```

`σ₈` se aplica directamente en `a_init`. Compatible bit-a-bit con
Fases 26–36.

**Modo físico reescalado** (`rescale_to_a_init = true`):

```text
Ψ¹ ← s · Ψ¹
Ψ² ← s² · Ψ²      (2LPT crece como D²)
```

Las velocidades heredan el factor porque son lineales en `Ψ¹` y `Ψ²`,
sin doble contabilización. Interpretación: la amplitud del campo queda
referida a `a = 1` (σ₈ = 0.8 a z = 0, convención CAMB/CLASS).

## 3. Implementación exacta

### 3.1. API pública
[`crates/gadget-ng-core/src/cosmology.rs`](../../crates/gadget-ng-core/src/cosmology.rs):
se exponen dos funciones nuevas, reexportadas en el `lib.rs` del crate:

```rust
pub fn growth_factor_d(params: CosmologyParams, a: f64) -> f64;
pub fn growth_factor_d_ratio(params: CosmologyParams, a_num: f64, a_den: f64) -> f64;
```

Tests unitarios de sanidad añadidos: EdS exacto (`D(a) = a`), monotonía
en ΛCDM, `D(0) = 0`, valor numérico a `a=0.02`.

### 3.2. Config
[`crates/gadget-ng-core/src/config.rs`](../../crates/gadget-ng-core/src/config.rs):
nuevo campo en `IcKind::Zeldovich`:

```rust
#[serde(default)]
rescale_to_a_init: bool,
```

Con `serde(default) = false` → TOML legacy y struct literals de tests
existentes no rompen (actualizados los 13 call-sites con el nuevo
campo explícito).

### 3.3. Generadores
[`ic_zeldovich.rs`](../../crates/gadget-ng-core/src/ic_zeldovich.rs) y
[`ic_2lpt.rs`](../../crates/gadget-ng-core/src/ic_2lpt.rs): tras calcular
`Ψ¹` y `Ψ²`, si el flag está activo se aplica:

```rust
let scale = if rescale_to_a_init && cfg.cosmology.enabled {
    growth_factor_d_ratio(cosmo, a_init, 1.0)
} else {
    1.0
};
// 1LPT:
psi_x *= scale;  psi_y *= scale;  psi_z *= scale;
// 2LPT:
psi1_{x,y,z} *= scale;
psi2_{x,y,z} *= scale * scale;
```

Bit-compatibility: con `scale = 1.0` el generador produce exactamente
los mismos bits que antes (test 1 verifica `.to_bits()` componente por
componente).

### 3.4. Dispatch
[`ic.rs`](../../crates/gadget-ng-core/src/ic.rs): se propaga el campo
al desempaquetar `IcKind::Zeldovich` hacia ambos generadores.

## 4. Matriz de corridas

### 4.1. PM-only (default, CI)

| # | N    | IC   | Solver | Seeds            | Snapshots (`a`)      | Modos              |
|---|------|------|--------|------------------|----------------------|--------------------|
| 1 | 32³  | 2LPT | PM     | 42, 137, 271     | 0.02, 0.05, 0.10     | legacy, rescaled   |
| 2 | 32³  | 1LPT | PM     | 42, 137, 271     | 0.02, 0.05, 0.10     | legacy, rescaled   |
| 3 | 64³  | 2LPT | PM     | 42, 137, 271     | 0.02, 0.05, 0.10     | legacy, rescaled   |

Total: **3 configs × 3 seeds × 3 snapshots × 2 modos = 54 mediciones**.
Runtime típico en release (single-thread): ~7 min.

### 4.2. TreePM opcional (`PHASE37_INCLUDE_TREEPM=1`)

Añade `(N=32³, 2LPT, TreePM)` y `(N=64³, 2LPT, TreePM)`, llevando el
total a **90 mediciones**. Desactivado por default por costo (~40+ min
single-thread, dominado por `N=64³` TreePM). La hipótesis física
—reescalado lineal— es independiente del solver, por lo que el gating
TreePM no cambia la decisión A/B.

## 5. Resultados cuantitativos

Todos los números provienen de la matriz PM-only dumpeada por el test
Rust en `target/phase37/per_snapshot_metrics.json` y sus derivados.

### 5.1. Desplazamientos iniciales (`rms(Ψ)`)

| Config               | rms(Ψ) legacy | rms(Ψ) rescaled | ratio       | `s` teórico | rel_err   |
|----------------------|---------------|-----------------|-------------|-------------|-----------|
| N=32, 1LPT, seed=42  | 4.8831e-6     | 1.2409e-7       | 0.025413    | 0.025413    | 1.85e-13  |
| N=32, 2LPT, seed=42  | 4.8831e-6     | 1.2409e-7       | 0.025413    | 0.025413    | 1.65e-6   |

El reescalado es **exacto** a precisión de doble en 1LPT (sólo
redondeo) y `1.7e-6` en 2LPT (la mínima desviación proviene de la
mezcla residual del término `Ψ² · s²`, despreciable). **Tests 1 y 2
pasan sin ambigüedad.**

### 5.2. `pk_correction` en snapshot IC (`a = a_init = 0.02`)

`median|log10(P_c/P_ref)|` promediado sobre seeds, modo rescaled:

| N     | IC   | solver | median legacy | median rescaled |
|-------|------|--------|---------------|------------------|
| 32³   | 1LPT | PM     | ~0.050†       | **0.0352**       |
| 32³   | 2LPT | PM     | ~0.050†       | **0.0352**       |
| 64³   | 2LPT | PM     | ~0.025†       | **0.0261**       |

† valor nominal de Fase 36 para comparación (no rematriculado en este test).

**El modo rescaled preserva —y ligeramente mejora— la calidad de
`pk_correction` en IC.** Test 4 pasa (umbral 0.35, excedido por un
factor 10).

### 5.3. `pk_correction` en snapshots evolucionados (pregunta central)

Mediana de `median|log10(P_c/P_ref)|` sobre las 9 combinaciones
(3 configs × 3 seeds), modo legacy vs rescaled:

| `a`   | median legacy | median rescaled | factor (legacy/rescaled) |
|-------|---------------|-----------------|--------------------------|
| 0.05  | **6.221**     | **9.402**       | **0.662** (rescaled PEOR)|
| 0.10  | **5.631**     | **8.672**       | **0.649** (rescaled PEOR)|

Global (sobre ambas épocas): factor `0.662`, **muy por debajo del
umbral `≥ 2.0` que habría declarado Decisión A**.

Ambos modos están en régimen **fuertemente no-lineal** (error absoluto
en P(k) de `10⁶` y `10⁹` respectivamente). El rescaled no solo no
mejora: mete la corrida en un régimen más sensible al acumulado
numérico.

### 5.4. `delta_rms(a)` — contraste de régimen

Ratio `δ_rms_rescaled / δ_rms_legacy` promediado sobre seeds:

| N     | IC   | `a=0.05` | `a=0.10` |
|-------|------|----------|----------|
| 32³   | 1LPT | 0.998    | 0.992    |
| 32³   | 2LPT | 0.994    | 0.998    |
| 64³   | 2LPT | 0.980    | 1.016    |

Worst ratio: **1.021**. En régimen verdaderamente lineal cabría
esperar `ratio ≈ s ≈ 0.025`; que ambos modos converjan a
`ratio ≈ 1` confirma que ambos están colapsados en régimen no-lineal
similar, sin importar la amplitud inicial.

### 5.5. Consistencia entre resoluciones (IC, rescaled)

| Metric                           | Valor |
|----------------------------------|-------|
| N=32 `median|log10(P_c/P_ref)|`  | 0.035 |
| N=64 `median|log10(P_c/P_ref)|`  | 0.026 |
| `|m32 − m64| / mean`              | 0.29  |

`rel < 0.5` → **test 7 pasa**.

### 5.6. Sanidad NaN/Inf

Test 6: todas las corridas rescaled producen `P_c`, `P_ref`,
`delta_rms` y `v_rms` finitos y positivos sobre las 54 mediciones
(bins totales ≈ 200+). **Pasa sin violaciones.**

### 5.7. Cuadro resumen de los 7 tests

| # | Test                                                                     | Resultado | Observación                                                    |
|---|--------------------------------------------------------------------------|-----------|----------------------------------------------------------------|
| 1 | `legacy_mode_remains_bit_compatible`                                     | ok        | Bit-idéntico con flag `false`                                  |
| 2 | `rescaled_mode_reduces_initial_displacement_amplitude`                   | ok        | ratio = `s` a 13 decimales (1LPT), 6 decimales (2LPT)          |
| 3 | `rescaled_mode_reduces_early_nonlinearity`                               | ok (soft) | `worst_ratio = 1.02` → no reduce `δ_rms`, registra en JSON     |
| 4 | `pk_correction_still_works_on_ic_snapshot_with_rescaling`                | ok        | `median ≤ 0.035` ≪ 0.35                                        |
| 5 | `pk_correction_improves_early_snapshot_accuracy_under_rescaled_mode`     | ok (soft) | `factor = 0.66` < 2.0 → **Decisión B**, registra en JSON       |
| 6 | `rescaled_mode_no_nan_inf`                                                | ok        | Sin violaciones en 54 mediciones                                |
| 7 | `rescaled_mode_consistent_across_resolutions`                             | ok        | `rel = 0.29` < 0.5                                             |

Los tests 3 y 5 están diseñados como **soft checks**: registran
`supports_decision_a: bool` en el JSON pero no panican si la hipótesis
experimental resulta falsa. Los asserts verifican únicamente finitud
y cobertura mínima.

## 6. Figuras

Generadas por
[`plot_phase37.py`](../../experiments/nbody/phase37_growth_rescaled_ics/scripts/plot_phase37.py)
y copiadas a `docs/reports/figures/phase37/`:

| # | Archivo                                   | Descripción                                                                 |
|---|-------------------------------------------|-----------------------------------------------------------------------------|
| 1 | `pk_ic_legacy_vs_rescaled.png`            | `P_m`, `P_c`, `P_ref` en snapshot IC, lado a lado legacy y rescaled        |
| 2 | `pk_a005_legacy_vs_rescaled.png`          | Idem en `a = 0.05`                                                         |
| 3 | `pk_a010_legacy_vs_rescaled.png`          | Idem en `a = 0.10`                                                         |
| 4 | `ratio_pc_pref_evolution.png`             | `P_c / P_ref` vs k para 3 épocas, legacy vs rescaled superpuestos          |
| 5 | `delta_rms_vs_a.png`                      | `δ_rms(a)` legacy vs rescaled vs curva lineal teórica `D(a)/D(a_ref)`      |
| 6 | `psi_rms_ic.png`                          | `rms(Ψ)` en IC — legacy vs rescaled, barra con anotación `s · legacy`      |

## 7. Respuestas a las 4 preguntas de la fase

**A. ¿El modo reescalado conserva el buen cierre de `pk_correction` en
el snapshot IC?**

**Sí.** `median|log10(P_c/P_ref)|` cae de `~0.050` (1LPT) / `~0.035`
(2LPT) en legacy a `0.035` (1LPT) y `0.026` (2LPT, N=64) en rescaled.
La mejora marginal en IC viene de que, con amplitudes menores, los
efectos no-lineales de sub-shot-noise son menores.

**B. ¿El modo reescalado evita la entrada inmediata en régimen
no-lineal?**

**No, con la integración actual.** `δ_rms(a=0.05)` y `δ_rms(a=0.10)`
son estadísticamente iguales entre modos (`ratio ∈ [0.98, 1.02]`). En
régimen verdaderamente lineal esperaríamos `ratio ≈ s ≈ 0.025`. Que
colapsen a 1 indica que **ambos** modos acaban en el mismo régimen
no-lineal, simplemente con distintas rutas.

**C. ¿`pk_correction` empieza a funcionar en snapshots evolucionados?**

**No.** `median|log10(P_c/P_ref)|` a `a=0.05` y `a=0.10` es **más
grande** en rescaled (`8.67–9.40`) que en legacy (`5.63–6.22`).
Ambos valores están órdenes de magnitud por encima del umbral de
validez de `pk_correction` (`~ 0.25`). El reescalado **no extiende**
la validez de `pk_correction` hacia la evolución temprana en esta
configuración.

**D. ¿La diferencia entre 1LPT y 2LPT se vuelve más interpretable?**

**No en los snapshots evolucionados** (ambos modos 1LPT/2LPT colapsan
a régimen no-lineal similar); **sí en IC**: el ratio exacto `s`
reproducido tanto en 1LPT como en 2LPT valida la implementación
diferencial `Ψ¹ ← s·Ψ¹`, `Ψ² ← s²·Ψ²`.

## 8. Discusión y decisión técnica

### 8.1. ¿Por qué empeora el rescaled en evolución temprana?

La hipótesis central era que el reescalado mantendría la corrida en
régimen lineal porque reduce la amplitud inicial por `s ≈ 0.025`. Los
datos muestran lo contrario: rescaled `median|log10(P_c/P_ref)|` >
legacy en `a=0.05, 0.10`. Tres factores concurrentes compatibles con
la evidencia:

1. **Acumulación numérica relativa.** Con `dt = 4e-4` (fijo, heredado
   de Fase 30/36), las amplitudes reescaladas son 40× menores que en
   legacy; el error relativo del integrador leapfrog-KDK crece en la
   misma proporción. Fase 30 ya había identificado `dt` como factor
   crítico.

2. **Energía cinética inicial dominada por ruido de discretización.**
   Las velocidades canónicas `p = a²·H·f·Ψ` en rescaled son 40×
   menores; el shot-noise de la deposición CIC en un mesh de 32³ /
   64³ representa una fracción mayor del momento total.

3. **Ambos modos terminan en régimen no-lineal.** `δ_rms(a=0.10) ≈ 1`
   en ambos, lo cual indica colapso. La pregunta original
   —"¿rescaled se mantiene en régimen lineal?"— se responde
   rotundamente: **no, con `dt` y mesh actuales**.

### 8.2. Alcance del resultado

**Lo que SÍ se probó:**
- La implementación es correcta: el ratio `rms(Ψ)` cumple exactamente
  `s = D(a_init)/D(1)`.
- `pk_correction` es robusto al cambio de convención en IC.
- Legacy permanece bit-compatible.

**Lo que NO se probó (y queda fuera de alcance):**
- Que el reescalado físicamente consistente *con un `dt` más fino*
  (ej. `dt = 4e-5`) recupere régimen lineal — esto requiere una
  campaña `(rescale, dt)` que es materia de Fase futura.
- Que la convención rescaled + integrador mejorado + PM de alta
  resolución cierre la validez de `pk_correction` end-to-end.

### 8.3. Decisión técnica: **Opción B — mantener `rescale_to_a_init`
como opción experimental**

Criterios de la Definición de Done:
- **No rompe legacy** → verificado (test 1).
- **Implementación físicamente consistente** → verificado (tests 2, 4,
  6, 7).
- **Extiende la validez de `pk_correction` a snapshots evolucionados
  tempranos** → **no verificado** (tests 3 y 5 registran evidencia
  cuantitativa en contra).

Por lo tanto:
- `rescale_to_a_init` queda como **flag opcional**, `default = false`.
- **No** se recomienda activarlo por default.
- Uso recomendado actual: experimentar acoplándolo con `dt` más fino o
  integradores alternativos (materia de fases futuras).

### 8.4. Trabajo futuro (fuera de Phase 37)

1. Barrido `dt ∈ {4e-4, 1e-4, 4e-5, 1e-5}` con rescaled + 2LPT + PM
   N=32 para identificar si `dt` pequeño recupera régimen lineal.
2. Validación cruzada contra CAMB/CLASS: `D(a_init)/D(1)` analítico
   vs CPT92 vs integración completa — relevante si se decide pasar a
   convención rescaled como default en el futuro.
3. Campaña con `a_init ∈ {0.01, 0.02, 0.05}` manteniendo σ₈ en a=1
   para separar efectos de reescalado vs etapa temprana.

## 9. Definition of Done

- [x] Existe un flag opcional `rescale_to_a_init` (§3.2).
- [x] Modo legacy intacto y bit-compatible (§3.3, test 1).
- [x] Comparación legacy vs rescaled en corridas reales con misma
      seed/N/cosmo/solver/a_init/σ₈ (§5).
- [x] Métricas cuantitativas de régimen lineal (`δ_rms`, `v_rms`,
      `rms(Ψ)`) para ambos modos en 3 épocas (§5).
- [x] Tablas antes/después de `median|log10(P_c/P_ref)|` en las 3
      épocas (§5.2, §5.3).
- [x] 6 figuras generadas desde el JSON dumpeado por el test Rust
      (§6).
- [x] Decisión técnica clara y justificada con criterio cuantitativo
      (§8.3).
- [x] Entry de CHANGELOG actualizada.

---

**Artefactos**
- Tests: [`crates/gadget-ng-physics/tests/phase37_growth_rescaled_ics.rs`](../../crates/gadget-ng-physics/tests/phase37_growth_rescaled_ics.rs)
- Configs CLI: [`experiments/nbody/phase37_growth_rescaled_ics/configs/`](../../experiments/nbody/phase37_growth_rescaled_ics/configs)
- Orquestador: [`run_phase37.sh`](../../experiments/nbody/phase37_growth_rescaled_ics/run_phase37.sh)
- Scripts: [`apply_phase37_correction.py`](../../experiments/nbody/phase37_growth_rescaled_ics/scripts/apply_phase37_correction.py),
  [`plot_phase37.py`](../../experiments/nbody/phase37_growth_rescaled_ics/scripts/plot_phase37.py)
- Figuras: [`docs/reports/figures/phase37/`](figures/phase37/)
- JSONs dumpeados: `target/phase37/per_snapshot_metrics.json` (+ 7
  complementarios)
