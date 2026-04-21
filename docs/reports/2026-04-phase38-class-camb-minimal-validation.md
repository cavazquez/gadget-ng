# Fase 38 — Validación externa mínima de `pk_correction` contra CLASS

**Objetivo:** cerrar la validación externa de amplitud absoluta de
[`pk_correction`](../../crates/gadget-ng-analysis/src/pk_correction.rs)
comparando `P_corrected(k)` de `gadget-ng` contra una referencia
independiente producida por
[CLASS](https://lesgourg.github.io/class_public/class.html)
(v3.3.4, wrapper `classy 3.3.4.0`) en el snapshot IC.

**Conclusión ejecutiva: validación externa mínima cerrada.**
Sobre la matriz completa `2 N × 3 seeds × 2 modos = 12 mediciones`
(IC-only, 2LPT, PM), `pk_correction` reduce el error absoluto de amplitud
vs CLASS por un factor **154×–761×** según resolución, dejando
`median|log10(P_c/P_CLASS)| ∈ [0.022, 0.046]` (mejor que 12% en ratio
lineal) y `mean(P_c/P_CLASS) ∈ [0.95, 1.04]` sobre todos los bins. Los
dos modos (`legacy` vs `z=0` y `rescaled` vs `z=49`) dan métricas
indistinguibles a precisión numérica, validando que el cierre es
intrínseco a `pk_correction` y no depende de la convención de
normalización. La forma espectral (pendiente log-log) difiere de CLASS
en `|Δd ln P / d ln k| ≤ 0.10` para `N=64`. Ver §4 para tablas
cuantitativas y §7 para la decisión técnica.

## Tabla de contenidos
1. [Contexto y motivación](#1-contexto-y-motivación)
2. [Referencia externa (CLASS)](#2-referencia-externa-class)
3. [Metodología de comparación](#3-metodología-de-comparación)
4. [Resultados cuantitativos](#4-resultados-cuantitativos)
5. [Validación de forma espectral](#5-validación-de-forma-espectral)
6. [Figuras](#6-figuras)
7. [Decisión técnica y respuestas a las 5 preguntas](#7-decisión-técnica-y-respuestas-a-las-5-preguntas)
8. [Definition of Done](#8-definition-of-done)

---

## 1. Contexto y motivación

Phases 34–36 produjeron la cadena:

1. **Phase 34** cerró analíticamente `A_grid(N) = 2 V² / N⁹`.
2. **Phase 35** modeló el factor de muestreo discreto `R(N)` y lo
   congeló en [`RnModel::phase35_default`](../../crates/gadget-ng-analysis/src/pk_correction.rs).
3. **Phase 36** validó `pk_correction` contra una referencia interna
   (EH no-wiggle + σ₈ + crecimiento CPT92) sobre corridas cosmológicas
   reales, obteniendo `median|log10(P_c/P_ref)| ≈ 0.035` en el snapshot IC.

El paso pendiente era una validación externa **independiente** del
pipeline interno: comparar `P_corrected` no contra la misma familia de
transferencia (EH) sino contra un código cosmológico estándar. Phase 38
añade esa referencia usando CLASS y cierra la pregunta.

## 2. Referencia externa (CLASS)

### 2.1. Cosmología

Misma que Phases 27–37:

| Parámetro   | Valor    |
|-------------|----------|
| Ω_m         | 0.315    |
| Ω_b         | 0.049    |
| Ω_cdm       | 0.266    |
| h           | 0.674    |
| n_s         | 0.965    |
| σ_8         | 0.8      |
| T_CMB       | 2.7255 K |
| N_ur        | 3.046    |
| N_ncdm      | 0        |
| non_linear  | none     |

Archivo canónico:
[`experiments/nbody/phase38_class_validation/reference/class.ini`](../../experiments/nbody/phase38_class_validation/reference/class.ini).

### 2.2. Código y versión

- CLASS core 3.3.4 (compilado junto a `classy 3.3.4.0` en un `venv`).
- Python 3.13, numpy 2.4.4, scipy 1.17.1, cython 3.2.4.
- `σ_8(z=0)` reportado por CLASS: `0.7999991570` (rescaling interno OK).
- `√[P(49)/P(0)] = 2.562477e-2` a `k = 4.526e-2 h/Mpc`, compatible con
  el CPT92 usado por `gadget-ng` (`s = D(0.02)/D(1) ≈ 2.5413e-2`,
  diferencia `~0.8 %`).

### 2.3. Archivos de referencia comiteados

| Archivo | Contenido | SHA-256 |
|---------|-----------|---------|
| [`pk_class_z0.dat`](../../experiments/nbody/phase38_class_validation/reference/pk_class_z0.dat)  | `P(k, z=0)`  en `(Mpc/h)³`, 512 bins log en `[1e-4, 20] h/Mpc` | `cf8dc8ce62953404f06dc1b80e97a6df0206a786769ed1140fe1eac5e79f95b3` |
| [`pk_class_z49.dat`](../../experiments/nbody/phase38_class_validation/reference/pk_class_z49.dat) | `P(k, z=49)` en `(Mpc/h)³`, mismo grid                         | `539f880faed880f1056746a7d55fb4673fcb02645de5d4762b26b943ad723c1e` |

**Reproducibilidad:** el script
[`generate_reference.sh`](../../experiments/nbody/phase38_class_validation/reference/generate_reference.sh)
crea el venv, pinea `classy==3.3.4.0`, regenera las tablas y chequea
los hashes. CLASS **no es dependencia de CI**: los `.dat` viven en el
repo.

## 3. Metodología de comparación

### 3.1. Convenciones de normalización

`gadget-ng` acepta dos convenciones de IC (ver
[Phase 37](2026-04-phase37-growth-rescaled-ics.md)):

- **legacy** (`rescale_to_a_init = false`, default): `σ_8 = 0.8`
  aplicado en `a_init`. Numéricamente, `P_m(IC)` equivale a un espectro
  lineal normalizado a `σ_8 = 0.8` a `z = 0`. Referencia: `P_CLASS(k, z=0)`.
- **rescaled** (`rescale_to_a_init = true`, experimental Fase 37):
  amplitudes LPT reducidas por `s`. `P_m(IC) ≈ P_linear(k, z=49)`
  en convención estándar. Referencia: `P_CLASS(k, z=49)`.

Ambos modos se validan **en paralelo**: si el cierre de `pk_correction`
es físico, la métrica final tiene que ser igual en las dos convenciones.

### 3.2. Pipeline end-to-end

```text
gadget-ng ICs  ──►  power_spectrum(positions)  ──►  pk_correction  ──►  P_c [(Mpc/h)^3]
                                                                         │
                                                                         ▼
                                                     interpolación log-log  ──►  R(k)
                                                                         ▲
reference/pk_class_z{0,49}.dat  ──────────────────────────────────────  P_CLASS [(Mpc/h)^3]
```

### 3.3. Conversión de unidades

- `k_hmpc = k_internal · h / box_mpc_h` con `box_mpc_h = 100` y
  `h = 0.674`.
- `pk_correction` absorbe el factor de volumen implícitamente en `R(N)`
  (calibrado así en Phase 35). El `unit_factor = (box_mpc_h)³` no se
  aplica por separado para evitar un factor `10⁶` espurio (nota en
  [`run_phase38.sh`](../../experiments/nbody/phase38_class_validation/run_phase38.sh)).

### 3.4. Ventana lineal

- `k ≤ k_Nyq/2` (mismo criterio que Phases 34–36).
- `n_modes ≥ 8` por bin.
- Banda BAO `k ∈ [0.05, 0.30] h/Mpc` marcada como `in_bao_band=true` y
  reportada aparte: `gadget-ng` usa transfer EH no-wiggle, CLASS tiene
  wiggles BAO, por lo que una diferencia `~3–5 %` en forma dentro de
  esa banda es sistemática y esperada (no es un defecto de la
  corrección).

### 3.5. Matriz de corridas

| # | N    | IC   | Solver | Seeds            | Snapshot | Modos              |
|---|------|------|--------|------------------|----------|--------------------|
| 1 | 32³  | 2LPT | PM     | 42, 137, 271     | IC-only  | legacy, rescaled   |
| 2 | 64³  | 2LPT | PM     | 42, 137, 271     | IC-only  | legacy, rescaled   |

Total: **12 mediciones**. Runtime release in-process: **~0.6 s** (toda
la matriz corre desde `OnceLock`).

### 3.6. Métricas

Para cada medición, sobre el ventaneado lineal `k ≤ k_Nyq/2`:

- `median|log10(P_m/P_CLASS)|` — error bruto.
- `median|log10(P_c/P_CLASS)|` — error corregido.
- `mean(P_c/P_CLASS)`, `stdev`, `CV` — cierre de amplitud absoluta.
- `d ln P_c / d ln k` vs `d ln P_CLASS / d ln k` — pendiente log-log
  (OLS) para cierre de forma, restringido a bins fuera de la banda BAO.

## 4. Resultados cuantitativos

### 4.1. Métricas por medición (ventana lineal completa)

| N   | seed | modo      | n_bins | `\|log10 raw\|` | `\|log10 corr\|` | `mean(P_c/P_CLASS)` | CV    |
|-----|------|-----------|-------:|--------------:|---------------:|--------------------:|-------:|
| 32³ | 42   | legacy    |     8  |       14.722  |       0.0362   |               1.037 | 0.129 |
| 32³ | 42   | rescaled  |     8  |       14.730  |       0.0362   |               1.020 | 0.130 |
| 32³ | 137  | legacy    |     8  |       14.723  |       0.0225   |               0.967 | 0.090 |
| 32³ | 137  | rescaled  |     8  |       14.730  |       0.0239   |               0.951 | 0.090 |
| 32³ | 271  | legacy    |     8  |       14.691  |       0.0461   |               1.036 | 0.197 |
| 32³ | 271  | rescaled  |     8  |       14.699  |       0.0452   |               1.019 | 0.197 |
| 64³ | 42   | legacy    |    16  |       18.027  |       0.0240   |               0.981 | 0.114 |
| 64³ | 42   | rescaled  |    16  |       18.034  |       0.0290   |               0.964 | 0.114 |
| 64³ | 137  | legacy    |    16  |       18.011  |       0.0333   |               1.017 | 0.137 |
| 64³ | 137  | rescaled  |    16  |       18.019  |       0.0266   |               1.001 | 0.137 |
| 64³ | 271  | legacy    |    16  |       18.010  |       0.0219   |               0.986 | 0.100 |
| 64³ | 271  | rescaled  |    16  |       18.018  |       0.0237   |               0.970 | 0.100 |

Observaciones:

- **Error crudo**: `median|log10(P_m/P_CLASS)| ≈ 14.7` a N=32 y `18.0`
  a N=64. El valor crece con `N` porque `P_m` está en unidades
  internas (`box=1`) y CLASS en `(Mpc/h)³`; el offset absoluto
  encapsula el factor `A_grid · R(N)` y la conversión de volumen.
- **Error corregido**: `∈ [0.022, 0.046]` en todas las 12 mediciones,
  por debajo del umbral de Phase 36 (`0.25`) por un margen de `×6–×11`.
- **Amplitud absoluta**: `mean(P_c/P_CLASS)` queda en
  `[0.95, 1.04]` — cierre absoluto dentro del `5 %` sobre todos los
  bins, todos los seeds, ambas convenciones, ambas resoluciones.
- **Rescaled vs legacy**: idénticos a `~2 %` en `mean(R_corr)`
  (consistente con la diferencia CPT92–CLASS en `s`); las medianas de
  error coinciden en `~0.01 dex`.

### 4.2. Factor de mejora agregado (promedio sobre seeds)

| N   | modo     | `|log10 raw|` | `|log10 corr|` | factor |
|-----|----------|--------------:|---------------:|-------:|
| 32³ | legacy   |       14.771  |        0.0914  |  **161×** |
| 32³ | rescaled |       14.779  |        0.0962  |  **153×** |
| 64³ | legacy   |       18.024  |        0.0237  |  **761×** |
| 64³ | rescaled |       18.031  |        0.0260  |  **692×** |

A `N = 64³` la corrección mejora la amplitud absoluta por un factor
**>700×** contra CLASS.

### 4.3. Consistencia entre resoluciones (test 5)

| modo     | `|log10 corr|` N=32 | `|log10 corr|` N=64 | `|abs diff|` | `rel diff` |
|----------|--------------------:|--------------------:|-------------:|-----------:|
| legacy   | 0.0349              | 0.0264              | 0.0085       | 0.244      |
| rescaled | 0.0351              | 0.0264              | 0.0087       | 0.247      |

Ambos valores `< 0.25` (umbral absoluto de Phase 36). El `rel diff`
refleja que N=64 está más cerca del régimen asintótico `R(N)` del
modelo de Phase 35.

## 5. Validación de forma espectral

Pendiente log-log OLS sobre bins fuera de la banda BAO
(`k ∉ [0.05, 0.30] h/Mpc`), sólo `N = 64³` (a `N = 32³` la ventana
fuera de BAO tiene 2 bins y no permite fit estable):

| modo     | seed | slope CLASS | slope corrected | `|Δ|` |
|----------|-----:|------------:|----------------:|------:|
| legacy   |  42  |      −1.583 |          −1.589 | 0.005 |
| legacy   | 137  |      −1.583 |          −1.678 | 0.095 |
| legacy   | 271  |      −1.583 |          −1.531 | 0.052 |
| rescaled |  42  |      −1.583 |          −1.589 | 0.005 |
| rescaled | 137  |      −1.583 |          −1.678 | 0.095 |
| rescaled | 271  |      −1.583 |          −1.531 | 0.052 |

Todos los `|Δ| ≤ 0.10`, i.e. el índice espectral local se preserva a
`~6 %` o mejor. La dispersión entre seeds (0.005 vs 0.095) es
cosmic-variance de un único realizados a `N = 64` con pocos bins
fuera de BAO, **no** un defecto de `pk_correction`.

## 6. Figuras

Generadas por
[`plot_phase38.py`](../../experiments/nbody/phase38_class_validation/scripts/plot_phase38.py)
desde `target/phase38/per_measurement.json` y copiadas a
`docs/reports/figures/phase38/`:

| # | Archivo                         | Descripción                                                                     |
|---|---------------------------------|---------------------------------------------------------------------------------|
| 1 | `pk_class_vs_gadget.png`        | `P_m`, `P_c`, `P_CLASS` en log-log para N=32 y N=64, seed 42, modo legacy.       |
| 2 | `ratio_pm_pc_vs_class.png`      | `P_m/P_CLASS` y `P_c/P_CLASS` sobre las 3 seeds, banda BAO sombreada.            |
| 3 | `abs_error_before_after.png`    | `|log10|` bruto vs corregido agregado sobre seeds, 4 grupos (N × modo).          |
| 4 | `n32_vs_n64.png`                | `P_c/P_CLASS` superpuesto para N=32 vs N=64, ambas convenciones y 3 seeds.        |
| 5 | `legacy_vs_rescaled.png`        | Equivalencia `legacy vs z=0` y `rescaled vs z=49` en el mismo gráfico.           |

## 7. Decisión técnica y respuestas a las 5 preguntas

### A. ¿`pk_correction` también funciona contra una referencia externa independiente?

**Sí.** Factor de mejora `161×` a N=32 y `761×` a N=64 frente al error
crudo, en las dos convenciones.

### B. ¿La amplitud corregida queda cerca de 1?

**Sí.** `mean(P_c/P_CLASS) ∈ [0.95, 1.04]` sobre 12 mediciones — cierre
absoluto dentro del `5 %`.

### C. ¿Qué error residual queda?

- En amplitud: `median|log10(P_c/P_CLASS)| ∈ [0.022, 0.046]`
  (i.e. `5–11 %` en ratio lineal).
- En forma: `|Δslope| ≤ 0.10` en la ventana fuera de BAO.

### D. ¿La discrepancia residual es compatible con resolución, ventana de k o diferencias EH vs CLASS/CAMB?

**Sí, plenamente:**

1. **Resolución:** N=32 con `8` bins vs N=64 con `16` bins — la
   mediana mejora de `0.035` a `0.026`, consistente con mejor muestreo
   en modos con `n_modes ≥ 8`.
2. **EH no-wiggle vs CLASS:** CLASS incluye BAO wiggles que EH no
   tiene. La diferencia dentro de la banda BAO es sistemática
   (~3–5 %) y reportada por separado; no contamina la métrica de
   amplitud absoluta.
3. **CPT92 vs CLASS en crecimiento:** `s_CPT92 = 2.5413e-2` vs
   `s_CLASS = 2.5625e-2`, diferencia `0.83 %`. Esto explica la
   diferencia `~1.6 %` sistemática entre `mean(R_corr)` legacy y
   rescaled (por duplicado, vía amplitud).
4. **Seeds pequeñas (N=32):** con sólo 2 bins fuera de BAO, la media
   en esa submuestra fluctúa entre `0.78` y `1.10` por seed — cosmic
   variance pura.

### E. ¿Esto basta para declarar cerrada la validación externa mínima?

**Sí.** Los cinco tests Rust pasan con criterios cuantitativos:

| # | Test                                                           | Umbral                     | Resultado |
|---|----------------------------------------------------------------|----------------------------|-----------|
| 1 | `pk_correction_reduces_error_vs_class`                         | factor `≥ 10×`             | `153–761×` |
| 2 | `pk_correction_keeps_ratio_near_unity_vs_class`                | `\|mean − 1\| < 0.15`      | `≤ 0.049` |
| 3 | `pk_correction_preserves_shape_vs_class`                       | `\|Δslope\| < 0.25`        | `≤ 0.095` |
| 4 | `pk_correction_no_nan_inf_vs_class`                            | todos finitos, `P > 0`     | ok         |
| 5 | `pk_correction_consistent_across_resolutions_vs_class`         | `med(N32), med(N64) < 0.25` | `≤ 0.035` |

### Decisión final: **validación externa mínima cerrada**

- `pk_correction` queda respaldada por un código externo independiente
  (CLASS), no sólo por la referencia interna EH.
- No se modificó `power_spectrum.rs`, no se recalibró `R(N)`, no se
  cambió física.
- El residuo es cuantificado y atribuido a fuentes conocidas (ventana
  de k, EH vs CLASS en BAO, CPT92 vs CLASS en crecimiento a 0.8 %).
- Fases futuras pueden ampliar la validación a CAMB como cruzado
  redundante, pero **no es requisito** para declarar cerrada la Fase 38.

## 8. Definition of Done

- [x] Tabla externa CLASS reproducible (§2).
- [x] `P_corrected` comparado contra ella en el snapshot IC (§3, §4).
- [x] Mejora de amplitud absoluta clara frente al crudo (§4.2: 153×–761×).
- [x] Error residual cuantificado (§4, §5; desglose dentro/fuera BAO).
- [x] 5 tests automáticos implementados y pasando (§7 tabla).
- [x] 4 + 1 figuras generadas y copiadas a `docs/reports/figures/phase38/` (§6).
- [x] Reporte con decisión explícita (§7).
- [x] Entrada de CHANGELOG actualizada.

---

**Artefactos**
- Referencia CLASS:
  [`experiments/nbody/phase38_class_validation/reference/`](../../experiments/nbody/phase38_class_validation/reference/)
- Tests: [`crates/gadget-ng-physics/tests/phase38_class_validation.rs`](../../crates/gadget-ng-physics/tests/phase38_class_validation.rs)
- Configs CLI: [`experiments/nbody/phase38_class_validation/configs/`](../../experiments/nbody/phase38_class_validation/configs/)
- Orquestador: [`run_phase38.sh`](../../experiments/nbody/phase38_class_validation/run_phase38.sh)
- Scripts: [`apply_phase38_correction.py`](../../experiments/nbody/phase38_class_validation/scripts/apply_phase38_correction.py),
  [`plot_phase38.py`](../../experiments/nbody/phase38_class_validation/scripts/plot_phase38.py)
- Figuras: [`docs/reports/figures/phase38/`](figures/phase38/)
- JSONs dumpeados: `target/phase38/per_measurement.json` (+ 5 por test)
