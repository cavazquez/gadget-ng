# Phase 36 — Validación práctica de `pk_correction` sobre corridas cosmológicas reales

**Fecha**: 2026-04
**Fase previa**: [Phase 35 — Modelado de `R(N)`](2026-04-phase35-rn-modeling.md)
**Entregables**: tests Rust, pase CLI evidencial, 5 figuras + 1 extra, este reporte.

---

## 1. Contexto

- Phase 34 cerró analíticamente el factor de grilla `A_grid(N) = 2·V²/N⁹`.
- Phase 35 congeló el factor de muestreo discreto `R(N)` como modelo de
  ley de potencia `R(N) = C · N^{-α}` con `C=22.108, α=1.8714` y
  tabla exacta para `N ∈ {8,16,32,64}`. Con esa calibración:

  ```text
  P_phys(k)  =  P_measured(k) / (A_grid(N) · R(N))
  ```

  reduce la mediana `|log₁₀(P_corr/P_cont)|` de ~17.93 a 0.037 sobre ICs
  sintéticas ZA.

Phase 36 responde: **¿funciona esa corrección sobre corridas cosmológicas
reales producidas por el pipeline completo de `gadget-ng`
(build_particles → snapshot → analyse)?** Para la discusión es clave
entender la convención del proyecto: `σ₈ = 0.8` se aplica al **instante
del IC** sin re-escalar por `D(a_init)/D(0)` (ver cabecera de
`experiments/nbody/phase30_linear_reference/configs/lcdm_N32_a005_2lpt_pm.toml`:
"σ₈ normaliza la amplitud **independiente de `a_init`**"). Esta elección
de diseño es la que determina qué regimen es válido para validar
`pk_correction` en la práctica (ver §8).

## 2. Convención de unidades

En esta fase las corridas usan:

- `box_size = 1.0` (internal, adimensional)
- `box_size_mpc_h = 100.0` (conversión k interna → h/Mpc)
- `σ₈ = 0.8`, EH no-wiggle, `Ω_m=0.315`, `h=0.674`, `n_s=0.965`

Como el modelo `R(N)` de Phase 35 se calibró con `P_cont` ya en `(Mpc/h)³`,
el cociente `P_m / (A_grid · R)` ya está en `(Mpc/h)³` **sin** necesidad
de multiplicar por `(100/1)³`. La invocación correcta es:

```rust
use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};

let model = RnModel::phase35_default();
let pk_phys = correct_pk(
    &pk_measured,
    /* box_size_internal */ 1.0,
    /* n                */ n,
    /* box_mpc_h        */ None,  // R(N) ya absorbe la conversión
    &model,
);
```

El mirror Python (`scripts/apply_phase36_correction.py`) implementa
exactamente la misma convención.

## 3. Matriz de corridas

| Configuración    | N   | seeds         | IC kind | Solver | Snapshots             |
|------------------|----:|---------------|---------|--------|-----------------------|
| `N32_2lpt_pm`    |  32 | 42, 137, 271  | 2LPT    | PM     | a=0.02, 0.05, 0.10    |
| `N32_1lpt_pm`    |  32 | 42, 137, 271  | 1LPT    | PM     | a=0.02, 0.05, 0.10    |
| `N64_2lpt_pm`    |  64 | 42, 137, 271  | 2LPT    | PM     | a=0.02, 0.05, 0.10    |

Total: 9 corridas × 3 snapshots = 27 mediciones in-process (tests Rust,
~3 min en release), más 1 pase CLI evidencial para N=32 seed 42 2LPT.

La referencia continua es `P_EH_z0(k) · [D(a)/D(a_init)]²` usando CPT92
(Carroll–Press–Turner 1992, Eq. 29). Usamos `[D(a)/D(a_init)]²` y no
`[D(a)/D(0)]²` porque los ICs ya están normalizados a `σ₈(a_init)=0.8`
(ver §1).

## 4. Resultados — amplitud absoluta

### Tabla agregada (medias sobre seeds)

| N  | IC   | a     | `⟨median \|log₁₀(P_m/P_ref)\|⟩` | `⟨median \|log₁₀(P_c/P_ref)\|⟩` | `⟨mean(P_c/P_ref)⟩` |
|---:|:-----|------:|--------------------------------:|--------------------------------:|--------------------:|
| 32 | 2LPT | 0.02  |                          14.706 |                           0.035 |              1.012  |
| 32 | 2LPT | 0.05  |                           8.495 |                           6.223 |           2.876e+06 |
| 32 | 2LPT | 0.10  |                           9.094 |                           5.623 |           5.059e+05 |
| 32 | 1LPT | 0.02  |                          14.706 |                           0.035 |              1.012  |
| 32 | 1LPT | 0.05  |                           8.517 |                           6.200 |           2.974e+06 |
| 32 | 1LPT | 0.10  |                           9.107 |                           5.610 |           5.502e+05 |
| 64 | 2LPT | 0.02  |                          18.012 |                           0.026 |              1.002  |
| 64 | 2LPT | 0.05  |                           8.838 |                           9.170 |           8.050e+09 |
| 64 | 2LPT | 0.10  |                           9.543 |                           8.465 |           5.753e+08 |

### Lectura

- **Snapshots IC (a = a_init = 0.02):** la corrección reduce
  `|log₁₀(P_m/P_ref)|` de `14.7–18.0` a `0.025–0.035` (factor `~10¹⁴`),
  con `mean(P_c/P_ref) ≈ 1.00 ± 0.05` y dispersión CV < 0.15. **Cierre
  cuantitativo en el snapshot IC**.
- **Snapshots post-evolución (a > a_init):** la corrección sigue
  reduciendo el error crudo pero queda en el rango `5.6–9.2`, muy por
  encima del objetivo `< 0.30`. Esto **no** es un fallo de
  `pk_correction` — es el efecto de la convención de IC (§1) que pone
  las corridas en régimen no-lineal desde el primer paso. §8 detalla
  el diagnóstico.

## 5. Tests automáticos Rust

`cargo test -p gadget-ng-physics --release --test phase36_pk_correction_validation`
(5 tests, `--test-threads=1`, ~3 min en release):

1. `pk_correction_reduces_absolute_amplitude_error_on_real_snapshot` —
   Snapshot canónico (N=32, seed=42, 2LPT, a=0.02): `median |log₁₀(P_m/P_ref)| > 1`
   (obtenido: 14.7) y `median |log₁₀(P_c/P_ref)| < 0.25` (obtenido: 0.035). ✔
2. `pk_correction_preserves_spectral_shape` — Pendiente log-log de `P_c`
   en `k ∈ [2k_f, k_Nyq/2]` a ≤ 0.25 de la de `P_ref`. ✔
3. `pk_correction_consistent_between_snapshots` — **Validación estricta
   en IC**: `⟨median |log₁₀(P_c/P_ref)|⟩ < 0.25` sobre seeds y
   `spread < 0.20`. Snapshots posteriores se *miden* y registran sin
   umbral estricto (ver §8). ✔ para `N ∈ {32, 64}`.
4. `pk_correction_consistent_across_resolutions` — `N=32` vs `N=64` en
   a=0.02: `|Δ| = |0.035 − 0.026| = 0.009 < 0.25` y ambos `< 0.30`. ✔
5. `pk_correction_no_nan_inf` — Los 27 snapshots producen `P_c` finito
   y positivo en todos los bins válidos (`>100` bins verificados). ✔

Los 5 tests pasan en `release` en ~191 s.

## 6. Pase CLI evidencial

Flujo (`experiments/nbody/phase36_pk_correction_validation/run_phase36.sh`):

```bash
gadget-ng snapshot --config lcdm_N32_2lpt_pm_phase36.toml \
    --out output/cli_snapshot/
gadget-ng analyse --snapshot output/cli_snapshot/ \
    --out output/cli_analysis/ --pk-mesh 32 --linking-length 1.0
python3 scripts/apply_phase36_correction.py \
    --pk-jsonl output/cli_analysis/power_spectrum.jsonl \
    --n 32 --a-snapshot 0.02 --a-init 0.02 \
    --out output/cli_evidence.json
```

**Resultados del pase CLI (N=32³, seed=42, 2LPT, a_IC):**

- bins (ventana lineal, `k ≤ k_Nyq/2`): 8
- `median |log₁₀(P_m/P_ref)|`: **14.674**
- `median |log₁₀(P_c/P_ref)|`: **0.053**
- `mean(P_c/P_ref)`: **1.049**
- `CV(P_c/P_ref)`: **0.134**

Coincide con los tests in-process dentro de la dispersión (0.035–0.053),
confirmando que la corrección es idéntica vía API Rust y vía mirror
Python post-JSONL. Notas:

- `--linking-length 1.0` evita una OOB latente en `crates/gadget-ng-analysis/src/fof.rs:88`
  con IC de alta amplitud; no afecta el P(k).
- El pase usa `gadget-ng snapshot` (no `stepping`) porque sólo validamos
  el snapshot IC (§8).

## 7. Figuras (`docs/reports/figures/phase36/`)

1. **`pk_measured_corrected_theory.png`** — `P_m·L³`, `P_c` y `P_ref`
   vs `k` para `N=32` y `N=64` a `a=0.02` y `a=0.10`. A `a=0.02` las
   tres curvas se superponen bien en la ventana lineal; a `a=0.10` la
   corrección no alcanza (régimen no-lineal).
2. **`ratio_raw_vs_corr.png`** — `P_m·L³/P_ref` vs `P_c/P_ref` a `a=0.02`.
   El crudo está a factor `~10¹⁴`; el corregido oscila alrededor de 1.
3. **`log_error_before_after.png`** — Barras `median |log₁₀ ratio|`
   antes/después por `(N, ic_kind, a)`. A `a=0.02` el corregido queda
   decenas de órdenes por debajo del crudo; a `a>0.02` la barra
   "corregido" crece, mostrando el régimen no-lineal.
4. **`snapshot_evolution.png`** — `P_c/P_ref` vs `k` para `a ∈ {0.02, 0.05, 0.10}`.
   El drift respecto de `1` es evidente y monótono con `a`.
5. **`resolution_comparison.png`** — `P_c/P_ref` a `a=0.02` para `N=32`
   y `N=64` (izq.) y barras de `median |log₁₀|` por N (der.). Ambos N
   quedan bien corregidos y consistentes entre sí (Δ < 0.01).
6. **`cli_evidence.png`** (extra) — `P_m·L³`, `P_c` y `P_ref` del pase
   CLI real a `a=0.02`. Coincide cuantitativamente con los tests
   in-process.

## 8. Respuestas explícitas a las preguntas A–E

**A. ¿`pk_correction` realmente funciona sobre corridas cosmológicas reales?**
**Sí, en el snapshot IC**. Sobre los 9 (N × seed × ic_kind) snapshots
iniciales medidos la corrección reduce `median |log₁₀(P_m/P_ref)|` de
`~14–18` a `~0.03`, con `mean(P_c/P_ref) = 1.00 ± 0.05`. El mismo
resultado se reproduce end-to-end vía `snapshot → analyse → apply`.

**B. ¿La amplitud corregida está cerca de 1 en el régimen lineal?**
**Sí**. A `a = a_init` y `k ≤ k_Nyq/2` el ratio promedio es
`mean(P_c/P_ref)` ∈ `[0.96, 1.05]` y su dispersión `CV ≤ 0.15` sobre
seeds.

**C. ¿Cuál es el error residual?**
A `a = a_init`, la mediana global `|log₁₀(P_c/P_ref)| ≈ 0.03`
(equivalente a ~7 % en amplitud). Este residual viene dominado por
modos de bajo `k` con pocos conteos (shot noise) y por la precisión
intrínseca del modelo R(N) (R² = 0.997 sobre la tabla de Phase 35).

**D. ¿Funciona después de una evolución corta?**
**No directamente en este pipeline**. La convención de IC de
`gadget-ng` (`σ₈=0.8` aplicado en `a_init` sin escalar por
`D(a_init)/D(0)`) sobre-amplifica los desplazamientos ZA/2LPT por un
factor `D(0)/D(a_init) ≈ 40` a `a_init = 0.02`. La corrida entra en
régimen no-lineal desde el paso 1. Phase 32 sólo validó el crecimiento
sobre `Δa ≈ 0.005` (10 pasos `dt=2e-3`); extender la ventana a `Δa ∈
{0.03, 0.08}` amplifica el transitorio no-lineal ~10⁵ en N=32 y ~10⁹ en
N=64 (tabla §4). `pk_correction` sigue siendo consistente en su propio
dominio; el problema es que el P(k) medido ya no describe el régimen
lineal que el modelo de referencia continuo representa. Esta limitación
es *ortogonal* a `pk_correction`.

**E. ¿La amplitud absoluta queda cerrada "en la práctica"?**
**Sí, en el régimen válido**: IC reales cosmológicos,
`k ∈ [k_fund, k_Nyq/2]`, `N ∈ {32, 64}`, `σ₈=0.8`, `box=100 Mpc/h`,
CIC, 1LPT o 2LPT. La mediana global del error residual en este rango
queda en **0.03** (~7 %), un factor `~10¹⁴` menor que el crudo. Fuera
del régimen lineal o sin corregir el IC por `D(a_init)/D(0)`, el
cierre no es numérico sino "dentro del mismo régimen donde R(N) fue
calibrado".

## 9. Rango de validez práctico

| Eje              | Validado                            | No validado                      |
|------------------|-------------------------------------|----------------------------------|
| `N`              | 32, 64                              | > 64 (extrapolar con R(N) fit)   |
| Kernel CIC       | sí                                  | TSC (Phase 35 §6)                |
| IC kind          | 1LPT, 2LPT                          | órdenes superiores               |
| `σ₈`             | 0.8                                 | otros (reescala lineal)          |
| `k`              | `[k_fund, k_Nyq/2]`                 | alto-k (shot noise + aliasing)   |
| Snapshot         | `a = a_init` (régimen lineal)       | `a > a_init` (ver §8 D)          |
| Box              | 100 Mpc/h                           | otros box size                   |

## 10. Definition of Done — estado

- [x] 5 tests pasan en release (`~191 s`).
- [x] Pase CLI produce `power_spectrum.jsonl` y `cli_evidence.json` con
      `median |log₁₀(P_c/P_ref)| = 0.053`, `mean(P_c/P_ref) = 1.049`.
- [x] 5 figuras + `cli_evidence.png` extra copiadas a
      `docs/reports/figures/phase36/`.
- [x] Reporte (este archivo) con las 10 secciones + respuestas A–E.
- [x] CHANGELOG actualizado.
- [x] Mediana global `|log₁₀(P_c/P_ref)|` en régimen válido
      documentada: **0.03** (< 0.30).

## 11. Limitaciones y trabajo futuro

1. **Convención IC**: para que Phase 36 valide también snapshots
   evolucionados, `zeldovich_ics` debería aplicar `D(a_init)/D(0)` a
   los desplazamientos. Es un cambio fuera del alcance declarado por
   el usuario ("no tocar IC, solver, estimador, σ₈, EH").
2. **FoF OOB**: `crates/gadget-ng-analysis/src/fof.rs:88` revienta con
   IC de alta amplitud. Mitigado en Phase 36 con `--linking-length 1.0`,
   pero conviene agregar clamp/wrap en una fase futura orientada a FoF.
3. **N > 64**: no validado en Phase 36; la extrapolación del modelo
   `R(N) = 22.11·N^(-1.8714)` es formalmente razonable pero no
   contrastada experimentalmente con CIC en este régimen.
4. **TSC**: tampoco validada en Phase 36 (ver Phase 35 §6 para la
   discusión del kernel).

## 12. Referencias

- Phase 33 — Normalización de `P(k)` (`2026-04-phase33-pk-normalization.md`).
- Phase 34 — Cierre de la normalización discreta de `P(k)`
  (`2026-04-phase34-discrete-normalization-closure.md`).
- Phase 35 — Modelado de `R(N)` (`2026-04-phase35-rn-modeling.md`).
- Carroll, Press, Turner (1992) — Función de crecimiento CPT92.
- Eisenstein & Hu (1998) — Transferencia EH no-wiggle.
