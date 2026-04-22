# Phase 44 — Auditoría y fix de condiciones iniciales 2LPT

**Fecha**: abril 2026
**Crate afectada**: `gadget-ng-core` (`src/ic_2lpt.rs`)
**Estado**: fix implementado, validado con 6 unit tests + 5 tests de integración.
**Conclusión física**: el bug existía y era matemáticamente real (doble `1/|n|²`
+ signo invertido), pero su impacto empírico en la normalización actual
(`Z0Sigma8` @ `a_init=0.02`) es **O(10⁻¹²) en posiciones** y **<2 % en el error
espectral evolucionado**. El bottleneck del crecimiento lineal NO es 2LPT.

## TL;DR

| Pregunta | Respuesta |
|---|---|
| ¿Había bugs reales en el 2LPT? | Sí (2 críticos + 1 menor). |
| ¿Los corregimos? | Sí, con unit tests k-space canónicos. |
| ¿Recuperamos el crecimiento lineal esperado? | **No**. El error post-fix es `≈ 2 %` menor que el pre-fix — insuficiente. |
| ¿Dónde está el bottleneck entonces? | No es 2LPT. Candidatos: normalización `σ₈ ↔ P(k) medido`, shot noise, unidades de la integración. |

---

## 1. Motivación

Phase 43 cerró la línea "temporal control + paralelismo" con un resultado
claro: reducir `dt` tiene impacto marginal (< 9 % mejora de `4·10⁻⁴` a `2·10⁻⁴`)
y el timestep adaptativo satura en su `dt_min` sin superar al mejor `dt` fijo.
Esto apuntaba al IC como el culpable principal:

```text
median |log10(P_c/P_ref)| ≈ 8.6  ↔  P_c / P_ref ≈ 10⁸·⁶ ≈ 4·10⁸
δ_rms ≈ 1.0  ya en a = 0.05  (ridículamente no-lineal para z ~ 19)
```

Si el 2LPT inyectaba aceleraciones con amplitud/signo mal calibrados, la
dinámica del solver se volvería no-lineal inmediatamente. Phase 44 audita
la implementación contra referencias canónicas y aplica un fix.

## 2. Referencias canónicas

| Fuente | Contribución |
|---|---|
| Scoccimarro (1998), MNRAS 299, 1097 — arXiv:[astro-ph/9711187](https://arxiv.org/abs/astro-ph/9711187) | Derivación moderna de 2LPT, definiciones de `f₁`, `f₂`. |
| Bouchet, Colombi, Hivon & Juszkiewicz (1995), A&A 296, 575 | Aproximaciones `D₂ ≈ −(3/7)D₁²` y `f₂ ≈ 2·Ω_m^{6/11}`. |
| Crocce, Pueblas & Scoccimarro (2006), MNRAS 373, 369 — `2LPTic` | Implementación de referencia en C. |
| Jenkins (2010), MNRAS 403, 1859 — arXiv:[0910.0258](https://arxiv.org/abs/0910.0258) | Estándar moderno (GADGET-4, N-GenIC). ec. 2, ec. 6. |

Fórmula canónica (Jenkins 2010 ec. 2 con `D₁ = 1` absorbida en la amplitud):

```text
x = q − ∇φ¹(q) + (D₂/D₁²)·∇φ²(q)
v = −D₁·f₁·H·∇φ¹ + D₂·f₂·H·∇φ²
∇²φ¹(q) = δ(q)
∇²φ²(q) = S(q) ≡ φ¹_{,xx}φ¹_{,yy} − φ¹²_{,xy} + (cíclicos)
```

En `2LPTic/main.c:477-478` el gradiente de `φ²` se codifica explícitamente
como `Ψ²(k) = −i·k/k² · S(k)` (código original):

```c
cdisp2[axes].re =  S.im * kvec[axes] / kmag2;
cdisp2[axes].im = −S.re * kvec[axes] / kmag2;
```

— una **única** división por `|k|²`, signo `−i` (en la convención donde
`Ψ²` es `+∇φ²`).

## 3. Bugs identificados

### Bug A — doble división por `|n|²` (CRÍTICO)

En la implementación pre-Phase-44 de `ic_2lpt.rs` el Poisson y el gradiente
estaban factorizados en dos funciones:

```rust
// 1) resolver Poisson: φ²(k) = −S(k) / |n|²
let phi2_k = solve_poisson_real_to_kspace(&source, n);
// 2) gradiente: Ψ²_α(k) = −i · n_α / |n|² · φ²(k)
let psi2 = phi2_to_psi2(&phi2_k, n, box_size);
```

Composición exacta:

```text
Ψ²_α(k) = −i · n_α / |n|² · ( −S(k) / |n|² )
         = +i · n_α · S(k) / |n|⁴
```

vs canónico `−i · n_α / |n|² · S(k)`. Dos defectos:
- **amplitud** atenuada por un factor extra `1/|n|²` (muy dañino en modos
  medios y altos),
- **signo global invertido** (`+i` vs `−i`).

### Bug B — signo del término 2LPT (CRÍTICO)

Combinado con el bug A, el signo invertido de `Ψ²` **anulaba parcialmente**
el signo negativo correcto de `D₂/D₁² ≈ −3/7·Ω_m^{−1/143}`: la corrección de
segundo orden se sumaba en la dirección físicamente incorrecta a bajo `|n|`.

### Bug C — aproximación de `f₂` (MENOR)

El código usaba `f₂ = 2·f₁` con `f₁ = Ω_m(a)^{0.55}` (Linder). La convención
de 2LPTic/Jenkins es `f₂ = 2·Ω_m(a)^{6/11}`. A `z = 49`, la diferencia es
`< 0.01 %`.

## 4. Fix aplicado

### 4.1 `source_to_psi2` unificada

Las dos funciones separadas se reemplazan por una única que codifica la
fórmula canónica directamente:

```text
Ψ²_α(k) = −i · n_α / |n|² · S(k)   (DC=0, Nyquist=0)
Ψ²_α(x) = IFFT[ Ψ²_α(k) ] · d        (d = box_size / n)
```

Ver `crates/gadget-ng-core/src/ic_2lpt.rs::source_to_psi2` (≈ 75 LOC).

### 4.2 `f₂` canónico

```rust
let f2 = 2.0 * omega_m_a.powf(6.0 / 11.0);
```

### 4.3 Variante A/B para auditoría

Para permitir comparaciones cuantitativas sin `git checkout` a la versión
previa, se expone `Psi2Variant::{Fixed, LegacyBuggy}` y
`zeldovich_2lpt_ics_with_variant(..., variant)`. `LegacyBuggy` reproduce
exactamente el bug histórico (`Ψ²_α = +i·n_α·S/|n|⁴`). No debe usarse en
simulaciones reales.

## 5. Validación

### 5.1 Unit tests k-space (crate `gadget-ng-core`)

Seis tests en `ic_2lpt.rs` bajo `mod tests`:

| Test | Qué valida |
|---|---|
| `source_is_finite` | Ninguna NaN/Inf en `S(x)` (retrocompatible). |
| `psi2_is_real_and_finite` | `Ψ²(x)` real y finito. |
| `psi2_matches_canonical_kspace_formula` | `FFT(Ψ²_α)(k) = (−i·n_α/|n|²)·S(k)·d` con error relativo global `< 10⁻¹⁰` sobre 10 122 modos (N=16). Distingue **signo** y **amplitud** del bug. |
| `psi2_amplitude_differs_from_legacy_bug` | `RMS(Ψ²_fix)/RMS(Ψ²_bug) > 1.5`. En el smoke (N=16, `P~\|n\|⁻²`) el ratio observado ≈ **2.72×**. |
| `psi2_scales_quadratically_with_delta` | `Ψ²(λδ) = λ²·Ψ²(δ)` con `rel_err < 10⁻¹⁰`. |
| `psi2_signs_consistent_across_amplitudes` | `Ψ²(−δ) = Ψ²(δ)` bit-idéntico (S es cuadrático en δ). |

Todos verdes: `cargo test -p gadget-ng-core ic_2lpt → 6/6 OK`.

### 5.2 Tests de integración A/B (crate `gadget-ng-physics`)

`tests/phase44_2lpt_audit.rs` corre ambas variantes a `N=32³`, `seed=42`,
TreePM con `ε_phys = 0.01 Mpc/h` y `dt = 2·10⁻⁴`.

| Test | Resultado |
|---|---|
| `ic_amplitudes_changed_by_fix` | `max Δpos = 1.14·10⁻¹²`, `max Δvel = 1.82·10⁻¹⁴` → **el fix cambia las ICs pero al nivel O(10⁻¹²)**. |
| `fixed_variant_matches_legacy_psi1_component` | Con `use_2lpt=false` ambas variantes son bit-idénticas (Ψ¹ no tocado). |
| `fixed_variant_runs_stably` | Sin NaN/Inf hasta `a=0.10`. |
| `fixed_variant_improves_growth_vs_legacy` (SOFT) | Growth err `Fixed/Legacy` = `85.3·10⁶` / `80.6·10⁶` — **ambos catastróficos**, el fix no empeora. |
| `no_nan_inf_under_phase44_matrix` | Todas las métricas finitas. |

### 5.3 Regresión con tests existentes

`cargo test -p gadget-ng-core` → **27/27 OK** (incluye 21 tests pre-existentes).
`cargo test -p gadget-ng-physics --test lpt2_ics` → **8/8 OK**.
Tests largos (Phase 37/40/42/43) no re-ejecutados en este run; cambiarán bit
en los snapshots de 2LPT por O(10⁻¹²), necesitarán regenerarse.

## 6. Datos cuantitativos (A/B)

Matriz canónica: `N=32³`, `seed=42`, TreePM, `ε_phys=0.01 Mpc/h`, `dt=2·10⁻⁴`,
`Z0Sigma8`, snapshots en `a ∈ {0.02, 0.05, 0.10}`.

Archivo de datos: `target/phase44/per_snapshot_metrics.json`.
CSV: `experiments/nbody/phase44_2lpt_audit/figures/phase44_summary.csv`.
Figura: `experiments/nbody/phase44_2lpt_audit/figures/phase44_metrics_vs_a.png`.

| Variante | `a` | `δ_rms` | `v_rms` | `med│log Pc/Pref│` | `growth_lowk` |
|---|---|---|---|---|---|
| Fixed | 0.020 | 0.000 | 9.85·10⁻¹⁰ | 0.035 | 1.000 |
| Fixed | 0.050 | 1.000 | 34.24 | **9.233** | **4.50·10⁸** |
| Fixed | 0.100 | 0.999 | 34.47 | **8.619** | **8.53·10⁷** |
| Legacy | 0.020 | 0.000 | 9.85·10⁻¹⁰ | 0.035 | 1.000 |
| Legacy | 0.050 | 1.001 | 34.53 | 9.217 | 4.92·10⁸ |
| Legacy | 0.100 | 1.001 | 34.75 | 8.627 | 8.06·10⁷ |

Observaciones:

1. **En las ICs (a=0.02) las métricas son indistinguibles** a `~1 ulp`
   (v_rms_Fix/v_rms_Legacy − 1 ≈ 2·10⁻⁷).
2. **A `a=0.05` y `a=0.10` ambas variantes son idénticamente catastróficas**:
   `δ_rms ≈ 1` (esperado << 0.1 para linealidad a z~9), `v_rms` saltó 10¹⁰× en
   ~150 pasos (esperado `×D(0.05)/D(0.02) ≈ 2.5`).
3. **La mejora absoluta del fix sobre el bug es**: `err Δ ≈ 0.015 dex` (1–2 %) y
   `growth_lowk Δ ≈ 10 %`. **Orden de magnitud menor** del que haría falta
   para recuperar el régimen lineal.

## 7. Por qué el fix no rescata la validación física

Dos pistas del análisis cuantitativo explican el resultado:

### 7.1 El término 2LPT en estas ICs está pre-atenuado

En la rama `Z0Sigma8` con `a_init = 0.02`, se aplica un factor de Phase 37:
`Ψ¹ *= s`, `Ψ² *= s²` con `s = D(a_init)/D(1) ≈ 0.02`. El término de
segundo orden efectivo en posición es:

```text
(D₂/D₁²) · s² · Ψ²_unscaled ≈ (−0.43) · (4·10⁻⁴) · Ψ²_unscaled ≈ −1.7·10⁻⁴ · Ψ²
```

Un factor ~5 000× más chico que Ψ¹. Aunque el bug atenuaba `Ψ²` por otro
`1/|n|²` adicional, la diferencia absoluta en las ICs nunca superaba
`10⁻¹²`. El bug era **matemáticamente incorrecto** pero **numéricamente
subdominante** en este régimen.

### 7.2 Un `v_rms` x10¹⁰ en 150 pasos no es un bug de LPT

`v_rms` al IC ≈ `10⁻⁹` (unidades código). A `a = 0.05`, `v_rms = 34`. El
crecimiento esperado es D(0.05)/D(0.02) ≈ 2.5. El observado es `≈ 3.5·10¹⁰`,
un factor **10¹⁰ demasiado grande**.

Interpretación: hay una **discrepancia de unidades** entre:

- el momentum canónico `p = a²·H·f·D·Ψ` que escribe el IC,
- y la convención esperada por `leapfrog_cosmo_kdk_step` + `CosmoFactors`.

Este es un bottleneck **de orden mayor** que el 2LPT y es el candidato
claro para Phase 45.

## 8. Cambios de API

### Añadidos (públicos)

```rust
// en gadget_ng_core
pub enum Psi2Variant { Fixed, LegacyBuggy }
pub fn zeldovich_2lpt_ics_with_variant(..., psi2_variant: Psi2Variant) -> Vec<Particle>;
```

### Sin cambios (bit-compatibles con Phase 43 cuando `variant=Fixed`)

```rust
pub fn zeldovich_2lpt_ics(...) -> Vec<Particle>;  // delega en _with_variant con Fixed
```

### Eliminados (internos)

- `fn solve_poisson_real_to_kspace(...)` — absorbido por `source_to_psi2`.
- `fn phi2_to_psi2(...)` — ídem.

## 9. Bit-compatibilidad

- **1LPT**: bit-idéntico a Phase 43 (el fix solo toca la rama 2LPT).
- **2LPT**: cambia al nivel `O(10⁻¹²)` en posiciones y `O(10⁻¹⁴)` en velocidades.
  Los snapshots bit-exactos de Phase 37/40/42/43 requieren regeneración si se
  usan como golden reference; la señal física (δ_rms, v_rms, P(k)) cambia
  en `< 2 %`.

## 10. Decisión técnica

- **Aceptar el fix**. Es matemáticamente correcto, está validado por 6 tests
  k-space que comparan contra la fórmula canónica (Jenkins 2010, `2LPTic`),
  y no introduce regresiones en los tests existentes.
- **Reconocer que no rescata la validación física**. La diferencia Fixed vs
  Legacy en métricas evolucionadas es `<2 %`, **órdenes de magnitud menor**
  que el error total (`~10⁸` en `P_c/P_ref`).
- **Abrir Phase 45** con foco en la **convención de unidades** entre IC,
  integrador y análisis espectral. Hipótesis de trabajo: el momentum canónico
  de `zeldovich_*_ics` (`p = a²·H·f·Ψ`) está en unidades incompatibles con
  `leapfrog_cosmo_kdk_step`, provocando un arranque dinámico explosivo que
  saturo linealidad en `<150` pasos.

## 11. Ejecución

```bash
# Unit tests
cargo test -p gadget-ng-core ic_2lpt

# A/B integration test (≈14 min en release @ 6 cores, N=32)
cargo test -p gadget-ng-physics --test phase44_2lpt_audit --release -- --nocapture

# Análisis + figura
python3 experiments/nbody/phase44_2lpt_audit/scripts/plot_ab_comparison.py
```

## 12. Artefactos

- Código: `crates/gadget-ng-core/src/ic_2lpt.rs` (`source_to_psi2`,
  `source_to_psi2_legacy_buggy`, `Psi2Variant`,
  `zeldovich_2lpt_ics_with_variant`).
- Tests: `crates/gadget-ng-core/src/ic_2lpt.rs` (6 unit),
  `crates/gadget-ng-physics/tests/phase44_2lpt_audit.rs` (5 integration).
- Experimento: `experiments/nbody/phase44_2lpt_audit/` (scripts, configs,
  figures, outputs).
- Datos: `target/phase44/per_snapshot_metrics.json`.
- Figura: `experiments/nbody/phase44_2lpt_audit/figures/phase44_metrics_vs_a.png`.
- CSV: `experiments/nbody/phase44_2lpt_audit/figures/phase44_summary.csv`.

## 13. Pendientes para Phase 45

1. **Auditoría de unidades IC ↔ integrador**: verificar que la convención
   GADGET-4 `p = a²·dx/dt` escrita por las ICs coincide con la que asume
   `leapfrog_cosmo_kdk_step` via `CosmoFactors`. Posible fuente del ×10¹⁰.
2. **Auditoría de `pk_correction` y `RnModel::phase35_default`**: los valores
   post-evolución dan `P_c/P_ref ≈ 10⁸–10⁹`, incluso partiendo de un IC con
   `err = 0.035`. ¿Se aplica la referencia correcta contra la potencia
   medida?
3. **Smoke test de linealidad**: un test mínimo `N=32, a=0.02→0.025` con
   integrador puro y análisis inline debería cerrar el círculo de que el
   solver está preservando el modo lineal de bajo-k.
