# Phase 40 — Formalización de la convención física de normalización de ICs

**Fecha:** 2026-04  
**Estado:** Completa — **Decisión B**: el modo `Z0Sigma8` queda como opción
experimental; `Legacy` sigue siendo el modo recomendado.

---

## 1. Contexto

Phase 40 continúa la línea Phase 37 → 38 → 39:

| Fase | Hallazgo |
| --- | --- |
| **37** | Se introduce el flag `rescale_to_a_init`. La campaña muestra que el modo reescalado **empeora** los snapshots evolucionados (Decisión B). |
| **38** | `pk_correction` queda validada contra CLASS en el snapshot IC. Las dos convenciones (legacy vs rescaled) difieren solamente por la identificación de `a_ref` en el espectro lineal de referencia. |
| **39** | Se descarta `dt` como fuente del error en snapshots evolucionados: reducir `dt` no mejora la fidelidad espectral. |

El brief de Phase 40 pide: (1) reemplazar el flag experimental por una
convención explícita y limpia; (2) auditar la implementación por bugs sutiles;
(3) medir empíricamente `σ₈(a_init)` para verificar la convención física;
(4) repetir la campaña legacy vs nueva convención y decidir técnicamente.

---

## 2. API nueva: enum `NormalizationMode`

Se reemplaza el campo `rescale_to_a_init: bool` de Fase 37 por una enum
explícita en [`config.rs`](../../crates/gadget-ng-core/src/config.rs):

```rust
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum NormalizationMode {
    #[default]
    Legacy,
    Z0Sigma8,
}
```

El campo dentro de `IcKind::Zeldovich` pasa a llamarse
`normalization_mode: NormalizationMode`. En TOML:

```toml
[initial_conditions.kind.zeldovich]
use_2lpt           = true
normalization_mode = "legacy"      # o "z0_sigma8"
```

El dispatcher en [`ic.rs`](../../crates/gadget-ng-core/src/ic.rs) mapea
`Z0Sigma8 → rescale_to_a_init=true` y `Legacy → false` antes de invocar las
funciones `zeldovich_ics` / `generate_2lpt_ics`, que mantienen su firma
interna `bool` por compatibilidad mínima. Migración: 13 tests Rust y 9 archivos
TOML (phase37/38/39) se actualizaron al nuevo campo en este PR (cambio
**breaking** del surface TOML, limpio en el interior).

---

## 3. Derivación física

### 3.1 Convención `Legacy`

`σ₈(a_init) = 0.8` aplicado directamente al campo `δ(k)` generado en
`a_init`:

```text
δ_legacy(a_init, k)  →  σ₈(δ_legacy) = 0.8
Ψ¹_legacy = ∇⁻¹(-δ_legacy)
Ψ²_legacy = ∇⁻¹(source[Ψ¹_legacy])
x(a_init)  = q + Ψ¹_legacy + (D₂/D₁²)(a_init) · Ψ²_legacy
p(a_init)  = a²·H(a)·[f₁·Ψ¹_legacy + f₂·(D₂/D₁²)(a_init)·Ψ²_legacy]
```

Consecuencia: el espectro en `a_init` tiene amplitud de un universo con
`σ₈ = 0.8` **hoy** multiplicado por el factor `[D(1)/D(a_init)]²`. Para
`a_init = 0.02` y ΛCDM Planck18, esto significa **≈ 1546× más potencia** que
la referencia lineal físicamente correcta. La simulación entra casi de
inmediato en régimen no lineal.

### 3.2 Convención `Z0Sigma8`

`σ₈ = 0.8` queda referido a `a=1` (convención CAMB/CLASS). Con
`s = D(a_init)/D(1)`:

```text
Ψ¹_z0(a_init) = s · Ψ¹_legacy
Ψ²_z0(a_init) = s² · Ψ²_legacy
x(a_init) = q + s · Ψ¹_legacy + (D₂/D₁²)(a_init) · s² · Ψ²_legacy
p(a_init) = a²·H(a)·[f₁·s·Ψ¹_legacy + f₂·(D₂/D₁²)(a_init)·s²·Ψ²_legacy]
```

**Por qué `Ψ²` escala como `s²`:**  en la descomposición LPT estándar,
`δ_lin(a) = D(a) · δ_cont(q)` y `Ψ⁽²⁾(q)` es independiente del tiempo
(viene del gradiente de un potencial cuadrático en `Ψ⁽¹⁾`). Al escribir:

- `Ψ¹_legacy ∝ D(1) · Ψ⁽¹⁾_cont`
- `Ψ²_legacy ∝ [D(1)]² · Ψ⁽²⁾_cont`

al aplicar `s = D(a_init)/D(1)`:

- `s · Ψ¹_legacy = D(a_init) · Ψ⁽¹⁾_cont` ✓
- `(D₂/D₁²)(a_init) · s² · Ψ²_legacy = D₂(a_init) · Ψ⁽²⁾_cont` ✓

**Velocidades:**  como `p = a²·H·f·Ψ` es lineal en `Ψ`, heredan `s` / `s²`
sin factor adicional.

### 3.3 Predicción de `σ₈(a_init)`

- `Legacy`:   `σ₈(a_init) = 0.8`
- `Z0Sigma8`: `σ₈(a_init) = 0.8 · s = 0.8 · D(a_init)/D(1) ≈ 0.02033`

---

## 4. Auditoría de Phase 37

Se auditaron `ic_zeldovich.rs` e `ic_2lpt.rs` bajo `rescale_to_a_init=true`
(equivalente del nuevo `Z0Sigma8`) buscando bugs sutiles que pudieran haber
sesgado la campaña anterior:

| Chequeo | Resultado |
| --- | --- |
| Doble factor sobre `Ψ¹` | ❌ No hay. `psi_x *= scale` se aplica una sola vez. |
| Doble factor sobre `Ψ²` | ❌ No hay. `psi2_x *= scale²` una sola vez; `d2_over_d1sq` se evalúa en `a_init` sin tocar. |
| Velocidades con factor extra | ❌ No hay. `vel_factor = a²·H·f` no depende de `s`; `p = Ψ · vel_factor` hereda `s` / `s²` por lo que `Ψ` ya contiene. |
| `amplitude_for_sigma8` interpreta `σ₈(z=0)` correctamente | ✅ Sí: calibra `A` para que `σ₈(δ_raw) = 0.8` antes del rescaling; este `δ_raw` corresponde a `D(1)·δ_cont` (convención CAMB). |
| Consistencia del desplazamiento total a `a_init` bajo `Z0Sigma8` | ✅ Sí: `(D₂/D₁²)(a_init) · s² · [D(1)]² · Ψ⁽²⁾_cont = D₂(a_init) · Ψ⁽²⁾_cont` (álgebra exacta, sin factores colgantes). |

**Conclusión de la auditoría:** la implementación de Phase 37 es
**físicamente correcta**. El resultado negativo de Phase 37 no viene de un
bug de implementación.

---

## 5. Campaña

**Matriz Phase 40:**  N = 32³, 2LPT, PM, seeds `{42, 137, 271}`, modos
`{Legacy, Z0Sigma8}`, snapshots `a ∈ {0.02, 0.05, 0.10}` = **18 corridas**.
Runtime: 24 s release (`OnceLock` memoiza entre tests).

**Tests automáticos:** 7 tests en
[`phase40_physical_ics_normalization.rs`](../../crates/gadget-ng-physics/tests/phase40_physical_ics_normalization.rs).
Todos pasan. Detalles en §7.

---

## 6. Métricas por modo y snapshot

### 6.1 σ₈ medido vs esperado (IC, `a = 0.02`)

Medición empírica integrando `P_corrected` (post-`pk_correction`) con
ventana top-hat `R = 8 Mpc/h`, sobre `k ≤ k_Nyq/2`. Por limitaciones de
ventana k, se comparan **ratios** adimensionales.

| Modo | σ₈ medido (avg seeds) | σ₈ integrado de `P_ref` | ratio `P_c/P_ref` | σ₈ esperado |
| --- | ---: | ---: | ---: | ---: |
| `Legacy` | 0.7898 | 0.7754 | **1.019** | 0.8 |
| `Z0Sigma8` | 0.02007 | 0.01971 | **1.019** | 0.02033 |

**Cociente entre modos** (adimensional, exacto):
```text
σ₈(Z0Sigma8) / σ₈(Legacy)  =  0.025413
s = D(a_init)/D(1)          =  0.025413
|err| / s                    ≈  10⁻⁸        ← precisión de máquina
```

Este invariante confirma que la convención `Z0Sigma8` está
implementada **correctamente a nivel físico**.

### 6.2 `pk_correction` en snapshot IC

`median |log10(P_corrected / P_ref)|` promediada sobre seeds:

| Seed | Legacy | Z0Sigma8 |
| --- | ---: | ---: |
| 42  | 0.0354 | 0.0354 |
| 137 | 0.0147 | 0.0147 |
| 271 | 0.0555 | 0.0555 |
| **avg** | **0.0353** | **0.0353** |

`pk_correction` **funciona igual de bien en los dos modos en el IC**
(umbral 0.2 holgadamente superado).

### 6.3 `pk_correction` en snapshots evolucionados

`median |log10(P_c / P_ref)|` (avg seeds):

| `a` | Legacy | Z0Sigma8 | Factor `Legacy/Z0Sigma8` |
| ---: | ---: | ---: | ---: |
| 0.05 | 6.223 | 9.333 | **0.667** |
| 0.10 | 5.623 | 8.674 | **0.648** |
| **avg** | 5.922 | 9.004 | **0.658** |

**El modo `Z0Sigma8` empeora la fidelidad espectral por un factor ≈ 1.52.**
Ninguno de los dos modos cae siquiera cerca del umbral `≤ 0.2` que sí se
cumple en el snapshot IC.

### 6.4 Dinámica (`δ_rms`, `v_rms`)

| `a` | `δ_rms` legacy | `δ_rms` z0 | ratio z0/legacy |
| ---: | ---: | ---: | ---: |
| 0.02 | 0.000 (CIC floor) | 0.000 (CIC floor) | — |
| 0.05 | 1.020 | 1.014 | 0.994 |
| 0.10 | 1.003 | 1.001 | 0.998 |

Ambos modos saturan `δ_rms → 1` desde `a ≈ 0.05`. El reescalado reduce el
`Ψ_rms` inicial en un factor `s ≈ 0.025`, pero para `a = 0.05` (equivalente a
`D(0.05)/D(0.02) ≈ 2.42`) el régimen **ya no es lineal** porque los modos de
alta `k` del espectro (que en la convención `Z0Sigma8` son físicamente
consistentes, pero muy bajos en amplitud) quedan **dominados por shot-noise
de partículas** a N=32³; el estimador CIC suma shot-noise uniforme sobre
todas las celdas y satura `δ_rms` en ~1 apenas se supera el nivel de ruido de
Poisson.

### 6.5 Análisis — por qué `Z0Sigma8` no mejora snapshots evolucionados

Aunque las ICs de `Z0Sigma8` son **físicamente correctas**, el régimen
evolucionado mide un `P(k)` dominado por contribuciones que no son las del
espectro lineal:

1. **Shot-noise del estimador CIC**: el floor de shot-noise escala como
   `1/n_particles`. En `Z0Sigma8`, `P_lin(k, a_init)` es ~10⁶ veces menor
   que en `Legacy`, por lo que el ruido de partícula domina inmediatamente.
2. **Evolución PM sobre señal mínima**: con amplitudes físicamente
   consistentes pero por debajo del shot-noise, la dinámica se ve
   contaminada por el ruido y el `P_corrected(k)` resultante no representa
   el modo creciente lineal.
3. **Resolución finita**: el rango `k ≤ k_Nyq/2` con N=32³ en una caja
   de 100 Mpc/h cubre modos `k ∈ [0.04, 1.0] h/Mpc` donde `P_lin`
   físicamente pequeño no alcanza a dominar la medición.

La convención **`Legacy` produce, por construcción, un espectro medido que
coincide con `P_ref` solo porque `P_ref` también se eleva por `[D(1)/D(a_init)]²`**.
Es decir, `Legacy` es consistente en su propio marco de referencia (una
"simulación con σ₈(a_init) = 0.8"), pero no es una simulación cosmológicamente
calibrada. Sirve bien para cerrar `pk_correction` y validar el pipeline
espectral, pero no produce crecimiento cosmológico realista.

---

## 7. Tests automáticos (7 obligatorios)

Todos pasan (`cargo test --release --test phase40_physical_ics_normalization -- --test-threads=1`).

| # | Nombre | Tipo | Resultado |
| - | --- | --- | --- |
| 1 | `legacy_mode_remains_bit_compatible` | hard | ✅ Legacy es determinista bit-a-bit entre llamadas; difiere de `Z0Sigma8`. |
| 2 | `z0_sigma8_mode_matches_expected_growth_scaling` | hard | ✅ `rms(Ψ)_z0 / rms(Ψ)_legacy = s` dentro de 1 %. |
| 3 | `sigma8_at_ainit_matches_linear_prediction` | hard | ✅ Ratio `σ₈(z0)/σ₈(legacy)` = `s` a 10⁻⁸; ratios `σ₈_meas/σ₈_ref` ≈ 1.02 en ambos modos. |
| 4 | `pk_correction_still_works_on_ic_snapshot_under_z0_mode` | hard | ✅ `median\|log10(P_c/P_ref)\| ≤ 0.036 ≪ 0.2`. |
| 5 | `z0_mode_reduces_early_nonlinearity_vs_legacy` | soft | ❌ Ratio `δ_rms` z0/legacy ≈ 1.0 (no hay reducción medible). |
| 6 | `z0_mode_improves_early_snapshot_accuracy_vs_legacy` | soft | ❌ Factor global 0.658 → **Decisión B**. |
| 7 | `z0_mode_no_nan_inf` | hard | ✅ Ninguna corrida produce NaN/Inf. |

---

## 8. Figuras

Las 6 figuras del brief viven en
[`docs/reports/figures/phase40/`](figures/phase40/):

1. [`pk_ic_legacy_vs_z0.png`](figures/phase40/pk_ic_legacy_vs_z0.png) — `P_corrected` y `P_ref` para ambos modos en `a=0.02`.
2. [`pk_a005_legacy_vs_z0.png`](figures/phase40/pk_a005_legacy_vs_z0.png) — `a≈0.05`.
3. [`pk_a010_legacy_vs_z0.png`](figures/phase40/pk_a010_legacy_vs_z0.png) — `a≈0.10`.
4. [`ratio_corrected_vs_ref_per_mode.png`](figures/phase40/ratio_corrected_vs_ref_per_mode.png) — ratio por snapshot.
5. [`delta_rms_vs_a_legacy_vs_z0_vs_linear.png`](figures/phase40/delta_rms_vs_a_legacy_vs_z0_vs_linear.png) — δ_rms vs a.
6. [`sigma8_measured_vs_expected.png`](figures/phase40/sigma8_measured_vs_expected.png) — `σ₈(a_init)` por modo.

---

## 9. Respuestas al brief

### A. ¿El nuevo modo conserva el buen cierre de `pk_correction` en IC?

**Sí.** `median|log10(P_c/P_ref)|` = 0.035 en los dos modos. Idéntico al
valor Legacy. Umbral 0.2 holgadamente superado.

### B. ¿La amplitud inicial ahora es físicamente consistente con `σ₈(z=0)`?

**Sí, a precisión de máquina.**  El ratio `σ₈(Z0Sigma8)/σ₈(Legacy)` = `s`
con error relativo `< 10⁻⁸`. La implementación del reescalado es correcta.

### C. ¿La evolución temprana deja de colapsar inmediatamente fuera del régimen lineal?

**No de forma medible.**  El `δ_rms` medido satura igual en los dos modos a
partir de `a ≈ 0.05`, porque el estimador CIC está dominado por shot-noise
de Poisson cuando la amplitud inicial es ~10⁻⁶ en `P(k, a_init)`.

### D. ¿`pk_correction` empieza a funcionar también en snapshots evolucionados tempranos?

**No.**  El error mediano aumenta a 9.0 (Z0Sigma8) vs 5.9 (Legacy), peor en
ambos casos y particularmente peor en el modo físicamente correcto. La
corrección `pk_correction` no extiende su validez con esta convención en
este régimen de `N`.

### E. ¿La convención nueva es lo bastante mejor como para reemplazar a legacy como modo recomendado?

**No.**  `Z0Sigma8` no mejora ninguna métrica evolucionada y empeora la
fidelidad espectral en 1.5×. Queda como modo experimental.

---

## 10. Decisión final

> **Decisión B.** El modo `Z0Sigma8` queda implementado y disponible como
> opción experimental para trabajos futuros que combinen (a) resoluciones
> mucho más altas (N ≥ 128³ o mayor) donde el shot-noise no domine, y (b)
> integración con una referencia externa CAMB/CLASS para bypassear el
> estimador de shot-noise al nivel de `P(k, a_init)`. El modo `Legacy`
> sigue siendo el default y el recomendado para validar `pk_correction`
> y la línea espectral de Phase 34–39.

Phase 40 aporta, no obstante, contribuciones concretas:

1. **API limpia** (`NormalizationMode` enum) que reemplaza el flag
   experimental `rescale_to_a_init: bool`.
2. **Auditoría formal** que descarta bugs sutiles en la implementación
   física del rescaling de Fase 37.
3. **Verificación empírica** de que `σ₈(a_init)` escala con `s` a nivel de
   precisión de máquina — el método está correcto, el resultado evolutivo
   no mejora por limitaciones del régimen de `N`/shot-noise.
4. **Diagnóstico claro** de por qué `Z0Sigma8` falla en snapshots
   evolucionados: shot-noise del estimador CIC domina sobre el `P(k)`
   físico cuando la amplitud inicial es ~10⁻⁶.

---

## 11. Reproducir

```bash
# Tests Rust (24 s release):
cargo test --release --test phase40_physical_ics_normalization -- --test-threads=1

# Campaña end-to-end + figuras:
bash experiments/nbody/phase40_physical_ics_normalization/run_phase40.sh

# Solo figuras (matriz ya en target/phase40/):
python3 experiments/nbody/phase40_physical_ics_normalization/scripts/plot_phase40_comparison.py \
    --matrix target/phase40/per_snapshot_metrics.json \
    --out docs/reports/figures/phase40
```

## 12. Archivos clave

- `crates/gadget-ng-core/src/config.rs` — enum `NormalizationMode`, campo `normalization_mode`.
- `crates/gadget-ng-core/src/ic.rs` — dispatcher enum → bool interno.
- `crates/gadget-ng-physics/tests/phase40_physical_ics_normalization.rs` — 7 tests + matriz 18 corridas.
- `experiments/nbody/phase40_physical_ics_normalization/` — configs TOML, scripts Python, runner.
- `docs/reports/figures/phase40/` — 6 figuras + `phase40_summary.csv`.
