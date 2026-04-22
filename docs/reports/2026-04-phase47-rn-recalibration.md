# Phase 47 — Recalibración R(N) y corrección P(k) post-Phase 45

**Fecha:** 2026-04-22
**Autor:** gadget-ng development log
**Estado:** ✅ Completa — compilación limpia, 8 tests automáticos pasan.

---

## Pregunta central

> Phase 35 calibró `R(N)` para `N ∈ {8, 16, 32, 64}`. Phase 45 corrigió el
> acoplamiento gravitacional de `G/a` a `G·a³` (QKSL). ¿Invalida ese cambio
> la calibración existente? ¿Y cómo se extiende R(N) a grids grandes (N=128)?

**Respuesta:**
- R(N) no se invalida por Phase 45: captura la respuesta CIC+discreción a t=0
  (antes de ninguna integración gravitacional), que es independiente de la física.
- R(N=128) se mide por primera vez: `R(128) = 0.002252` (campaña 4 seeds), con
  el fit {32,64,128} dando `α=1.953` (ligeramente diferente de `α=1.871` de Phase 35
  sobre {8,16,32,64}).
- El pipeline P(k) corregido con el acoplamiento QKSL correcto reproduce la teoría
  lineal con `error < 2%` en evoluciones cortas (test de humo).

---

## Contexto previo

| Phase | Logro |
|-------|-------|
| 34 | A_grid(N) = 2V²/N⁹ demostrado analíticamente |
| 35 | R(N) calibrado para N ∈ {8,16,32,64}, fit C·N^{-α} |
| 36 | Validación P(k) corregido ≈ P_lineal en régimen lineal |
| 45 | Corrección del acoplamiento QKSL: g_cosmo = G·a³ en lugar de G/a |

Phase 47 cierra el ciclo: extiende la tabla R(N) a N=128 y valida el pipeline
de corrección con la física QKSL correcta.

---

## ¿Por qué R(N) no se invalida por Phase 45?

R(N) es el cociente:

```text
R(N) = mean_k[ P_measured(k, t=0) / (A_grid(N) · P_cont(k)) ]
```

Se mide sobre ICs ZA **sin evolución** (cero pasos de integración). La física
gravitacional nunca interviene en esta medición. Phase 45 corrigió la integración
temporal, no el campo de densidad en t=0. Por tanto:

> Los valores R(N=32)=0.033752 y R(N=64)=0.008834 de Phase 35 son **idénticos**
> en la campaña Phase 47 (mismas semillas, mismo setup).

---

## Campaña de medición N=128

**Parámetros** (idénticos a Phase 35 para comparabilidad):

| Parámetro | Valor |
|-----------|-------|
| Box_size interno | 1.0 |
| Box_mpc_h | 100.0 |
| σ₈ | 0.8 |
| n_s | 0.965 |
| Ω_m, Ω_b, h | 0.315, 0.049, 0.674 |
| Seeds | {42, 137, 271, 314} |
| N³ partículas | 128³ = 2 097 152 |

**Resultados:**

| N | R(N) medido | CV | R(N) fit Phase35 | δ vs fit |
|---|-------------|-----|------------------|----------|
| 32 | 0.033752 | 3.6% | 0.033752 | 0% (tabla) |
| 64 | 0.008834 | 1.4% | 0.008834 | 0% (tabla) |
| 128 | **0.002252** | **1.0%** | 0.002518 | **10.6%** |

El fit Phase 35 sobreestima R(128) un 10.6 %. Esto es consistente con que el
fit se realizó sobre {8,16,32,64} y extrapola a N=128. El exponente α medido
sobre {32,64,128} es **1.953** vs 1.871 de Phase 35 sobre {8,16,32,64}.

### Interpretación física

R(N) = C·N^{-α} refleja que la dispersión por aliasing CIC decrece más rápido
que lo que predice la extrapolación de N pequeños. El CV del 1.0% a N=128 (vs
22.9% a N=8 en Phase 35) confirma la reducción de ruido estadístico con
partículas más densas.

---

## API añadida: `pk_correction.rs`

### `measure_rn()`

```rust
pub fn measure_rn(
    n: usize,
    seeds: &[u64],
    box_size: f64,
    box_mpc_h: f64,
    sigma8: f64,
    n_s: f64,
    eh: &EisensteinHuParams,
) -> (f64, f64)   // (r_mean, cv)
```

Genera ICs ZA in-process, mide P_measured con `power_spectrum`, y devuelve el
R(N) promediado sobre seeds y bins k≤k_Nyq/2. Permite recalibrar R(N) para
cualquier N sin salir del workspace Rust.

### `correct_pk_with_shot_noise()`

```rust
pub fn correct_pk_with_shot_noise(
    pk_bins: &[PkBin],
    box_size_internal: f64,
    n: usize,
    box_mpc_h: Option<f64>,
    n_particles: usize,
    model: &RnModel,
) -> Vec<PkBin>
```

Sustrae el ruido de Poisson `P_shot = V/N_part` antes de aplicar la
corrección `A_grid·R(N)`. Bins donde `P_measured ≤ P_shot` quedan en cero
(no negativos).

### `RnModel::phase47_default()`

```rust
pub fn phase47_default() -> Self {
    Self {
        c: 29.77,
        alpha: 1.953,  // fit OLS log-log sobre {32,64,128}
        table: vec![
            (8,  0.415381..),  // idéntico a Phase 35
            (16, 0.139628..),  // idéntico a Phase 35
            (32, 0.033752..),  // idéntico a Phase 35
            (64, 0.008834..),  // idéntico a Phase 35
            (128, 0.002252),   // Phase 47, medición directa
        ],
        ..
    }
}
```

---

## Convención de unidades en `correct_pk`

**Importante:** `R(N)` fue calibrado comparando P_measured (interno, box=1) con
P_cont en **(Mpc/h)³**. La corrección ya absorbe el factor de volumen. Por tanto:

```rust
// CORRECTO: box_mpc_h=None devuelve P en (Mpc/h)³
let pk_phys = correct_pk(&pk_raw, 1.0, n, None, &model);

// INCORRECTO: box_mpc_h=Some(100) aplicaría 100³ extra
// let pk_phys = correct_pk(&pk_raw, 1.0, n, Some(100.0), &model);  // WRONG
```

Este comportamiento está documentado en Phase 36 §2 y confirmado en Phase 47.

---

## Validación end-to-end con física QKSL

Tres tests validan el pipeline completo:

| Test | Setup | Resultado |
|------|-------|-----------|
| `phase47_pk_correction_at_ics` | ICs ZA Legacy (z=0 amp), N=32 | err mediana = **8.2%** vs P_EH(z=0) |
| `phase47_pk_growth_qksl` | PM + G·a³, a=0.02→0.022, N=32 | err crecimiento = **1.4%** vs [D/D₀]² |
| `shot_noise_correction_is_proportional` | Sintético P=2·P_shot | ratio = **0.5000** exacto |

El 8.2% en ICs refleja la dispersión estadística de Phase 35 (R² = 0.997, una seed).
El 1.4% en crecimiento confirma que QKSL preserva el régimen lineal correctamente.

---

## Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-analysis/src/pk_correction.rs` | +`measure_rn()`, +`correct_pk_with_shot_noise()`, +`RnModel::phase47_default()` |
| `crates/gadget-ng-analysis/src/lib.rs` | Re-export de nuevas funciones |
| `crates/gadget-ng-physics/tests/phase47_rn_calibration.rs` | Campaña N∈{32,64,128}, 3 tests |
| `crates/gadget-ng-physics/tests/phase47_pk_evolution.rs` | Validación end-to-end, 3 tests |

---

## Sucesores sugeridos

| Issue | Dificultad | Impacto |
|-------|-----------|---------|
| R(N=256,512) con `measure_rn` | Media | Simulaciones de alta resolución |
| Corrección TSC/PCS (kernels de orden superior) | Alta | Precisión en k alto |
| P(k) a escala no-lineal: comparación con Halofit | Alta | Física más allá del lineal |
