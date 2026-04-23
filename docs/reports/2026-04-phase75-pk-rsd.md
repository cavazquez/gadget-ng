# Phase 75 — P(k,μ) + Multipoles P₀/P₂/P₄ en Espacio de Redshift

**Crate**: `gadget-ng-analysis/src/pk_rsd.rs`  
**Fecha**: 2026-04

## Resumen

Implementación del espectro de potencia en espacio de redshift P(k,μ) y sus
multipoles de Legendre P₀/P₂/P₄, integrando el efecto RSD (Kaiser 1987) mediante
desplazamiento de posiciones a lo largo de la línea de visión.

## Algoritmo

El método sigue Hamilton (1992) y Kaiser (1987):

1. **Desplazamiento RSD**: `s = r + v_los / (a × H(a)) × ê_los`
2. **CIC deposit** del campo desplazado en un grid de resolución `n_mesh`.
3. **FFT 3D** + deconvolución CIC (corrección de ventana sinc).
4. **Binning (k, μ)**: para cada modo, `k = |k|`, `μ = k_los/|k|`.
5. **Multipoles**: integración numérica sobre μ con polinomios de Legendre:
   - P₀(k) = ⟨P(k,μ)⟩ — monopolo
   - P₂(k) = (5/2) ⟨P(k,μ) × L₂(μ)⟩ — cuadrupolo (efecto Kaiser)
   - P₄(k) = (9/2) ⟨P(k,μ) × L₄(μ)⟩ — hexadecapolo

## Física

En el régimen lineal (Kaiser 1987), los ratios de los multipoles satisfacen:

```
P₀/P_lin = 1 + 2β/3 + β²/5
P₂/P_lin = 4β/3 + 4β²/7
P₄/P_lin = 8β²/35
```

donde `β = f/b`, `f = Ω_m^0.55` es la tasa de crecimiento lineal y `b` el bias de galaxias.

## Nuevos tipos

```rust
pub struct PkRsdBin   { k: f64, mu: f64, pk: f64, n_modes: u64 }
pub struct PkMultipoleBin { k: f64, p0: f64, p2: f64, p4: f64, n_modes: u64 }
pub struct PkRsdParams  { n_k_bins, n_mu_bins, los: LosAxis, scale_factor, hubble_a }
pub enum   LosAxis      { X, Y, Z }
```

## Funciones exportadas

| Función | Descripción |
|---|---|
| `pk_redshift_space(...)` | P(k,μ) completo en espacio de redshift |
| `pk_multipoles(...)` | Multipoles P₀/P₂/P₄ desde P(k,μ) |
| `compute_pk_multipoles(...)` | Combinado: posiciones → multipoles |
| `kaiser_multipole_ratios(β)` | Ratios teóricos Kaiser (validación) |

## Integración in-situ

Se extiende `InsituAnalysisSection` con `pk_rsd_bins: usize` y se agrega al
`insitu_NNNNNN.json`:

```json
{
  "pk_rsd":       [{ "k": ..., "mu": ..., "pk": ... }],
  "pk_multipoles": [{ "k": ..., "p0": ..., "p2": ..., "p4": ... }]
}
```

## Tests

7 tests unitarios en `pk_rsd::tests`:
- `pk_rsd_zero_vel_equals_real_space`
- `pk_rsd_bins_have_increasing_k`
- `pk_multipoles_p0_positive`
- `kaiser_ratios_beta_zero` (β=0 → P₀=P_lin, P₂=P₄=0)
- `kaiser_ratios_beta_positive` (β>0 → P₂>0)
- `pk_rsd_los_axes_different`
- `pk_multipole_bin_serializes`
