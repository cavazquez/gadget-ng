# Phase 58 — c(M) y perfiles NFW desde N-body + función de correlación ξ(r)

**Fecha:** abril 2026  
**Crate principal:** `gadget-ng-analysis`  
**Archivo de test:** `crates/gadget-ng-physics/tests/phase58_nfw_concentration.rs`

---

## Contexto

Phase 53 implementó el ajuste NFW y la relación c(M) de Duffy+2008 y Bhattacharya+2013.
Phase 58 añade:

1. La relación c(M) de **Ludlow et al. 2016**, calibrada sobre simulaciones Planck2015 ΛCDM.
2. La **función de correlación de 2 puntos ξ(r)** mediante dos métodos independientes.

---

## Física

### Relación c(M) de Ludlow+2016

Forma de potencia doble calibrada sobre Millennium-XXL + Bolshoi-Planck:

```
c_200(M, z) = 3.395 × (M / 10¹⁴ M☉/h)^{−0.215} × (1+z)^{−0.642}
```

A z=0 predice c ~ 5–7 para halos de grupo/clúster. Difiere de Duffy+2008 en
un factor ~0.6–2.0 dependiendo de la masa, con mejor acuerdo a masas intermedias.

| M [M☉/h] | c(Duffy) | c(Ludlow) | ratio |
|----------|----------|-----------|-------|
| 10¹¹     | 7.34     | 14.99     | 2.04  |
| 10¹²     | 6.05     | 9.14      | 1.51  |
| 10¹³     | 4.99     | 5.57      | 1.12  |
| 10¹⁴     | 4.11     | 3.40      | 0.83  |
| 10¹⁵     | 3.39     | 2.07      | 0.61  |

### Función de correlación de 2 puntos ξ(r)

**Método FFT** — transformada de Hankel discreta desde P(k):

```
ξ(r) = (1/2π²) Σ_k  k² P(k) sinc(k·r) Δk
```

Complejidad O(N_k × N_r), válida para cajas periódicas.

**Método de conteo de pares** — estimador de Davis-Peebles:

```
ξ(r) = DD/RR − 1
```

con RR analítico para distribución uniforme en caja periódica.
Complejidad O(N²); apto para N < 10⁴.

---

## Implementación

Nuevo archivo `crates/gadget-ng-analysis/src/correlation.rs`:

```rust
pub struct XiBin { pub r: f64, pub xi: f64, pub n_pairs: u64 }

pub fn two_point_correlation_fft(pk: &[PkBin], box_size: f64, n_r_bins: usize) -> Vec<XiBin>
pub fn two_point_correlation_pairs(
    positions: &[Vec3], box_size: f64, r_min: f64, r_max: f64, n_bins: usize
) -> Vec<XiBin>
```

Expuestas en `lib.rs` junto con `concentration_ludlow2016`.

---

## Tests de integración (phase58)

| Test | Descripción |
|------|-------------|
| `phase58_concentration_vs_theory` | Simulación N=32³ → FoF → NFW fit; c_fit/c_duffy ∈ [0.1, 10] |
| `phase58_xi_fft_finite` | ξ(r) via FFT produce valores finitos |
| `phase58_xi_pairs_finite` | ξ(r) via pares produce valores finitos |
| `phase58_ludlow_vs_duffy_range` | Ludlow/Duffy ∈ [0.3, 3.5] para M ∈ [10¹¹, 10¹⁵] |

Controlado con `PHASE58_SKIP=1`.
