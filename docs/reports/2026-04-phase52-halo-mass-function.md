# Phase 52 — Función de masa de halos: Press-Schechter / Sheth-Tormen

**Fecha:** Abril 2026  
**Autor:** gadget-ng development

---

## Contexto

Las Phases 50–51 resolvieron la inconsistencia de unidades (`G_consistente`)
y la integraron en el motor de producción. Phase 52 implementa la primera
herramienta analítica de estadísticas de halos: la función de masa de halos
(HMF), que predice `dn/d ln M` —la densidad numérica comoving de halos por
intervalo logarítmico de masa.

La HMF es un observable cosmológico clave para:

- **Calibrar simulaciones** contra catálogos de cúmulos de galaxias.
- **Constrainar cosmología** (Ω_m, σ₈, w) con conteos de cúmulos.
- **Validar resultados FoF** de gadget-ng con predicciones analíticas.

---

## Física

### Varianza σ(M, z)

```
R(M) = (3M / 4πρ̄_m)^{1/3}                                [radio de Lagrange]

σ²(M, z=0) = (1/2π²) ∫₀^∞ k³ P_lin(k) W²(kR) dk         [integral top-hat]

σ(M, z) = σ(M, 0) × D(z)/D(0)                             [factor de crecimiento]
```

donde `W(x) = 3(sin x − x cos x)/x³` es el filtro esférico top-hat.

**Normalización**: `P_lin(k) = amp² × k^n_s × T²_EH(k)` con
`amp = sigma8 / sqrt((1/2π²)∫k^(n_s+3)T²W²dk)`. Esto asegura
`σ(R=8 Mpc/h) = σ₈` con error < 0.01 %.

### Press-Schechter (1974)

```
f_PS(σ) = √(2/π) × (δ_c/σ) × exp(−δ_c²/2σ²)     [δ_c = 1.686]

dn/d ln M = (ρ̄_m / M) × |d ln σ⁻¹ / d ln M| × f_PS(σ)
```

La derivada `d ln σ / d ln M = (1/3) d ln σ / d ln R` se calcula por
diferencias finitas centrales con eps=2%.

PS satisface `∫₀^∞ f dν = 1` (con factor 2 incluido), verificado numéricamente:
`∫ f_PS d(ln σ⁻¹) = 0.999` en σ ∈ [0.001, 1000].

### Sheth-Tormen (1999)

```
f_ST(σ) = A × √(2a/π) × ν × [1 + (aν²)^{-p}] × exp(−aν²/2)

     ν = δ_c/σ,   a = 0.707,   p = 0.3,   A = 0.3222
```

ST modela el colapso elipsoidal (en lugar de esférico de PS) y está
calibrado contra simulaciones N-body. `∫ f_ST d(ln σ⁻¹) = 0.953`.

---

## Implementación

### Archivos nuevos

| Archivo | Descripción |
|---------|-------------|
| `crates/gadget-ng-analysis/src/halo_mass_function.rs` | Módulo principal |
| `crates/gadget-ng-physics/tests/phase52_mass_function.rs` | 7 tests de integración |

### Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-analysis/src/lib.rs` | Export de `halo_mass_function` |
| `docs/roadmap.md` | Entrada Phase 52 |
| `docs/user-guide.md` | Sección de uso y API |

### API pública (`gadget_ng_analysis::halo_mass_function`)

```rust
pub struct HmfParams { omega_m, omega_lambda, h, sigma8, n_s, omega_b, t_cmb }
pub struct HmfBin { log10_m, m_msun_h, r_hmpc, sigma, dlns_inv_dlnm, n_ps, n_st }

pub fn sigma_m(m_msun_h: f64, params: &HmfParams, z: f64) -> f64
pub fn lagrange_radius(m: f64, rho_bar: f64) -> f64
pub fn multiplicity_ps(sigma: f64) -> f64
pub fn multiplicity_st(sigma: f64) -> f64
pub fn hmf_press_schechter(m, sigma, dlns_inv, rho_bar) -> f64
pub fn hmf_sheth_tormen(m, sigma, dlns_inv, rho_bar) -> f64
pub fn mass_function_table(params, m_min, m_max, n_bins, z) -> Vec<HmfBin>
pub fn total_halo_density(table: &[HmfBin]) -> (f64, f64)

pub const RHO_CRIT_H2: f64 = 2.775e11;   // (M_sun/h)/(Mpc/h)³
pub const DELTA_C: f64 = 1.686;
```

---

## Resultados

### Tabla HMF z=0 (Planck 2018)

| log₁₀(M [M_sun/h]) | σ(M)   | dn/d ln M — PS [h³/Mpc³] | dn/d ln M — ST [h³/Mpc³] |
|---------------------|--------|--------------------------|--------------------------|
| 10.0                | 3.745  | 3.012×10⁻¹               | 2.346×10⁻¹               |
| 11.0                | 2.843  | 3.919×10⁻²               | 2.815×10⁻²               |
| 12.0                | 2.054  | 4.992×10⁻³               | 3.357×10⁻³               |
| 13.0                | 1.389  | 5.469×10⁻⁴               | 3.655×10⁻⁴               |
| 14.2                | 0.857  | 3.037×10⁻⁵               | 2.522×10⁻⁵               |

### Abundancia de cúmulos de galaxias

```
n(>10¹³ M_sun/h) :  PS = 6.676×10⁻⁴  ST = 4.622×10⁻⁴  [h³/Mpc³]
n(>10¹⁴ M_sun/h) :  PS = 3.234×10⁻⁵  ST = 2.844×10⁻⁵  [h³/Mpc³]
```

El valor n(>10¹⁴) ≈ 3×10⁻⁵ h³/Mpc³ es coherente con los observados en
catálogos ACT-DR5, SPT-SZ y eROSITA para cosmología Planck 2018.

### Evolución con redshift

```
σ(10¹⁴ M_sun/h):   z=0 → 0.933    z=1 → 0.567    z=3 → 0.295
f_PS(10¹⁴):        z=0 → 2.818e-1  z=1 → 2.852e-2  ratio ≈ 0.10
D(z=1)/D(0) = 0.608
```

La reducción de ~10× en f_PS de z=0 a z=1 para M=10¹⁴ muestra la
formación jerárquica: los cúmulos masivos se forman mucho más tarde.

---

## Tests

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `phase52_sigma_normalization` | σ(R=8)=σ₈ < 0.01% error | ✓ |
| `phase52_sigma_profile` | σ(M) monótono, rangos físicos | ✓ |
| `phase52_multiplicity_normalization` | ∫f_PS≈0.999, ∫f_ST≈0.953 | ✓ |
| `phase52_hmf_table_z0` | Tabla coherente, 25 bins | ✓ |
| `phase52_hmf_redshift_evolution` | σ(z=0)>σ(z=1)>σ(z=3) | ✓ |
| `phase52_hmf_cluster_abundance` | n(>10¹⁴)∈[1e-8,1e-2] h³/Mpc³ | ✓ |
| `phase52_fof_vs_hmf_qualitative` | ICs Zel'dovich + FoF + HMF | ✓ |

**Tiempo total de tests**: < 0.05 s (debug, todos analíticos salvo test 7).

---

## Limitaciones conocidas

1. **Transfer function**: Se usa Eisenstein-Hu no-wiggle. Para mayor precisión
   se puede integrar CAMB/CLASS pero requiere dependencias externas.

2. **Factor de crecimiento**: Aproximación Carroll, Press & Turner (1992),
   precisa al ~1 % para ΛCDM. Para w≠−1 se necesita integración numérica.

3. **Comparación FoF cuantitativa**: Requiere N ≥ 128³ y L ≥ 100 Mpc/h para
   estadística suficiente. El test 7 solo verifica coherencia cualitativa.

4. **Bariones**: La HMF implementada es CDM puro. Los efectos de bariones
   (feedback AGN, feedback estelar) pueden modificar la HMF en ~10-30 %.

---

## Próximos pasos recomendados

- **Phase 53A**: Validación D²(a) cuantitativa con `auto_g=true` y N=64 en release.
- **Phase 53B**: Perfiles de halos NFW y relación c(M).
- **Phase 53C**: Comparación estadística FoF vs HMF con cajas grandes (release).
