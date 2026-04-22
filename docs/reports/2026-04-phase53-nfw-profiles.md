# Phase 53 — Perfiles de halos NFW y relación c(M)

**Fecha:** Abril 2026  
**Autor:** gadget-ng development

---

## Contexto

Phase 52 implementó la función de masa de halos (HMF) analítica. Phase 53
añade la infraestructura para describir la **estructura interna** de los halos
mediante el perfil NFW (Navarro, Frenk & White 1996/1997) y la relación
concentración-masa c(M, z) de Duffy et al. (2008).

---

## Física implementada

### Perfil NFW

```
ρ(r) = ρ_s / [(r/r_s)(1 + r/r_s)²]
```

Propiedades analíticas verificadas:
- **Pendiente interna**: ρ ∝ r⁻¹ para r ≪ r_s  
- **Pendiente de escala**: ρ ∝ r⁻² en r = r_s  
- **Pendiente externa**: ρ ∝ r⁻³ para r ≫ r_s  

### Masa encerrada

```
M(<r) = 4π ρ_s r_s³ g(r/r_s)
g(x) = ln(1+x) − x/(1+x)
```

La función `g(x)` usa una expansión de Taylor para x < 10⁻⁴ para evitar
cancelación numérica, verificada en x = 10⁻⁸ con error relativo < 10⁻⁶.

### Radio y densidad virial

```
r_200 = (3 M_200 / 4π × 200 × ρ_crit)^{1/3}
ρ_s = (200/3) × ρ_crit × c³ / g(c)
```

Verificado exactamente: M(<r_200) = M_200 y ρ_mean(<r_200) = 200 ρ_crit con
error relativo < 10⁻¹⁰.

### Relación c(M, z) de Duffy et al. (2008)

```
c_200(M, z) = 5.71 × (M / 2×10¹² M_sun/h)^{-0.084} × (1+z)^{-0.47}
```

Calibrada sobre simulaciones N-body con WMAP5 para halos "all" (relaxed + unrelaxed).
Válida para M ∈ [10¹¹, 10¹⁵] M_sun/h y z ∈ [0, 2].

### Velocidad circular máxima

```
v_c(r) = sqrt(G M(<r) / r)
```

El máximo de v_c se da en r = 2.163 r_s (verificado numéricamente, error < 0.3 r_s).

---

## Resultados

### Tabla de propiedades NFW (Planck 2018, z=0)

| log₁₀(M) | c     | r₂₀₀ [Mpc/h] | r_s [Mpc/h] | ρ_s [(M_sun/h)/(Mpc/h)³] |
|-----------|-------|--------------|-------------|--------------------------|
| 8.0       | 13.12 | 0.0075       | 5.8×10⁻⁴    | 2.4×10¹⁶                 |
| 10.0      | 8.91  | 0.0350       | 3.9×10⁻³    | 9.4×10¹⁵                 |
| 12.0      | 6.05  | 0.1626       | 0.0269      | 3.7×10¹⁵                 |
| 14.0      | 4.11  | 0.7549       | 0.1836      | 1.6×10¹⁵                 |
| 15.0      | 3.39  | 1.6263       | 0.4800      | 1.0×10¹⁵                 |

### Evolución de c con z

```
c(10¹³ M_sun/h):   z=0 → 4.99   z=1 → 3.60   z=2 → 2.98
```

### Ajuste de concentración desde datos sintéticos

Con 30000 partículas muestreadas de un perfil NFW con c_true = 5.0:
- c_fit = 5.27 (error < 6%)
- χ²_red = 0.002 (excelente ajuste log-space)

---

## Implementación

### Archivos nuevos

| Archivo | Descripción |
|---------|-------------|
| `crates/gadget-ng-analysis/src/nfw.rs` | Módulo NFW completo |
| `crates/gadget-ng-physics/tests/phase53_nfw_profiles.rs` | 6 tests de integración |

### Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-analysis/src/lib.rs` | Export de `nfw` |
| `docs/roadmap.md` | Entrada Phase 53 |
| `docs/user-guide.md` | Sección NFW |

### API pública

```rust
pub struct NfwProfile { pub rho_s: f64, pub r_s: f64 }

impl NfwProfile {
    pub fn from_m200_c(m200, c, rho_crit) -> Self
    pub fn density(r) -> f64
    pub fn mass_enclosed(r) -> f64
    pub fn r200(rho_crit) -> f64
    pub fn concentration(rho_crit) -> f64
    pub fn circular_velocity_sq_over_g(r) -> f64
}

pub fn r200_from_m200(m200, rho_crit) -> f64
pub fn rho_crit_z(omega_m, omega_lambda, z) -> f64
pub fn concentration_duffy2008(m200, z) -> f64
pub fn concentration_bhattacharya2013(m200, z) -> f64
pub fn measure_density_profile(radii, m_part, r_min, r_max, n_bins, profile?) -> Vec<DensityBin>
pub fn fit_nfw_concentration(bins, m200, rho_crit, c_min, c_max, n_c) -> Option<NfwFitResult>
```

---

## Tests

| Test | Descripción | Resultado |
|------|-------------|-----------|
| `phase53_nfw_virial_properties` | M(<r_200)=M_200 exacto, ρ_mean=200ρ_crit | ✓ |
| `phase53_concentration_mass_relation` | c decrece con M y z, valores Duffy+2008 | ✓ |
| `phase53_nfw_profile_shapes` | pendiente -1/-2/-3 verificada | ✓ |
| `phase53_mass_concentration_table` | tabla 10⁸–10¹⁵ M_sun/h coherente | ✓ |
| `phase53_circular_velocity_peak` | v_c_max en r/r_s = 2.163 ± 0.3 | ✓ |
| `phase53_density_profile_from_fof_halo` | infraestructura FoF + perfil | ✓ |
| Unitarios en `nfw.rs` | 8 tests: g_nfw, masas, pendientes, ajuste, sampling | ✓ |

**Total: 14 tests (6 integración + 8 unitarios), todos pasan.**  
**Tiempo: < 0.15 s en debug.**

---

## Limitaciones

1. **c(M) solo WMAP5**: Duffy+2008 fue calibrada con WMAP5 (σ₈=0.817). Para
   Planck 2018 (σ₈=0.811) existen relaciones más modernas como Diemer & Joyce (2019).

2. **Sin triaxialidad**: el perfil NFW esférico es una idealización. Halos reales
   son triaxiales (eje mayor ≈ 1.5× eje menor).

3. **Ajuste solo en c**: el método `fit_nfw_concentration` fija r_200 desde la masa
   FoF y varía solo c. Para datos ruidosos podría requerir fit conjunto (ρ_s, r_s).

4. **FoF vs NFW**: el radio FoF con b=0.2 corresponde aproximadamente a M_200
   (sobredensidad 200× la media), pero la correspondencia exacta depende de c.

---

## Próximos pasos

- **Phase 54A**: Triaxialidad y formas de halos (tensor de inercia).
- **Phase 54B**: Comparación estadística FoF vs c(M) con simulaciones completas.
- **Phase 54C**: Perfiles de velocidad (dispersión σ_r(r), anisotropía β).
