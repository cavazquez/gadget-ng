# Physics Roadmap — gadget-ng

> **Actualizado:** 2026-04-23 (post Phase 113 — Capa 1 bariónica completada)
>
> Este documento cataloga la física ya implementada, evalúa su nivel de completitud respecto a los
> códigos de referencia (GADGET-4, AREPO/IllustrisNG, EAGLE), y proyecta qué nuevos módulos se
> pueden agregar y a qué costo.

---

## 1. Inventario de física implementada (Phases 1–113)

### 1.1 Gravedad

| Módulo | Estado | Referencia |
|--------|--------|------------|
| N-cuerpos directo O(N²) | Completo | Phase 1 |
| Barnes-Hut árbol + FMM octupolo | Completo | Phases 2, 44 |
| PM (Particle-Mesh) periódico | Completo | Phases 5–6 |
| TreePM (fuerzas cortas + largas) | Completo | Phase 23 |
| SFC domain decomposition | Completo | Phase 56 |
| Block timesteps jerárquicos | Completo | Phase 56 |
| Softening comóvil → físico | Completo | Phase 101 |
| Gravedad MPI distribuida (LET) | Completo | Phase 61 |

**Nivel:** comparable con GADGET-4 en funcionalidad de gravedad.

### 1.2 Hidrodinámica SPH

| Módulo | Estado | Parámetros config |
|--------|--------|------------------|
| SPH cosmológico (leapfrog KDK) | Completo | `sph.enabled`, `gamma`, `alpha_visc` |
| Kernel Wendland C2 3D | Completo | `n_neigh` |
| Densidad adaptativa (Newton-Raphson) | Completo | `n_neigh` |
| Fuerzas SPH simétricas (Springel & Hernquist 2002) | Completo | `alpha_visc` |
| EOS adiabática P=(γ-1)ρu | Completo | `gamma` |
| Integrador KDK con factores cosmológicos | Completo | |
| Enfriamiento atómico H/He (Katz 1996) | Básico | `cooling = "atomic_hhe"`, `t_floor_k` |

### 1.3 Formación estelar y feedback

| Módulo | Estado | Modelo | Parámetros |
|--------|--------|--------|------------|
| SFR por umbral de densidad (Schmidt-Kennicutt) | Completo | Springel & Hernquist (2003) | `rho_sf`, `sfr_min` |
| Feedback SN estocástico (kick + energía térmica) | Completo | Dalla Vecchia & Schaye (2012) | `eps_sn`, `v_kick_km_s` |
| Vientos galácticos (factor de carga η) | Completo | Springel & Hernquist (2003) | `wind.v_wind_km_s`, `wind.mass_loading` |
| **Enriquecimiento metálico SPH (SN II + AGB)** | **✅ Phase 110** | Woosley & Weaver (1995) | `enrichment.yield_snii`, `yield_agb` |
| **Enfriamiento por metales** | **✅ Phase 111** | Sutherland & Dopita (1993) | `cooling = "metal_cooling"` |
| **Partículas estelares reales (spawning)** | **✅ Phase 112** | Springel & Hernquist (2003) | `m_star_fraction`, `m_gas_min` |
| **SN Ia con DTD power-law** | **✅ Phase 113** | Maoz & Mannucci (2012) | `a_ia`, `t_ia_min_gyr`, `e_ia_code` |

### 1.4 Agujeros negros (AGN)

| Módulo | Estado | Modelo | Parámetros |
|--------|--------|--------|------------|
| Acreción de Bondi-Hoyle | Completo | Springel (2005) | `agn.m_seed`, `agn.eps_feedback` |
| Crecimiento del BH (Ṁ → masa) | Completo | | |
| Feedback térmico + cinético (jets) | Completo | | `agn.v_kick_agn`, `agn.r_influence` |
| Semillas FoF en centros de halos | Completo | Phase 100 | `agn.n_agn_bh` |
| Persistencia en checkpoint | Completo | Phase 106 | |

### 1.5 Transferencia radiativa y química

| Módulo | Estado | Modelo | Parámetros |
|--------|--------|--------|------------|
| RT M1 (closure de Eddington) | Completo | González et al. (2007) | `rt.c_red_factor`, `kappa_abs` |
| Fotoionización + fotocalentamiento | Completo | | |
| Acoplamiento radiación-gas | Completo | | |
| Química H/He no-equilibrio (6 especies) | Completo | Anninos et al. (1997) | |
| Reionización EoR con fuentes UV puntuales | Completo | Phase 87 | `reionization.*` |
| Temperatura IGM T(z) | Completo | Phase 88 | |
| Estadísticas 21cm (δT_b, P(k)) | Completo | Phase 97 | |
| RT MPI distribuida | Completo | Phase 81 | |

### 1.6 Análisis y I/O

| Módulo | Estado |
|--------|--------|
| FoF halo finder + membresía | Completo |
| Merger trees con matching real | Completo (Phase 107) |
| P(k), ξ(r), bispectrum, HMF | Completo |
| Perfiles NFW, c(M), spin λ | Completo |
| HDF5 GADGET-4 (PartType0/1) | Completo (Phase 102) |
| JSONL snapshots con campos SPH | Completo (Phase 105) |
| Checkpoint/restart con estado completo | Completo (Phase 106) |
| CLI analyze (cm21, igm_temp, agn_stats, eor) | Completo (Phase 104) |

---

## 2. Comparación con códigos de referencia

```
gadget-ng vs GADGET-4 / AREPO (IllustrisNG) / EAGLE
────────────────────────────────────────────────────────────────
Gravedad           ████████████  Comparable
SPH básico         ████████████  Comparable
Enfriamiento       ████░░░░░░░░  Básico (H/He atómico)
Metales            ░░░░░░░░░░░░  No implementado
SFR + SN           ████████░░░░  Funcional, sin IMF detallada
Partículas estelares░░░░░░░░░░░  No implementado
AGN                ████████░░░░  Funcional, sin modo radio
RT / Reionización  ████████████  Comparable o superior
21cm               ████████████  Comparable
────────────────────────────────────────────────────────────────
```

---

## 3. Física nueva disponible — catálogo por capas

Las capas están ordenadas por **impacto físico y dependencia**. Cada módulo en la Capa N puede
depender de los de la Capa N-1.

---

### Capa 1 — Bariónica faltante (mayor impacto, base para todo lo demás)

> **✅ CAPA 1 COMPLETADA** (Phases 109–113, 2026-04-23)

Estas son las piezas que mayor diferencia hacen en la predicción de propiedades de galaxias.

#### 1A. Metales y enriquecimiento químico ✅ Phase 109-110

**Estado**: **IMPLEMENTADO**

- `Particle::metallicity: f64` y `Particle::stellar_age: f64` (`#[serde(default)]`).
- `ParticleType::Star` — nueva variante (gravedad sí, SPH no).
- `Particle::new_star(...)` y `Particle::is_star()`.
- `EnrichmentSection` con `yield_snii = 0.02`, `yield_agb = 0.04`.
- `apply_enrichment(particles, sfr, dt, cfg)` en `enrichment.rs` — distribución SPH kernel.
- 9 + 6 tests.

**Referencias**: Woosley & Weaver (1995); Wiersma et al. (2009).

---

#### 1B. Enfriamiento por metales ✅ Phase 111

**Estado**: **IMPLEMENTADO** (fitting analítico; tablas completas en Capa 2)

- `CoolingKind::MetalCooling` — nueva variante.
- `cooling_rate_metal(u, rho, Z, γ, T_floor)` — Sutherland & Dopita (1993) fitting analítico.
- `Λ(T,Z) = Λ_HHe(T) + (Z/Z_sun) × Λ_metal(T)` con tres regímenes de T.
- Despacho automático en `apply_cooling`.
- 6 tests.

**Siguiente paso (Capa 2)**: tablas completas `Λ(n_H, T, Z, z)` de Wiersma+09 o CLOUDY.

**Referencias**: Sutherland & Dopita (1993) ApJS 88, 253.

---

#### 1C. Partículas estelares reales (stellar spawning) ✅ Phase 112

**Estado**: **IMPLEMENTADO**

- `ParticleType::Star` con `metallicity` y `stellar_age`.
- `spawn_star_particles(particles, sfr, dt, seed, cfg, next_gid) -> (Vec<Particle>, Vec<usize>)`.
- Estrellas heredan metalicidad y no participan en SPH.
- Gas padre pierde masa; gas residual < `m_gas_min` → eliminado.
- Integrado en `engine.rs` (macro `maybe_sph!`).
- 7 tests.

---

#### 1D. Modelo ISM multifase frío-caliente

**Estado**: PENDIENTE (Capa 2)

**Descripción**: El ISM real coexiste en fases frías (T~100 K, nubes moleculares) y calientes
(T~10⁷ K, gas difuso). El modelo de Springel & Hernquist (2003) parametriza esto con una
"presión efectiva" Q*(ρ) que evita inestabilidades numéricas.

**Impacto**: Mejora la relación Kennicutt-Schmidt y la morfología del disco galáctico.

**Costo estimado**: 2–3 sesiones

**Cambios principales**:
- Campo `u_cold: f64` en `GasData` para la componente fría.
- Presión efectiva: `effective_pressure(rho, u, q_star) -> f64`.
- Parámetro `q_star: f64` en `SphSection`.

**Referencias**: Springel & Hernquist (2003) MNRAS 339, 289.

---

#### 1E. SN tipo Ia con distribución de retraso temporal (DTD) ✅ Phase 113

**Estado**: **IMPLEMENTADO**

- `apply_snia_feedback(particles, dt_gyr, seed, cfg)` con DTD `R ∝ t^{-1}`.
- `advance_stellar_ages(particles, dt_gyr)` — incrementa edad estelar cada paso.
- Parámetros: `a_ia = 2e-3`, `t_ia_min_gyr = 0.1`, `e_ia_code`.
- Distribución de Fe (~0.002 M_sun/SN Ia) a vecinos de gas.
- 7 tests.

**Referencias**: Maoz & Mannucci (2012) PASA 29, 447.

---

### Capa 2 — Física avanzada de submalla ✅ COMPLETADA (Phases 114-119)

Módulos que requieren la Capa 1 o añaden fenómenos a escalas no resueltas.

#### 2A. Vientos estelares de estrellas masivas (pre-SN) ✅ Phase 115

**Descripción**: Estrellas OB y Wolf-Rayet inyectan energía mecánica y metales ~10–30 Myr
antes de la SN. Afectan la estructura de las nubes GMC.

**Costo estimado**: 1–2 sesiones (sobre 1C y 1A)

---

#### 2B. Modo radio AGN (bubble feedback) ✅ Phase 116

**Descripción**: A tasas de acreción bajas (Ṁ/Ṁ_Edd < 0.01), los AGN inyectan jets
mecánicos en lugar de feedback térmico. Relevante para quenching de galaxias masivas.

**Costo estimado**: 2–3 sesiones

**Cambios principales**:
- En `agn.rs`: bifurcar entre modo quasar (térmico, Ṁ alto) y modo radio (mecánico, Ṁ bajo).
- `bubble_feedback(bh, particles, params, dt)`: depositar energía en burbuja esférica.
- Parámetro `f_edd_threshold` en `AgnSection`.

---

#### 2C. Rayos cósmicos (CR) básicos ✅ Phase 117

**Descripción**: Los CRs contribuyen presión no térmica, aceleran vientos galácticos y
suprimen la formación estelar. Un modelo simple: fluido CR con presión P_CR.

**Costo estimado**: 2–3 sesiones

**Cambios principales**:
- Campo `e_cr: f64` (energía CR específica) en `GasData`.
- Ecuación de difusión/advección CR; `apply_cr_diffusion(particles, dt, kappa_cr)`.
- Inyección de CRs en SNe: fracción ε_CR de E_SN.

---

#### 2D. Magnetohidrodinámica (MHD ideal)

**Descripción**: Campo magnético acoplado al gas: inducción, presión magnética, fuerzas de
Lorentz. Relevante para morfología de discos, jets AGN, y CR.

**Estado**: ✅ **INFRAESTRUCTURA BASE COMPLETADA (Phases 123–126)**

**Implementado**:
- Crate `gadget-ng-mhd` con módulos `induction`, `pressure`, `cleaning`.
- Campo `b_field: Vec3` y `psi_div: f64` en `Particle` (Phases 123, 125).
- `advance_induction`: ecuación SPH de Morris & Monaghan (1997).
- `apply_magnetic_forces`: tensor de Maxwell con conservación de momento.
- `dedner_cleaning_step`: esquema de Dedner para div-B (Phase 125).
- Macro `maybe_mhd!` en engine.rs, activada por `[mhd] enabled = true` (Phase 126).
- 25 tests en total (Phases 123–126).

**Pendiente (fases futuras)**:
- ICs magnetizadas cosmológicas.
- Ondas de Alfvén 3D en caja periódica.
- Validación vs. resultados analíticos (shear Alfvén, magnetosónica).
- Acoplamiento con rayos cósmicos y conducción térmica.

**Referencias**: Dolag & Stasyszyn (2009), Tricco & Price (2012), Price & Monaghan (2005).

---

#### 2E. Conducción térmica (ICM)

**Descripción**: En cúmulos de galaxias el ICM conduce calor eficientemente (suprimido por B).
Relevante para perfiles de temperatura y *cool-core* clusters.

**Estado**: ✅ **COMPLETADA (Phase 121)**

**Implementado**:
- `ConductionSection` en `SphSection` (enabled, kappa_spitzer, psi_suppression).
- `apply_thermal_conduction(particles, cfg, gamma, t_floor_k, dt)` en `thermal_conduction.rs`.
- Loop SPH simétrico (i < j) con conservación exacta de energía.
- 6 tests en `phase121_thermal_conduction.rs`.

---

#### 2F. Polvo (destrucción/creación básica)

**Descripción**: El polvo modifica la opacidad, absorbe UV y emite en IR. Un modelo simple:
razón polvo-gas D/G como función de Z y T.

**Costo estimado**: 1–2 sesiones

---

### Capa 3 — Observables sintéticos

Módulos para comparar directamente con surveys. Requieren Capas 1 (especialmente 1C).

#### 3A. SED y función de luminosidad

Asignar luminosidades L(λ) a partículas estelares usando tablas SPS (BC03, FSPS, BPASS)
como función de edad y metalicidad.

**Costo estimado**: 2–3 sesiones

---

#### 3B. Líneas de emisión (Hα, [OIII], [NII])

Desde temperatura SPH + fracción ionizada (ya calculada por la química). Permite
comparaciones directas con SDSS, DESI.

**Costo estimado**: 1–2 sesiones

---

#### 3C. Emisión de rayos X (cúmulos)

T_X y L_X desde temperatura SPH + densidad electrónica. Comparación con eROSITA, XMM.

**Costo estimado**: 1 sesión

---

#### 3D. Mock catalogues con efectos de selección

Generar catálogos sintéticos de galaxias con magnitudes, colores y redshift para comparar
con DES, Euclid, DESI, Roman.

**Costo estimado**: 2–3 sesiones

---

### Capa 4 — Física de frontera / investigación

Módulos de interés para papers específicos; generalmente bajos en costo técnico pero
con alto valor científico.

#### 4A. Neutrinos masivos

Partículas de N-cuerpos adicionales (DM "tibia"). El efecto es una supresión de P(k)
en escalas pequeñas. Solo requiere cambiar las ICs y el P(k) inicial.

**Costo estimado**: 1–2 sesiones

---

#### 4B. Energía oscura dinámica w(z)

Ecuación de estado `w(a) = w₀ + wₐ(1-a)` (Chevallier-Polarski-Linder). Solo cambia
`advance_a()` en `cosmology.rs`.

**Costo estimado**: 1 sesión

---

#### 4C. Gravedad modificada f(R) / DGP

Fuerza gravitacional modificada con screening chameleon o Vainshtein. Relevante para
tests de gravedad con LSS.

**Costo estimado**: 3–5 sesiones

---

#### 4D. Materia oscura auto-interactuante (SIDM)

Colisiones partícula-partícula con sección eficaz σ/m. La infraestructura del árbol ya
existe; se añade un paso de Monte Carlo de scattering.

**Costo estimado**: 2–3 sesiones

---

#### 4E. Formación de cúmulos estelares (GMC collapse)

Fragmentación del gas frío en sub-partículas estelares múltiples (IMF sampling explícito).

**Costo estimado**: 4–6 sesiones (sobre Capa 1)

---

## 4. Resumen cuantitativo

| Capa | Módulos | Sesiones estimadas | Dependencias |
|------|---------|-------------------|--------------|
| 1 — Bariónica faltante | 5 módulos | **13–17 sesiones** | Independiente |
| 2 — Física avanzada | 6 módulos | **22–38 sesiones** | Parcialmente Capa 1 |
| 3 — Observables | 4 módulos | **6–9 sesiones** | Requiere Capa 1C |
| 4 — Física frontera | 5 módulos | **7–13 sesiones** | Independiente |
| **Total** | **20 módulos** | **48–77 sesiones** | |

> MHD (Capa 2D) está desglosado separadamente por ser un proyecto de 1–2 meses propio.

---

## 5. Ruta recomendada (máximo impacto físico primero)

```
Prioridad 1 (siguiente bloque, ~6 sesiones)
├── 1A. Metales y enriquecimiento         → habilita toda la metalicidad
└── 1B. Enfriamiento por tablas de metales → corrección de 1-2 órdenes en M_*

Prioridad 2 (~8 sesiones)
├── 1C. Partículas estelares reales       → habilita observables
└── 1E. SN Ia con DTD                    → corrección de [Fe/H]

Prioridad 3 (~8 sesiones)
├── 3A. SED y función de luminosidad     → comparación con surveys
└── 3B. Líneas de emisión                → comparación con espectros

Prioridad 4 — Física nueva de interés
├── 2B. Modo radio AGN                   → quenching de galaxias masivas
├── 2C. Rayos cósmicos                   → vientos + SF
└── 4B. Energía oscura dinámica          → 1 sesión, alto impacto en papers
```

Con esta ruta, en **~22 sesiones** gadget-ng pasa de ser un código de investigación avanzado
a un código de **formación de galaxias básico pero publicable** con comparaciones directas
a observaciones de surveys modernos.

---

## 6. Referencia de archivos a modificar

| Módulo nuevo | Archivo(s) principal(es) |
|--------------|--------------------------|
| Metales | `gadget-ng-core/src/particle.rs`, nuevo `gadget-ng-sph/src/enrichment.rs` |
| Enfriamiento tablas | `gadget-ng-sph/src/cooling.rs`, `gadget-ng-core/src/config.rs` |
| Partículas estelares | `gadget-ng-core/src/particle.rs` (`ParticleType::Star`), `gadget-ng-sph/src/feedback.rs` |
| ISM multifase | `gadget-ng-sph/src/feedback.rs`, `GasData` en `gadget-ng-sph/src/particle.rs` |
| SN Ia DTD | `gadget-ng-sph/src/feedback.rs` (nueva función) |
| Modo radio AGN | `gadget-ng-sph/src/agn.rs`, `gadget-ng-core/src/config.rs` (`AgnSection`) |
| Rayos cósmicos | nuevo `gadget-ng-sph/src/cosmic_rays.rs` |
| MHD | nuevo crate `gadget-ng-mhd` |
| SED/luminosidades | nuevo crate `gadget-ng-obs` |
| w(z) dinámico | `gadget-ng-core/src/cosmology.rs` (modificar `advance_a`) |
| SIDM | `gadget-ng-tree/src/` (paso de scattering en el árbol) |
