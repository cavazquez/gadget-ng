# Physics Roadmap — gadget-ng

> **Actualizado:** 2026-04-23 (post Phase 108)
>
> Este documento cataloga la física ya implementada, evalúa su nivel de completitud respecto a los
> códigos de referencia (GADGET-4, AREPO/IllustrisNG, EAGLE), y proyecta qué nuevos módulos se
> pueden agregar y a qué costo.

---

## 1. Inventario de física implementada (Phases 1–108)

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

Estas son las piezas que mayor diferencia hacen en la predicción de propiedades de galaxias.

#### 1A. Metales y enriquecimiento químico

**Descripción**: Rastrear la metalicidad Z y abundancias individuales (O, Fe, Mg, Si) en cada
partícula de gas y estelar. Los yields se calculan por SN II, SN Ia y AGB.

**Impacto**: Habilita comparación directa con observaciones ([O/Fe] vs [Fe/H],
función de distribución de metalicidades, Z–SFR).

**Costo estimado**: 3–4 sesiones

**Cambios principales**:
- Campo `metallicity: f64` en `Particle` (o `GasData`).
- Tabla de yields por tipo de estrella: `yields_snii(m_star)`, `yields_snia()`, `yields_agb(m_star, z)`.
- Distribución de metales en el vecindario SPH al explotar (kernel smoothing).
- Módulo `gadget-ng-sph/src/enrichment.rs`.

**Referencias**: Wiersma et al. (2009), Karakas (2010).

---

#### 1B. Enfriamiento por metales (tablas)

**Descripción**: A temperaturas T < 10⁷ K, el gas con Z > 0 se enfría 10–100× más rápido
que el gas primordial. La diferencia es la mayor fuente de error en simulaciones sin metales.

**Impacto**: Corrección de 1–2 órdenes de magnitud en la masa estelar final de una galaxia.
Sin esto, galaxias predichas son demasiado calientes.

**Costo estimado**: 2–3 sesiones

**Cambios principales**:
- Ampliar `CoolingKind` en `config.rs`: `CoolingKind::MetalTables`.
- Leer tabla pre-calculada `Λ(n_H, T, Z, z)` en formato HDF5/binario.
- Interpolación bilineal en `apply_cooling`.
- Archivo de datos: tablas de Wiersma+09 o CLOUDY.

**Referencias**: Wiersma et al. (2009) MNRAS 393, 99.

---

#### 1C. Partículas estelares reales (stellar spawning)

**Descripción**: En lugar de que las partículas de gas sean "fuentes de SN" abstractas, convertir
estocásticamente partículas de gas en partículas estelares con edad y metalicidad registradas.

**Impacto**: Habilita función de luminosidad, colores de galaxias, perfiles de masa estelar,
observables de surveys (DES, Euclid, DESI).

**Costo estimado**: 4–5 sesiones

**Cambios principales**:
- Nuevo `ParticleType::Star` con campos `age_gyr: f64` y `metallicity: f64`.
- Función `spawn_star_particles(gas, sfr, dt, seed) -> Option<Particle>` en `feedback.rs`.
- Las estrellas spawneadas no participan en SPH pero sí en gravedad y enriquecimiento.
- El motor necesita manejar listas de partículas crecientes.

---

#### 1D. Modelo ISM multifase frío-caliente

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

#### 1E. SN tipo Ia con distribución de retraso temporal (DTD)

**Descripción**: Las SN Ia explotan ~1 Gyr después de la formación estelar (binarios WD).
Son la principal fuente de Fe. Sin DTD, los modelos predicen [O/Fe] incorrecto.

**Impacto**: Corrección importante para galaxias masivas y química estelar.

**Costo estimado**: 1–2 sesiones

**Cambios principales**:
- Función `snia_rate(stellar_age, mass_formed) -> f64` con DTD power-law o exponencial.
- Inyección diferida en `apply_sn_feedback` según la edad de la partícula estelar.
- Requiere partículas estelares (1C).

---

### Capa 2 — Física avanzada de submalla

Módulos que requieren la Capa 1 o añaden fenómenos a escalas no resueltas.

#### 2A. Vientos estelares de estrellas masivas (pre-SN)

**Descripción**: Estrellas OB y Wolf-Rayet inyectan energía mecánica y metales ~10–30 Myr
antes de la SN. Afectan la estructura de las nubes GMC.

**Costo estimado**: 1–2 sesiones (sobre 1C y 1A)

---

#### 2B. Modo radio AGN (bubble feedback)

**Descripción**: A tasas de acreción bajas (Ṁ/Ṁ_Edd < 0.01), los AGN inyectan jets
mecánicos en lugar de feedback térmico. Relevante para quenching de galaxias masivas.

**Costo estimado**: 2–3 sesiones

**Cambios principales**:
- En `agn.rs`: bifurcar entre modo quasar (térmico, Ṁ alto) y modo radio (mecánico, Ṁ bajo).
- `bubble_feedback(bh, particles, params, dt)`: depositar energía en burbuja esférica.
- Parámetro `f_edd_threshold` en `AgnSection`.

---

#### 2C. Rayos cósmicos (CR) básicos

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

**Impacto**: Cambio estructural profundo; requiere reformular el solver de fuerzas SPH o
cambiar a SPH-MHD (Dedner divergence cleaning o SPH con campo B).

**Costo estimado**: 15–25 sesiones (proyecto mayor)

**Cambios principales**:
- Campo `b_field: Vec3` en `Particle`.
- Nuevo crate `gadget-ng-mhd` con solver de inducción y presión magnética.
- Modificar `compute_sph_forces` para incluir tensor de Maxwell.
- Condition: div B = 0 vía Dedner cleaning o Euler potentials.

**Referencias**: Dolag & Stasyszyn (2009), Tricco & Price (2012).

---

#### 2E. Conducción térmica (ICM)

**Descripción**: En cúmulos de galaxias el ICM conduce calor eficientemente (suprimido por B).
Relevante para perfiles de temperatura y *cool-core* clusters.

**Costo estimado**: 1–2 sesiones

**Cambios principales**:
- `apply_thermal_conduction(particles, kappa_cond, dt)` en `cooling.rs`.
- Coeficiente de Spitzer con factor de supresión ψ (por B o turbulencia).

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
