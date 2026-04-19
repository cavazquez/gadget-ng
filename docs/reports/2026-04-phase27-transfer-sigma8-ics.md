# Phase 27: Función de Transferencia Eisenstein–Hu y Normalización σ₈

**Fecha**: abril 2026  
**Estado**: implementación completa con tests automáticos  
**Archivo de plan**: `~/.cursor/plans/phase_27_transfer_σ₈_ics_a5ec3991.plan.md`

---

## 1. Resumen ejecutivo

Esta fase implementa una función de transferencia cosmológica realista (Eisenstein–Hu 1998,
aproximación sin-wiggle) y la normalización física del espectro de potencia inicial mediante
el parámetro σ₈, integrándolas al generador de ICs de Zel'dovich de la Fase 26.

El resultado es que `gadget-ng` puede ahora generar condiciones iniciales con:

- **Espectro físicamente plausible**: P(k) = A² · k^{n_s} · T²_EH(k)
- **Amplitud física**: A calculada para que σ(8 Mpc/h) = σ₈_target
- **Retrocompatibilidad completa**: las ICs de Fase 26 (`transfer = "power_law"`) siguen
  funcionando sin cambios en los archivos TOML
- **Tests automáticos**: 8 tests en `gadget-ng-physics/tests/transfer_sigma8_ics.rs`

---

## 2. Formulación matemática

### 2.1 Espectro de potencia con función de transferencia

El espectro de potencia inicial del campo de densidad es:

```
P(k) = A² · k^{n_s} · T²(k)
```

donde:

| Símbolo | Significado | Unidades |
|---------|-------------|---------|
| k | número de onda | h/Mpc |
| n_s | índice espectral primordial | adimensional |
| T(k) | función de transferencia | adimensional, T ∈ (0, 1] |
| A | amplitud normalizada por σ₈ | [(Mpc/h)^{3/2} · (h/Mpc)^{-n_s/2}] |

### 2.2 Función de transferencia Eisenstein–Hu (no-wiggle)

Implementamos la aproximación sin oscilaciones acústicas bariónicas (no-wiggle) de
Eisenstein & Hu (1998, ApJ 496, 605), ecuaciones 29–31.

**Parámetros de entrada**: Ω_m, Ω_b, h (k en h/Mpc)

**Formulación** (con k en h/Mpc):

```
ωm = Ω_m h²,   ωb = Ω_b h²,   fb = Ω_b/Ω_m

s      = 44.5 ln(9.83/ωm) / sqrt(1 + 10 ωb^{3/4})        [Mpc, horizonte de sonido]
α_Γ    = 1 − 0.328 ln(431 ωm) fb + 0.38 ln(22.3 ωm) fb²

k_Mpc  = k_hmpc × h                                        [Mpc⁻¹]
ks     = 0.43 k_Mpc s                                      [adimensional]
Γ_eff  = Ω_m h [α_Γ + (1 − α_Γ) / (1 + ks⁴)]            [h/Mpc]

q      = k_hmpc / Γ_eff                                    [adimensional]
L₀     = ln(2e + 1.8 q)
C₀     = 14.2 + 731/(1 + 62.5 q)
T(k)   = L₀ / (L₀ + C₀ q²)                               [∈ (0, 1]]
```

**Comportamiento asintótico**:
- T(k → 0) → 1 (modos super-horizonte)
- T(k → ∞) ∝ ln(k)/k² (supresión tipo CDM)

**Valores de referencia para Planck18** (Ω_m=0.315, Ω_b=0.049, h=0.674):

| k [h/Mpc] | T(k) |
|-----------|------|
| 1×10⁻⁴   | ≈ 0.9997 |
| 1×10⁻³   | ≈ 0.992 |
| 1×10⁻²   | ≈ 0.79 |
| 0.1       | ≈ 0.14 |
| 1.0       | ≈ 0.003 |
| 10.0      | ≈ 1×10⁻⁴ |

La escala de igualdad k_eq ≈ Ω_m h² × [conversión] ≈ 0.015 h/Mpc. Para k > k_eq
el espectro está significativamente suprimido respecto al primordial.

**Nota**: la versión no-wiggle es apropiada para N=32³ porque la resolución de la
caja (k_Nyq ≈ 1 h/Mpc con box=100 Mpc/h) no puede resolver las oscilaciones BAO
(longitud de onda ~0.06 h/Mpc ↔ 150 Mpc/h).

### 2.3 Normalización por σ₈

El parámetro σ₈ mide las fluctuaciones de densidad suavizadas en R = 8 Mpc/h:

```
σ²(R) = (1/2π²) ∫₀^∞ k² P(k) W²(kR) dk
```

con el filtro top-hat en k-space:

```
W(x) = 3 [sin(x) − x cos(x)] / x³,   W(0) = 1
```

**Procedimiento de normalización**:

1. Calcular σ²_unit = σ²(8 Mpc/h) con P(k) = k^{n_s} · T²(k) y A = 1
   (integración numérica en log-k con regla del trapecio, k ∈ [10⁻⁵, 5×10²] h/Mpc,
   N = 8192 pasos)

2. Determinar la amplitud: **A = σ₈_target / sqrt(σ²_unit)**

3. El espectro normalizado cumple exactamente σ(8 Mpc/h) = σ₈_target

**Verificación**: el test `amplitude_for_sigma8_is_consistent` confirma la
autoconsistencia con error relativo < 0.1%.

### 2.4 Conversión de unidades k_grid → k_phys

El generador de ICs trabaja con un grid de tamaño N³ y caja interna de tamaño L = 1.
La función de transferencia requiere k en h/Mpc. La conversión es:

```
k_phys [h/Mpc] = 2π × |n| × h / box_size_mpc_h
```

donde:
- `|n|` = módulo del vector de modo entero (|n|² = nx² + ny² + nz²)
- `h` = parámetro de Hubble adimensional H₀/(100 km/s/Mpc)
- `box_size_mpc_h` = tamaño de la caja en Mpc/h (parámetro nuevo en la config)

**Importante**: `box_size_mpc_h` no afecta el sistema de unidades interno de
`gadget-ng`. Es un parámetro exclusivo del generador de ICs para la conversión de k.

### 2.5 Closure de espectro

La refactorización central de esta fase es que `generate_delta_kspace` ahora
acepta un closure genérico `spectrum_fn: impl Fn(f64) -> f64` en lugar de
parámetros fijos de amplitud/índice:

```
σ(|n|) = spectrum_fn(|n|)
```

Para Eisenstein–Hu con σ₈:
```
σ(|n|) = A · k_phys(|n|)^{n_s/2} · T(k_phys(|n|)) / sqrt(N³)
```

donde P(k) = σ²(|n|) × N³ = A² · k^{n_s} · T²(k) ✓

Para ley de potencia (legacy):
```
σ(|n|) = amplitude · |n|^{n_s/4} / sqrt(N³)
```

donde P(k) = amplitude² · |n|^{n_s/2} (en unidades de grid) ✓

---

## 3. Decisiones de diseño

### 3.1 Eisenstein–Hu no-wiggle vs alternativas

| Opción | Pros | Contras |
|--------|------|---------|
| **EH98 no-wiggle** | estándar moderno, incluye corrección bariónica, formulación cerrada | T_CMB no usada, sin wiggles BAO |
| BBKS | más simple | sin corrección bariónica, menos precisa |
| EH98 full (wiggles) | física más completa | oscilaciones BAO no resueltas con N=32³ |
| CAMB tabulado | exacto | requiere dependencia externa |

La elección de EH98 no-wiggle es la correcta para esta fase.

### 3.2 h separado de h0

La configuración tiene dos parámetros distintos:
- `cosmology.h0`: H₀ en unidades internas de tiempo (1/t_sim), depende del sistema
  de unidades de la simulación
- `IcKind::Zeldovich.h`: parámetro adimensional h = H₀/(100 km/s/Mpc), solo para
  la función de transferencia

Esta separación evita ambigüedad entre el sistema de unidades interno y la cosmología física.

### 3.3 Retrocompatibilidad

Todos los campos nuevos en `IcKind::Zeldovich` tienen `#[serde(default)]`:
- `transfer`: default = `PowerLaw` (comportamiento Fase 26)
- `sigma8`: default = `None` (usa `amplitude`)
- `omega_b`, `h`, `t_cmb`: defaults Planck18
- `box_size_mpc_h`: default = `None` (causa error si se usa EH sin especificarlo)

Los configs TOML de Fase 26 siguen funcionando sin ningún cambio.

---

## 4. Archivos creados/modificados

### Nuevos
- `crates/gadget-ng-core/src/transfer_fn.rs`:
  `EisensteinHuParams`, `transfer_eh_nowiggle`, `sigma_sq_unit`,
  `amplitude_for_sigma8`, `tophat_window`, `sigma_from_pk_bins`
  + 8 tests unitarios internos

- `crates/gadget-ng-physics/tests/transfer_sigma8_ics.rs`:
  8 tests de integración

- `experiments/nbody/phase27_transfer_sigma8/`:
  3 configs TOML + 3 scripts Python + `run_phase27.sh`

- `docs/reports/2026-04-phase27-transfer-sigma8-ics.md` (este archivo)

### Modificados
- `crates/gadget-ng-core/src/config.rs`:
  enum `TransferKind`, campos nuevos en `IcKind::Zeldovich` con defaults

- `crates/gadget-ng-core/src/ic_zeldovich.rs`:
  `generate_delta_kspace` → acepta closure `spectrum_fn`,
  nueva función `build_spectrum_fn`, `zeldovich_ics` extendida con parámetros EH,
  función wrapper `zeldovich_ics_power_law` para compatibilidad

- `crates/gadget-ng-core/src/ic.rs`:
  match arm de `IcKind::Zeldovich` extendido con nuevos campos

- `crates/gadget-ng-core/src/lib.rs`:
  `pub mod transfer_fn`, exports de `TransferKind` y funciones de `transfer_fn`

- `crates/gadget-ng-physics/tests/zeldovich_ics.rs`:
  añadidos campos `transfer`, `sigma8`, `omega_b`, `h`, `t_cmb`, `box_size_mpc_h`
  con defaults legacy en construcciones directas de `IcKind::Zeldovich`

---

## 5. Resultados de validación

### 5.1 Tests automáticos

Todos los tests pasan:

```
gadget-ng-core (17 tests, 0 failed):
  transfer_fn::tests::eh_transfer_low_k_is_one    ✓
  transfer_fn::tests::eh_transfer_high_k_suppressed ✓
  transfer_fn::tests::eh_transfer_range_valid      ✓
  transfer_fn::tests::eh_transfer_reference_value  ✓
  transfer_fn::tests::sigma_sq_unit_is_positive_finite ✓
  transfer_fn::tests::amplitude_for_sigma8_is_consistent ✓
  transfer_fn::tests::tophat_window_at_zero        ✓
  transfer_fn::tests::tophat_window_decays         ✓
  ic_zeldovich::tests (4)                          ✓

gadget-ng-physics (8 tests, 0 failed):
  eh_transfer_low_k_is_one                         ✓
  eh_transfer_high_k_suppressed                    ✓
  sigma8_normalization_matches_target              ✓  (error < 0.1%)
  eh_spectrum_differs_from_power_law               ✓  (T²(k=1) ≈ 9×10⁻⁶)
  legacy_amplitude_still_works                     ✓  (retrocompatibilidad)
  positions_in_box_with_eh                         ✓
  pm_run_stable_with_eh_ics                        ✓  (10 pasos, sin NaN/Inf)
  treepm_run_stable_with_eh_ics                    ✓  (10 pasos, sin NaN/Inf)

gadget-ng-physics/zeldovich_ics (10 tests, 0 failed):  [tests Fase 26]
  (todos los tests de Fase 26 siguen pasando)          ✓
```

### 5.2 Verificación de σ₈

Para Planck18 (Ω_m=0.315, Ω_b=0.049, h=0.674, n_s=0.965, σ₈_target=0.8):

```
σ²_unit(R=8 Mpc/h) = σ²(8, A=1) = [valor calculado por integración numérica]
A = σ₈_target / sqrt(σ²_unit)
σ₈_recuperado = A × sqrt(σ²_unit) = 0.800000  (error relativo < 0.01%)
```

La verificación empírica completa (σ₈ medido desde el P(k) del campo de partículas
con N = 32³ y caja = 100 Mpc/h) requiere ejecutar `run_phase27.sh` y el script
`validate_sigma8.py`. Con N = 32³, la resolución espectral es limitada y se espera
un error del ~20-40% debido a la discretización del grid (k_fund ≈ 0.063 h/Mpc,
k_Nyq ≈ 1.0 h/Mpc, solo ~10 bins en P(k)).

Para validación precisa de σ₈ se recomienda N ≥ 128³ o 256³.

### 5.3 Cambio de pendiente del espectro

La relación T²(k=1 h/Mpc) ≈ 9×10⁻⁶ confirma que el espectro EH está
fuertemente suprimido respecto a la ley de potencia en k = 1 h/Mpc.

La comparación visual (scripts/compare_spectra.py) muestra:
- Para k < 0.01 h/Mpc: P(k)_EH ≈ P(k)_PL (T ≈ 1)
- Para k ≈ 0.1 h/Mpc: P(k)_EH ≈ 0.02 × P(k)_PL (T² ≈ 0.02)
- Para k ≈ 1 h/Mpc: P(k)_EH ≈ 9×10⁻⁶ × P(k)_PL

(Los ratios son para la misma amplitud A; en la práctica las amplitudes difieren
porque la normalización σ₈ del caso EH es diferente a la `amplitude` manual del caso PL.)

### 5.4 Estabilidad PM y TreePM

Ambos solvers integran 10 pasos de leapfrog cosmológico con ICs EH sin producir
NaN/Inf. Los campos de velocidad permanecen finitos y en el régimen lineal esperado.

---

## 6. Configuración de ejemplo

### Fase 27 (EH + σ₈):

```toml
[simulation]
particle_count         = 32768     # 32³
box_size               = 1.0
dt                     = 0.0004
num_steps              = 50
softening              = 0.01
gravitational_constant = 1.0
seed                   = 42

[initial_conditions]
kind = { zeldovich = {
    seed            = 42,
    grid_size       = 32,
    spectral_index  = 0.965,
    amplitude       = 1.0e-4,      # ignorado cuando sigma8 está definido
    transfer        = "eisenstein_hu",
    sigma8          = 0.8,
    omega_b         = 0.049,
    h               = 0.674,
    t_cmb           = 2.7255,
    box_size_mpc_h  = 100.0
} }

[cosmology]
enabled      = true
periodic     = true
omega_m      = 0.315
omega_lambda = 0.685
h0           = 0.1
a_init       = 0.02
```

### Fase 26 (ley de potencia, retrocompatibilidad):

```toml
[initial_conditions]
kind = { zeldovich = { seed = 42, grid_size = 32, spectral_index = -2.0, amplitude = 1.0e-4 } }
```

Ambas configuraciones funcionan sin cambios.

---

## 7. Limitaciones explícitas

### 7.1 Limitaciones actuales (a corregir en fases futuras)

| Limitación | Impacto | Prioridad |
|-----------|---------|-----------|
| **Sin 2LPT** | La aproximación de Zel'dovich (1LPT) sobrestima las perturbaciones iniciales para n_s > 0 y z < 50 | Alta |
| **T_CMB no usada** | La versión no-wiggle de EH no incluye dependencia en T_CMB | Baja (efecto < 0.5%) |
| **Sin oscilaciones BAO** | La forma del espectro es suave; las oscilaciones BAO (~150 Mpc/h) no están presentes | Media para cajas > 300 Mpc/h |
| **Generador serial** | El campo k-space completo se genera en todos los rangos MPI | Impacto en N > 64³ |
| **N = 32³ es pequeño** | σ₈ empírico tiene error ~30% por falta de resolución espectral | Para validación: usar N ≥ 128³ |
| **h0 vs h ambigüedad** | El usuario debe especificar h adimensional explícitamente en los ICs | Documentar claramente |

### 7.2 Qué falta para ICs cosmológicas publicables

1. **2LPT**: reducción del error de posicionamiento de ~100% a ~10% para z < 100
2. **Generador MPI nativo**: actualmente O(N³) por rango; debe ser O(N³/n_rank)
3. **Transfer function exacta (CAMB/CLASS)**: incluyendo wiggles BAO, neutrinos, w(z)
4. **Interpolación del campo de crecimiento**: D(a) numérico en lugar de Ω_m^0.55
5. **Campo de velocidad 2LPT**: momenta de segundo orden para mejor conservación
6. **Corrección de aliasing**: el generador actual suprime Nyquist pero no aplica
   corrección de aliasing de segundo orden
7. **σ₈ validado con N ≥ 256³**: con la implementación actual, la resolución
   del grid limita la precisión de σ₈ medido a ~30%

---

## 8. Referencias

- **Eisenstein & Hu (1998)**: "Power Spectra for Cold Dark Matter and Its Variants",
  ApJ 496, 605. [arXiv:astro-ph/9709066]
  - Ecuaciones 29–31: función de transferencia no-wiggle
  - Ecuación 28: shape parameter q
  - Ecuación 31: Γ_eff(k)

- **Carroll, Press & Turner (1992)**: "The Cosmological Constant", ARA&A 30, 499.
  Factor de crecimiento D(a) en ΛCDM.

- **Linder (2005)**: "Cosmic growth history and expansion history",
  PRD 72, 043529. Aproximación f(a) ≈ Ω_m(a)^{0.55}.

- **Planck Collaboration (2018)**: Planck 2018 results VI, A&A 641, A6.
  Parámetros de referencia: Ω_m=0.315, Ω_b=0.049, h=0.674, n_s=0.965, σ₈=0.811.

- **Zel'dovich (1970)**: "Gravitational instability: An approximate theory for large
  density perturbations". A&A 5, 84.

---

## 9. Comparación con estado anterior (Fase 26)

| Capacidad | Fase 26 | Fase 27 |
|-----------|---------|---------|
| Espectro | P(k) ∝ k^{n_s} (ley de potencia) | P(k) = A² k^{n_s} T²_EH(k) |
| Normalización | `amplitude` manual adimensional | σ₈ físico (integral numérica) |
| Función de transferencia | ninguna (T = 1) | EH98 no-wiggle |
| Cosmología | EdS (Ω_m=1, Ω_Λ=0) | ΛCDM general |
| Unidades de k | adimensionales (unidades de grid) | h/Mpc (conversión via box_size_mpc_h) |
| Tests automáticos | 10 tests de IC | + 8 tests de EH + σ₈ |
| Retrocompatibilidad | N/A | 100% (campo `transfer = "power_law"` es default) |

---

*Reporte generado automáticamente como parte del proceso de desarrollo de gadget-ng.*  
*Commit de referencia: ver `git log --oneline -5` en el momento de la generación.*
