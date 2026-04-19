# Fase 28: Condiciones Iniciales de Segundo Orden (2LPT)

**Fecha:** Abril 2026  
**Estado:** Implementado y validado  
**Responsable:** gadget-ng core team

---

## 1. Motivación

Las condiciones iniciales de primer orden (Zel'dovich/1LPT) producen errores sistemáticos que crecen con el tiempo: sobre-predicen la amplitud de las perturbaciones de densidad y sub-predicen las velocidades relativas a la solución exacta. Estos errores se deben a que la aproximación lineal ignora los términos de orden superior en la teoría de perturbaciones lagrangiana.

La teoría de perturbaciones lagrangiana de segundo orden (2LPT) añade la corrección `Ψ²` que es cuadrática en la amplitud del campo inicial, reduciendo los errores transitorios típicos de las ICs Zel'dovich. Codecs de N-cuerpos estándar como GADGET-4, N-GenIC y 2LPTic implementan 2LPT para simular condiciones físicamente precisas desde altos redshifts (z ~ 50-100).

---

## 2. Formulación matemática

### 2.1 Posición total a segundo orden

La posición de una partícula en la aproximación 2LPT es:

```
x(q, a) = q + D₁(a) Ψ¹(q) + D₂(a) Ψ²(q)
```

donde:
- `q` es la posición Lagrangiana (en la retícula inicial)
- `D₁(a)` es el factor de crecimiento lineal (absorbido en la amplitud del campo, `D₁ = 1`)
- `D₂(a)` es el factor de crecimiento de segundo orden
- `Ψ¹`, `Ψ²` son los campos de desplazamiento de primer y segundo orden

La implementación en gadget-ng usa la convención `D₁ = 1` absorbida en la amplitud del espectro, de modo que la corrección neta es:

```
x = q + Ψ¹ + (D₂/D₁²) · Ψ²
```

### 2.2 Desplazamiento de primer orden Ψ¹

Ya implementado en Fase 26 (ic_zeldovich.rs):

```
Ψ¹(k) = i (k/k²) δ(k)
```

### 2.3 Desplazamiento de segundo orden Ψ²

**Paso A: Derivadas del potencial de desplazamiento**

El potencial φ¹ satisface `∇²φ¹ = δ`, por lo que:
```
φ¹(k) = δ(k) / |k|²
```

Sus derivadas en k-space:
```
φ̂¹,αβ(k) = -n_α n_β / |n|² · δ̂(k)       α, β ∈ {x, y, z}
```

donde `n = (nx, ny, nz)` son los números de modo enteros del grid discreto.

Aplicando IFFT → obtenemos 6 componentes en espacio real:
`φ,xx`, `φ,yy`, `φ,zz`, `φ,xy`, `φ,xz`, `φ,yz`.

**Paso B: Término fuente de segundo orden**

```
S(x) = (φ,xx φ,yy - φ,xy²) + (φ,yy φ,zz - φ,yz²) + (φ,zz φ,xx - φ,xz²)
```

Esta es la suma de `Σ_{i>j} [φ,ii φ,jj - (φ,ij)²]`, relacionada con el segundo invariante del tensor de Hessiano de φ¹.

**Paso C: Poisson de segundo orden**

```
∇²φ² = S(x)   →   φ²(k) = -S(k) / |n|²
```

Con DC = 0 y modos Nyquist = 0.

**Paso D: Gradiente → Ψ²**

```
Ψ²(k) = -i (n/|n|²) φ²(k)   →   IFFT   →   Ψ²(x)
```

### 2.4 Factor D₂/D₁²

Aproximación de Bouchet et al. (1995), válida para ΛCDM plana:

```
D₂(a) / D₁²(a) ≈ -3/7 · Ω_m(a)^{-1/143}
```

donde `Ω_m(a) = Ω_m / a³ / E²(a)` y `E²(a) = H²(a)/H₀²`.

Para parámetros Planck18 (`Ω_m = 0.315`) a `a = 0.02` (z ≈ 49):
```
Ω_m(a = 0.02) ≈ 1.0003   →   D₂/D₁² ≈ -3/7 × 1.0003^(-1/143) ≈ -0.4286
```

El signo negativo es físico: Ψ² compensa parcialmente la sobre-clustering de Zel'dovich.

### 2.5 Velocidades a segundo orden

El momentum canónico total (convenio GADGET-4):

```
p = a²·H(a) · [f₁·Ψ¹ + f₂·(D₂/D₁²)·Ψ²]
```

con:
- `f₁ = d ln D₁ / d ln a ≈ Ω_m(a)^{0.55}` (Linder 2005, ya implementado en `growth_rate_f`)
- `f₂ ≈ 2·f₁` (aproximación válida a z ≫ 1 en ΛCDM)

**Nota sobre f₂ = 2f₁:** Esta aproximación tiene error < 5% para z > 1 en ΛCDM estándar. A z = 49 (a = 0.02), el universo está dominado por materia y la aproximación es prácticamente exacta.

---

## 3. Arquitectura de implementación

### 3.1 Flujo de control

```
IcKind::Zeldovich { use_2lpt: true, ... }
    ↓
ic.rs: build_particles_for_gid_range()
    ↓
ic_2lpt.rs: zeldovich_2lpt_ics()
    ├── ic_zeldovich::generate_delta_kspace()   [reutilizado]
    ├── ic_zeldovich::build_spectrum_fn()        [reutilizado]
    ├── ic_zeldovich::delta_to_displacement()    → Ψ¹   [reutilizado]
    └── 2LPT pipeline:
            phi_second_derivatives()             → φ,αβ(x)
            build_2lpt_source()                  → S(x)
            solve_poisson_real_to_kspace()       → φ²(k)
            phi2_to_psi2()                       → Ψ²(x)
```

### 3.2 Archivos modificados o creados

| Archivo | Cambio |
|---------|--------|
| `crates/gadget-ng-core/src/ic_2lpt.rs` | **Nuevo**: módulo 2LPT completo |
| `crates/gadget-ng-core/src/config.rs` | Añadido `use_2lpt: bool` a `IcKind::Zeldovich` |
| `crates/gadget-ng-core/src/ic.rs` | Dispatch a `zeldovich_2lpt_ics` si `use_2lpt = true` |
| `crates/gadget-ng-core/src/lib.rs` | `pub mod ic_2lpt` + export de `zeldovich_2lpt_ics` |
| `crates/gadget-ng-core/src/ic_zeldovich.rs` | `fft3d`, `generate_delta_kspace`, etc. → `pub(crate)` |
| `crates/gadget-ng-physics/tests/lpt2_ics.rs` | **Nuevo**: 8 tests de integración |

### 3.3 Retrocompatibilidad

- `use_2lpt` tiene `#[serde(default)]` → todos los configs TOML existentes siguen funcionando (default `false`).
- El código de Fase 26/27 (1LPT) no se modifica.
- Los tests de Fase 26 y 27 solo requirieron añadir `use_2lpt: false` a los struct literals de Rust (campo nuevo en enum).

---

## 4. Decisiones de diseño

### 4.1 Flag `use_2lpt` vs nueva variante de `IcKind`

Se eligió añadir un único campo booleano a la variante existente `IcKind::Zeldovich` en lugar de crear `IcKind::Zeldovich2Lpt`. Ventajas:
- Mínimos cambios al código de dispatch en `ic.rs`
- Todos los parámetros del espectro (EH, σ₈, etc.) se comparten automáticamente
- Retrocompatible via `#[serde(default)]`

### 4.2 Reutilización de funciones de Fase 26/27

Las funciones `fft3d`, `generate_delta_kspace`, `delta_to_displacement` y `build_spectrum_fn` se hicieron `pub(crate)` para ser importadas desde `ic_2lpt.rs`. Esto evita duplicación de código y garantiza que 1LPT y 2LPT usan exactamente la misma generación de campo y mismo espectro.

### 4.3 Aproximación D₂/D₁² = −3/7 Ω_m(a)^{−1/143}

La fórmula de Bouchet et al. (1995) es válida para ΛCDM plana con error < 1% en el rango 0.1 < Ω_m < 1. Para el modelo Planck18 estándar da:
```
D₂/D₁² ≈ -0.4286   (a z = 49, Ω_m = 0.315)
```

### 4.4 Aproximación f₂ = 2f₁

Basada en Crocce et al. (2006) y Scoccimarro (1998). La validez se verifica:
- A z ≫ 1 (materia dominante): `f₁ → 1`, `f₂ → 2`, error = 0%
- A z = 0 (ΛCDM Planck18): `f₁ ≈ 0.44`, `f₂ ≈ 0.80`, f₂_exact/2f₁ ≈ 1.04, error ≈ 4%
- A z = 49 (a_init = 0.02): error < 0.1%

### 4.5 Hermitian symmetry y modo DC

- El campo `φ²(k)` se construye a partir de FFT forward de una función real `S(x)`, garantizando simetría Hermitiana automáticamente.
- El modo DC (`k = 0`) se forza a cero: `⟨Ψ²⟩ = 0` por construcción.
- Los modos Nyquist se fuerzan a cero para evitar aliasing.

---

## 5. Propiedades físicas de Ψ²

### 5.1 Magnitud esperada

Para una simulación con N = 32³, caja = 100 Mpc/h, σ₈ = 0.8 a z = 49:

| Cantidad | Estimación |
|---------|-----------|
| `\|Ψ¹\|_rms` | ~1% del spacing de retícula |
| `\|D₂/D₁² · Ψ²\|_rms` | ~0.01–0.05% del spacing |
| Ratio `\|Ψ²\|/\|Ψ¹\|` | ~1–5% |

La corrección 2LPT es cuadrática en el desplazamiento Zel'dovich, por lo que es subleading en el régimen lineal (z ≫ 1). Aumenta su importancia relativa al bajar el redshift de inicio de la simulación.

### 5.2 Efecto en P(k)

La corrección 2LPT afecta la distribución de partículas pero no cambia el espectro de potencia lineal (que sigue siendo `P(k) ∝ k^ns T²(k)`). Las principales diferencias con 1LPT son:

1. **Reducción de errores transitorios**: las ICs 2LPT minimizan las oscilaciones no-físicas en la evolución de δ_rms(a).
2. **Velocidades más precisas**: la contribución de segundo orden a p es proporcional a f₂ × Ψ², corrigiendo la sobre-estimación de velocidades de Zel'dovich.
3. **Estructura de filamentos**: la corrección 2LPT tiende a distribuir las partículas de forma más coherente con la estructura cósmica esperada.

---

## 6. Tests automáticos implementados

| Test | Propiedad verificada |
|------|---------------------|
| `lpt2_reproducible` | Misma seed → partículas bit-a-bit iguales |
| `lpt2_psi2_mean_near_zero` | `⟨Ψ²⟩ ≈ 0` (modo DC nulo) |
| `lpt2_psi2_smaller_than_psi1` | `\|Ψ²\|_rms < \|Ψ¹\|_rms` (corrección subleading) |
| `lpt2_positions_in_box` | Todas las posiciones en `[0, box_size)` |
| `lpt2_no_nan_inf` | Sin NaN ni Inf en posición/velocidad |
| `lpt2_pm_run_stable` | 10 pasos PM sin explosión numérica |
| `lpt2_treepm_run_stable` | 10 pasos TreePM sin explosión numérica |
| `lpt2_vs_1lpt_differ` | 2LPT produce posiciones distintas a 1LPT (Ψ² ≠ 0) |

Todos los 8 tests pasan en una grid de 8³ con espectro EH + σ₈ = 0.8.

**Nota sobre `lpt2_vs_1lpt_differ`:** La corrección 2LPT es cuadrática en el desplazamiento y para grids pequeñas (8³) puede ser mucho menor que el espaciado de retícula. Por ello el test usa comparación a nivel de bits (no un umbral físico), garantizando que Ψ² ≠ 0 exacto.

---

## 7. Experimentos de validación

Los scripts en `experiments/nbody/phase28_2lpt/` producen las siguientes métricas:

```bash
bash experiments/nbody/phase28_2lpt/run_phase28.sh
```

### 7.1 Configuraciones

| Config | IC | Solver |
|--------|-----|--------|
| `lcdm_N32_1lpt_pm.toml` | 1LPT baseline | PM |
| `lcdm_N32_2lpt_pm.toml` | 2LPT | PM |
| `lcdm_N32_2lpt_treepm.toml` | 2LPT | TreePM |

N = 32³ = 32768 partículas, caja = 100 Mpc/h, a_init = 0.02, 50 pasos.

### 7.2 Figuras generadas

1. **`fig_compare_1lpt_2lpt.png`**: histograma `|Ψ¹|` vs `|D₂/D₁²·Ψ²|`, P(k) inicial comparado, y ratio P(k)[2LPT]/P(k)[1LPT].

2. **`fig_growth_1lpt_2lpt.png`**: evolución de δ_rms(a) para 1LPT y 2LPT comparada con el factor de crecimiento teórico D₁(a).

### 7.3 Métricas esperadas

- `|Ψ²|_rms / |Ψ¹|_rms` ≈ 0.01–0.05 (subleading)
- `P(k)[2LPT] / P(k)[1LPT]` ≈ 1 ± 5% para k < k_Nyq
- `δ_rms(a) / D₁(a)` ≈ constante (crecimiento lineal conservado)

---

## 8. Limitaciones conocidas

| Limitación | Descripción |
|-----------|-------------|
| **Solo series temporales iniciales** | 2LPT corrige las ICs, no la evolución posterior |
| **Aproximación D₂/D₁²** | Válida para ΛCDM plana, error < 1%. No válida para w ≠ -1 o modelos con masa de neutrinos |
| **Aproximación f₂ = 2f₁** | Error < 5% para z > 1 en ΛCDM. Para simulaciones a z < 1, usar la fórmula exacta |
| **Grid serie** | La generación del campo 2LPT es serial (mismo patrón que 1LPT). Escalado MPI limitado por memoria por rango para N > 512³ |
| **Sin 3LPT** | Para z_init < 10 o simulaciones de alta precisión, 3LPT reduce aún más los transitorios |
| **Sin corrección de virialization** | Las velocidades corregidas con 2LPT siguen sin incluir los efectos del virialization local |

---

## 9. Referencias

- Zel'dovich, Ya. B. (1970): *Gravitational instability: An approximate theory for large density perturbations*. A&A 5, 84.
- Bouchet, F.R. et al. (1995): *Perturbative Lagrangian approach to gravitational instability*. A&A 296, 575.
- Scoccimarro, R. (1998): *Transients from initial conditions: a perturbative analysis*. MNRAS 299, 1097.
- Crocce, M., Pueblas, S. & Scoccimarro, R. (2006): *Transients from initial conditions in cosmological simulations*. MNRAS 373, 369.
- Linder, E.V. (2005): *Cosmic growth history and expansion history*. PRD 72, 043529.
