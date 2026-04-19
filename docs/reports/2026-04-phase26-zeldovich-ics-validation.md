# Fase 26: Condiciones Iniciales de Zel'dovich — Validación Física

**Fecha**: abril 2026  
**Estado**: implementado, tests automáticos aprobados  
**Rama**: main

---

## Objetivo

Implementar un generador de condiciones iniciales (ICs) de Zel'dovich (1LPT) para
`gadget-ng` y validar que:

1. El espectro de potencia inicial medido sigue el espectro objetivo.
2. La evolución es estable en el régimen lineal con PM y TreePM.
3. Las velocidades son físicamente consistentes con la teoría lineal.
4. El sistema es reproducible y compatible con la arquitectura MPI existente.

---

## Formulación matemática

### Aproximación de Zel'dovich (1LPT)

La aproximación de Zel'dovich relaciona las posiciones Lagrangianas **q** con las
posiciones Eulerianas **x** a través del campo de desplazamiento **Ψ**:

```
x_i = q_i + Ψ(q_i)
```

El campo de desplazamiento se obtiene del potencial de desplazamiento Φ:

```
∇²Φ = −δ   ⟹   Φ̂(k) = δ̂(k) / k²

Ψ = −∇Φ    ⟹   Ψ̂_α(k) = i · k_α / k² · δ̂(k)
```

donde `δ` es el contraste de densidad y `k_α` es la componente α del vector de onda.

### Velocidades (formulación momentum canónico GADGET-4)

En la formulación GADGET-4, el momentum canónico almacenado es `p = a² · dx_c/dt`,
donde `x_c` son coordenadas comóviles. Para la 1LPT:

```
dx_c/dt = f(a) · H(a) · Ψ(q)

⟹  p_i = a² · f(a) · H(a) · Ψ(q_i)
```

donde:
- `f(a) = d ln D / d ln a ≈ Ω_m(a)^{0.55}` (tasa de crecimiento, aproximación de Linder 2005)
- `H(a) = H₀ √(Ω_m a⁻³ + Ω_Λ)` (parámetro de Hubble)
- `D(a)` es el factor de crecimiento lineal (incluido en la amplitud de Ψ)

Para EdS (Ω_m=1, Ω_Λ=0): `f = 1` exacto, `D(a) = a`.

### Espectro de potencia inicial

Se usa un espectro de ley de potencia controlado:

```
P(k) = amplitude² · |n|^{spectral_index}
```

donde `n = (nx, ny, nz)` es el vector de modo entero (`|n|² = nx² + ny² + nz²`)
y `amplitude` es un parámetro adimensional. La varianza por modo es:

```
σ²(k) = P(k) / N³
```

El campo gaussiano `δ̂(k)` se genera con esta varianza en cada modo.

---

## Convenciones y normalización

### Unidades

| Cantidad | Unidades |
|----------|----------|
| `k` (vector de modo físico) | `2π/L` por modo entero |
| `P(k)` (estimador gadget-ng) | `L³` |
| `amplitude` | adimensional |
| Desplazamientos `Ψ` | `d = L/N` (spacing de retícula) |
| Momentum canónico `p` | `[L/t]` internas |

### Convención FFT (rustfft)

Se usa DFT forward con exponente negativo: `f̂[j] = Σ f[n] · exp(−2πi·jn/N)`.

Índices de modo con signo: `n_α = j` para `j ≤ N/2`, `n_α = j − N` para `j > N/2`.

La IFFT de rustfft **no normaliza** → se aplica factor `1/N³` manualmente.

### Simetría Hermitiana

Para garantizar que `Ψ` sea real tras la IFFT, se impone:

```
δ̂(−k) = conj(δ̂(k))
```

El modo DC (`k=0`) se fija a cero para evitar drift sistemático de todas las partículas.
Los modos Nyquist (`|n_α| = N/2`) se fijan a cero para evitar aliasing.

### Reproducibilidad

Cada modo `(nx, ny, nz)` recibe su propia semilla derivada de la semilla global:

```
seed_modo = hash_u64(seed_global XOR hash_u64(ix·N² + iy·N + iz))
```

Esto garantiza reproducibilidad bit-a-bit independientemente del orden de evaluación
y del número de rangos MPI.

### Estrategia MPI

En esta fase, el campo completo `N³` se genera en **todos los rangos** (serial),
y luego cada rango extrae su segmento `[lo, hi)` de `gid`. Esto es correcto y
reproducible para `N ≤ 64³` (campo < 200 MB por componente). La distribución
completa del generador (sin redundancia) se deja para una fase futura.

---

## Implementación

### Archivos modificados

| Archivo | Cambio |
|---------|--------|
| `gadget-ng-core/Cargo.toml` | Agrega dependencia `rustfft = "6.4.1"` |
| `gadget-ng-core/src/config.rs` | Variante `IcKind::Zeldovich { seed, grid_size, spectral_index, amplitude }` |
| `gadget-ng-core/src/cosmology.rs` | Función `growth_rate_f(params, a)` |
| `gadget-ng-core/src/ic.rs` | Brazo `IcKind::Zeldovich`, error `ZeldovichGridMismatch` |
| `gadget-ng-core/src/lib.rs` | Export de `growth_rate_f` y módulo `ic_zeldovich` |
| `gadget-ng-physics/Cargo.toml` | Agrega `gadget-ng-analysis` como dependencia |

### Archivos creados

| Archivo | Contenido |
|---------|-----------|
| `gadget-ng-core/src/ic_zeldovich.rs` | Generador completo: campo gaussiano, Ψ, partículas |
| `gadget-ng-physics/tests/zeldovich_ics.rs` | 10 tests automáticos |
| `experiments/nbody/phase26_zeldovich_ics/` | Configs, scripts, run script |
| `experiments/nbody/phase26_zeldovich_ics/scripts/validate_pk.py` | Validación P(k) |
| `experiments/nbody/phase26_zeldovich_ics/scripts/plot_growth.py` | Crecimiento D(a) |
| `experiments/nbody/phase26_zeldovich_ics/scripts/compare_pm_treepm.py` | PM vs TreePM |
| `experiments/nbody/phase26_zeldovich_ics/scripts/plot_density_slice.py` | Campo 2D |

---

## Tests automáticos

Los tests se ejecutan con:

```bash
cargo test -p gadget-ng-physics --test zeldovich_ics
```

| Test | Qué verifica | Resultado |
|------|-------------|-----------|
| `zel_reproducible` | Misma seed → partículas idénticas bit-a-bit | ✓ |
| `zel_mean_displacement_zero` | `⟨Ψ⟩ ≈ 0` (modo DC nulo) | ✓ |
| `zel_dc_mode_zero` | COM cerca del centro de caja | ✓ |
| `zel_positions_in_box` | Todas las posiciones en `[0, L)` | ✓ |
| `zel_displacement_rms_linear_regime` | `Ψ_rms/d < 0.3` (régimen lineal) | ✓ |
| `zel_pk_follows_power_law` | Pendiente P(k) compatible con n_s=-2 (±2) | ✓ |
| `zel_pm_short_run_stable` | 10 pasos PM sin NaN/Inf | ✓ |
| `zel_treepm_short_run_stable` | 10 pasos TreePM sin NaN/Inf | ✓ |
| `zel_gid_range_consistent` | Rango MPI produce partículas idénticas al build completo | ✓ |
| `zel_velocities_linear_theory` | Velocidades no nulas, consistentes con f·H·Ψ | ✓ |

Todos los tests **pasan** en CI (serial, sin MPI).

---

## Configuración TOML

```toml
[initial_conditions]
kind = { zeldovich = { seed = 42, grid_size = 32, spectral_index = -2.0, amplitude = 1.0e-4 } }
```

Restricción: `particle_count` debe ser igual a `grid_size³`. Si no, se lanza el error
`ZeldovichGridMismatch` con información explícita.

---

## Parámetros y sus efectos

| Parámetro | Rol | Rango típico |
|-----------|-----|-------------|
| `seed` | Reproducibilidad | cualquier u64 |
| `grid_size` | N de la retícula (N³ partículas) | 8–128 |
| `spectral_index` | Pendiente P(k) ∝ k^n_s | -3 a 1 |
| `amplitude` | Amplitud adimensional del campo | 1e-5 a 1e-2 |

**Régimen lineal**: se garantiza cuando `Ψ_rms/d < 0.3`, equivalente a
`amplitude < ~0.1 · N^{|n_s|/2}` para modos dominantes.

---

## Limitaciones y trabajo futuro

### Limitaciones actuales

1. **No es 2LPT**: solo se implementa la aproximación de Zel'dovich (1er orden).
   Para ICs "de producción" se necesita 2LPT (corrección de 2o orden) que corrige
   sobredensidades en filamentos y subdens en vacíos.

2. **Espectro simplificado**: se usa `P(k) ∝ k^{n_s}` sin función de transferencia.
   Para cosmología realista se necesita la función de transferencia de Eisenstein & Hu
   (o BBKS) que modela la igualdad materia-radiación y el amortiguamiento acústico.

3. **Normalización absoluta no calibrada**: el parámetro `amplitude` es adimensional
   y no está conectado a σ_8 ni a ninguna normalización del CMB. Para comparar con
   simulaciones publicadas se necesita un paso de calibración de amplitude contra σ_8.

4. **Generador serial en MPI**: todos los rangos generan el campo completo. Para
   N > 64³ esto puede ser un cuello de botella de memoria. La solución correcta
   es un generador paralelo por rango en k-space.

5. **f(a) aproximado**: se usa `f ≈ Ω_m(a)^{0.55}`. Para mayor precisión se debería
   integrar numéricamente `D(a) = H(a) ∫₀ᵃ da'/(a'H(a'))³`.

6. **Sin transfer function de velocidades**: las velocidades usan el mismo `f(a)`
   para todos los modos. En cosmologías con calor oscuro (HDM) se necesita una
   función de transferencia de velocidades modo-dependiente.

### Para ICs cosmológicas "de producción"

| Característica | Estado | Prioridad |
|----------------|--------|-----------|
| 2LPT | ✗ no implementado | Alta |
| Transfer function (Eisenstein-Hu) | ✗ no implementado | Alta |
| Calibración con σ_8 | ✗ no calibrado | Alta |
| f(a) numérico (D(a) integrado) | ✗ usa aproximación | Media |
| Generador paralelo en MPI | ✗ serial | Media |
| Velocidades CDM + HDM | ✗ no implementado | Baja |

---

## Flujo de datos del generador

```
seed, n_s, A, N
        ↓
 generate_delta_kspace()      ← campo gaussiano en k-space con simetría Hermitiana
        ↓                       σ(k) = A · |n|^(n_s/2) / N^(3/2)
 delta_to_displacement()      ← Ψ̂_α = i·n_α/|n|² · δ̂, IFFT3D, × d
        ↓
 [Ψ_x, Ψ_y, Ψ_z]            ← campo real en unidades físicas
        ↓
 zeldovich_ics()             ← x = q + Ψ, p = a²·f·H·Ψ, wrap periódico
        ↓
 Vec<Particle>               ← partículas con gid ∈ [lo, hi)
```

---

## Validación de tests unitarios del módulo

Los tests en `ic_zeldovich.rs` (módulo interno) verifican:

- `delta_dc_mode_is_zero`: `δ̂(k=0) = 0` exactamente
- `delta_hermitian_symmetry`: `|δ̂(−k) − conj(δ̂(k))| < 10⁻¹⁴`
- `displacement_field_is_real`: `|Im(Ψ)| < 10⁻¹⁰` tras IFFT
- `displacement_mean_near_zero`: `|⟨Ψ_α⟩| < 10⁻¹²`

---

## Relación con la arquitectura existente

La implementación de Zel'dovich:

- **No modifica** la arquitectura PM/TreePM distribuida (fases 18–25)
- **No requiere** cambios en los integradores cosmológicos
- **Extiende** `IcKind` en `config.rs` con una nueva variante
- **Reutiliza** `build_particles_for_gid_range` para compatibilidad MPI
- **Reutiliza** `power_spectrum` de `gadget-ng-analysis` para validación
- **Agrega** `growth_rate_f` a `cosmology.rs` (útil también para diagnósticos futuros)

---

## Conclusión

`gadget-ng` puede ahora generar condiciones iniciales de Zel'dovich físicamente
consistentes con:

- Espectro de ley de potencia controlado
- Velocidades consistentes con la teoría lineal de primer orden
- Reproducibilidad exacta con seed fija
- Compatibilidad con PM y TreePM periódicos
- 10 tests automáticos que pasan en CI

**El sistema NO afirma todavía cosmología "publicable"** sin:

1. Función de transferencia realista (Eisenstein-Hu o CAMB)
2. Calibración de normalización con σ_8
3. 2LPT para corrección de segundo orden
4. Integración numérica de D(a)

Este reporte documenta explícitamente qué está hecho, qué funciona, y qué falta
para el siguiente nivel de realismo cosmológico.
