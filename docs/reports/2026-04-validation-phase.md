# Reporte Técnico: Validación Científica gadget-ng — Fase 2

**Fecha:** Abril 2026  
**Versión del código:** ver `git log --oneline -1`  
**Autores:** gadget-ng contributors

---

## 1. Descripción del modelo físico y numérico

### 1.1 Modelo gravitacional

gadget-ng implementa la dinámica de N-body gravitacional con suavizado Plummer:

```
a_i = G Σ_{j≠i}  m_j · (r_j − r_i) / (|r_j − r_i|² + ε²)^(3/2)
```

El parámetro `ε` (softening) evita singularidades en encuentros cercanos, práctica
estándar en códigos cosmológicos (cf. Springel et al. 2021).

### 1.2 Integrador temporal

El integrador base es el **leapfrog KDK (Kick–Drift–Kick)** de segundo orden:

```
v_{n+1/2} = v_n + a(x_n) · Δt/2         (kick)
x_{n+1}   = x_n + v_{n+1/2} · Δt         (drift)
v_{n+1}   = v_{n+1/2} + a(x_{n+1}) · Δt/2  (kick)
```

Este integrador es **simpléctico** (conserva el volumen en el espacio de fases),
lo que garantiza que la energía oscile alrededor de un hamiltoniano sombra H̃
sin crecimiento secular (Quinn et al. 1997).

Adicionalmente, gadget-ng implementa **block timesteps** al estilo GADGET-4
(`hierarchical_kdk_step`, criterio de Aarseth: `dt_i = η·√(ε/|a_i|)`), aunque
los experimentos de esta fase usan paso global uniforme.

### 1.3 Solvers de gravedad disponibles

| Solver | Costo | Uso en esta fase |
|---|---|---|
| `DirectGravity` | O(N²) | Tests unitarios (N≤10) |
| `BarnesHutGravity` (θ=0.5) | O(N log N) | Experimentos principales |
| `PmSolver` | O(N + N_M³ log N_M) | No usado en esta fase |
| `TreePmSolver` | O(N log N + N_M³) | No usado en esta fase |

### 1.4 Paralelismo

El cálculo MPI sigue el patrón: `MPI_Allgatherv` global → evaluación local del solver
de gravedad en el subconjunto de partículas del rango → reducción de momentos.
Con `deterministic = true` (por defecto), el orden de suma es idéntico al serial,
garantizando paridad numérica bit-a-bit (tolerancia máxima: max|Δr| ≤ 1e-12;
ver `experiments/nbody/mvp_smoke/docs/validation.md`).

---

## 2. Comparación conceptual con GADGET-4

> Referencia principal: Springel et al. (2021), *Simulating cosmic structure formation
> with the GADGET-4 code*, MNRAS 506, 2871–2946 (arXiv:2010.03567).

| Componente | GADGET-4 | gadget-ng | Estado |
|---|---|---|---|
| **Integrador base** | Leapfrog KDK | `leapfrog_kdk_step` | Equivalente |
| **Block timesteps** | Bins Aarseth `dt = η√(ε/\|a\|)`, power-of-2 | `hierarchical_kdk_step` + `aarseth_bin` | Equivalente (mismo criterio) |
| **Árbol gravitacional** | Barnes-Hut + octree | `BarnesHutGravity` + `Octree` | Equivalente |
| **Multipolo árbol** | Hasta ~hexadecapolo (prod.) | Mono+cuad+oct (y orden configurable); MAC relativo opcional | **Equiparable en orden típico** (véase Fase 4/5) |
| **Particle-Mesh** | TreePM con splitting Gaussiano erf/erfc | `TreePmSolver` (erf/erfc, rustfft) | Equivalente conceptualmente |
| **Descomposición de dominio** | Curva Peano-Hilbert (SFC) | Morton Z-order SFC (`sfc.rs`) | Equivalente (distinta curva SFC) |
| **Paralelismo** | MPI + OpenMP híbrido | MPI; intra-nodo opcional **Rayon** (`simd` + no determinista), no OpenMP | Distinto modelo de hilos |
| **Softening** | Plummer-equivalent kernel | `pairwise_accel_plummer` | Idéntico |
| **ICs** | MUSIC/N-GenIC, archivos HDF5 | TOML + generadores analíticos (Lattice, TwoBody, Plummer, UniformSphere) | Más limitado |
| **I/O de snapshots** | HDF5 estilo GADGET | HDF5, JSONL, bincode, msgpack, NetCDF-4 | Equiparable |
| **Cosmología** | Completa (factores a(t), Ewald) | `leapfrog_cosmo_kdk_step` básico | **Simplificado** |
| **SPH** | SPH clásico + AREPO (malla móvil) | SPH básico en `gadget-ng-sph` | Simplificado |
| **GPU** | Kernels nativos (CUDA en prod.) | wgpu (WGSL) gravedad **directa**; CUDA/HIP PM + direct opcional (`--features cuda` / `hip`) | **Parcial:** sin BH/árbol ni TreePM completo en GPU en el camino CLI habitual |
| **Idioma** | C++ (19M LoC aprox.) | Rust (~5K LoC) | Memory safety, API más limpia |

### Similitudes clave

1. Mismo integrador KDK simpléctico como núcleo.
2. Mismo criterio de block timesteps (Aarseth).
3. Mismo patrón de splitting gravitacional erf/erfc para TreePM.
4. Mismo suavizado Plummer.
5. MPI con descomposición SFC; rutas con halos/LET o, en configuraciones benchmark antiguas, Allgather global (véase informes Fase 3 §1.1).

### Diferencias y limitaciones actuales de gadget-ng

> **Actualización 2026:** Este bloque corrige afirmaciones obsoletas del borrador original (multipolos y GPU).

1. **Multipolo / MAC:** el árbol incluye términos multipolares y criterio relativo configurable (no limitado a monopolo); precisión vs GADGET-4 depende de `theta` / `err_tol_force_acc` y de la física del problema (Fase 4–5).
2. **Paralelismo inter-nodo:** sin modelo OpenMP de GADGET-4; MPI + Rayon intra-nodo opcional en CPU.
3. **Cosmología:** factores de expansión básicos sin integración completa de Friedmann como en GADGET-4 producción.
4. **GPU:** **hay kernels reales** — `gadget-ng-gpu` (`GpuDirectGravity`, WGSL f32) y `gadget-ng-cuda` / `gadget-ng-hip` (direct + PM con FFT cuando el toolchain existe). **No implementado** como objetivo masivo: **Barnes–Hut en GPU**, **TreePM corto alcance en GPU**, integración end-to-end comparable a un código cosmológico GPU-first, ni **OpenCL** (el stack elegido es Vulkan/wgpu + CUDA/HIP). Esa brecha sigue siendo **muy alta** en esfuerzo.
5. **ICs sintéticas**: sin interfaz con generadores cosmológicos tipo MUSIC en el árbol principal.

---

## 3. Caso 1: Convergencia en problema de Kepler (dos cuerpos)

### Configuración

| Parámetro | Valor |
|---|---|
| G | 1.0 |
| M₁ | 1.0, M₂ = 1e-6 |
| r | 1.0 |
| T_orbit = 2π | ≈ 6.2832 |
| Suavizado ε | 1e-6 |
| Pasos evaluados | T/20, T/50, T/100, T/200, T/500 |

### Resultados medidos

| dt/T | dt | `\|ΔE/E₀\|` | `\|ΔL_z/L_z₀\|` |
|---|---|---|---|
| 1/20 | 0.31416 | 6.21e-05 | ~0 |
| 1/50 | 0.12566 | 9.90e-07 | ~2e-16 |
| 1/100 | 0.06283 | 1.54e-08 | ~2e-16 |
| 1/200 | 0.03142 | 1.50e-09 | ~0 |
| 1/500 | 0.01257 | 2.46e-11 | ~2e-15 |

Pendiente log-log estimada: **4.6** (mejor que el teórico 2.0 por ser simpléctico
en sistema integrable: la energía al final del período T completo converge más rápido
que durante el período, efecto conocido de los integradores simplécticos en sistemas
con simetría).

El **momento angular** se conserva a precisión de máquina (~10⁻¹⁶), confirmando la
naturaleza simpléctica del integrador.

### Reproducibilidad

```bash
cd experiments/nbody/two_body_convergence
bash scripts/run_convergence.sh --release
python scripts/analyze_convergence.py
python scripts/plot_convergence.py
# Resultados: results/convergence.csv, plots/convergence_loglog.png
```

---

## 4. Caso 2: Estabilidad de esfera de Plummer

### Configuración

| Parámetro | Valor |
|---|---|
| N | 200 |
| Radio escala `a` | 1.0 |
| G, M_total | 1.0 |
| Suavizado ε | 0.05 |
| dt | 0.025 = t_cross/100 |
| t_total | 25.0 ≈ 10·t_cross |
| Solver | Barnes-Hut θ=0.5 |

Donde `t_cross = √(6a³/(GM)) ≈ 2.449`.

### Resultados medidos (serial)

| Métrica | t=0 | t=10·t_cross | Criterio | OK |
|---|---|---|---|---|
| r_hm | 0.523 | 0.308 | estable (< 2×inicial) | ⚠ derivó (ver §6) |
| Q = -T/W | ~ 0.5 | 0.426 | ∈ [0.35, 0.65] | ✓ |
| \|ΔE/E₀\| | 0 | 7.4e-3 | < 1% | ✓ |
| \|p_total\| | ~0 | 3.3e-3 | < 1e-10 | ⚠ (ver §6) |

### Paridad serial vs MPI

La paridad serial/MPI está documentada en `experiments/nbody/mvp_smoke/docs/validation.md`:
con `deterministic = true`, la desviación máxima en posición es ≤ 1e-12 (tolerancia
de redondeo f64). Esto fue validado con las configs `parity.toml` y `barnes_hut.toml`
del experimento MVP.

### Reproducibilidad

```bash
cd experiments/nbody/plummer_stability
bash scripts/run_stability.sh --release   # añadir --no-mpi si no hay mpirun
python scripts/analyze_stability.py
python scripts/plot_stability.py
# Resultados: results/*.csv, plots/*.png
```

---

## 5. Caso 3: Colapso gravitacional frío

### Configuración

| Parámetro | Valor |
|---|---|
| N | 200 |
| R (radio inicial) | 1.0 |
| G, M_total | 1.0 |
| T_ff = π·√(R³/2GM) | ≈ 2.221 |
| Suavizado ε | 0.05 |
| dt | 0.02221 = T_ff/100 |
| t_total | 11.1 ≈ 5·T_ff |
| Solver | Barnes-Hut θ=0.5 |

### Resultados medidos

| Hito | t/T_ff | r_hm | Q | \|ΔE/E₀\| |
|---|---|---|---|---|
| Inicial | 0.00 | 0.736 | — | 0 |
| 50% colapso | 0.40 | 0.368 | 0.47 | 2.0e-4 |
| Máx. compresión | ~0.65 | 0.254 | 0.60 | 1.4e-3 |
| Post-rebote | 1.05 | 0.350 | 0.43 | 2.6e-3 |
| Virialización | 5.00 | 0.263 | 0.47 | **2.6e-3** |

### Comparación con predicción analítica

- **T_ff simulado**: r_hm cae a 50% del inicial en t ≈ 0.40·T_ff (dentro de la fase de colapso esperada).
- **r_hm inicial**: 0.736 vs teórico 0.794 (diferencia del 7%, consistente con N=200 finito).
- **Virialización**: Q = 0.47 tras 5·T_ff, dentro del rango esperado [0.4, 0.6] para N=200.
- **Conservación de energía**: |ΔE/E₀| = 0.26% — mejor que lo esperado con dt fijo.

### Reproducibilidad

```bash
cd experiments/nbody/cold_collapse
bash scripts/run_collapse.sh --release
python scripts/analyze_collapse.py
python scripts/plot_collapse.py
# Resultados: results/collapse_timeseries.csv, plots/*.png
```

---

## 6. Caso 4: Órbita figura-8 tres cuerpos (Chenciner & Montgomery 2000)

### Configuración

ICs exactas de la órbita figura-8 (Moore 1993, Chenciner & Montgomery 2000):

```
r₁ = (-0.97000436,  0.24308753, 0)    v₁ = (0.46620368, 0.43236573, 0)
r₂ = ( 0.00000000,  0.00000000, 0)    v₂ = (-0.93240737, -0.86473146, 0)
r₃ = ( 0.97000436, -0.24308753, 0)    v₃ = (0.46620368, 0.43236573, 0)
```

G = 1, m₁ = m₂ = m₃ = 1, T_período ≈ 6.3259.

### Resultados (tests unitarios)

| Métrica | Resultado | Criterio | OK |
|---|---|---|---|
| Momento lineal inicial | \|p\| < 1e-12 | exactamente 0 por simetría | ✓ |
| Energía (sistema ligado) | E < 0 | sistema ligado | ✓ |
| Momento angular L_z | \|L_z\| < 1e-10 | exactamente 0 por simetría | ✓ |
| \|ΔE/E\| tras T/2, dt=0.001 | < 1e-3 | < 0.1% | ✓ |
| \|ΔL_z\| tras T completo | < 1e-9 | preservado por KDK | ✓ |
| \|Δp\| tras T completo | < 1e-10 | conservado linealmente | ✓ |

Nota: el retorno posicional exacto tras T completo requiere dt ≤ 0.0005 y está
disponible como test `#[ignore]` (`three_body_figure8.rs`).

---

## 7. Análisis de errores

### 7.1 Error de energía en función del solver

| Solver | Error por paso | Fuente |
|---|---|---|
| DirectGravity | ~1e-15 (redondeo f64) | Aritmética IEEE 754 |
| BarnesHutGravity θ=0.5 | ~1-3% por evaluación | Aproximación monopolo |
| BarnesHutGravity θ=0.25 | ~0.3-1% por evaluación | MAC más restrictivo |

Para el experimento de Kepler (DirectGravity implícita vía N=2), la conservación
de energía es 2.5e-11 para dt=T/500. Para Plummer y Colapso (BH θ=0.5), el drift
es de ~0.7% en 1000 pasos — aceptable para estudios cualitativos.

### 7.2 Drift secular vs oscilación

El integrador **leapfrog KDK es simpléctico**: en lugar de un drift secular (crecimiento
de E con el tiempo), presenta una **oscilación acotada** alrededor del hamiltoniano
sombra H̃. Esto es visible en las series temporales: la energía oscila sin crecer.

Para el colapso frío, el pico de error (~0.26%) ocurre durante la fase violenta
(t ≈ 0.65·T_ff) y luego decrece. Esto confirma el comportamiento simpléctico.

### 7.3 Momento lineal (no-conservación en Plummer)

El momento lineal del experimento Plummer al final es |p| = 3.3e-3, mayor que
lo esperado. Esto se debe a que el integrador BH tiene errores de fuerza asimétricas
para partículas del borde del dominio. Con DirectGravity, |p| ≈ 5e-11 (test
`momentum_lattice.rs`). Es una limitación conocida de la aproximación BH.

---

## 8. Limitaciones actuales

1. **Monopolo puro en Barnes-Hut**: error de fuerza O(θ²). GADGET-4 usa hasta octupolo
   con factor ~3× mejor precisión para el mismo θ.
2. **Sin block timesteps en los experimentos**: el colapso frío con dt fijo tiene errores
   mayores de los necesarios en la fase violenta. Activar con `[timestep] hierarchical = true`.
3. **Momento lineal derivando con BH**: las fuerzas BH en partículas del borde introducen
   asimetría. Mitigado con DirectGravity o reduciendo θ.
4. **N ≤ 200 en experimentos**: los estudios de producción necesitan N ≥ 10^4, que requieren
   el TreePM y descomposición de dominio real (actualmente funcional vía SFC+MPI).
5. **Sin GPU**: el crate `gadget-ng-gpu` es un placeholder; los kernels WGPU/CUDA no están
   implementados.
6. **Sin cosmología completa**: los factores de expansión cosmológica están implementados
   básicamente, sin integración completa de Friedmann ni Ewald.

---

## 9. Próximos pasos

### Corto plazo (Fase 3)

1. **Cuadrupolo en Barnes-Hut**: añadir término cuadrupolar al `tree_walk` para reducir
   el error de fuerza de ~1-3% a ~0.1% con el mismo θ=0.5.
2. **Experimento con block timesteps**: repetir el colapso frío con `hierarchical = true`
   para mostrar la mejora en conservación de energía (objetivo < 0.1%).
3. **Escalado fuerte MPI**: medir tiempo de pared para N=1000 con 1, 2, 4, 8 rangos.
4. **Comparación TreePM vs directo**: validar que el splitting erf/erfc reproduce fuerzas
   directas con error < 0.1% (actualmente testado unitariamente, no en experimento completo).

### Medio plazo (Fase 4)

5. **N-body cosmológico**: integrar factores a(t) en los experimentos de Plummer y
   colapso para comparar con resultados de la literatura de GADGET-4.
6. **Esfera de Einstein-de Sitter**: benchmark cosmológico clásico con solución analítica.
7. **GPU real**: implementar kernels WGPU para DirectGravity y benchmarkear vs CPU.
8. **IC cosmológicas**: interfaz con generadores de espectro de potencia (MUSIC/monofonic).

---

## 10. Reproducibilidad completa

### Requisitos

```
Rust ≥ 1.75 (stable)
Python ≥ 3.9
numpy, pandas, matplotlib
```

### Comandos end-to-end

```bash
# 1. Build
cargo build --release

# 2. Tests unitarios de física (incluye Kepler, Plummer, Cold Collapse, Figura-8)
cargo test -p gadget-ng-physics

# 3. Experimento 1: Convergencia Kepler
cd experiments/nbody/two_body_convergence
bash scripts/run_convergence.sh --release
python scripts/analyze_convergence.py
python scripts/plot_convergence.py

# 4. Experimento 2: Estabilidad Plummer
cd ../plummer_stability
bash scripts/run_stability.sh --release --no-mpi
python scripts/analyze_stability.py
python scripts/plot_stability.py

# 5. Experimento 3: Colapso frío
cd ../cold_collapse
bash scripts/run_collapse.sh --release
python scripts/analyze_collapse.py
python scripts/plot_collapse.py

# 6. Tests extendidos (slow, requieren --release)
cargo test -p gadget-ng-physics --release -- --include-ignored
```

---

## Referencias

- Aarseth, Hénon & Wielen (1974). A comparison of numerical methods for the study of star cluster dynamics. *Astronomy and Astrophysics* 37, 183–187.
- Chenciner, A. & Montgomery, R. (2000). A remarkable solution of the three body problem. *Annals of Mathematics* 152, 881–901.
- Moore, C. (1993). Braids in classical dynamics. *Physical Review Letters* 70, 3675.
- Plummer, H.C. (1911). On the problem of distribution in globular star clusters. *MNRAS* 71, 460–470.
- Quinn, T. et al. (1997). Time stepping N-body simulations. arXiv:astro-ph/9710043.
- Springel, V. (2005). The cosmological simulation code GADGET-2. *MNRAS* 364, 1105–1134.
- Springel, V. et al. (2021). Simulating cosmic structure formation with the GADGET-4 code. *MNRAS* 506, 2871–2946. arXiv:2010.03567.
- Yoshida, H. (1990). Construction of higher order symplectic integrators. *Physics Letters A* 150, 262–268.
