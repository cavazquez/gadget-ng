# 🌌 gadget-ng

> Simulador **N-body + SPH + MHD cosmológico** en Rust, inspirado conceptualmente en la
> arquitectura y prácticas de [GADGET-4](https://wwwmpa.mpa-garching.mpg.de/gadget4/),
> sin compartir código ni historial git con el proyecto original.
>
> Cubre el pipeline completo de una corrida cosmológica contemporánea:
> **ICs 1LPT/2LPT con transferencia Eisenstein–Hu y normalización σ₈
> (`Legacy`/`Z0Sigma8`) → integrador leapfrog/Yoshida4 con factores de
> drift/kick cosmológicos → PM periódico / TreePM / Barnes–Hut / Direct, con
> versiones distribuidas (allreduce, slab alltoall, pencil 2D, scatter-gather)
> → SPH cosmológico con gas + DM + estrellas + metales + SN feedback + AGN
> → MHD ideal + SRMHD + turbulencia Ornstein-Uhlenbeck + reconexión Sweet-Parker
>    + Braginskii + plasma de dos fluidos T_e ≠ T_i
> → transferencia radiativa M1 multifrecuencia + reionización EoR z=6–12 + estadísticas 21cm
> → rayos cósmicos + conducción térmica anisótropa + polvo intersticial UV/IR
> → Pop III, química D/HD, WDM/FDM y f(R) con screening PM
> → análisis in-situ (FoF, P(k), P_B(k), HMF, ξ(r), c(M) Ludlow+2016)
> → corrección absoluta de `P(k)` vía `pk_correction` (Phase 34–36)
> → validación externa contra CLASS (Phase 38)
> → block timesteps jerárquicos + LET distribuido (Phase 56)
> → solver PM GPU CUDA/HIP opcional (Phase 57)
> → CLI `analyze` con pipeline completo FoF+P(k)+ξ(r)+c(M) y render PPM**.
>
> **Estado:** Phases 1–185 completadas · **v0.2.0** (CHANGELOG) · **cartera Physics Extensions cerrada** (D/HD, Pop III, RT multifrecuencia, polvo IR, AGN spin/mergers, WDM/FDM y f(R) no lineal PM) · **SPH Gadget-2 completo** (entropía + Balsara + colapso de Evrard) ·
> GPU kernels reales CUDA/HIP N² · MHD 3D solenoidal · `cargo test -p gadget-ng-physics` en ~3.5 min.
>
> **Mayo 2026 — validación tipo paper:** transferencia **tabulada** CLASS/CAMB en ICs, **pancake** Zel’dovich analítico,
> MAC **geométrico bmax** + **relativo** en árbol local y LET, backend FFT opcional **`fftw`** en PM con tests de paridad,
> workflow nocturno [**physics-validation**](.github/workflows/physics-validation.yml), script local
> [`scripts/check-physics.sh`](scripts/check-physics.sh) y módulo **lightcone** (κ/γ Born) en análisis.

## 🧰 Herramientas y tecnologías

[![🦀 Rust](https://img.shields.io/badge/🦀_Rust-1.85%2B_%28edition_2024%29-orange?logo=rust&logoColor=white)](https://www.rust-lang.org/)
[![🐍 Python](https://img.shields.io/badge/🐍_Python-3.10%2B-yellow?logo=python&logoColor=white)](https://www.python.org/)
[![🔌 MPI](https://img.shields.io/badge/🔌_MPI-OpenMPI%2FMPICH-red)](https://www.open-mpi.org/)
[![⚡ wgpu](https://img.shields.io/badge/⚡_wgpu-Vulkan%2FMetal%2FDX12-purple)](https://wgpu.rs/)
[![🟢 CUDA](https://img.shields.io/badge/🟢_CUDA-nvcc%2BcuFFT-76B900?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![🔴 HIP](https://img.shields.io/badge/🔴_HIP-ROCm%2BrocFFT-ED1C24?logo=amd&logoColor=white)](https://rocm.docs.amd.com/)
[![🌐 HDF5](https://img.shields.io/badge/🌐_HDF5-1.10%2B-blue)](https://www.hdfgroup.org/solutions/hdf5/)
[![📊 NumPy](https://img.shields.io/badge/📊_NumPy-SciPy%2FMatplotlib-blue?logo=numpy&logoColor=white)](https://numpy.org/)
[![🔭 CLASS](https://img.shields.io/badge/🔭_CLASS-classy_3.3%2B-green)](https://lesgourg.github.io/class_public/class.html)
[![🧲 MHD](https://img.shields.io/badge/🧲_MHD-Ideal%2BSRMHD%2BTurbulencia-blueviolet)](crates/gadget-ng-mhd/)
[![⚛️ Plasma2F](https://img.shields.io/badge/⚛️_Plasma-2_Fluidos_T_e≠T_i-teal)](crates/gadget-ng-mhd/src/two_fluid.rs)
[![🧪 GitHub_Actions](https://img.shields.io/badge/🧪_GitHub_Actions-CI-lightgrey?logo=githubactions&logoColor=white)](.github/workflows/ci.yml)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/cavazquez/gadget-ng)
[![📜 License](https://img.shields.io/badge/📜_License-GPL--3.0-blue)](LICENSE)

| Ámbito | Herramientas |
|---|---|
| 🦀 **Core** | Rust 1.85+ (edición 2024), workspace multi-crate, `serde`, `toml`, `clap`, `rayon` |
| 🔌 **Cómputo paralelo** | MPI (`OpenMPI` / `MPICH` vía `mpi` crate), SFC Hilbert 3D, LET, alltoall/allreduce/scatter-gather |
| ⚡ **GPU (wgpu)** | `wgpu` con shaders WGSL (Vulkan / Metal / DX12 / WebGPU) |
| 🟢 **GPU CUDA** | Solver PM NVIDIA CUDA + cuFFT; kernels CIC + Poisson + FFT 3D; degradación elegante sin toolchain |
| 🔴 **GPU HIP/ROCm** | Solver PM AMD HIP + rocFFT; misma arquitectura que CUDA; `CUDA_SKIP` / `HIP_SKIP` para CI |
| 🧲 **MHD** | MHD ideal SPH, SRMHD, Braginskii, reconexión Sweet-Parker, turbulencia Ornstein-Uhlenbeck, flux-freeze ICM |
| ⚛️ **Plasma 2 fluidos** | `T_e ≠ T_i`, acoplamiento Coulomb implícito, jets AGN relativistas, espectro P_B(k) |
| 💫 **Física bariónica** | SPH Wendland C2, cooling metálico, SF estocástica, vientos, SN Ia+II, AGN bimodal + spin/mergers, rayos cósmicos, polvo UV/IR, Pop III |
| 🌅 **Radiación** | RT M1 (HLL + Levermore), grupos HI/HeI/HeII/LW/IR, reionización EoR z=6–12, señal 21cm δT_b, T(z) IGM |
| 🧬 **Física de frontera** | Química primordial 12 especies D/HD, WDM/FDM en ICs, f(R) Hu-Sawicki con boost PM y screening no lineal de malla |
| 🗜️ **I/O** | JSONL, `bincode`, `hdf5` estilo GADGET-4 (22 attrs, yt/pynbody compatible), `msgpack`, `netcdf` |
| 🔭 **Cosmología de referencia** | `classy` 3.3+ (CLASS) para validación externa (Phase 38), Eisenstein–Hu interno |
| 📊 **Postproceso** | Python 3.10+, NumPy, SciPy, Matplotlib — mirrors de `pk_correction` y figuras por fase |
| 🖼️ **Visualización PPM** | `render_ppm` + `write_ppm` sin dependencias externas; salida P6 legible por GIMP/ImageMagick |
| 🧪 **CI / calidad** | GitHub Actions (`ci.yml` + **physics-validation** nocturno), `cargo fmt`, `cargo clippy --workspace` (0 warnings), `cargo test --release`; snapshot advisory `clippy --all-targets` (ver [backlog](docs/reports/2026-05-clippy-all-targets-backlog.md)) |
| 📁 **Experimentos** | TOML + orquestadores Bash + volcados JSON cacheables en `target/phaseNN/` |

---

> **¿Primera vez aquí?** Lee [docs/getting-started.md](docs/getting-started.md)
> para tener tu primera simulación corriendo en menos de 10 minutos.
> Los [notebooks interactivos](notebooks/) te guían paso a paso desde Python.

Índice de documentación: [docs/README.md](docs/README.md). Referencias rápidas: [Guía de usuario](docs/user-guide.md) · [Desde GADGET-4](docs/from-gadget4.md) · [Validación vs referencia GADGET-4](docs/runbooks/validation-vs-gadget4-reference.md) · [Arquitectura](docs/architecture.md).

## Tabla de contenidos

1. [🧰 Herramientas y tecnologías](#-herramientas-y-tecnologías)
2. [Características](#características)
3. [Inicio rápido](#inicio-rápido)
4. [Configuración TOML](#configuración-toml)
5. [Crates del workspace](#crates-del-workspace)
6. [Hitos de desarrollo](#hitos-de-desarrollo)
7. [Arquitectura de comunicación PM](#arquitectura-de-comunicación-pm)
8. [Condiciones iniciales y validación cosmológica](#condiciones-iniciales-y-validación-cosmológica)
9. [Corrección absoluta de `P(k)` (`pk_correction`)](#corrección-absoluta-de-pk-pk_correction)
10. [Convención de ICs y validación dinámica (Phase 37–42)](#convención-de-ics-y-validación-dinámica-phase-3742)
11. [Análisis post-procesamiento (Phase 56–60 + Rápidas)](#análisis-post-procesamiento-phase-5660--rápidas)
12. [Estadísticas avanzadas y transferencia radiativa (Phases 71–82)](#estadísticas-avanzadas-y-transferencia-radiativa-phases-7182)
13. [Reionización y RT (Phases 87–92)](#reionización-y-rt-phases-8792)
14. [Estadísticas 21cm y EoR (Phases 94–95)](#estadísticas-21cm-y-eor-phases-9495)
15. [Feedback AGN (Phase 96)](#feedback-agn-phase-96)
16. [SPH Gadget-2 (Phase 166)](#sph-gadget-2-phase-166)
17. [Physics Extensions cerradas (Phases 177–185)](#physics-extensions-cerradas-phases-177185)
18. [MHD Completo + Plasma 2F (Phases 123–150)](#mhd-completo--plasma-2f-phases-123150)
19. [Tests automáticos](#tests-automáticos)
20. [Reportes técnicos](#reportes-técnicos)
21. [Features opcionales](#features-opcionales)
22. [Calidad y CI](#calidad-y-ci)
23. [Estructura de experimentos](#estructura-de-experimentos)
24. [Licencia](#licencia)

---

## Características

### Avances mayo 2026 (validación tipo paper)

| Tema | Qué hay y dónde |
|------|-----------------|
| **ICs tabuladas** | `TransferKind::Tabulated { path }`: archivo texto `k [h/Mpc]`, `T(k)` (p. ej. CLASS/CAMB), interpolación log–log monotónica (PCHIP). Unidad: `ic_zeldovich::tests::tabulated_transfer_reconstructs_knots_and_midpoints`. |
| **Pancake Zel’dovich** | Tests `zeldovich_pancake` en `gadget-ng-physics`; reporte [phase160-zeldovich-pancake](docs/reports/2026-05-phase160-zeldovich-pancake.md). |
| **MAC árbol** | `OpeningCriterion::GeometricBmax` (criterio `b_max/d`) además de `geometric` y `relative`; comparación error/costo en `gadget-ng-tree` → `mac_error_cost`. |
| **PM / FFT** | Feature opcional **`fftw`** en `gadget-ng-pm`: segundo backend con **paridad numérica** frente a RustFFT en `fft_poisson` / `slab_fft` (gancho listo para FFTW nativo en host). |
| **CI física** | [`.github/workflows/physics-validation.yml`](.github/workflows/physics-validation.yml): corrida programada + artefactos (pancake, growth largo, producción); clippy `--all-targets` en modo advisory. |
| **Chequeo local** | `bash scripts/check-physics.sh` — transfer tabulada (unit) + pancake + suite CLASS completa con `--include-ignored`. |
| **Lightcone / lensing** | `gadget_ng_analysis::lightcone`: cruces de cono de luz y acumulación Born para mapas de convergencia y shear (κ, γ). |
| **Clippy all-targets** | Deuda y plan documentados en [2026-05-clippy-all-targets-backlog](docs/reports/2026-05-clippy-all-targets-backlog.md). |

| Componente | Descripción |
|---|---|
| **Integradores** | Leapfrog KDK + **Yoshida4** KDK, newtonianos **y cosmológicos** (drift/kick `∫dt'/a²`, `∫dt'/a`); timestep adaptativo estilo Aarseth |
| **Gravedad directa** | Pares Plummer-suavizados O(N²) — `DirectGravity` con SoA + SIMD |
| **Barnes–Hut + FMM** | Octree en arena, MAC `s/d < θ`, criterio **`b_max/d`** (`GeometricBmax`), MAC **relativo** tipo GADGET-4; monopolo + cuadrupolo + **octupolo STF**, suavizado multipolar |
| **PM periódico** | CIC + FFT Poisson 3D periódica; solver `pm` (Fase 18); backend FFT por defecto RustFFT, opcional **`fftw`** (`gadget-ng-pm`) |
| **PM distribuido (allreduce)** | `allreduce_sum_f64_slice` O(nm³) — elimina `allgather` O(N·P) (Fase 19) |
| **PM slab (alltoall)** | Slab decomposition Z: FFT 3D distribuida con `alltoall_transpose`, grid **no replicado** (Fase 20) |
| **PM pencil 2D** | Descomposición 2D; escala hasta `P ≤ nm²` en lugar de `P ≤ nm`; `alltoallv_f64_subgroup` (Phase 46) |
| **TreePM** | Barnes–Hut short-range + PM long-range; versión distribuida (Fase 21–25) con scatter-gather (Fase 24); softening Plummer absoluto `ε_phys` (Mpc/h) independiente de `N` (Phase 42) |
| **Cosmología ΛCDM** | Friedmann ΛCDM, factor de escala `a(t)` por RK4, momentum canónico, diagnósticos `a/z/v_rms/δ_rms`; fallback EdS |
| **ICs cosmológicas** | Retícula cúbica + ZA (**1LPT**) y **2LPT**; transfer **Eisenstein–Hu no-wiggle**, **tabulada** (CLASS/CAMB) o ley de potencia; normalización σ₈ con `NormalizationMode { Legacy, Z0Sigma8 }` (Phase 40) |
| **WDM/FDM en ICs** | `[dark_matter]` aplica cutoff warm/fuzzy sobre amplitudes ZA/2LPT; half-mode y proxy de presión cuántica FDM (Phase 184) |
| **P(k) + corrección** | Estimador CIC + deconvolución; módulo [`pk_correction`](crates/gadget-ng-analysis/src/pk_correction.rs) con `A_grid = 2·V²/N⁹` (Phase 34) + `R(N)` (Phase 35) para amplitud absoluta, validado vs 🔭 CLASS (Phase 38) y en alta resolución hasta `N=128³` (Phase 41) |
| **MPI** | `ParallelRuntime` con SFC (**Hilbert 3D**), Locally Essential Trees (LET), overlap compute/comm |
| **SPH Gadget-2** | Kernel Wendland C2, densidad adaptativa; **entropía A=P/ρ^γ** (Springel & Hernquist 2002); **viscosidad de señal** + **limitador de Balsara** (∇·v, ∇×v); integrador KDK-entropía + `courant_dt`; validado en tubo de Sod + **colapso de Evrard** (Phase 166) |
| **GPU** | Compute shader WGSL vía `wgpu` (Vulkan/Metal/DX12/WebGPU); fallback CPU automático |
| **Análisis in-situ** | FoF (halos), espectro de potencia P(k), catálogos JSONL; HMF Press-Schechter/Sheth-Tormen; perfiles NFW + c(M) Duffy/Bhattacharya/Ludlow+2016; Halofit no-lineal (Takahashi+2012); función de correlación ξ(r) FFT + pares; **lightcone** + mapas Born κ/γ (`gadget-ng-analysis`) |
| **🟢🔴 GPU PM** | Solver PM CUDA/HIP opcional (cuFFT/rocFFT); CIC+Poisson+FFT 3D en GPU; degradación elegante automática si no hay toolchain |
| **⏱️ Block timesteps** | Integrador jerárquico acoplado al árbol LET distribuido; evaluación de fuerzas solo para partículas activas (`active_local`), O(N_active) por subnivel |
| **Checkpointing** | Guarda/reanuda desde snapshots comprimidos (`--resume`); continuidad bit-a-bit verificada; `SfcDecomposition` reconstruida automáticamente post-restart |
| **🔀 Rebalanceo adaptativo** | `rebalance_imbalance_threshold` configurable: si `max/min(walk_ns) > threshold` → rebalanceo inmediato, independientemente del intervalo fijo |
| **Visualización** | Render CPU a PNG, proyecciones XY/XZ/YZ, colormap Viridis; `render_ppm`/`write_ppm` PPM sin dependencias externas |
| **📊 CLI `analyze`** | `gadget-ng analyze`: FoF + P(k) + ξ(r) + c(M) desde cualquier snapshot → `results.json` estructurado |
| **Configuración** | TOML + variables de entorno `GADGET_NG_*` |
| **Snapshots** | JSONL (default), **bincode** o **HDF5** estilo GADGET-4 (22 attrs, yt/pynbody compatible) + `provenance.json` |
| **Unidades físicas** | Sección `[units]` opcional: kpc/M☉/km·s⁻¹ y `G` coherente; `auto_g = true` calcula `G = 3Ω_mH₀²/(8π)` automáticamente (Phase 50–51) |
| **🧲 MHD ideal** | Ecuación de inducción SPH, limpieza de div-B Dedner, presión + tensor Maxwell, ICs magnetizadas BFieldKind{None,Uniform,Random,Spiral}, CFL Alfvén (Phase 123–132) |
| **⚡ SRMHD** | MHD especial-relativista: factor de Lorentz γ, primitivización Newton-Raphson, advance_srmhd; jets AGN bipolares v_jet=0.3–0.9c (Phase 139 + 148) |
| **🌪️ Turbulencia MHD** | Forzado Ornstein-Uhlenbeck, espectro k^{-5/3}, semilla reproducible, números de Mach sónico y Alfvénico (Phase 140) |
| **🔁 Reconexión** | Modelo Sweet-Parker SPH: B antiparalelos en 2h → libera ΔE_heat, reduce \|B\| (Phase 145) |
| **🌡️ Braginskii** | Viscosidad anisótropa π_ij = −η (b̂⊗b̂ − δ/3) ∇·v; máxima difusión ∥B, nula ⊥B (Phase 146) |
| **⚛️ Plasma 2F** | T_e independiente de T_i; acoplamiento Coulomb implícito ν_ei ∝ n_e/T_e^{3/2}; Particle.t_electron (Phase 149) |
| **❄️ Flux-freeze ICM** | B ∝ ρ^{2/3} en celdas con β > β_freeze; conservación de flujo magnético en fases de baja ionización (Phase 138) |
| **📡 P_B(k)** | Espectro de potencia magnético por bins logarítmicos de k ∝ 2π/h_i; diagnóstico de magnetogénesis primordial (Phase 147) |
| **💥 Conducción anisótropa** | Difusión térmica y CR ∥B con factores kappa_par/kappa_perp; supresión CR por β_plasma (Phase 133) |
| **🌑 Polvo + RT UV/IR** | Ratio polvo/gas D/G, absorción UV, presión radiativa, temperatura de granos y emisión IR a `PhotonGroup::Infrared` (Phases 130, 137, 182) |
| **🧪 Química primordial + Pop III** | `ChemState` 12 especies con H⁻/H₂/H₂⁺/D/D⁺/HD, cooling HD, Pop III top-heavy + PISN y LW fotodisociación (Phases 178–181) |
| **🕳️ AGN avanzado** | Bondi + modo radio/quasar, spin Kerr escalar, eficiencia dependiente de spin, mergers BH y recoil (Phases 96, 116, 183) |
| **🌀 f(R) PM no lineal** | Hu-Sawicki f(R): boost PM homogéneo `4/3` y screening chameleon espacial en malla PM `ρ×S(x)/3` (Phases 178, 185) |

### Matriz de paridad de aceleradores

Estado resumido de implementación por backend. Para el backlog vivo ver
[`docs/reports/2026-05-accelerator-parity-pending.md`](docs/reports/2026-05-accelerator-parity-pending.md)
y la cobertura detallada en
[`docs/reports/2026-05-simd-cuda-coverage.md`](docs/reports/2026-05-simd-cuda-coverage.md).

| Área / módulo | CPU sin Rayon | CPU con Rayon | SIMD sin Rayon AVX2/AVX512 | CUDA |
|---|---:|---:|---:|---:|
| Gravedad directa `O(N²)` | ✅ | ✅ | ✅ AVX2 + AVX512 | ✅ |
| Barnes-Hut / Tree local | ✅ | ✅ | ✅ AVX2 + AVX512 local walk | ⚠️ kernel monopole parity |
| Tree LET / RMN SoA | ✅ | ✅ | ✅ AVX2 + AVX512 | ❌ full LET traversal |
| TreePM corto alcance | ✅ | ✅ | ✅ AVX2 + AVX512 SR kernel | ⚠️ wgpu/CUDA híbrido parcial |
| PM CIC assign/interp | ✅ | ✅ | ✅ AVX2 + AVX512 | ✅ |
| PM FFT/Poisson | ✅ | ✅ k-space + PM path | ✅ AVX2 + AVX512 spectral kernel | ✅ |
| SPH density | ✅ | ✅ | ✅ Wendland AVX2 + AVX512 batch | ✅ |
| SPH forces clásico | ✅ | ✅ | ✅ Wendland AVX2 + AVX512 batch | ✅ |
| SPH Gadget-2/Balsara | ✅ | ✅ | ✅ Wendland AVX2 + AVX512 batch | ✅ |
| Cooling H/He/metales/UVB | ✅ | ✅ | ✅ AVX2 + AVX512 per-particle batch; MetalTabular logT lookup batched | ✅ |
| Dust update / radiation pressure | ✅ | ✅ | ✅ AVX2 + AVX512 growth/sputtering/radiation kick | ✅ |
| Molecular H₂ / shielding | ✅ | ✅ | ✅ AVX2 + AVX512 H₂ + dust shielding | ✅ |
| MHD induction/resistivity | ✅ | ✅ | ✅ AVX2 + AVX512 induction and resistivity pair accumulation | ✅ smoke/parity kernel |
| MHD magnetic forces | ✅ | ✅ | ✅ AVX2 + AVX512 pair accumulation | ✅ smoke/parity kernel |
| MHD Dedner cleaning | ✅ | ✅ paralelo por partícula; pares escalar | ✅ AVX2 + AVX512 density + pairwise inner batch (Wendland kernel) + final-update | ✅ |
| MHD anisotropic conduction / CR diffusion | ✅ | ✅ | ✅ AVX2 + AVX512 conduction + CR diffusion pair accumulation | ✅ scalar diffusion surface |
| MHD Braginskii | ✅ | ✅ | ✅ AVX2 + AVX512 anisotropic pair accumulation | ✅ |
| MHD reconnection | ✅ | ✅ | ✅ AVX2 + AVX512 pair prefilter/update | ✅ combined kernel |
| MHD CR streaming / dynamo | ✅ | ✅ | ✅ AVX2 + AVX512 streaming local update + dynamo B-field update + energy ratio | ✅ combined kernel |
| MHD ambipolar diffusion (nonideal) | ✅ | ✅ | ✅ AVX2 + AVX512 B-field damping + ionization proxy + heating | ✅ |
| MHD two-fluid (e-i coupling) | ✅ | ✅ | ✅ AVX2 + AVX512 Coulomb coupling + T_e/T_i reduction | ✅ |
| SPH cooling (atomic/metal/UVB) | ✅ | ✅ | ✅ AVX2 + AVX512 per-particle batch | ✅ |
| MHD flux-freeze / stats | ✅ | ✅ | ✅ AVX2 + AVX512 (flux-freeze scaling + mean density); b-field stats real AVX512 8-lane | ✅ |
| RT M1 diagnostics/photoheating | ✅ | ✅ | ✅ AVX2 + AVX512 diagnostics/photoheating | ✅ |
| RT full M1 advection | ✅ | ✅ advección + update | ✅ final update AVX2 + AVX512 | ❌ |
| RT chemistry rates/cooling | ✅ | ✅ | ✅ AVX2 + AVX512 photoionization rates + cooling | ❌ |
| RT chemistry stiff solver | ✅ | ✅ | ✅ AVX2 + AVX512 masked-lane dispatch; stiff update scalar-per-lane with chunk/tail parity tests | ❌ |
| RT IGM temperature profile | ✅ | ✅ | ⚠️ scalar-optimal por diseño (ChemState AoS / química; no es un olvido) | ❌ |
| RT reionization state | ✅ | ✅ | ✅ AVX2 + AVX512 reductions | ❌ |
| RT 21cm | ✅ | ✅ | ✅ AVX2 + AVX512 field reductions | ❌ |
| Analysis spin/luminosity/SED | ✅ | ✅ | ✅ AVX2 + AVX512 reductions | ❌ |
| SIDM | ✅ | ✅ density + pair evaluation | ✅ AVX2 + AVX512 density/pair prefilter | ✅ smoke/parity kernel |
| f(R) / modified gravity PM | ✅ | ✅ via PM path | ✅ PM spectral path | ⚠️ PM CUDA only |
| Runtime CLI wiring | ✅ | ✅ | ✅ `simd` separado de `rayon` y propagado a SPH/MHD | ⚠️ gravedad/PM/SPH/cooling/dust/H₂/RT/MHD parcial |

Leyenda: ✅ implementado y validable localmente; ⚠️ parcial, smoke/parity surface o eje mezclado; ❌ no implementado todavía.

Nota MHD Dedner con Rayon: la columna «CPU con Rayon» indica que el paso paralelo
(`dedner_cleaning_step_par`) reparte el trabajo por partícula gas; el bucle de
pares `i`–`j` sigue escalar. Las rutas AVX2/AVX512 de densidad, acumulación por
lotes y actualización final aplican en **CPU sin Rayon** con `feature = "simd"`.

Nota RT chemistry: `rates/cooling` está vectorizado con AVX2/AVX512 en la ruta
CPU sin Rayon. El paso stiff (`solve_chemistry_implicit`) ya usa dispatch SIMD
enmascarado por lote y actualización stiff escalar por lane, con tests de paridad
chunk/cola; la complejidad restante es adaptativa por partícula (subciclos,
ramas moleculares/D/HD, clamps), no la ausencia de SIMD en CPU. Lo que sigue
abierto en la matriz es sobre todo CUDA y el backlog en
[`docs/reports/2026-05-accelerator-parity-pending.md`](docs/reports/2026-05-accelerator-parity-pending.md).

---

## Inicio rápido

### Compilar

```bash
# Mínimo (CPU serial, sin GPU ni MPI):
cargo build --release -p gadget-ng-cli

# Con MPI (requiere libmpi-dev):
cargo build --release -p gadget-ng-cli --features mpi

# Con GPU (wgpu — Vulkan/Metal/DX12):
cargo build --release -p gadget-ng-cli --features gpu

# Con snapshots HDF5 (requiere libhdf5-dev):
cargo build --release -p gadget-ng-cli --features hdf5

# Todo activado:
cargo build --release -p gadget-ng-cli --features full
```

El binario queda en `target/release/gadget-ng`.

### Subcomandos disponibles

```bash
./target/release/gadget-ng --help
```

| Subcomando | Uso |
|---|---|
| `config` | Valida y muestra la configuración efectiva (TOML + env `GADGET_NG_*`) |
| `snapshot` | Escribe un snapshot del estado IC resuelto (sin evolucionar) |
| `stepping` | Integra `num_steps` pasos leapfrog KDK; opcional `--snapshot` y `--vis-snapshot` final |
| `analyse` | FoF (halos) + espectro de potencia P(k) a partir de un snapshot |
| `📊 analyze` | Pipeline completo: FoF + P(k) + ξ(r) + c(M) → `results.json` |
| `visualize` | Renderiza un snapshot a PNG (proyección XY/XZ/YZ) |

### Ejecutar una simulación

```bash
# Esfera de Plummer (512 partículas, Barnes-Hut)
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --snapshot

# Cosmológica EdS con PM periódico (serial)
./target/release/gadget-ng stepping \
  --config experiments/nbody/phase18_periodic_pm/configs/eds_N512_pm.toml \
  --out runs/cosmo_pm

# Cosmológica ΛCDM con 2LPT + PM, snapshot inicial
./target/release/gadget-ng snapshot \
  --config experiments/nbody/phase30_linear_reference/configs/lcdm_N32_a002_2lpt_pm.toml \
  --out runs/lcdm_ic_2lpt

# Análisis de un snapshot (FoF + P(k))
./target/release/gadget-ng analyse \
  --snapshot runs/lcdm_ic_2lpt \
  --out runs/lcdm_ic_2lpt/analysis \
  --pk-mesh 32

# Cosmológica ΛCDM con PM slab distribuido (MPI, Fase 20)
mpirun -n 4 ./target/release/gadget-ng stepping \
  --config experiments/nbody/phase20_slab_pm/configs/lcdm_N2000_slab.toml \
  --out runs/cosmo_slab
```

### Con MPI

```bash
mpirun -n 4 ./target/release/gadget-ng stepping \
  --config examples/nbody_bh_1k.toml \
  --out runs/mpi --snapshot
```

### Reanudar desde un checkpoint

```bash
./target/release/gadget-ng stepping \
  --config examples/plummer_sphere.toml \
  --out runs/plummer --resume runs/plummer
```

Los snapshots incluyen `checkpoint.json` (paso, factor de escala `a`,
hash de config), `particles.jsonl|bin|h5` y, si aplica,
`hierarchical_state.json` para el integrador jerárquico.

---

## Configuración TOML

### Simulación básica

```toml
[simulation]
particle_count         = 512
box_size               = 1.0
dt                     = 0.005
num_steps              = 100
softening              = 0.0
gravitational_constant = 1.0
seed                   = 42

[initial_conditions]
# Opciones:
#   kind = "lattice"
#   kind = { two_body = { … } }
#   kind = { plummer   = { a = 1.0 } }
#   kind = { perturbed_lattice = { amplitude=0.05, velocity_amplitude=0.01 } }
#   kind = { zeldovich = { seed=42, grid_size=32, spectral_index=0.965,
#                          amplitude=1e-4, transfer="eisenstein_hu",
#                          sigma8=0.8, omega_b=0.049, h=0.674, t_cmb=2.7255,
#                          box_size_mpc_h=100.0, use_2lpt=true } }
kind = { perturbed_lattice = { amplitude = 0.05, velocity_amplitude = 0.01 } }

[gravity]
# Opciones: "direct" | "barnes_hut" | "pm" | "tree_pm"
solver       = "pm"
pm_grid_size = 32

[cosmology]
enabled       = true
omega_m       = 0.3
omega_lambda  = 0.7
h0            = 0.1   # unidades internas (ver `[units]`)
a_init        = 0.05
periodic      = true
```

### ICs cosmológicas 2LPT + EH + σ₈ (Fases 26–29)

```toml
[initial_conditions.kind.zeldovich]
seed            = 42
grid_size       = 32
spectral_index  = 0.965
amplitude       = 1.0e-4
transfer        = "eisenstein_hu"   # "power_law" | "eisenstein_hu"
sigma8          = 0.8               # normaliza la amplitud a z=0
omega_b         = 0.049
h               = 0.674
t_cmb           = 2.7255
box_size_mpc_h  = 100.0
use_2lpt        = true              # 1LPT=false / 2LPT=true
```

Transferencia tabulada (CLASS/CAMB u otra tabla `k[h/Mpc]  T(k)`):

```toml
[initial_conditions.kind.zeldovich]
# … mismos campos que arriba …
transfer = { tabulated = { path = "path/to/class_transfer.dat" } }
```

Cutoff WDM/FDM en ICs cosmológicas:

```toml
[dark_matter]
enabled = true
model = "warm"   # "cold" | "warm" | "fuzzy"
m_wdm_kev = 3.0
# m_fdm_22 = 1.0  # para model = "fuzzy" en unidades de 10^-22 eV
```

Criterio de apertura **b_max** en Barnes–Hut / TreePM:

```toml
[gravity]
opening_criterion = "geometric_bmax"   # "geometric" | "geometric_bmax" | "relative"
```

f(R) Hu-Sawicki en PM con screening de malla:

```toml
[modified_gravity]
enabled = true
f_r0 = 1.0e-6
n = 1.0
nonlinear_mesh = true
mesh_iterations = 4
screening_smoothing = 0.5
```

### PM distribuido (Fase 19 — allreduce grid)

```toml
[gravity]
solver          = "pm"
pm_grid_size    = 32
pm_distributed  = true   # allreduce O(nm³) elimina allgather O(N·P) de partículas
```

### PM slab con FFT distribuida (Fase 20 — alltoall)

```toml
[gravity]
solver       = "pm"
pm_grid_size = 32
pm_slab      = true   # FFT distribuida: O(nm³/P) por alltoall transpose
                      # requiere pm_grid_size % n_ranks == 0
```

### Unidades físicas (opcional)

```toml
[units]
enabled              = true
length_in_kpc        = 1.0e3   # box_size · length_in_kpc = kpc
mass_in_msun         = 1.0e10
velocity_in_km_s     = 1.0
```

Cuando `enabled = true`, `RunConfig::effective_g()` calcula `G` coherente
con `G = 4.3009e-6 kpc·M☉⁻¹·(km/s)²` y `meta.json` incluye el bloque
completo.

---

## Crates del workspace

```
gadget-ng/
├── crates/
│   ├── gadget-ng-core          # Vec3, Particle, RunConfig, CosmologyParams,
│   │                           # wrap_position, ic_zeldovich (1LPT+2LPT),
│   │                           # transfer EH no-wiggle + amplitude_for_sigma8; WDM/FDM cutoff;
│   │                           # rebalance_imbalance_threshold (Phase 60)
│   ├── gadget-ng-tree          # Octree + Barnes–Hut + FMM (cuadrupolo + octupolo STF),
│   │                           # SoA + SIMD, LET; MAC geometric / bmax / relative
│   ├── gadget-ng-integrators   # leapfrog_kdk / yoshida4_kdk (newton + cosmológico),
│   │                           # integrador jerárquico + block timesteps LET (Phase 56),
│   │                           # Aarseth timestep
│   ├── gadget-ng-parallel      # SerialRuntime / MpiRuntime, SFC Hilbert 3D,
│   │                           # alltoallv, allreduce, exchange_domain_{x,z}, halos
│   ├── gadget-ng-io            # Snapshots JSONL / Bincode / HDF5 estilo GADGET + Provenance
│   ├── gadget-ng-pm            # PM: CIC, FFT Poisson periódica, slab_fft, slab_pm; feature `fftw`;
│   │                           # f(R) PM homogéneo + screening chameleon no lineal en malla
│   ├── gadget-ng-treepm        # TreePM: BH short-range + PM long-range (serial + dist)
│   ├── gadget-ng-gpu           # Compute shaders WGSL vía wgpu (Vulkan/Metal/DX12)
│   ├── gadget-ng-cuda          # 🟢 Solver PM NVIDIA CUDA + cuFFT (Phase 57); CIC+Poisson+FFT 3D
│   ├── gadget-ng-hip           # 🔴 Solver PM AMD HIP + rocFFT (Phase 57); misma API que CUDA
│   ├── gadget-ng-analysis      # FoF halos + P(k) + pk_correction (Phase 34–36);
│   │                           # NFW + c(M) Duffy/Bhattacharya/Ludlow+2016 (Phase 53/58);
│   │                           # ξ(r) FFT + pares Davis-Peebles (Phase 58); lightcone + Born κ/γ
│   ├── gadget-ng-sph           # SPH: Wendland C2, densidad adaptativa, visc. Monaghan;
│   │                           # ⭐ stellar feedback (Phase 78); AGN Bondi/spin/mergers (Phase 96/183);
│   │                           # metales/SFR/SN Ia+II (Phase 109–115); CRs (Phase 117);
│   │                           # conducción Spitzer (Phase 121); gas molecular (Phase 122);
│   │                           # 🌊 entropía A=P/ρ^γ + Balsara + visc. señal (Phase 166)
│   ├── gadget-ng-mhd           # 🧲 MHD ideal + SRMHD + turbulencia + reconexión +
│   │                           # Braginskii + flux-freeze + P_B(k) + jets AGN + plasma 2F
│   │                           # (Phases 123–150); benchmarks Criterion avanzados
│   ├── gadget-ng-rt            # 🌅 Transferencia radiativa M1 (Phase 81):
│   │                           # solver Godunov HLL, cierre Levermore 1984, acoplamiento SPH;
│   │                           # reionización EoR + fuentes UV + 21cm + T(z) IGM (Phase 89–95);
│   │                           # RT multifrecuencia HI/HeI/HeII/LW/IR + LW H2/HD (Phase 181)
│   ├── gadget-ng-vis           # Visualización CPU: proyecciones, Viridis, PNG;
│   │                           # 🖼️ render_ppm / write_ppm PPM sin dependencias (Phase Rápidas)
│   ├── gadget-ng-physics       # Tests de validación física/cosmológica (Kepler,
│   │                           # Plummer, PM, TreePM, ICs, ensembles, pk_correction,
│   │                           # checkpoint Phase 59, rebalanceo adaptativo Phase 60,
│   │                           # Phases 97–150: SPH bariónica, MHD, SRMHD, plasma 2F)
│   └── gadget-ng-cli           # Binario gadget-ng (clap), subcomandos config/snapshot/
│                               # stepping/analyse/analyze/visualize;
│                               # macros maybe_sph!/maybe_mhd!/maybe_rt! en engine.rs;
├── examples/                   # Configuraciones TOML comentadas
├── experiments/nbody/          # Benchmarks y resultados por fase (60+ experimentos)
└── docs/reports/               # Reportes técnicos de cada fase (60+ reportes)
```

---

## Hitos de desarrollo

El proyecto lleva **185 fases completadas** cubriendo N-body, cosmología, PM/TreePM,
MPI/GPU, SPH, MHD, RT, física bariónica avanzada y la cartera Physics Extensions 2026.

> Para el historial completo de fases y reportes técnicos asociados, ver
> [docs/development-history.md](docs/development-history.md).

Resumen de bloques principales:

| Bloque | Fases | Contenido |
|--------|-------|-----------|
| N-body core | 1–16 | Leapfrog, Yoshida4, Barnes-Hut, SIMD, SFC |
| Cosmología | 17–55 | ΛCDM, PM/TreePM, 2LPT, P(k), FoF, HMF |
| HPC avanzado | 56–70 | Block timesteps, GPU CUDA/HIP, AMR, checkpoint |
| Estadísticas | 71–83 | Bispectrum, RSD, assembly bias, P_B(k), in-situ |
| RT + Reionización | 81–95 | M1, EoR z=6–12, señal 21cm, química HII |
| Física bariónica | 96–122 | AGN, feedback, metales, SN Ia, ISM multifase, CR |
| MHD completo | 123–166 | Ideal, SRMHD, turbulencia O-U, Braginskii, SPH Gadget-2 |
| Physics Extensions | 177–185 | Cooling/SF producción, D/HD, Pop III, RT multifrecuencia, polvo IR, AGN spin, WDM/FDM, f(R) no lineal PM |

Phases recientes:

| Fase | Descripción | Estado |
|------|-------------|--------|
| **177** | Cooling+SF+feedback de producción: UVB, ley de presión y feedback térmico estocástico | ✅ |
| **178** | Cierre Physics Extensions: PM f(R) homogéneo, química primordial 9 especies, lightcones documentados | ✅ |
| **179** | Química D/HD + cooling molecular HD | ✅ |
| **180** | Pop III / primeras estrellas con IMF top-heavy y feedback PISN | ✅ |
| **181** | RT multifrecuencia HI/HeI/HeII/LW/IR y fotodisociación LW de H₂/HD | ✅ |
| **182** | Polvo IR: temperatura de granos, luminosidad greybody y depósito en `PhotonGroup::Infrared` | ✅ |
| **183** | AGN spin + mergers: eficiencia Kerr, spin-up, fusiones BH y recoil | ✅ |
| **184** | Warm/fuzzy dark matter: cutoff WDM/FDM en ICs ZA/2LPT y proxy de presión cuántica | ✅ |
| **185** | f(R) no lineal en malla PM con screening chameleon espacial | ✅ |

| Fase | Descripción | Estado |
|------|-------------|--------|
| **1–2** | N-body directo O(N²), integrador leapfrog | ✅ |
| **3** | Benchmark vs GADGET-4 (fuerza, energía) | ✅ |
| **4** | Suavizado de multipolos, MAC mejorado | ✅ |
| **5** | Consistencia energía + MAC en distribuciones reales | ✅ |
| **6** | Integrador Yoshida4, convergencia de orden 4 | ✅ |
| **7** | Timestep adaptativo estilo Aarseth | ✅ |
| **8–9** | HPC: SFC Z-order, LET distribuido, halos p2p | ✅ |
| **10–11** | LetTree: árbol remoto compacto, validación paralela | ✅ |
| **12** | Reducción comunicación LET (`let_theta_export_factor`) | ✅ |
| **13** | Hilbert 3D SFC: balance de dominio mejorado vs Morton | ✅ |
| **14** | SoA + SIMD: kernels calientes en layout columnar | ✅ |
| **15–16** | SIMD explícito: tiling 4×N_i, leaf-max sweep | ✅ |
| **17a** | Cosmología serial: Friedmann ΛCDM, momentum canónico, `G/a` | ✅ |
| **17b** | Cosmología distribuida MPI con SFC+LET | ✅ |
| **18** | PM periódico: CIC + FFT Poisson, `wrap_position`, `minimum_image` | ✅ |
| **19** | PM distribuido sin allgather: `allreduce_sum_f64_slice` O(nm³) | ✅ |
| **20** | PM slab: FFT distribuida alltoall O(nm³/P), grid no replicado | ✅ |
| **21** | TreePM distribuido: BH local + PM slab (first light) | ✅ |
| **22** | TreePM: halos 3D (x,y,z) para short-range periódico | ✅ |
| **23** | TreePM: dominio 3D con SFC, métricas de balance | ✅ |
| **24** | TreePM PM scatter-gather (reemplaza allgather de densidad) | ✅ |
| **25** | Validación MPI de TreePM scatter-gather end-to-end | ✅ |
| **26** | ICs Zel'dovich 1LPT con power-law y validación numérica | ✅ |
| **27** | Transfer Eisenstein–Hu no-wiggle + normalización σ₈ | ✅ |
| **28** | ICs 2LPT (Buchert–Ehlers, corrección `-3/7` sobre 1LPT) | ✅ |
| **29** | Validación cruzada 1LPT vs 2LPT en crecimiento lineal | ✅ |
| **30** | Referencia lineal ΛCDM: P(k) medido vs EH + D₁(a)² | ✅ |
| **31** | Ensemble a resolución media: CV(R(k)) y shape | ✅ |
| **32** | Ensemble alta-res N=32³ · 6 seeds: crecimiento + PM/TreePM | ✅ |
| **33** | Análisis de normalización absoluta de `P(k)` (offset 17×) | ✅ |
| **34** | Cierre analítico del factor de grilla `A_grid = 2·V²/N⁹` | ✅ |
| **35** | Modelado del factor de muestreo `R(N) = C·N^(-α)` | ✅ |
| **36** | Validación práctica de `pk_correction` en corridas reales | ✅ |
| **37** | Reescalado físico opcional de ICs por `D(a_init)/D(1)` (exp.) | ✅ |
| **38** | 🔭 Validación externa mínima vs CLASS (IC snapshot) | ✅ |
| **39** | Convergencia temporal del integrador (barrido `dt`) | ✅ |
| **40** | Formalización de la convención física (`NormalizationMode`) | ✅ |
| **41** | Validación de alta resolución y transición shot-noise↔señal | ✅ |
| **42** | Regularización física vía TreePM + softening absoluto `ε_phys` | ✅ |
| **43** | Control temporal TreePM + paralelismo Rayon en PM; timestep adaptativo cosmológico | ✅ |
| **44** | Auditoría y fix de ICs 2LPT (doble división k², signo global, `f₂`) | ✅ |
| **45** | Auditoría y corrección de unidades IC ↔ integrador; fix `g_cosmo = G·a³` (QKSL) | ✅ |
| **46** | PM pencil 2D: FFT distribuida hasta `P ≤ nm²`; `alltoallv_f64_subgroup` | ✅ |
| **47** | Corrección P(k) recalibrada: `R(N)` in-process, sustracción shot-noise Poisson | ✅ |
| **48** | Halofit no-lineal (Takahashi+2012): `halofit_pk`, `k_sigma`, `n_eff`, boost vs ΛCDM | ✅ |
| **49** | Fix integrador cosmológico: `gravity_coupling_qksl` en `cosmo_pm.rs` y tests anteriores | ✅ |
| **50** | Unidades físicamente consistentes: `g_code_consistent(Ω_m, H₀)`, diagnóstico de inconsistencia | ✅ |
| **51** | G auto-consistente en motor de producción (`auto_g = true`, warn si G manual difiere > 1 %) | ✅ |
| **52** | Función de masa de halos Press-Schechter / Sheth-Tormen; σ(M,z), tabla dn/d ln M | ✅ |
| **53** | Perfiles NFW y relación concentración-masa c(M); Duffy+2008, Bhattacharya+2013 | ✅ |
| **54** | Validación cuantitativa D²(a) con G consistente; N ∈ {64,128,256}, 6 snapshots | ✅ |
| **55** | Comparación FoF vs HMF hasta z=0 (BOX=300 Mpc/h); ratio dn/dlnM(FoF)/dn/dlnM(ST) | ✅ |
| **56** | ⏱️ Block timesteps jerárquicos acoplados al árbol LET distribuido; `active_local` O(N_active) | ✅ |
| **57** | 🟢🔴 PM solver CUDA/HIP: segunda cadena de compilación; cuFFT/rocFFT; degradación elegante | ✅ |
| **58** | 📐 c(M) y perfiles NFW desde N-body: fit NFW → FoF halos, c(M) vs Duffy/Ludlow; ξ(r) FFT+pares | ✅ |
| **59** | 💾 Restart/checkpoint bit-a-bit: guarda/restaura estado completo (pos, vel, paso, SFC) | ✅ |
| **60** | 🔀 Domain decomposition adaptativa por costo: `should_rebalance` + `rebalance_imbalance_threshold` | ✅ |
| **Rápidas** | 📊 CLI `gadget-ng analyze` (FoF+P(k)+ξ(r)+c(M)→JSON); 🖼️ `gadget-ng-vis` render PPM sin deps | ✅ |
| **61–65** | Grandes: SPH cosmológico, merger trees MAH, SUBFIND, N=256³ producción, AMR-PM | ✅ |
| **66** | 🌊 SPH cosmológico KDK: gas+DM, cooling atómico, separación de partículas | ✅ |
| **67** | 🌳 Merger trees + Mass Accretion History (MAH, McBride 2009) | ✅ |
| **68** | 🔍 SUBFIND: subestructura intra-halo via Union-Find + binding energy | ✅ |
| **69** | 🚀 Producción N=256³: config TOML, scripts PBS, notebook post-proceso | ✅ |
| **70** | 🔲 AMR-PM: parches adaptativos 2 niveles, refinamiento por sobredensidad | ✅ |
| **71** | 📊 Bispectrum B(k₁,k₂,k₃): estadística de orden 3, CIC+FFT+shell-filter | ✅ |
| **72** | 🌀 Spin de halos λ Peebles+Bullock: momento angular FoF | ✅ |
| **73** | 📐 Perfiles de velocidad σ_v(r): σ_r, σ_t, β(r) de Binney | ✅ |
| **74** | 💾 HDF5 GADGET-4 estándar: 22 atributos, compatible con yt/pynbody | ✅ |
| **75** | 📈 P(k,μ) RSD: espacio de redshift, multipoles P₀/P₂/P₄ (Hamilton 1992) | ✅ |
| **76** | 🔗 Assembly bias: Spearman spin/concentración vs δ_env suavizado | ✅ |
| **77** | 📁 Catálogo de halos HDF5: FoF+SUBFIND, /Header, /Halos, /Subhalos | ✅ |
| **78** | ⭐ Stellar feedback estocástico: SFR+SN kicks, `compute_sfr`, `apply_sn_feedback` | ✅ |
| **79** | 🔬 Validación N=128³: config Planck18, scripts PBS, validate_pk_hmf.py | ✅ |
| **80** | 🔲 AMR jerárquico N-nivel: `build_amr_hierarchy`, `amr_pm_accels_multilevel` | ✅ |
| **81** | 🌅 Transferencia radiativa M1: solver HLL, cierre Levermore 1984, acoplamiento SPH | ✅ |
| **82** | 🔁 Integración automática: `maybe_sph!`, `maybe_rt!`, bispectrum+assembly_bias in-situ, `--hdf5-catalog` | ✅ |
| **83** | 📊 Post-procesamiento: `postprocess_insitu.py` — P(k), σ₈(z), multipoles, B(k) | ✅ |
| **84** | 🌐 RT MPI slab: `RadiationFieldSlab`, `allreduce_radiation`, `exchange_radiation_halos` | ✅ |
| **85** | 🌐 AMR MPI: `AmrPatchMessage`, `broadcast_patch_forces`, `amr_pm_accels_multilevel_mpi` | ✅ |
| **86** | 🧪 Química no-equilibrio HII/HeII/HeIII: `ChemState`, solver implícito, `apply_chemistry` | ✅ |
| **87** | 🌐 MPI real (rsmpi): `allreduce_radiation_mpi`, `exchange_radiation_halos_mpi`, `broadcast_patch_forces_mpi` | ✅ |
| **88** | 📈 Benchmarks GPU vs CPU (Criterion) + CI `--release` con tests integración MPI | ✅ |
| **89** | ☀️ Reionización del Universo: `UvSource`, `deposit_uv_sources`, `reionization_step`, R_Strömgren | ✅ |
| **90** | 🌡️ Perfil de temperatura IGM T(z): `IgmTempBin`, filtrado por densidad SPH, percentiles 16/84 | ✅ |
| **91** | 📄 Paper draft JOSS: `docs/paper/paper.md` + `paper.bib` (15 refs BibTeX) | ✅ |
| **92** | 📊 Benchmarks formales: `bench_mpi_scaling.sh`, `bench_pk_vs_gadget4.py`, rsmpi verificado | ✅ |
| **93** | 📝 README final + figuras JOSS (P(k), HMF, Strömgren) + submission checklist | ✅ |
| **94** | 📡 Estadísticas 21cm: `brightness_temperature`, P(k)₂₁cm, `Cm21Output` in-situ | ✅ |
| **95** | 🌌 EoR z=6–12: `maybe_reionization!` en engine, `uv_from_halos`, test fase completa | ✅ |
| **96** | 🕳️ Feedback AGN: `BlackHole`, `bondi_accretion_rate`, `apply_agn_feedback`, `AgnSection` | ✅ |
| **97–99** | SPH avanzado: `Particle.z_metal`, `MetalCooling`, enriquecimiento Q, SN Ia DTD; `apply_metal_cooling`, `compute_metallicity` | ✅ |
| **100** | AGN con halos FoF: AGN en halos masivos, `bondi_agn_halo`, `agn_halos_from_fof` | ✅ |
| **101** | Fix softening comóvil→físico: `epsilon_phys = epsilon_comoving * a`, corrección unitaria | ✅ |
| **102** | HDF5 layout GADGET-4 completo: grupos `/PartType0-5`, atributos Header, campos Bfld, ChemAb | ✅ |
| **103** | Domain decomp con coste medido: `cpu_time_tree_ns` por partícula, SFC ponderado | ✅ |
| **104** | CLI extendida: `gadget-ng postprocess`, `--phases`, logging estructurado JSON | ✅ |
| **105** | JSONL con campos SPH: `u_therm`, `rho`, `h_sml`, `z_metal`, `sfr`, `t_star` | ✅ |
| **106** | Restart con SPH state completo: `u_therm`, `rho`, campos MHD en checkpoint | ✅ |
| **107** | Merger trees con FoF real: árbol de fusiones usando IDs de halos FoF, `MergerTree`, `progenitor_map` | ✅ |
| **108** | Vientos galácticos: `apply_galactic_winds`, `v_wind ∝ σ_dm`, mass-loading `η_w` | ✅ |
| **109** | Metales en Particle + `ParticleType::Star`: `z_alpha`, `z_fe`, spawning de partículas estelares | ✅ |
| **110** | Enriquecimiento químico SPH: yields O+Fe de SNII+SNIa distribuidos a vecinos SPH | ✅ |
| **111** | Enfriamiento por metales (`MetalCooling`): tablas Z-dependientes, interp. bilineal | ✅ |
| **112** | Partículas estelares reales (spawning): `spawn_star_particles`, SSP Kroupa | ✅ |
| **113** | SN Ia con DTD power-law: `snia_rate_dtd`, yields Fe, calor térmico | ✅ |
| **114** | ISM Multifase fría-caliente: `TwoPhaseISM`, fracción fría `x_cold`, `u_hot` vs `u_cold` | ✅ |
| **115** | Vientos estelares pre-SN: `apply_stellar_winds`, masa perdida via `mass_loss_rate` | ✅ |
| **116** | Modo radio AGN (bubble feedback): `inject_agn_bubble`, cavidades de entalpía | ✅ |
| **117** | Rayos cósmicos básicos: `Particle.e_cr`, `inject_cr_sn`, difusión isotrópica | ✅ |
| **118** | Función de luminosidad y colores galácticos: `galaxy_luminosity`, magnitudes B/V/R | ✅ |
| **119** | Enfriamiento tabulado Sutherland-Dopita 1993: `cooling_sd93`, grilla [T, Z] | ✅ |
| **120** | Engine integration bariónica: `maybe_sph!` coordina SPH+quím+metales+feedback por paso | ✅ |
| **121** | Conducción térmica ICM Spitzer: `apply_spitzer_conduction`, κ ∝ T^{5/2} anisótropo | ✅ |
| **122** | Gas molecular HI→H₂: `MolecularFraction`, umbral de densidad, shielding UV | ✅ |
| **123** | 🧲 Crate `gadget-ng-mhd` + `b_field` en Particle + ecuación de inducción SPH | ✅ |
| **124** | Presión magnética + tensor Maxwell en fuerzas SPH: `f_lorentz`, `p_mag = B²/2` | ✅ |
| **125** | Dedner div-B cleaning: `psi_div` advectado, decaimiento `ch/cp²`, residuos < 1 % | ✅ |
| **126** | Integración MHD en engine + macro `maybe_mhd!` + validación onda Alfvén | ✅ |
| **127** | ICs magnetizadas + CFL magnético: `BFieldKind`, `alfven_dt`, `cfl_mhd` | ✅ |
| **128** | Validación MHD 3D Alfvén + Brio-Wu 1D: tests de onda y choque MHD | ✅ |
| **129** | Acoplamiento CR–B: difusión CR suprimida por β_plasma, `diffuse_cr_anisotropic` | ✅ |
| **130** | Polvo intersticial básico: `Particle.dust_fraction`, `apply_dust_growth`, D/G ratio | ✅ |
| **131** | HDF5 campos MHD + SPH completos: `/Bfld`, `/DivB`, `/Psi`, `/ECr`, `/Dust` en snapshot | ✅ |
| **132** | Benchmark MHD Criterion + CFL unificado: `bench mhd`, `cfl_mhd` unifica dt | ✅ |
| **133** | MHD anisótropo: difusión térmica + CR paralela a B, `kappa_par/kappa_perp` | ✅ |
| **134** | Cooling magnético: `apply_magnetic_cooling`, emisión sincrotrón ∝ B² γ_e² | ✅ |
| **135** | Resistividad numérica artificial: `apply_artificial_resistivity`, `alpha_b` adaptativo | ✅ |
| **136** | MHD cosmológico end-to-end: `lcdm_mhd_N64`, crecimiento de B de semilla, test B_rms(a) | ✅ |
| **137** | Polvo + RT: absorción UV kappa_dust×D/G×ρ×h, M1Params.sigma_dust, τ_dust | ✅ |
| **138** | Freeze-out de B en ICM: `apply_flux_freeze`, B ∝ ρ^{2/3}, `beta_freeze` | ✅ |
| **139** | SRMHD — MHD especial-relativista: γ Lorentz, primitivización NR, `advance_srmhd` | ✅ |
| **140** | Turbulencia MHD: forzado Ornstein-Uhlenbeck, P_B(k) ∝ k^{-5/3}, semilla reproducible | ✅ |
| **141** | Tests de integración MHD avanzados (Phases 133–140); 48 tests; 0 regresiones | ✅ |
| **142** | Engine: RMHD + turbulencia en `maybe_mhd!`/`maybe_sph!`; hooks B+plasma | ✅ |
| **143** | Benchmarks Criterion avanzados: turbulencia, flux-freeze, SRMHD primitivas | ✅ |
| **144** | Clippy cero warnings en todo el workspace: 15+ lints corregidos | ✅ |
| **145** | Reconexión magnética Sweet-Parker: `apply_magnetic_reconnection`, `sweet_parker_rate` | ✅ |
| **146** | Viscosidad Braginskii anisótropa: `apply_braginskii_viscosity`, tensor π_ij | ✅ |
| **147** | Corrida cosmológica MHD completa + P_B(k): `magnetic_power_spectrum`, test end-to-end | ✅ |
| **148** | Jets AGN relativistas: `inject_relativistic_jet`, halos FoF masivos, v_jet 0.3–0.9c | ✅ |
| **149** | Plasma de dos fluidos T_e ≠ T_i: `apply_electron_ion_coupling`, `mean_te_over_ti` | ✅ |
| **150** | Reportes 142–149, CHANGELOG, roadmap, commit final | ✅ |
| **165** | 🟢🔴 Kernels CUDA/HIP N² reales (`CudaDirectGravity::compute`, `HipDirectGravity::compute`); MHD 3D solenoidal `primordial_bfield_ic_3d` con ∇·B < 1e-14; 5 tests GPU activados | ✅ |
| **166** | 🌊 **SPH Gadget-2**: entropía A=P/ρ^γ · limitador Balsara · viscosidad señal · `sph_kdk_step_gadget2` · `courant_dt`; test tubo de Sod + colapso de Evrard; ~50 tests lentos marcados `#[ignore]` → `cargo test -p gadget-ng-physics` en ~3.5 min | ✅ |

---

## Arquitectura de comunicación PM

| Path | Activar | Comm/rank/paso | Solve |
|------|---------|----------------|-------|
| PM clásico (Fase 18) | `solver="pm"` | O(N·P) — allgather | Serial replicado |
| PM distribuido (Fase 19) | `pm_distributed=true` | O(nm³) — allreduce | Serial replicado |
| **PM slab (Fase 20)** | **`pm_slab=true`** | **O(nm³/P) — alltoall** | **Distribuido** |
| TreePM scatter-gather (Fase 24) | `solver="tree_pm"` + `pm_slab=true` | O(nm³/P) + halos BH | Distribuido |

Ejemplos concretos (bytes/rank/paso con nm=32):

| Ranks (P) | Fase 19 | Fase 20 |
|-----------|---------|---------|
| 1 | 262 KB | 262 KB (serial) |
| 2 | 262 KB | 131 KB |
| 4 | 262 KB |  66 KB |
| 8 | 262 KB |  33 KB |

La Fase 20 introduce `alltoall_transpose` entre slabs `z → y → x`,
permitiendo escalar la FFT 3D sin replicar el grid en cada rank.
La Fase 24 usa un scatter-gather equivalente para el cálculo de la
densidad desde partículas → grid sin `allgather` intermedio.

---

## Condiciones iniciales y validación cosmológica

Las Fases 26–32 cubren todo el pipeline de ICs cosmológicas y su
validación física:

- **Fase 26** — Zel'dovich 1LPT: `δ(k) → Ψ(k) = i·k/k²·δ(k)`,
  normalización por `σ(|n|)` en modo `power_law`.
- **Fase 27** — Transferencia EH no-wiggle + `amplitude_for_sigma8`
  (convierte σ₈ en amplitud absoluta resolviendo `σ²(R=8)=σ₈²`).
- **2026-05** — Transferencia **tabulada** (`k`, `T(k)` desde CLASS/CAMB u otras tablas) con interpolación log–log monotónica; **pancake** Zel’dovich analítico ([reporte](docs/reports/2026-05-phase160-zeldovich-pancake.md)).
- **Fase 28** — 2LPT (Buchert–Ehlers): segundo orden en el
  desplazamiento, corrige transitorios de 1LPT a `a_init` alto.
- **Fase 29** — Validación 1LPT vs 2LPT: comparación de crecimiento
  `D₁(a)` y residuos al snapshot inicial.
- **Fase 30** — Referencia lineal ΛCDM: `P(k)` medido vs EH + CPT92
  `D(z)/D(0)` a `z ≈ 49`.
- **Fase 31** — Ensemble N=16³ · 4 seeds: CV de `R(k) = P_m/P_EH`.
- **Fase 32** — Ensemble alta-resolución N=32³ · 6 seeds: 10 tests
  sobre shape espectral, crecimiento, PM vs TreePM y reproducibilidad
  bit-idéntica.

Para configurar un IC cosmológico completo:

```toml
[initial_conditions.kind.zeldovich]
seed            = 42
grid_size       = 32
spectral_index  = 0.965
transfer        = "eisenstein_hu"
sigma8          = 0.8
omega_b         = 0.049
h               = 0.674
t_cmb           = 2.7255
box_size_mpc_h  = 100.0
use_2lpt        = true
```

---

## Corrección absoluta de `P(k)` (`pk_correction`)

Las Fases 33–36 cierran la normalización absoluta del estimador de
`P(k)`. El problema identificado en Phase 33 — un offset de amplitud
`~10¹⁴` entre `P(k)` medido y la referencia continua — se decompone
en dos factores:

```text
P_measured(k) = A_grid(N) · R(N) · P_cont(k)

  A_grid(N) = 2·V²/N⁹       (Phase 34 — cerrado analíticamente)
  R(N)      = C · N^(-α)    (Phase 35 — fit log-log, R² = 0.997)
              con C = 22.108, α = 1.8714, tabla exacta
              para N ∈ {8, 16, 32, 64}
```

La corrección se aplica en postproceso vía la API pública de
[`gadget_ng_analysis::pk_correction`](crates/gadget-ng-analysis/src/pk_correction.rs):

```rust
use gadget_ng_analysis::pk_correction::{correct_pk, RnModel};
use gadget_ng_analysis::power_spectrum::power_spectrum;

let mesh  = 32;
let pk    = power_spectrum(&positions, &masses, box_size, mesh);
let model = RnModel::phase35_default();
let pk_phys = correct_pk(&pk, box_size, mesh, /* box_mpc_h */ None, &model);
// pk_phys[i].pk está ahora en (Mpc/h)³ y compara directamente con EH·D(a)²
```

Phase 36 validó la corrección end-to-end sobre 27 snapshots reales
(N=32³/64³, 3 seeds, 1LPT y 2LPT, PM): la mediana
`|log₁₀(P_m/P_ref)|` baja de **14.7–18.0** a **0.03** en el snapshot
IC — una mejora de factor `~10¹⁴`, con `mean(P_c/P_ref) = 1.00 ± 0.05`
y `CV ≤ 0.15`. El mismo resultado se reproduce vía CLI real
(`gadget-ng snapshot → analyse → apply_phase36_correction.py`).

🔭 **Phase 38** cierra la validación externa contra CLASS (`classy 3.3.4.0`,
referencia reproducible en
[`experiments/nbody/phase38_class_validation/reference/`](experiments/nbody/phase38_class_validation/reference/)):
`median|log10(P_c/P_CLASS)| ∈ [0.022, 0.046]` y `mean(P_c/P_CLASS) ∈ [0.95,
1.04]` sobre 12 mediciones, factor de mejora **161× a N=32³** y **761× a
N=64³** frente a `P_measured/P_CLASS`.

📈 **Phase 41** extiende la validación a alta resolución (N ∈ {32, 64, 128})
y demuestra empíricamente la transición **shot-noise ↔ señal**
(`P_shot = V/N_p`):

| `N`   | `P_shot` [(Mpc/h)³] | `S/N(k_min)` en IC (Z0Sigma8) |
|-------|---------------------|-------------------------------|
|  32   | 30.52               | 0.374 → shot-noise domina     |
|  64   | 3.815               | 2.21  → transición            |
| 128   | 0.4768              | 16.06 → señal limpia          |

`pk_correction` cierra en IC a `median|log10(P_c/P_ref)| ≤ 0.049` para los
tres N, confirmando que el resultado de Phase 38 se extiende a `N = 128³`.

Para más detalles, ver
[`docs/reports/2026-04-phase36-pk-correction-validation.md`](docs/reports/2026-04-phase36-pk-correction-validation.md),
[`docs/reports/2026-04-phase38-class-camb-minimal-validation.md`](docs/reports/2026-04-phase38-class-camb-minimal-validation.md)
y
[`docs/reports/2026-04-phase41-high-resolution-validation.md`](docs/reports/2026-04-phase41-high-resolution-validation.md).

---

## Convención de ICs y validación dinámica (Phase 37–42)

Las Phases 37–42 formalizan la convención de normalización de ICs
cosmológicas, exploran sus límites dinámicos y prueban la
regularización física de fuerzas:

- **Phase 37** introduce el flag experimental `rescale_to_a_init` (luego
  renombrado a `NormalizationMode` en Phase 40): aplica `Ψ¹ ← s·Ψ¹` y
  `Ψ² ← s²·Ψ²` con `s = D(a_init)/D(1)` (CPT92). Decisión: **B** — la
  implementación es exacta (residuo `< 2·10⁻⁶`) pero no mejora snapshots
  evolucionados a `N=32³`, queda experimental.
- **Phase 38** valida `pk_correction` contra 🔭 CLASS en ambas convenciones
  (`legacy` vs `z=0`, `rescaled` vs `z=49`) — confirma que el cierre es
  intrínseco a la corrección y no depende de la normalización.
- **Phase 39** barre `dt ∈ {4·10⁻⁴, 2·10⁻⁴, 1·10⁻⁴, 5·10⁻⁵}` y demuestra
  que reducir `dt` **no** reduce el error espectral (pendiente log-log
  observada ≈ `−0.06` vs predicción `+2` para KDK). Decisión: mantener
  `dt = 4·10⁻⁴` como default.
- **Phase 40** reemplaza el booleano `rescale_to_a_init` por una enum
  tipada `NormalizationMode { Legacy, Z0Sigma8 }` (⚠️ breaking change en
  TOML), audita la implementación LPT (sin bugs) y verifica
  empíricamente `σ₈(Z0Sigma8)/σ₈(Legacy) = s` a precisión de máquina.
  Decisión: **B** — `Z0Sigma8` queda experimental por dominancia de
  shot-noise a `N=32³`.
- **Phase 41** resuelve Phase 40-B: a `N ≥ 64³` la señal supera el
  shot-noise floor (`S/N(k_min) = 2.21` en IC) y a `N = 128³` el margen
  es ×16. Decisión: **cierre parcial** — el eje señal/ruido queda
  cerrado; el eje evolución lineal/no-lineal requiere softening físico y/o
  integrador adaptativo (trabajo futuro).
- **Phase 42** ataca el eje dinámica lineal↔no-lineal abierto por
  Phase 41 introduciendo softening físico absoluto `ε_phys ∈
  {0.01, 0.02, 0.05} Mpc/h` (independiente de `N`) y `TreePmSolver` con
  kernel `erfc` + Plummer. Matriz `{PM, TreePM × 3 ε}` a `N=32³` (smoke)
  y `N=64³` (2h 18min wall / 22.3h CPU con rayon). Decisión: **A_partial
  (softening confirmado como palanca correcta)** — `TreePM ε=0.01`
  mejora el error de crecimiento lineal **×345** vs PM a `N=64³` (vs
  ×3.5 a `N=32³`), con efecto que crece con `N`. La magnitud absoluta
  del error sigue invalidando lectura lineal hasta `N=128³` (diferido
  al pipeline TreePM distribuido).

```toml
# Convención Phase 40+ — reemplaza `rescale_to_a_init = true/false`
[initial_conditions.kind.zeldovich]
# ...
normalization_mode = "legacy"     # default, bit-compatible con Phases 26–39
# normalization_mode = "z0_sigma8" # σ₈ referido a a=1 (CAMB/CLASS), Phase 40+
```

Reportes: [Phase 37](docs/reports/2026-04-phase37-growth-rescaled-ics.md) ·
[Phase 38](docs/reports/2026-04-phase38-class-camb-minimal-validation.md) ·
[Phase 39](docs/reports/2026-04-phase39-dt-convergence.md) ·
[Phase 40](docs/reports/2026-04-phase40-physical-ics-normalization.md) ·
[Phase 41](docs/reports/2026-04-phase41-high-resolution-validation.md) ·
[Phase 42](docs/reports/2026-04-phase42-tree-short-range.md).

---

## Análisis post-procesamiento (Phase 56–60 + Rápidas)

### ⏱️ Block timesteps jerárquicos (Phase 56)

Los block timesteps jerárquicos acoplan el árbol LET distribuido con el integrador de pasos múltiples. En cada nivel de bloque solo se evalúan fuerzas para las partículas activas (`active_local`), reduciendo el costo por subnivel a O(N_active):

```toml
[simulation]
hierarchical_timesteps = true
n_levels               = 4     # 2^4 = 16 subniveles de dt
```

### 🟢🔴 GPU PM CUDA/HIP (Phase 57)

Solver PM opcional que usa la GPU para CIC + FFT Poisson 3D. Compilado en crates separados (`gadget-ng-cuda`, `gadget-ng-hip`) para no contaminar el árbol de compilación principal:

```bash
# NVIDIA (requiere CUDA toolkit + nvcc):
cargo build --release -p gadget-ng-cuda

# AMD (requiere ROCm + hipcc):
cargo build --release -p gadget-ng-hip

# CI sin GPU disponible:
CUDA_SKIP=1 cargo test -p gadget-ng-cuda
HIP_SKIP=1  cargo test -p gadget-ng-hip
```

### 📐 c(M) y ξ(r) desde N-body (Phase 58)

```rust
use gadget_ng_analysis::{
    nfw::{fit_nfw_concentration, concentration_ludlow2016},
    correlation::{two_point_correlation_fft, two_point_correlation_pairs},
};

// Concentración c(M) — Ludlow et al. (2016) calibrada en Planck 2015 ΛCDM
let c = concentration_ludlow2016(m200_msun_h, z);

// ξ(r) desde P(k) medido (transformada de Hankel discreta)
let xi = two_point_correlation_fft(&pk_bins, box_size, n_r_bins);

// ξ(r) por conteo de pares Davis-Peebles
let xi = two_point_correlation_pairs(&positions, box_size, &r_edges);
```

### 💾 Checkpoint robusto (Phase 59)

El checkpoint guarda posiciones, velocidades, paso actual, factor de escala y el estado de la SFC. Al reanudar, se reconstruye `SfcDecomposition` automáticamente — la continuidad bit-a-bit está verificada por test de integración:

```bash
./gadget-ng stepping --config run.toml --out runs/cosmo --resume runs/cosmo
```

### 🔀 Rebalanceo adaptativo por costo (Phase 60)

Configura un umbral de desequilibrio de carga basado en el tiempo de caminata del árbol por partícula:

```toml
[performance]
rebalance_interval          = 50     # fijo: cada 50 pasos
rebalance_imbalance_threshold = 0.3  # también si max/min(walk_ns) > 1.3
```

Si `threshold = 0.0` (default) se desactiva el rebalanceo por costo y solo aplica el intervalo fijo.

### 📊 CLI `gadget-ng analyze` (Rápidas)

```bash
./gadget-ng analyze \
  --snapshot runs/cosmo/snapshot \
  --out runs/cosmo/analysis \
  --fof-linking-length 0.2 \
  --pk-mesh 32 \
  --xi-bins 20 \
  --box-mpc-h 300.0
# Produce: runs/cosmo/analysis/results.json
# Campos: n_halos, pk[], xi[], halos[]{mass, c_duffy, c_ludlow}
```

### 🖼️ Render PPM sin dependencias (Rápidas)

```rust
use gadget_ng_vis::ppm::{render_ppm, write_ppm};

let pixels = render_ppm(&positions, box_size, 512, 512);
write_ppm(Path::new("snapshot.ppm"), &pixels, 512, 512)?;
```

---

## Estadísticas avanzadas y transferencia radiativa (Phases 71–82)

### 📊 Bispectrum B(k₁,k₂,k₃) (Phase 71)

Estadística de tercer orden; detector de no-gaussianidades primordiales.

```rust
use gadget_ng_analysis::{bispectrum_equilateral, BkBin};

let bk: Vec<BkBin> = bispectrum_equilateral(&positions, &masses, box_size, mesh, n_bins);
// bk[i].k: modo k; bk[i].bk: B(k,k,k); bk[i].n_triangles: número de triángulos
```

Activar en análisis in-situ (TOML):
```toml
[insitu_analysis]
bispectrum_bins = 20   # 0 = desactivado
```

### 🌀 Spin de halos λ (Phase 72)

Momento angular de halos FoF, definiciones Peebles y Bullock.

```rust
use gadget_ng_analysis::{halo_spin, SpinParams, HaloSpin};

let params = SpinParams::default();
let spin: Option<HaloSpin> = halo_spin(&positions, &velocities, &masses, &params);
// spin.lambda_peebles: |L| / (M × V_vir × R_vir)
// spin.lambda_bullock: |L| / (sqrt(2) × M × V_vir × R_vir)
```

### 📐 Perfiles de velocidad σ_v(r) (Phase 73)

Dispersión radial y tangencial dentro de halos; parámetro de anisotropía de Binney β(r).

```rust
use gadget_ng_analysis::{velocity_profile, VelocityProfileParams};

let params = VelocityProfileParams { n_bins: 12, log_bins: true, ..Default::default() };
let profile = velocity_profile(&positions, &velocities, &center, &params);
// profile[i].sigma_r, .sigma_t, .sigma_3d, .beta
```

### 💾 HDF5 compatible GADGET-4 (Phase 74)

22 atributos estándar de GADGET-4 en cada snapshot. Compatible con `yt`, `pynbody`, `h5py`.

```bash
# Compilar con soporte HDF5
cargo build --features hdf5

# Los snapshots incluyen /Header con NumPart_Total, OmegaBaryon, etc.
```

### 📈 P(k,μ) en espacio de redshift (Phase 75)

Espectro de potencia anisótropo con RSD (Hamilton 1992). Multipoles P₀/P₂/P₄.

```rust
use gadget_ng_analysis::{compute_pk_multipoles, PkRsdParams, LosAxis};

let params = PkRsdParams {
    n_k_bins: 32, n_mu_bins: 10, los: LosAxis::Z,
    scale_factor: 0.5, hubble_a: 67.4,
};
let multipoles = compute_pk_multipoles(&positions, &velocities, &masses, box, mesh, &params);
```

```toml
[insitu_analysis]
pk_rsd_bins = 10   # bins en μ para P(k,μ)
```

Los resultados se guardan en `insitu_NNNNNN.json` bajo las claves `pk_rsd` y `pk_multipoles`.

### 🔗 Assembly bias (Phase 76)

Correlación de Spearman entre propiedades de halos (spin, concentración) y densidad del entorno.

```rust
use gadget_ng_analysis::{compute_assembly_bias, AssemblyBiasParams};

let params = AssemblyBiasParams { smooth_radius: 5.0, mesh: 32, n_quartiles: 4 };
let result = compute_assembly_bias(&halo_pos, &halo_mass, &spins, &concs,
                                   &all_pos, &all_mass, box_size, &params);
// result.spearman_lambda, result.bias_vs_lambda
```

```toml
[insitu_analysis]
assembly_bias_enabled = true
assembly_bias_smooth_r = 5.0
```

### 📁 Catálogo de halos HDF5 (Phase 77)

Catálogo FoF + SUBFIND en formato HDF5 compatible con `Caesar`, `rockstar-galaxies`.

```bash
# Generar catálogo de halos al analizar un snapshot
gadget-ng analyze --snapshot out/snap --out analysis/ --hdf5-catalog
# → analysis/halos.hdf5  (con feature hdf5)
# → analysis/halos.jsonl (sin feature hdf5)
```

```rust
use gadget_ng_io::{write_halo_catalog_hdf5, HaloCatalogEntry, HaloCatalogHeader};
```

### ⭐ Stellar feedback estocástico (Phase 78)

Kicks de supernova aleatorios acoplados al módulo SPH. Activar en TOML:

```toml
[sph]
enabled = true

[sph.feedback]
enabled = true
sn_energy_erg    = 1.0e51
sn_efficiency    = 0.1
sfr_threshold    = 1.0
```

### 🔬 Validación N=128³ (Phase 79)

Configuración de producción con comparación contra Eisenstein-Hu y HMF analítica.

```bash
bash scripts/run_validation_128.sh --out runs/val128
python docs/scripts/validate_pk_hmf.py --dir runs/val128/insitu
```

### 🔲 AMR jerárquico multi-nivel (Phase 80)

Solver PM adaptativo con N niveles de refinamiento recursivos.

```rust
use gadget_ng_pm::{build_amr_hierarchy, amr_pm_accels_multilevel, AmrParams};

let params = AmrParams {
    overdensity_threshold: 5.0, patch_size: 8,
    max_levels: 3, refine_factor: 2,
};
let levels = build_amr_hierarchy(&density_grid, n, box_size, &params);
let accels = amr_pm_accels_multilevel(&particles, box_size, &params);
```

### 🌅 Transferencia radiativa M1 (Phase 81)

Solver Godunov de primer orden con cierre M1 (Levermore 1984). Acoplado a gas SPH.

```rust
use gadget_ng_rt::{RadiationField, M1Params, m1_update, radiation_gas_coupling_step};

let mut rf = RadiationField::uniform(32, 32, 32, dx, e0);
let m1 = M1Params { c_red_factor: 100.0, kappa_abs: 1.0, kappa_scat: 0.0, substeps: 5 };
m1_update(&mut rf, dt, &m1);
radiation_gas_coupling_step(&mut particles, &mut rf, &m1, dt, box_size);
```

Activar en TOML:
```toml
[rt]
enabled     = true
c_red_factor = 100.0
kappa_abs    = 1.0
rt_mesh      = 32
substeps     = 5
```

### 🔁 Integración automática in-situ (Phase 82)

Los módulos SPH, RT, bispectrum y assembly bias se ejecutan automáticamente en cada
paso de simulación cuando están activados. La macro `maybe_rt!()` y `maybe_sph!()` en
`engine.rs` coordinan la secuencia: gravedad → SPH → RT → análisis in-situ.

### 📊 Post-procesamiento automático (Phase 83)

```bash
# Procesar todos los insitu_*.json de una corrida
python docs/scripts/postprocess_insitu.py \
    --dir runs/cosmo/insitu \
    --out analysis/ \
    --box-size 100.0
# → analysis/pk_evolution.png
# → analysis/pk_multipoles.png
# → analysis/sigma8_evolution.png
# → analysis/halos_evolution.png
# → analysis/bispectrum.png
# → analysis/summary.json
```

---

## Tests automáticos

```bash
# Chequeo físico local recomendado (transfer tabulada unit + pancake + CLASS completo):
bash scripts/check-physics.sh

# Tests unitarios de todos los crates (tests lentos marcados #[ignore] → ~3.5 min)
cargo test

# Tests de validación física (N-body + cosmología) — rápidos por defecto (~3.5 min)
cargo test -p gadget-ng-physics --release

# Pancake Zel’dovich (analítico)
cargo test -p gadget-ng-physics --test zeldovich_pancake --release

# MAC relativo vs costo (árbol)
cargo test -p gadget-ng-tree --test mac_error_cost --release

# Incluir tests lentos (N=32³-64³, corridas largas):
cargo test -p gadget-ng-physics --release -- --include-ignored

# Tests específicos por fase
cargo test -p gadget-ng-physics --test cosmo_serial                       # Fase 17a
cargo test -p gadget-ng-physics --test cosmo_pm                           # Fase 18
cargo test -p gadget-ng-physics --test cosmo_pm_dist                      # Fase 19
cargo test -p gadget-ng-physics --test cosmo_pm_slab                      # Fase 20
cargo test -p gadget-ng-physics --test phase30_linear_reference --release # Fase 30
cargo test -p gadget-ng-physics --test phase31_ensemble          --release # Fase 31
cargo test -p gadget-ng-physics --test phase32_high_res_ensemble --release # Fase 32
cargo test -p gadget-ng-physics --test phase33_pk_normalization  --release # Fase 33
cargo test -p gadget-ng-physics --test phase34_discrete_normalization --release # Fase 34
cargo test -p gadget-ng-physics --test phase35_rn_modeling       --release # Fase 35
cargo test -p gadget-ng-physics --test phase36_pk_correction_validation --release # Fase 36
cargo test -p gadget-ng-physics --test phase37_growth_rescaled_ics --release # Fase 37
cargo test -p gadget-ng-physics --test phase38_class_validation  --release # Fase 38 (CLASS)
cargo test -p gadget-ng-physics --test phase39_dt_convergence    --release # Fase 39
cargo test -p gadget-ng-physics --test phase40_physical_ics_normalization --release # Fase 40
cargo test -p gadget-ng-physics --test phase41_high_resolution_validation --release # Fase 41
# Rerun rápido Phase 41 desde cache de disco (~0.7 s):
PHASE41_USE_CACHE=1 PHASE41_SKIP_N256=1 cargo test -p gadget-ng-physics \
    --test phase41_high_resolution_validation --release
cargo test -p gadget-ng-physics --test phase42_tree_short_range --release # Fase 42
# Smoke N=32 (~8 min) y corrida completa N=64 (~2h 18min):
PHASE42_QUICK=1 cargo test -p gadget-ng-physics --test phase42_tree_short_range --release
PHASE42_N=64 cargo test -p gadget-ng-physics --test phase42_tree_short_range --release
# Rerun rápido desde caché de disco:
PHASE42_USE_CACHE=1 cargo test -p gadget-ng-physics \
    --test phase42_tree_short_range --release

# Phase 58 — c(M) + ξ(r) (FoF → NFW → Ludlow vs Duffy + correlación)
cargo test -p gadget-ng-physics --test phase58_nfw_concentration --release
PHASE58_SKIP=1 cargo test -p gadget-ng-physics --test phase58_nfw_concentration --release

# Phase 59 — Checkpoint bit-a-bit (20 pasos → checkpoint a 10 → restart)
cargo test -p gadget-ng-physics --test phase59_checkpoint_continuity --release

# Phase 60 — Rebalanceo adaptativo (should_rebalance + imbalance_threshold)
cargo test -p gadget-ng-physics --test phase60_adaptive_rebalance --release

# GPU PM CUDA/HIP (requieren toolchain; SKIP=1 para CI sin GPU)
CUDA_SKIP=1 cargo test -p gadget-ng-cuda --release
HIP_SKIP=1  cargo test -p gadget-ng-hip  --release

# Analysis: ξ(r) FFT + pares, c(M) Ludlow+2016
cargo test -p gadget-ng-analysis --release
```

Tests de validación cubiertos:

- **Kepler**: conservación de energía y momento angular (1–2 cuerpos).
- **Plummer**: ratio virial Q ≈ 0.5 en equilibrio.
- **Cosmología serial**: EdS y ΛCDM, `a(t)` por RK4, sin NaN.
- **PM periódico** (Fase 18): CIC masa, Poisson sinusoidal, `G/a`, wrap.
- **PM distribuido** (Fase 19): equivalencia serial/MPI, allreduce.
- **PM slab** (Fase 20): `SlabLayout`, ghost CIC, transpose roundtrip,
  Poisson sanity a P = {1, 2, 4} ranks.
- **TreePM scatter-gather** (Fase 24–25): paridad con TreePM replicado.
- **ICs ZA + EH + σ₈** (Fase 26–27): transferencia, normalización,
  reproducibilidad bit-idéntica.
- **Transfer tabulada** (2026-05): reconstrucción PCHIP log–log en nodos y puntos intermedios (`gadget-ng-core`).
- **Pancake Zel’dovich** (2026-05): convergencia frente a solución analítica (`zeldovich_pancake`).
- **MAC bmax + relativo** (2026-05): `GeometricBmax` y comparación error/costo (`mac_error_cost`).
- **PM FFT** (2026-05): paridad numérica backend `fftw` vs RustFFT cuando la feature está activa (`gadget-ng-pm`).
- **2LPT** (Fase 28–29): crecimiento `D₁(a)`, residuos 1LPT vs 2LPT.
- **Ensembles cosmológicos** (Fase 30–32): shape espectral, `R(k)`,
  crecimiento lineal, PM vs TreePM, reproducibilidad.
- **`pk_correction`** (Fase 34–36): roundtrip DFT, `A_grid` cerrado,
  `R(N)` estable, corrección absoluta sobre snapshots reales.
- **Reescalado físico de ICs** (Fase 37): `s = D(a_init)/D(1)` exacto a
  `< 2·10⁻⁶`, crecimiento `D(a)` sensible a la convención.
- **Validación externa vs CLASS** (Fase 38): factor de mejora ≥161× y
  `median|log10(P_c/P_CLASS)| ≤ 0.046` en ambas convenciones.
- **Convergencia temporal** (Fase 39): barrido `dt` + diagnósticos
  `delta_rms(a)`, sin NaN/Inf.
- **`NormalizationMode`** (Fase 40): enum tipado, equivalencia bit-idéntica
  del modo `Legacy` y ratio `s` exacto del modo `Z0Sigma8`.
- **Alta resolución** (Fase 41): transición shot-noise ↔ señal a `N ≥ 64³`,
  cierre de `pk_correction` en IC hasta `N = 128³`, con caché JSON para
  rerun sub-segundo (`PHASE41_USE_CACHE=1`).
- **TreePM + softening absoluto** (Fase 42): matriz `PM + 3 TreePM ε_phys`
  a `N ∈ {32, 64}`, `δ_rms(a)`, `v_rms(a)`, error de crecimiento lineal
  y espectral. A `N=64³`, TreePM mejora el error de crecimiento **×345**
  vs PM; óptimo interior `ε_phys ≈ 0.01 Mpc/h`. Soporta `PHASE42_N=<N>`
  (16–256, potencia de 2) y `PHASE42_USE_CACHE=1`.
- **⏱️ Block timesteps + LET distribuido** (Phase 56): integrador jerárquico acoplado
  a árbol LET; fuerzas evaluadas solo para `active_local` partículas; validado
  contra integrador de paso único en corrida cosmológica de referencia.
- **🟢 GPU CUDA PM** (Phase 57): `gadget-ng-cuda` compila kernels CIC + FFT 3D en NVIDIA GPU;
  `CUDA_SKIP=1` permite CI sin toolchain; interfaz idéntica al solver CPU.
- **🔴 GPU HIP PM** (Phase 57): `gadget-ng-hip` replica la cadena CUDA para AMD ROCm;
  `HIP_SKIP=1` para CI; mismos tests que CUDA para paridad.
- **📐 c(M) + ξ(r) desde N-body** (Phase 58): PM 64³ → FoF → fit NFW → concentración
  medida vs Duffy+2008 y Ludlow+2016; ξ(r) FFT y pares validados en distribución uniforme.
- **💾 Checkpoint bit-a-bit** (Phase 59): 20 pasos PM → guarda en paso 10 → restart →
  partículas bit-idénticas en paso 20; `SfcDecomposition` reconstruida automáticamente.
- **🔀 Rebalanceo adaptativo** (Phase 60): `should_rebalance` con `cost_pending` override;
  `rebalance_imbalance_threshold = 0.0` desactiva el costo (solo intervalo fijo).
- **🌊 SPH Gadget-2** (Phase 166): entropía A=P/ρ^γ inicializada correctamente; Balsara 0 ≤ f ≤ 1;
  `courant_dt` finito; paso KDK energía acotada; tubo de Sod + colapso de Evrard (lentos, `--include-ignored`).

---

## Reportes técnicos

Los reportes en [`docs/reports/`](docs/reports/) documentan cada fase con contexto,
metodología, resultados y limitaciones.

> El índice completo de reportes está en
> [docs/development-history.md](docs/development-history.md#reportes-técnicos).

### N-body + HPC

| Reporte | Tema |
|---------|------|
| [`phase3-gadget4-benchmark`](docs/reports/2026-04-phase3-gadget4-benchmark.md) | Benchmark vs GADGET-4 |
| [`phase4-multipole-softening`](docs/reports/2026-04-phase4-multipole-softening.md) | Suavizado de multipolos |
| [`phase5-energy-mac-consistency`](docs/reports/2026-04-phase5-energy-mac-consistency.md) | Consistencia energía + MAC |
| [`phase6-higher-order-integrator`](docs/reports/2026-04-phase6-higher-order-integrator.md) | Yoshida4 |
| [`phase7-aarseth-timestep`](docs/reports/2026-04-phase7-aarseth-timestep.md) | Timestep adaptativo |
| [`phase8-hpc-scaling`](docs/reports/2026-04-phase8-hpc-scaling.md) | Escalado HPC |
| [`phase9-hpc-local`](docs/reports/2026-04-phase9-hpc-local.md) | SFC + halos locales |
| [`phase10-let-tree`](docs/reports/2026-04-phase10-let-tree.md) | LetTree compacto |
| [`phase11-let-tree-parallel-validation`](docs/reports/2026-04-phase11-let-tree-parallel-validation.md) | Validación LET paralelo |
| [`phase12-let-communication-reduction`](docs/reports/2026-04-phase12-let-communication-reduction.md) | Reducción comm LET |
| [`phase13-hilbert-decomposition`](docs/reports/2026-04-phase13-hilbert-decomposition.md) | Hilbert 3D SFC |
| [`phase14-soa-simd`](docs/reports/2026-04-phase14-soa-simd.md) | SoA + SIMD |
| [`phase15-explicit-simd`](docs/reports/2026-04-phase15-explicit-simd.md) | SIMD explícito |
| [`phase15b-leaf-max-sweep`](docs/reports/2026-04-phase15b-leaf-max-sweep.md) | Leaf-max sweep |
| [`phase16-tiled-simd-multi-i`](docs/reports/2026-04-phase16-tiled-simd-multi-i.md) | Tiling 4×N_i |

### Cosmología + PM + TreePM

| Reporte | Tema |
|---------|------|
| [`phase17a-cosmology-serial`](docs/reports/2026-04-phase17a-cosmology-serial.md) | Cosmología ΛCDM serial |
| [`phase17b-cosmology-distributed`](docs/reports/2026-04-phase17b-cosmology-distributed.md) | Cosmología MPI + SFC+LET |
| [`phase18-periodic-pm`](docs/reports/2026-04-phase18-periodic-pm.md) | PM periódico con CIC + FFT |
| [`phase19-distributed-pm`](docs/reports/2026-04-phase19-distributed-pm.md) | PM sin allgather |
| [`phase20-slab-distributed-pm`](docs/reports/2026-04-phase20-slab-distributed-pm.md) | PM slab FFT distribuida |
| [`phase21-distributed-treepm`](docs/reports/2026-04-phase21-distributed-treepm.md) | TreePM distribuido (first light) |
| [`phase22-treepm-3d-halo`](docs/reports/2026-04-phase22-treepm-3d-halo.md) | Halos 3D para SR periódico |
| [`phase23-treepm-sr-3d-domain`](docs/reports/2026-04-phase23-treepm-sr-3d-domain.md) | Dominio 3D con SFC |
| [`phase24-treepm-pm-scatter-gather`](docs/reports/2026-04-phase24-treepm-pm-scatter-gather.md) | Scatter-gather PM |
| [`phase25-treepm-scatter-gather-mpi-validation`](docs/reports/2026-04-phase25-treepm-scatter-gather-mpi-validation.md) | Validación MPI scatter-gather |

### ICs, ensembles y normalización de `P(k)`

| Reporte | Tema |
|---------|------|
| [`phase26-zeldovich-ics-validation`](docs/reports/2026-04-phase26-zeldovich-ics-validation.md) | ICs Zel'dovich (1LPT) |
| [`phase27-transfer-sigma8-ics`](docs/reports/2026-04-phase27-transfer-sigma8-ics.md) | Transfer EH + σ₈ |
| [`phase28-2lpt-ics`](docs/reports/2026-04-phase28-2lpt-ics.md) | ICs 2LPT |
| [`phase29-1lpt-vs-2lpt-validation`](docs/reports/2026-04-phase29-1lpt-vs-2lpt-validation.md) | 1LPT vs 2LPT |
| [`phase30-linear-reference-validation`](docs/reports/2026-04-phase30-linear-reference-validation.md) | Referencia lineal ΛCDM |
| [`phase31-ensemble-higher-resolution-validation`](docs/reports/2026-04-phase31-ensemble-higher-resolution-validation.md) | Ensemble N=16³ |
| [`phase32-high-resolution-ensemble-validation`](docs/reports/2026-04-phase32-high-resolution-ensemble-validation.md) | Ensemble N=32³ · 6 seeds |
| [`phase33-pk-normalization-analysis`](docs/reports/2026-04-phase33-pk-normalization-analysis.md) | Análisis del offset 17× |
| [`phase34-discrete-normalization-closure`](docs/reports/2026-04-phase34-discrete-normalization-closure.md) | Cierre `A_grid = 2·V²/N⁹` |
| [`phase35-rn-modeling`](docs/reports/2026-04-phase35-rn-modeling.md) | Modelado `R(N)` |
| [`phase36-pk-correction-validation`](docs/reports/2026-04-phase36-pk-correction-validation.md) | Validación `pk_correction` |
| [`phase37-growth-rescaled-ics`](docs/reports/2026-04-phase37-growth-rescaled-ics.md) | Reescalado físico `D(a)` experimental |
| [`phase38-class-camb-minimal-validation`](docs/reports/2026-04-phase38-class-camb-minimal-validation.md) | 🔭 Validación externa vs CLASS |
| [`phase39-dt-convergence`](docs/reports/2026-04-phase39-dt-convergence.md) | Barrido `dt` y diagnósticos dinámicos |
| [`phase40-physical-ics-normalization`](docs/reports/2026-04-phase40-physical-ics-normalization.md) | `NormalizationMode { Legacy, Z0Sigma8 }` |
| [`phase41-high-resolution-validation`](docs/reports/2026-04-phase41-high-resolution-validation.md) | 📈 Alta resolución y shot-noise↔señal |
| [`phase42-tree-short-range`](docs/reports/2026-04-phase42-tree-short-range.md) | 🌲 TreePM + softening absoluto `ε_phys` |
| [`phase160-zeldovich-pancake`](docs/reports/2026-05-phase160-zeldovich-pancake.md) | 🥞 Pancake Zel’dovich: convergencia vs solución analítica |

### Calidad / deuda (2026-05)

| Reporte | Tema |
|---------|------|
| [`clippy-all-targets-backlog`](docs/reports/2026-05-clippy-all-targets-backlog.md) | Snapshot `clippy --all-targets` y plan de endurecimiento |

### HPC avanzado, GPU y análisis (Phase 56–60 + Rápidas)

| Reporte | Tema |
|---------|------|
| [`phase56-hierarchical-let`](docs/reports/2026-04-phase56-hierarchical-let.md) | ⏱️ Block timesteps jerárquicos + árbol LET distribuido |
| [`phase57-cuda-hip-pm`](docs/reports/2026-04-phase57-cuda-hip-pm.md) | 🟢🔴 PM solver CUDA/HIP: segunda cadena de compilación |
| [`phase58-nfw-concentration-xi`](docs/reports/2026-04-phase58-nfw-concentration-xi.md) | 📐 c(M) desde N-body + ξ(r) FFT y pares |
| [`phase59-checkpoint-continuity`](docs/reports/2026-04-phase59-checkpoint-continuity.md) | 💾 Checkpoint/restart bit-a-bit robusto |
| [`phase60-adaptive-rebalance`](docs/reports/2026-04-phase60-adaptive-rebalance.md) | 🔀 Domain decomposition adaptativa por costo |
| [`rapidas-analyze-vis`](docs/reports/2026-04-rapidas-analyze-vis.md) | 📊🖼️ CLI `analyze` + PPM rendering sin dependencias |

### MHD + Física bariónica avanzada (Phases 123–150)

| Reporte | Tema |
|---------|------|
| [`phase123-mhd-crate`](docs/reports/2026-04-phase123-mhd-crate.md) | 🧲 Crate gadget-ng-mhd + inducción SPH |
| [`phase124-mhd-forces`](docs/reports/2026-04-phase124-mhd-forces.md) | Presión magnética + tensor Maxwell |
| [`phase125-dedner-divb`](docs/reports/2026-04-phase125-dedner-divb.md) | Cleaning div-B Dedner |
| [`phase126-mhd-engine`](docs/reports/2026-04-phase126-mhd-engine.md) | Integración en engine + onda Alfvén |
| [`phase127-magnetic-ics`](docs/reports/2026-04-phase127-magnetic-ics.md) | ICs magnetizadas + CFL magnético |
| [`phase128-mhd-validation`](docs/reports/2026-04-phase128-mhd-validation.md) | Validación Alfvén 3D + Brio-Wu 1D |
| [`phase129-cr-b-coupling`](docs/reports/2026-04-phase129-cr-b-coupling.md) | CR suprimidos por β_plasma |
| [`phase130-dust`](docs/reports/2026-04-phase130-dust.md) | 🌑 Polvo intersticial básico D/G |
| [`phase131-hdf5-mhd`](docs/reports/2026-04-phase131-hdf5-mhd.md) | HDF5 campos MHD+SPH (Bfld, Psi, ECr, Dust) |
| [`phase132-mhd-bench`](docs/reports/2026-04-phase132-mhd-bench.md) | Benchmarks Criterion MHD + CFL unificado |
| [`phase133-anisotropic-mhd`](docs/reports/2026-04-phase133-anisotropic-mhd.md) | Difusión ∥B térmica y CR anisótropa |
| [`phase134-magnetic-cooling`](docs/reports/2026-04-phase134-magnetic-cooling.md) | Cooling magnético (sincrotrón) |
| [`phase135-artificial-resistivity`](docs/reports/2026-04-phase135-artificial-resistivity.md) | Resistividad artificial α_b adaptativo |
| [`phase136-cosmo-mhd`](docs/reports/2026-04-phase136-cosmo-mhd.md) | MHD cosmológico end-to-end, B_rms(a) |
| [`phase137-dust-rt`](docs/reports/2026-04-phase137-dust-rt.md) | Polvo + RT UV: absorción kappa_dust×D/G |
| [`phase138-flux-freeze`](docs/reports/2026-04-phase138-flux-freeze.md) | ❄️ Flux-freeze B en ICM, beta_freeze |
| [`phase139-srmhd`](docs/reports/2026-04-phase139-srmhd.md) | ⚡ SRMHD: factor de Lorentz γ, primitivas NR |
| [`phase140-turbulence`](docs/reports/2026-04-phase140-turbulence.md) | 🌪️ Turbulencia O-U, P_B(k) ∝ k^{-5/3} |
| [`phase141-tests`](docs/reports/2026-04-phase141-tests.md) | Tests integración MHD avanzados (48 tests) |
| [`phase142-engine-rmhd-turb`](docs/reports/2026-04-phase142-engine-rmhd-turb.md) | Engine: RMHD + turbulencia integrados |
| [`phase143-advanced-bench`](docs/reports/2026-04-phase143-advanced-bench.md) | Benchmarks Criterion: turb, flux-freeze, SRMHD |
| [`phase144-clippy`](docs/reports/2026-04-phase144-clippy.md) | Clippy 0 warnings en todo el workspace |
| [`phase145-reconnection`](docs/reports/2026-04-phase145-reconnection.md) | 🔁 Reconexión Sweet-Parker: B antiparalelos |
| [`phase146-braginskii`](docs/reports/2026-04-phase146-braginskii.md) | 🌡️ Viscosidad Braginskii anisótropa π_ij |
| [`phase147-mhd-cosmo-full`](docs/reports/2026-04-phase147-mhd-cosmo-full.md) | Corrida MHD completa + P_B(k) end-to-end |
| [`phase148-rmhd-jets`](docs/reports/2026-04-phase148-rmhd-jets.md) | 🕳️ Jets AGN relativistas desde halos FoF |
| [`phase149-two-fluid`](docs/reports/2026-04-phase149-two-fluid.md) | ⚛️ Plasma 2F: T_e ≠ T_i, Coulomb implícito |

### Physics Extensions 2026 (Phases 177–185)

| Reporte | Tema |
|---------|------|
| [`phase177-cooling-sf-feedback-production`](docs/reports/2026-05-phase177-cooling-sf-feedback-production.md) | UVB, ley SF por presión y feedback térmico estocástico |
| [`phase178-physics-extensions-closure`](docs/reports/2026-05-phase178-physics-extensions-closure.md) | PM f(R) homogéneo, química primordial 9 especies y lightcones |
| [`phase179-deuterium-hd-cooling`](docs/reports/2026-05-phase179-deuterium-hd-cooling.md) | Química D/HD y cooling molecular HD |
| [`phase180-pop-iii-first-stars`](docs/reports/2026-05-phase180-pop-iii-first-stars.md) | Pop III, IMF top-heavy y feedback PISN |
| [`phase181-rt-multifrequency-lw`](docs/reports/2026-05-phase181-rt-multifrequency-lw.md) | Grupos HI/HeI/HeII/LW/IR y LW H₂/HD |
| [`phase182-dust-ir-thermal-emission`](docs/reports/2026-05-phase182-dust-ir-thermal-emission.md) | Temperatura de granos y emisión IR |
| [`phase183-agn-spin-mergers`](docs/reports/2026-05-phase183-agn-spin-mergers.md) | Spin Kerr, mergers BH y recoil |
| [`phase184-wdm-fdm`](docs/reports/2026-05-phase184-wdm-fdm.md) | Cutoff WDM/FDM en ICs y presión cuántica FDM |
| [`phase185-fr-nonlinear-mesh`](docs/reports/2026-05-phase185-fr-nonlinear-mesh.md) | f(R) no lineal en malla PM |

### Meta

| Reporte | Tema |
|---------|------|
| [`gadget-ng-treepm-evolution-paper`](docs/reports/2026-04-gadget-ng-treepm-evolution-paper.md) | Paper-style sobre evolución TreePM |
| [`validation-phase`](docs/reports/2026-04-validation-phase.md) | Protocolo general de validación |

---

## Features opcionales

| Feature | Descripción |
|---------|-------------|
| `mpi` | Enlaza a MPI para `MpiRuntime` con descomposición SFC Hilbert |
| `gpu` | Aceleración GPU vía `wgpu` (Vulkan/Metal/DX12/WebGPU) |
| `🟢 cuda` | Solver PM NVIDIA CUDA + cuFFT (`gadget-ng-cuda`; requiere CUDA toolkit) |
| `🔴 hip` | Solver PM AMD HIP + rocFFT (`gadget-ng-hip`; requiere ROCm) |
| `rayon` | Paralelismo intra-rango con Rayon |
| `simd` | Kernels SIMD explícitos / `#[target_feature]` sin activar Rayon |
| `pm-rayon` | Rayon en PM/TreePM (`gadget-ng-pm`, `gadget-ng-treepm`) |
| `bincode` | Snapshots binarios `particles.bin` |
| `hdf5` | Snapshots `snapshot.hdf5` estilo GADGET-4 (requiere `libhdf5-dev`) |
| `msgpack` | Snapshots compactos MessagePack |
| `netcdf` | Snapshots NetCDF4 (requiere `libnetcdf-dev`) |
| `full` | Todas las anteriores activadas |
| `fftw` | `gadget-ng-pm`: backend FFT alternativo con tests de paridad frente a RustFFT (API preparada; sin dependencia obligatoria de libfftw en el árbol actual) |

---

## Calidad y CI

```bash
cargo fmt --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --release
cargo build --release --features mpi
```

GitHub Actions: [`.github/workflows/ci.yml`](.github/workflows/ci.yml) (PR/push) y
[`.github/workflows/physics-validation.yml`](.github/workflows/physics-validation.yml) (nocturno: tests físicos largos + log advisory de `clippy --workspace --all-targets`).

Convención de estilo:

- Código y comentarios en **español** salvo nombres de items públicos.
- Tests de fase nuevos bajo `crates/gadget-ng-physics/tests/phaseNN_*.rs`
  con orquestador `experiments/nbody/phaseNN_*/run_phaseNN.sh`.
- Un reporte por fase en `docs/reports/2026-MM-phaseNN-*.md`.
- `CHANGELOG.md` actualizado bajo `[Unreleased]` antes de cerrar PR.

---

## Estructura de experimentos

Cada experimento vive en `experiments/nbody/phaseNN_*/` con una
estructura uniforme:

```
experiments/nbody/phaseNN_<slug>/
├── configs/           # Uno o varios .toml
├── scripts/           # Python de postproceso + generación de figuras
├── output/            # Artefactos de corrida (ignorado por git en general)
├── figures/           # PNGs generados
└── run_phaseNN.sh     # Orquestador: ejecuta, mide, plotea
```

Ejemplos actuales:

```
experiments/nbody/
├── phase18_periodic_pm/      # PM periódico: N=512..2000, grid 16³..32³
├── phase19_distributed_pm/   # PM allreduce: comparativa vs clásico
├── phase20_slab_pm/          # PM slab alltoall: escalado P=1,2,4
├── phase26_zeldovich_ics/    # Validación ZA + reproducibilidad
├── phase27_transfer_sigma8/  # EH + σ₈ + roundtrip σ
├── phase28_2lpt/             # 2LPT: corrección al campo de velocidad
├── phase29_1lpt_vs_2lpt/     # Crecimiento D₁(a) comparado
├── phase30_linear_reference/ # Referencia lineal vs EH + D(a)²
├── phase31_ensemble_higher_res/
├── phase33_pk_normalization/ # Caracterización del offset 17×
├── phase34_discrete_normalization/ # A_grid = 2·V²/N⁹
├── phase35_rn_modeling/      # Fit R(N), figuras, demo
├── phase36_pk_correction_validation/ # Validación práctica end-to-end
│   ├── configs/lcdm_N32_2lpt_pm_phase36.toml
│   ├── scripts/apply_phase36_correction.py
│   ├── scripts/plot_phase36.py
│   └── run_phase36.sh
├── phase37_growth_rescaled_ics/ # Reescalado físico opcional de ICs
├── phase38_class_validation/    # 🔭 Referencia CLASS + comparación por N
│   └── reference/               # JSON generado por classy 3.3.4.0
├── phase39_dt_convergence/      # Barrido dt ∈ {4e-4..5e-5}
├── phase40_physical_ics_normalization/ # Enum NormalizationMode
├── phase41_high_resolution_validation/ # N ∈ {32, 64, 128}, shot-noise
│   ├── configs/lcdm_N{128,256}_2lpt_pm_{legacy,z0_sigma8}.toml
│   ├── scripts/apply_phase41_correction.py
│   ├── scripts/plot_phase41_resolution.py
│   └── run_phase41.sh           # PHASE41_SKIP_N256=1, PHASE41_USE_CACHE=1
└── phase42_tree_short_range/    # 🌲 TreePM + softening absoluto ε_phys
    ├── configs/lcdm_N128_{pm_eps0,treepm_eps{001,002,005}}.toml
    ├── scripts/apply_phase42_correction.py
    ├── scripts/plot_phase42_short_range.py
    └── run_phase42.sh           # PHASE42_N=<N>, PHASE42_QUICK, PHASE42_USE_CACHE
```

---

## Reionización y RT (Phases 87–92)

Combinación completa de Transferencia Radiativa M1 + Química no-equilibrio + Reionización cósmica.

### Configuración TOML completa

```toml
[rt]
enabled      = true
rt_mesh      = 32
c_red_factor = 100.0
kappa_abs    = 1.0
substeps     = 5

[reionization]
enabled       = true
n_sources     = 8
uv_luminosity = 1.0
z_start       = 12.0
z_end         = 6.0

[insitu_analysis]
igm_temp_enabled = true
cm21_enabled     = true
```

### Ejemplo de uso

```rust
use gadget_ng_rt::{reionization_step, ReionizationParams, UvSource};
use gadget_ng_rt::{compute_igm_temp_profile, IgmTempParams};

// Fuentes UV puntuales (halos FoF como fuentes de ionización)
let sources = vec![
    UvSource { pos: [25.0, 25.0, 25.0], luminosity: 1.0 },
];
let params = ReionizationParams::default();
reionization_step(&mut rad_field, &mut chem_states, &sources, &m1_params, dt, box_size, z);

// Perfil de temperatura IGM
let igm_params = IgmTempParams { delta_max: 10.0 };
let bins = compute_igm_temp_profile(&particles, &chem_states, &igm_params);
// bins[i].z, bins[i].t_mean, bins[i].t_median, bins[i].t_sigma
```

### MPI real (rsmpi, Phases 87–88)

Las funciones MPI para RT y AMR usan `rsmpi` real bajo `--features mpi`:

```bash
cargo test -p gadget-ng-rt --features mpi
cargo test -p gadget-ng-pm --features mpi
```

---

## Estadísticas 21cm y EoR (Phases 94–95)

```toml
[insitu_analysis]
cm21_enabled = true

[reionization]
enabled      = true
uv_from_halos = true   # Fuentes UV desde halos FoF
z_start      = 12.0
z_end        = 6.0
```

La señal de temperatura de brillo δT_b ≈ 27 x_HI (1+δ) √((1+z)/10) mK se calcula
por partícula y se acumula en P(k)₂₁cm guardado en `insitu_*.json`.

---

## Feedback AGN (Phase 96)

```toml
[sph.agn]
enabled      = true
eps_feedback = 0.05
m_seed       = 1e5
v_kick_agn   = 500.0
spin_enabled = true
initial_spin = 0.3
mergers_enabled = true
merger_radius = 0.1
recoil_velocity_scale = 500.0
```

Tasa de acreción de Bondi-Hoyle y depósito de energía térmica en partículas vecinas:

```rust
use gadget_ng_core::Vec3;
use gadget_ng_sph::{AgnParams, BlackHole, apply_agn_feedback, bondi_accretion_rate};

let bh = BlackHole::with_spin(Vec3::zero(), 1e8, 0.3);
let params = AgnParams {
    eps_feedback: 0.05,
    m_seed: 1e5,
    v_kick_agn: 500.0,
    r_influence: 1.0,
};
let mdot = bondi_accretion_rate(&bh, rho_gas, c_sound);
apply_agn_feedback(&mut particles, &[bh], &params, dt);
```

---

## SPH Gadget-2 (Phase 166)

### Formulación de entropía

En lugar de integrar la energía interna `u`, el integrador `sph_kdk_step_gadget2` evoluciona
la entropía termodinámica `A = P/ρ^γ` (Springel & Hernquist 2002, MNRAS 333:649).
Esto garantiza segunda ley y conservación en flujos sin viscosidad.

```rust
use gadget_ng_sph::{sph_kdk_step_gadget2, courant_dt};

// Avance KDK con entropía + Balsara + viscosidad de señal
sph_kdk_step_gadget2(&mut particles, dt, |p| [0.0, 0.0, 0.0]);

// Timestep Courant adaptativo
let dt = courant_dt(&particles, 0.3);   // C_Courant = 0.3
```

### Limitador de Balsara

Suprime la viscosidad artificial en flujos de cizalla pura (galaxias en rotación, discos),
activándola solo donde hay compresión genuina:

```
f_i = |∇·v_i| / (|∇·v_i| + |∇×v_i| + ε · c_s/h)
```

### Viscosidad de señal (Gadget-2, Ec. 14)

```
v_sig = α(c_i + c_j − 3 w_ij) / 2    con  w_ij = min(v_ij·r̂_ij, 0)
Π_ij  = −(f_i+f_j)/2 · v_sig · w_ij / (ρ̄_ij)
```

### Tests de validación SPH

```bash
# Tests rápidos: entropía, Balsara acotado, Courant positivo, paso energía acotada
cargo test -p gadget-ng-physics --test gadget2_sph_validation --release

# Tests lentos (Sod + Evrard — requieren --include-ignored):
cargo test -p gadget-ng-physics --test gadget2_sph_validation --release -- --include-ignored
```

| Test | Descripción |
|------|-------------|
| `gadget2_entropy_initialized_correctly` | A_0 = (γ−1)·u/ρ^{γ−1} tras densidad inicial |
| `gadget2_balsara_bounded` | 0 ≤ f_i ≤ 1 para todos los gas particles |
| `gadget2_courant_dt_positive` | dt > 0 y finito desde velocidad de sonido |
| `gadget2_single_step_bounded_energy` | E_total acotada tras un paso KDK |
| `gadget2_sod_shock_compresses_right_region` *(ignore)* | tubo de Sod: región derecha se comprime |
| `gadget2_entropy_monotonically_nondecreasing` *(ignore)* | 2ª ley: S no decrece en Sod |
| `evrard_adiabatic_energy_conservation` *(ignore)* | colapso de Evrard: E_total acotada |
| `evrard_central_density_increases` *(ignore)* | densidad central crece en colapso |

---

## Physics Extensions cerradas (Phases 177–185)

La cartera documentada en [docs/roadmap-physics-extensions.md](docs/roadmap-physics-extensions.md)
quedó cerrada en código. Los módulos nuevos son retrocompatibles por defecto:
si las secciones TOML no se activan, las corridas legacy conservan su comportamiento.

```toml
[rt]
enabled = true
multifrequency_enabled = true
lw_h2_factor = 1.0
lw_hd_factor = 1.0

[sph.dust]
enabled = true
ir_emission_enabled = true
kappa_dust_ir = 10.0
ir_emissivity = 1.0

[sph.pop_iii]
enabled = true
critical_metallicity = 1.0e-4
min_h2_fraction = 1.0e-6
min_hd_fraction = 1.0e-9

[dark_matter]
enabled = true
model = "warm"      # "cold" | "warm" | "fuzzy"
m_wdm_kev = 3.0

[modified_gravity]
enabled = true
f_r0 = 1.0e-6
nonlinear_mesh = true
mesh_iterations = 4
screening_smoothing = 0.5
```

| Phase | API principal | Validación |
|-------|---------------|------------|
| 177 | `CoolingKind::UvBackground`, SF por presión, feedback térmico estocástico | `phase177_cooling_sf_feedback.rs` |
| 178 | `solve_forces_modified_gravity`, química primordial 9 especies | `gadget-ng-pm`, `phase158_modified_gravity` |
| 179 | `ChemState` 12 especies, `cooling_rate_hd` | `phase179_deuterium_hd.rs` |
| 180 | `PopIIISection`, `form_pop_iii_clusters`, PISN | `phase180_pop_iii.rs` |
| 181 | `MultiFrequencyField`, `PhotonGroup`, LW H₂/HD | `phase181_multifrequency_lw.rs` |
| 182 | `dust_equilibrium_temperature`, `deposit_dust_ir_emission` | `phase182_dust_ir_emission.rs` |
| 183 | `BlackHole::spin`, `merge_black_holes`, eficiencia Kerr | `phase183_agn_spin_mergers.rs` |
| 184 | `[dark_matter]`, `wdm_transfer_suppression`, `fdm_transfer_suppression` | `phase184_wdm_fdm.rs` |
| 185 | `solve_forces_fr_screened_mesh`, `fr_screening_field` | `phase185_fr_nonlinear_mesh.rs` |

---

## MHD Completo + Plasma 2F (Phases 123–150)

El crate `gadget-ng-mhd` implementa un stack MHD completo acoplado al motor SPH.

### Configuración TOML MHD

```toml
[mhd]
enabled             = true
alpha_b             = 0.5        # resistividad artificial adaptativa
beta_freeze         = 0.1        # umbral de flux-freeze en ICM
relativistic_mhd    = false      # activar SRMHD (Lorentz γ)
v_rel_threshold     = 0.1        # v/c mínima para régimen relativista
reconnection_enabled = true
f_reconnection      = 0.01       # fracción de E_mag liberada por reconexión
eta_braginskii      = 0.05       # coeficiente de viscosidad Braginskii
jet_enabled         = false      # jets AGN bipolares desde halos FoF
v_jet               = 0.5        # fracción de c para el jet (SRMHD)
n_jet_halos         = 1          # número de halos AGN activos

[mhd.initial_b_field]
kind = "uniform"
value = [1.0e-6, 0.0, 0.0]      # B_0 en unidades internas

[turbulence]
enabled         = true
amplitude       = 0.01
correlation_time = 0.1
k_min           = 1
k_max           = 4
spectral_index  = 1.666          # Kolmogorov -5/3

[two_fluid]
enabled      = true
nu_ei_coeff  = 1.0               # factor sobre ν_ei Coulomb
t_e_init_k   = 1.0e4             # T_e inicial en Kelvin
```

### Funciones públicas clave (`gadget-ng-mhd`)

```rust
// Turbulencia Ornstein-Uhlenbeck
gadget_ng_mhd::apply_turbulent_forcing(&mut particles, &cfg.turbulence, dt, step as u64);

// Flux-freeze ICM (B ∝ ρ^{2/3})
gadget_ng_mhd::apply_flux_freeze(&mut particles, rho_ref, cfg.mhd.beta_freeze);

// SRMHD — avanza campos relativistas
gadget_ng_mhd::advance_srmhd(&mut particles, dt, C_LIGHT);

// Reconexión Sweet-Parker
gadget_ng_mhd::apply_magnetic_reconnection(&mut particles, f_rec, gamma, dt);
// Tasa teórica de reconexión
gadget_ng_mhd::sweet_parker_rate(v_alfven, l_rec, eta_eff);

// Viscosidad Braginskii anisótropa
gadget_ng_mhd::apply_braginskii_viscosity(&mut particles, eta, dt);

// Espectro de potencia magnético P_B(k)
let bins = gadget_ng_mhd::magnetic_power_spectrum(&particles, box_size, n_bins);
// bins[i].k_center, .power

// Jets AGN relativistas desde halos FoF
gadget_ng_mhd::inject_relativistic_jet(&mut particles, &halo_centers, v_jet_frac, n_jet_halos, c, b_jet);

// Plasma de dos fluidos — acoplamiento Coulomb
gadget_ng_mhd::apply_electron_ion_coupling(&mut particles, &cfg.two_fluid, dt);
// diagnóstico T_e/T_i
let ratio = gadget_ng_mhd::mean_te_over_ti(&particles);
```

### Reportes técnicos MHD (Phases 123–150)

| Reporte | Tema |
|---------|------|
| [`phase123-mhd-crate`](docs/reports/2026-04-phase123-mhd-crate.md) | Crate gadget-ng-mhd + inducción SPH |
| [`phase124-mhd-forces`](docs/reports/2026-04-phase124-mhd-forces.md) | Presión magnética + tensor Maxwell |
| [`phase125-dedner-divb`](docs/reports/2026-04-phase125-dedner-divb.md) | Cleaning div-B Dedner |
| [`phase126-mhd-engine`](docs/reports/2026-04-phase126-mhd-engine.md) | Integración en engine + onda Alfvén |
| [`phase127-magnetic-ics`](docs/reports/2026-04-phase127-magnetic-ics.md) | ICs magnetizadas + CFL |
| [`phase128-mhd-validation`](docs/reports/2026-04-phase128-mhd-validation.md) | Validación Alfvén 3D + Brio-Wu 1D |
| [`phase129-cr-b-coupling`](docs/reports/2026-04-phase129-cr-b-coupling.md) | CR suprimidos por β_plasma |
| [`phase130-dust`](docs/reports/2026-04-phase130-dust.md) | Polvo intersticial D/G |
| [`phase131-hdf5-mhd`](docs/reports/2026-04-phase131-hdf5-mhd.md) | HDF5 con campos MHD + SPH |
| [`phase132-mhd-bench`](docs/reports/2026-04-phase132-mhd-bench.md) | Benchmark Criterion + CFL unificado |
| [`phase133-anisotropic-mhd`](docs/reports/2026-04-phase133-anisotropic-mhd.md) | Difusión ∥B térmica + CR |
| [`phase134-magnetic-cooling`](docs/reports/2026-04-phase134-magnetic-cooling.md) | Cooling magnético sincrotrón |
| [`phase135-artificial-resistivity`](docs/reports/2026-04-phase135-artificial-resistivity.md) | Resistividad artificial α_b |
| [`phase136-cosmo-mhd`](docs/reports/2026-04-phase136-cosmo-mhd.md) | MHD cosmológico end-to-end |
| [`phase137-dust-rt`](docs/reports/2026-04-phase137-dust-rt.md) | Polvo + RT UV |
| [`phase138-flux-freeze`](docs/reports/2026-04-phase138-flux-freeze.md) | Freeze-out de B en ICM |
| [`phase139-srmhd`](docs/reports/2026-04-phase139-srmhd.md) | SRMHD: γ Lorentz + primitivización NR |
| [`phase140-turbulence`](docs/reports/2026-04-phase140-turbulence.md) | Turbulencia O-U + P_B(k) |
| [`phase141-tests`](docs/reports/2026-04-phase141-tests.md) | 48 tests de integración avanzados |
| [`phase142-engine-rmhd-turb`](docs/reports/2026-04-phase142-engine-rmhd-turb.md) | Engine: RMHD + turbulencia integrados |
| [`phase143-advanced-bench`](docs/reports/2026-04-phase143-advanced-bench.md) | Benchmarks Criterion avanzados |
| [`phase144-clippy`](docs/reports/2026-04-phase144-clippy.md) | Clippy 0 warnings workspace |
| [`phase145-reconnection`](docs/reports/2026-04-phase145-reconnection.md) | Reconexión magnética Sweet-Parker |
| [`phase146-braginskii`](docs/reports/2026-04-phase146-braginskii.md) | Viscosidad Braginskii anisótropa |
| [`phase147-mhd-cosmo-full`](docs/reports/2026-04-phase147-mhd-cosmo-full.md) | Corrida MHD cosmológica + P_B(k) |
| [`phase148-rmhd-jets`](docs/reports/2026-04-phase148-rmhd-jets.md) | Jets AGN relativistas desde halos FoF |
| [`phase149-two-fluid`](docs/reports/2026-04-phase149-two-fluid.md) | Plasma 2F: T_e ≠ T_i, acoplamiento Coulomb |

---

## Tests MHD y plasma (Phases 142–150)

```bash
# Tests de cada phase MHD/plasma
cargo test -p gadget-ng-physics --test phase142_engine_rmhd_turb  --release
cargo test -p gadget-ng-physics --test phase143_advanced_bench    --release
cargo test -p gadget-ng-physics --test phase144_clippy_clean      --release
cargo test -p gadget-ng-physics --test phase145_reconnection      --release
cargo test -p gadget-ng-physics --test phase146_braginskii        --release
cargo test -p gadget-ng-physics --test phase147_mhd_cosmo_full    --release
cargo test -p gadget-ng-physics --test phase148_rmhd_jets         --release
cargo test -p gadget-ng-physics --test phase149_two_fluid         --release

# Benchmarks Criterion MHD avanzados
cargo bench -p gadget-ng-mhd --bench advanced_bench
```

Tests de validación cubiertos en Phases 142–150:

- **Phase 142**: `TwoFluidSection` defaults, turbulent forcing, flux-freeze, SRMHD sub-threshold, reconexión y Braginskii sin panics.
- **Phase 143**: correctitud de funciones benchmarked: `turb_n100_nonzero`, `flux_freeze_n1000_no_crash`, `conserved_to_primitive_1000_iter_finite`.
- **Phase 144**: 0 regresiones tras limpieza Clippy: `braginskii_eta_zero_is_noop`, `particle_constructors_t_electron_zero`.
- **Phase 145**: liberación de calor para B antiparalelos, sin calor para B paralelos, decremento de |B|, f_rec=0 es noop, `sweet_parker_rate` fórmula.
- **Phase 146**: transferencia de momentum ∥B, η=0 es noop, conservación de momentum, anisotropía ⊥B nula.
- **Phase 147**: `power_spectrum_has_variation`, B_rms ≠ 0, E_mag finita, max_v < c tras evolución MHD.
- **Phase 148**: inyección v_jet, B alineado con eje, energía relativista, n_jet=0 noop, v_jet=0 noop.
- **Phase 149**: T_e inicialización, acoplamiento reduce brecha, T_e ≥ 0, equilibrio T_e/T_i → 1, non-gas ignorados.

---

## Licencia

Este repositorio se distribuye bajo la
[GNU General Public License v3.0](LICENSE) (GPL-3.0).
