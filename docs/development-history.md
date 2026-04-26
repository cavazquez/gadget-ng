# Historial de desarrollo de gadget-ng

Este documento recoge la tabla de fases de desarrollo completadas y los
reportes técnicos asociados. Es material de referencia para contribuidores
y para reproducir experimentos de validación internos.

Para comenzar a **usar** el simulador, ve a [getting-started.md](getting-started.md).

---

## Hitos de desarrollo

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
| **56** | ⏱️ Block timesteps jerárquicos acoplados al árbol LET distribuido | ✅ |
| **57** | 🟢🔴 PM solver CUDA/HIP: segunda cadena de compilación; degradación elegante | ✅ |
| **58** | 📐 c(M) y perfiles NFW desde N-body: fit NFW → FoF halos; ξ(r) FFT+pares | ✅ |
| **59** | 💾 Restart/checkpoint bit-a-bit: guarda/restaura estado completo | ✅ |
| **60** | 🔀 Domain decomposition adaptativa por costo | ✅ |
| **Rápidas** | 📊 CLI `gadget-ng analyze`; 🖼️ `gadget-ng-vis` render PPM sin deps | ✅ |
| **61–65** | SPH cosmológico, merger trees MAH, SUBFIND, N=256³ producción, AMR-PM | ✅ |
| **66** | 🌊 SPH cosmológico KDK: gas+DM, cooling atómico | ✅ |
| **67** | 🌳 Merger trees + Mass Accretion History (MAH) | ✅ |
| **68** | 🔍 SUBFIND: subestructura intra-halo via Union-Find + binding energy | ✅ |
| **69** | 🚀 Producción N=256³: config TOML, scripts PBS, notebook post-proceso | ✅ |
| **70** | 🔲 AMR-PM: parches adaptativos 2 niveles, refinamiento por sobredensidad | ✅ |
| **71** | 📊 Bispectrum B(k₁,k₂,k₃): estadística de orden 3 | ✅ |
| **72** | 🌀 Spin de halos λ Peebles+Bullock: momento angular FoF | ✅ |
| **73** | 📐 Perfiles de velocidad σ_v(r): σ_r, σ_t, β(r) de Binney | ✅ |
| **74** | 💾 HDF5 GADGET-4 estándar: 22 atributos, compatible con yt/pynbody | ✅ |
| **75** | 📈 P(k,μ) RSD: espacio de redshift, multipoles P₀/P₂/P₄ | ✅ |
| **76** | 🔗 Assembly bias: Spearman spin/concentración vs δ_env suavizado | ✅ |
| **77** | 📁 Catálogo de halos HDF5: FoF+SUBFIND, /Header, /Halos, /Subhalos | ✅ |
| **78** | ⭐ Stellar feedback estocástico: SFR+SN kicks | ✅ |
| **79** | 🔬 Validación N=128³: config Planck18, scripts PBS, validate_pk_hmf.py | ✅ |
| **80** | 🔲 AMR jerárquico N-nivel: `build_amr_hierarchy` | ✅ |
| **81** | 🌅 Transferencia radiativa M1: solver HLL, cierre Levermore 1984, acoplamiento SPH | ✅ |
| **82** | 🔁 Integración automática: `maybe_sph!`, `maybe_rt!`, bispectrum+assembly_bias in-situ | ✅ |
| **83** | 📊 Post-procesamiento: `postprocess_insitu.py` — P(k), σ₈(z), multipoles, B(k) | ✅ |
| **84** | 🌐 RT MPI slab: `RadiationFieldSlab`, `allreduce_radiation` | ✅ |
| **85** | 🌐 AMR MPI: `AmrPatchMessage`, `broadcast_patch_forces` | ✅ |
| **86** | 🧪 Química no-equilibrio HII/HeII/HeIII: `ChemState`, solver implícito | ✅ |
| **87** | 🌐 MPI real (rsmpi): `allreduce_radiation_mpi`, tests end-to-end | ✅ |
| **88** | 📈 Benchmarks GPU vs CPU (Criterion) + CI `--release` con tests integración MPI | ✅ |
| **89** | ☀️ Reionización del Universo: `UvSource`, `deposit_uv_sources`, R_Strömgren | ✅ |
| **90** | 🌡️ Perfil de temperatura IGM T(z): `IgmTempBin`, filtrado por densidad SPH | ✅ |
| **91** | 📄 Paper draft JOSS: `docs/paper/paper.md` + `paper.bib` (15 refs BibTeX) | ✅ |
| **92** | 📊 Benchmarks formales: `bench_mpi_scaling.sh`, `bench_pk_vs_gadget4.py` | ✅ |
| **93** | 📝 README final + figuras JOSS (P(k), HMF, Strömgren) | ✅ |
| **94** | 📡 Estadísticas 21cm: `brightness_temperature`, P(k)₂₁cm, `Cm21Output` in-situ | ✅ |
| **95** | 🌌 EoR z=6–12: `maybe_reionization!` en engine, `uv_from_halos` | ✅ |
| **96** | 🕳️ Feedback AGN: `BlackHole`, `bondi_accretion_rate`, `apply_agn_feedback` | ✅ |
| **97–99** | SPH avanzado: `z_metal`, `MetalCooling`, enriquecimiento Q, SN Ia DTD | ✅ |
| **100** | AGN con halos FoF: `bondi_agn_halo`, `agn_halos_from_fof` | ✅ |
| **101** | Fix softening comóvil→físico: `epsilon_phys = epsilon_comoving * a` | ✅ |
| **102** | HDF5 layout GADGET-4 completo: grupos `/PartType0-5`, atributos Header | ✅ |
| **103** | Domain decomp con coste medido: `cpu_time_tree_ns` por partícula | ✅ |
| **104** | CLI extendida: `gadget-ng postprocess`, `--phases`, logging estructurado JSON | ✅ |
| **105** | JSONL con campos SPH: `u_therm`, `rho`, `h_sml`, `z_metal`, `sfr` | ✅ |
| **106** | Restart con SPH state completo: campos MHD en checkpoint | ✅ |
| **107** | Merger trees con FoF real: árbol de fusiones usando IDs de halos FoF | ✅ |
| **108** | Vientos galácticos: `apply_galactic_winds`, `v_wind ∝ σ_dm` | ✅ |
| **109** | Metales en Particle + `ParticleType::Star`: `z_alpha`, `z_fe` | ✅ |
| **110** | Enriquecimiento químico SPH: yields O+Fe de SNII+SNIa distribuidos a vecinos | ✅ |
| **111** | Enfriamiento por metales (`MetalCooling`): tablas Z-dependientes | ✅ |
| **112** | Partículas estelares reales (spawning): `spawn_star_particles`, SSP Kroupa | ✅ |
| **113** | SN Ia con DTD power-law: `snia_rate_dtd`, yields Fe, calor térmico | ✅ |
| **114** | ISM Multifase fría-caliente: `TwoPhaseISM`, fracción fría `x_cold` | ✅ |
| **115** | Vientos estelares pre-SN: `apply_stellar_winds`, masa perdida via `mass_loss_rate` | ✅ |
| **116** | Modo radio AGN (bubble feedback): `inject_agn_bubble`, cavidades de entalpía | ✅ |
| **117** | Rayos cósmicos básicos: `Particle.e_cr`, `inject_cr_sn`, difusión isotrópica | ✅ |
| **118** | Función de luminosidad y colores galácticos: magnitudes B/V/R | ✅ |
| **119** | Enfriamiento tabulado Sutherland-Dopita 1993: `cooling_sd93`, grilla [T, Z] | ✅ |
| **120** | Engine integration bariónica: `maybe_sph!` coordina SPH+quím+metales+feedback | ✅ |
| **121** | Conducción térmica ICM Spitzer: `apply_spitzer_conduction`, κ ∝ T^{5/2} | ✅ |
| **122** | Gas molecular HI→H₂: `MolecularFraction`, umbral de densidad, shielding UV | ✅ |
| **123** | 🧲 Crate `gadget-ng-mhd` + `b_field` en Particle + ecuación de inducción SPH | ✅ |
| **124** | Presión magnética + tensor Maxwell en fuerzas SPH: `f_lorentz` | ✅ |
| **125** | Dedner div-B cleaning: `psi_div` advectado, decaimiento `ch/cp²` | ✅ |
| **126** | Integración MHD en engine + macro `maybe_mhd!` + validación onda Alfvén | ✅ |
| **127** | ICs magnetizadas + CFL magnético: `BFieldKind`, `alfven_dt`, `cfl_mhd` | ✅ |
| **128** | Validación MHD 3D Alfvén + Brio-Wu 1D: tests de onda y choque MHD | ✅ |
| **129** | Acoplamiento CR–B: difusión CR suprimida por β_plasma | ✅ |
| **130** | Polvo intersticial básico: `Particle.dust_fraction`, `apply_dust_growth` | ✅ |
| **131** | HDF5 campos MHD + SPH completos: `/Bfld`, `/DivB`, `/Psi`, `/ECr`, `/Dust` | ✅ |
| **132** | Benchmark MHD Criterion + CFL unificado: `bench mhd`, `cfl_mhd` | ✅ |
| **133** | MHD anisótropo: difusión térmica + CR paralela a B | ✅ |
| **134** | Cooling magnético: `apply_magnetic_cooling`, emisión sincrotrón ∝ B² γ_e² | ✅ |
| **135** | Resistividad numérica artificial: `apply_artificial_resistivity`, `alpha_b` adaptativo | ✅ |
| **136** | MHD cosmológico end-to-end: `lcdm_mhd_N64`, crecimiento de B de semilla | ✅ |
| **137** | Polvo + RT: absorción UV kappa_dust×D/G×ρ×h, `tau_dust` | ✅ |
| **138** | Freeze-out de B en ICM: `apply_flux_freeze`, B ∝ ρ^{2/3}, `beta_freeze` | ✅ |
| **139** | SRMHD — MHD especial-relativista: γ Lorentz, primitivización NR | ✅ |
| **140** | Turbulencia MHD: forzado Ornstein-Uhlenbeck, P_B(k) ∝ k^{-5/3} | ✅ |
| **141** | Tests de integración MHD avanzados (48 tests) | ✅ |
| **142** | Engine: RMHD + turbulencia en `maybe_mhd!`/`maybe_sph!` | ✅ |
| **143** | Benchmarks Criterion avanzados: turbulencia, flux-freeze, SRMHD | ✅ |
| **144** | Clippy cero warnings en todo el workspace: 15+ lints corregidos | ✅ |
| **145** | Reconexión magnética Sweet-Parker: `apply_magnetic_reconnection` | ✅ |
| **146** | Viscosidad Braginskii anisótropa: `apply_braginskii_viscosity`, tensor π_ij | ✅ |
| **147** | Corrida cosmológica MHD completa + P_B(k) end-to-end | ✅ |
| **148** | Jets AGN relativistas: `inject_relativistic_jet`, halos FoF masivos | ✅ |
| **149** | Plasma de dos fluidos T_e ≠ T_i: `apply_electron_ion_coupling` | ✅ |
| **150** | Reportes 142–149, CHANGELOG, roadmap, commit final | ✅ |
| **165** | 🟢🔴 Kernels CUDA/HIP N² reales; MHD 3D solenoidal con ∇·B < 1e-14 | ✅ |
| **166** | 🌊 **SPH Gadget-2**: entropía A=P/ρ^γ · limitador Balsara · viscosidad señal · test Sod + Evrard | ✅ |

---

## Reportes técnicos

Los reportes en [`docs/reports/`](reports/) documentan cada fase con contexto,
metodología, resultados y limitaciones.

### N-body + HPC

| Reporte | Tema |
|---------|------|
| [`phase3-gadget4-benchmark`](reports/2026-04-phase3-gadget4-benchmark.md) | Benchmark vs GADGET-4 |
| [`phase4-multipole-softening`](reports/2026-04-phase4-multipole-softening.md) | Suavizado de multipolos |
| [`phase5-energy-mac-consistency`](reports/2026-04-phase5-energy-mac-consistency.md) | Consistencia energía + MAC |
| [`phase6-higher-order-integrator`](reports/2026-04-phase6-higher-order-integrator.md) | Yoshida4 |
| [`phase7-aarseth-timestep`](reports/2026-04-phase7-aarseth-timestep.md) | Timestep adaptativo |
| [`phase8-hpc-scaling`](reports/2026-04-phase8-hpc-scaling.md) | Escalado HPC |
| [`phase9-hpc-local`](reports/2026-04-phase9-hpc-local.md) | SFC + halos locales |
| [`phase10-let-tree`](reports/2026-04-phase10-let-tree.md) | LetTree compacto |
| [`phase11-let-tree-parallel-validation`](reports/2026-04-phase11-let-tree-parallel-validation.md) | Validación LET paralelo |
| [`phase12-let-communication-reduction`](reports/2026-04-phase12-let-communication-reduction.md) | Reducción comm LET |
| [`phase13-hilbert-decomposition`](reports/2026-04-phase13-hilbert-decomposition.md) | Hilbert 3D SFC |
| [`phase14-soa-simd`](reports/2026-04-phase14-soa-simd.md) | SoA + SIMD |
| [`phase15-explicit-simd`](reports/2026-04-phase15-explicit-simd.md) | SIMD explícito |
| [`phase16-tiled-simd-multi-i`](reports/2026-04-phase16-tiled-simd-multi-i.md) | Tiling 4×N_i |

### Cosmología + PM + TreePM

| Reporte | Tema |
|---------|------|
| [`phase17a-cosmology-serial`](reports/2026-04-phase17a-cosmology-serial.md) | Cosmología ΛCDM serial |
| [`phase17b-cosmology-distributed`](reports/2026-04-phase17b-cosmology-distributed.md) | Cosmología MPI + SFC+LET |
| [`phase18-periodic-pm`](reports/2026-04-phase18-periodic-pm.md) | PM periódico con CIC + FFT |
| [`phase19-distributed-pm`](reports/2026-04-phase19-distributed-pm.md) | PM sin allgather |
| [`phase20-slab-distributed-pm`](reports/2026-04-phase20-slab-distributed-pm.md) | PM slab FFT distribuida |
| [`phase21-distributed-treepm`](reports/2026-04-phase21-distributed-treepm.md) | TreePM distribuido (first light) |
| [`phase22-treepm-3d-halo`](reports/2026-04-phase22-treepm-3d-halo.md) | Halos 3D para SR periódico |
| [`phase23-treepm-sr-3d-domain`](reports/2026-04-phase23-treepm-sr-3d-domain.md) | Dominio 3D con SFC |
| [`phase24-treepm-pm-scatter-gather`](reports/2026-04-phase24-treepm-pm-scatter-gather.md) | Scatter-gather PM |
| [`phase25-treepm-scatter-gather-mpi-validation`](reports/2026-04-phase25-treepm-scatter-gather-mpi-validation.md) | Validación MPI scatter-gather |

### ICs, ensembles y normalización de P(k)

| Reporte | Tema |
|---------|------|
| [`phase26-zeldovich-ics-validation`](reports/2026-04-phase26-zeldovich-ics-validation.md) | ICs Zel'dovich (1LPT) |
| [`phase27-transfer-sigma8-ics`](reports/2026-04-phase27-transfer-sigma8-ics.md) | Transfer EH + σ₈ |
| [`phase28-2lpt-ics`](reports/2026-04-phase28-2lpt-ics.md) | ICs 2LPT |
| [`phase29-1lpt-vs-2lpt-validation`](reports/2026-04-phase29-1lpt-vs-2lpt-validation.md) | 1LPT vs 2LPT |
| [`phase30-linear-reference-validation`](reports/2026-04-phase30-linear-reference-validation.md) | Referencia lineal ΛCDM |
| [`phase38-class-camb-minimal-validation`](reports/2026-04-phase38-class-camb-minimal-validation.md) | Validación externa vs CLASS |
| [`phase42-tree-short-range`](reports/2026-04-phase42-tree-short-range.md) | TreePM + softening absoluto `ε_phys` |

### HPC avanzado, GPU y análisis (Phases 56–60)

| Reporte | Tema |
|---------|------|
| [`phase56-hierarchical-let`](reports/2026-04-phase56-hierarchical-let.md) | Block timesteps jerárquicos |
| [`phase57-cuda-hip-pm`](reports/2026-04-phase57-cuda-hip-pm.md) | PM solver CUDA/HIP |
| [`phase58-nfw-concentration-xi`](reports/2026-04-phase58-nfw-concentration-xi.md) | c(M) desde N-body + ξ(r) |
| [`phase59-checkpoint-continuity`](reports/2026-04-phase59-checkpoint-continuity.md) | Checkpoint/restart bit-a-bit |
| [`phase60-adaptive-rebalance`](reports/2026-04-phase60-adaptive-rebalance.md) | Domain decomposition adaptativa |
| [`rapidas-analyze-vis`](reports/2026-04-rapidas-analyze-vis.md) | CLI `analyze` + PPM rendering |

### MHD + Física bariónica avanzada (Phases 123–150)

| Reporte | Tema |
|---------|------|
| [`phase123-mhd-crate`](reports/2026-04-phase123-mhd-crate.md) | Crate gadget-ng-mhd + inducción SPH |
| [`phase125-dedner-divb`](reports/2026-04-phase125-dedner-divb.md) | Cleaning div-B Dedner |
| [`phase128-mhd-validation`](reports/2026-04-phase128-mhd-validation.md) | Validación Alfvén 3D + Brio-Wu 1D |
| [`phase136-cosmo-mhd`](reports/2026-04-phase136-cosmo-mhd.md) | MHD cosmológico end-to-end |
| [`phase139-srmhd`](reports/2026-04-phase139-srmhd.md) | SRMHD: factor de Lorentz γ |
| [`phase140-turbulence`](reports/2026-04-phase140-turbulence.md) | Turbulencia O-U, P_B(k) ∝ k^{-5/3} |
| [`phase145-reconnection`](reports/2026-04-phase145-reconnection.md) | Reconexión Sweet-Parker |
| [`phase149-two-fluid`](reports/2026-04-phase149-two-fluid.md) | Plasma 2F: T_e ≠ T_i |

### Meta

| Reporte | Tema |
|---------|------|
| [`gadget-ng-treepm-evolution-paper`](reports/2026-04-gadget-ng-treepm-evolution-paper.md) | Paper-style sobre evolución TreePM |
| [`validation-phase`](reports/2026-04-validation-phase.md) | Protocolo general de validación |
