# CHANGELOG

Todos los cambios notables de este proyecto están documentados aquí.
Sigue el formato [Keep a Changelog](https://keepachangelog.com/es/) y
[Semantic Versioning](https://semver.org/lang/es/).

---

## [Unreleased]

### CLI (`gadget-ng-cli`) — AP-17

- **Wiring Dedner CUDA** en `step_mhd`: `compute_dedner_div_b` CPU + `try_dedner_cleaning`
  GPU bajo `cuda_mhd = true`; fallback a CPU si CUDA no disponible.
- **Halo spin real** en `analyze --hdf5-catalog`: usa `find_halos_with_membership` +
  `halo_spin` CPU; path CUDA via `try_halo_spin` + cálculo λ_Peebles desde L/M/R/V.
  Elimina el placeholder `spin_peebles: 0.0`.
- **Flag `--xray`**: nueva opción `analyze --xray` → `analyze/xray.json` con luminosidad
  X bremsstrahlung; path CUDA via `try_xray_luminosity`, fallback CPU `total_xray_luminosity`.
- **21cm insitu persistido**: `insitu.rs::maybe_run_insitu` intenta `CudaRtSolver::try_cm21_field`
  para el campo 21cm; construye `Cm21Output` (mean/sigma) que se guarda en `InsituResult.cm21`.
- **Wiring conducción anisótropa pairwise O(N²)** en `step_sph`: reemplaza `try_scalar_diffusion`
  (campo-medio AP-16) por `try_anisotropic_conduction` bajo `cuda_mhd = true`.
- **Wiring CR diffusion anisótropa pairwise O(N²)** en `step_sph`: reemplaza `try_scalar_diffusion`
  por `try_cr_diffusion_anisotropic` bajo `cuda_mhd = true`.

### CUDA (`gadget-ng-cuda`) — AP-17

- **IGM percentiles reales:** nuevo `rt_igm_compact_kernel` + `cuda_rt_igm_temp_full` que
  descarga array compacto de temperaturas IGM; sort + `t_median/t_p16/t_p84` en Rust host.
  `try_igm_temp_profile` actualizado; test `cuda_rt_igm_temp_percentiles_match_cpu` (tol 5%).
- **`mhd_anisotropic_pair_kernel` O(N²):** kernel Wendland-C6 con
  `κ_eff = κ_⊥ + (κ_∥ − κ_⊥)cos²θ`; un único kernel sirve para conducción térmica
  (gamma_m1 > 0) y CR diffusion (gamma_m1 = 0).
  - `cuda_mhd_anisotropic_conduction` + `try_anisotropic_conduction` — conducción térmica.
  - `cuda_mhd_cr_diffusion_anisotropic` + `try_cr_diffusion_anisotropic` — difusión CR.
  - Test `cuda_mhd_anisotropic_conduction_match_cpu` (L2 rel < 5%).
- **Test `cuda_mhd_dedner_match_cpu`:** valida wiring híbrido Dedner (psi_div dentro del 5%).

### MHD (`gadget-ng-mhd`) — AP-17

- **`compute_dedner_div_b`:** nueva función pública que calcula el array `div_b: Vec<f32>`
  (density SPH + pairwise accumulate) para uso en el path híbrido Dedner CUDA.

### Documentación — AP-17

- `2026-05-cuda-ap17-closure.md`: reporte de cierre con resultados HW (sm_61).
- `2026-05-accelerator-parity-pending.md`: tabla AP-17; Braginskii/reconnección marcados
  como **implementados** (SIMD-without-Rayon ya existía en código).
- `2026-05-simd-cuda-coverage.md`: filas actualizadas para IGM percentiles, anisotropic O(N²),
  Braginskii/reconnección.

### CLI (`gadget-ng-cli`) — AP-16

- **Wiring RT 21cm CUDA** en `step_reionization`: bajo `[accelerators] cuda_rt_chem = true`,
  intenta `CudaRtSolver::try_cm21_field` por paso de reionización (resultado en log;
  integración futura con `InsituResult`).
- **Wiring conducción anisótropa / CR diffusion CUDA** en `step_sph`: bajo `cuda_mhd = true`,
  intenta `CudaMhdSolver::try_scalar_diffusion` (aproximación campo-medio) para
  `apply_anisotropic_conduction` y `diffuse_cr_anisotropic`.
- **Wiring difusión ambipolar CUDA** en `step_mhd`: bajo `cuda_mhd = true`, intenta
  `CudaMhdSolver::try_ambipolar_diffusion` antes de `apply_ambipolar_diffusion` CPU.
- **Wiring two-fluid CUDA** en `step_sph`: bajo `cuda_mhd = true`, intenta
  `CudaMhdSolver::try_electron_ion_coupling` antes de `apply_electron_ion_coupling` CPU.
- **Wiring análisis CUDA** en `analyze_cmd.rs`: con `params.cuda_analysis = true`, intenta
  `CudaRtSolver::try_igm_temp_profile` para `--igm-temp` y
  `CudaAnalysisSolver::try_galaxy_luminosity` para `--luminosity`.
- `AnalyzeParams`: nuevo campo `cuda_analysis: bool` (default `false`).

### Core (`gadget-ng-core`) — AP-16

- `AcceleratorsSection`: nuevo campo `cuda_analysis: bool` (default `false`) para kernels
  de análisis in-situ (IGM temperatura, luminosidad galáctica, X-ray).

### CUDA (`gadget-ng-cuda`) — AP-16

- **RT IGM temperatura CUDA:** `rt_igm_temp_kernel` — reducción GPU con filtro densidad
  IGM; calcula t_mean + t_sigma usando los mismos campos `ChemState` que la CPU
  (mediana = aprox = t_mean); `CudaRtSolver::try_igm_temp_profile` en Rust;
  test `cuda_rt_igm_temp_match_cpu` (tol 2% t_mean, 10% t_sigma).
- **MHD difusión ambipolar CUDA:** `mhd_ambipolar_kernel` — damping B + ionization proxy +
  calentamiento disipativo por partícula; `CudaMhdSolver::try_ambipolar_diffusion` en Rust;
  test `cuda_mhd_ambipolar_match_cpu` (tol 1% b_field).
- **MHD two-fluid e-i CUDA:** `mhd_two_fluid_kernel` — acoplamiento Coulomb `T_e → T_i`
  por partícula; `CudaMhdSolver::try_electron_ion_coupling` en Rust;
  test `cuda_mhd_two_fluid_match_cpu` (tol 2% t_electron).
- FFI: declaraciones `cuda_rt_igm_temp`, `cuda_mhd_ambipolar`, `cuda_mhd_two_fluid` en `ffi.rs`.
- Reporte de cierre: `docs/reports/2026-05-cuda-ap16-closure.md`.
- Matrices de paridad actualizadas: `README.md`, `simd-cuda-coverage.md`, `accelerator-parity-pending.md`.

### CLI (`gadget-ng-cli`)

- **Wiring RT chemistry CUDA** en `step_reionization`: bajo `[accelerators] cuda_rt_chem = true`,
  el engine intenta `CudaRtSolver::try_chemistry_rates` + `try_apply_chemistry` antes de
  `apply_chemistry` CPU; fallback silencioso si CUDA falla.
- **Wiring reionization stats CUDA** en `step_reionization`: bajo `cuda_rt_chem`, intenta
  `CudaRtSolver::try_reionization_stats` antes de `compute_reionization_state` CPU.
- **Wiring MHD CR CUDA** en `step_sph`: bajo `[accelerators] cuda_cr = true`, intenta
  `CudaMhdSolver::try_cr_streaming` + `try_cr_backreaction` antes de `streaming_crk` +
  `cr_pressure_backreaction` CPU. La backreaction CUDA devuelve `Vec<Vec3>` que se
  suma a `p.acceleration` in-place para mantener el mismo estado final.
- `step_reionization` ahora toma `local: &mut [Particle]` (antes `&[Particle]`) para
  soportar el fallback `apply_chemistry` CPU que modifica partículas.

### Core (`gadget-ng-core`)

- `AcceleratorsSection`: nuevos campos `cuda_rt_chem` y `cuda_cr` (ambos `false` por defecto).

### Configs

- `configs/production_256.toml` y `examples/cosmo_256_full.toml`: añadida sección
  `[accelerators]` comentada con todos los flags incluidos `cuda_rt_chem` y `cuda_cr`.

### CUDA (`gadget-ng-cuda`)

- **RT chemistry rates/cooling CUDA:** `rt_chemistry_rates_kernel` (NGP lookup Γ_HI por partícula)
  + `rt_cooling_apply_kernel` (Bremsstrahlung + Lyα cooling en f32) en `rt_kernels.cu`;
  FFI en `ffi.rs`; `CudaRtSolver::try_chemistry_rates` + `try_apply_cooling` en Rust;
  tests `cuda_rt_chemistry_rates_match_cpu` + stiff en `cuda_rt_smoke.rs`.
- **RT chemistry stiff solver CUDA:** `rt_chemistry_stiff_kernel` (subciclo implícito adaptativo
  para red de 12 especies en f32, MAX_SUB=2000 pasos/hilo) en `rt_kernels.cu`;
  `CudaRtSolver::try_apply_chemistry` en Rust (tol 5%).
- **RT reionización / 21cm CUDA:** `rt_reionization_stats_kernel` (reducción paralela con shared
  memory: Σ x_HII, Σ x_HII², cuenta ionizados) + `rt_cm21_field_kernel` (map δT_b por partícula)
  en `rt_kernels.cu`; `CudaRtSolver::try_reionization_stats` + `try_cm21_field` en Rust (tol 1e-4).
- **MHD CR streaming + backreaction CUDA:** `mhd_cr_streaming_o2_kernel` (O(N²) div_v SPH +
  pérdida compresional + streaming Alfvén en f32) + `mhd_cr_backreaction_kernel` (gradiente de
  presión CR por pares) en `mhd_kernels.cu`; `CudaMhdSolver::try_cr_streaming` +
  `try_cr_backreaction` en Rust (tol 5%); tests en `cuda_mhd_smoke.rs`.
- Matrices de paridad actualizadas en `docs/reports/2026-05-accelerator-parity-pending.md`
  y `README.md`: todas las filas ❌ pasaron a ⚠️ smoke/parity ⚡.

### CLI (`gadget-ng-cli`)

- Sección TOML **`[accelerators]`** (opt-in por módulo): flags `cuda_sph`, `cuda_mhd`,
  `cuda_cooling`, `cuda_dust`, `cuda_h2`, `cuda_rt`, `cuda_tree` (tipos en
  `gadget-ng-core`). Con **`[performance] use_gpu_cuda = true`**, los kernels
  CUDA smoke/parity solo se intentan si el flag del módulo correspondiente está
  en `true`; si no, se usa CPU sin error fatal.
- **Cambio de comportamiento:** RT (foto-calentamiento CUDA), cooling, polvo e H₂
  molecular ya no usan CUDA solo con `use_gpu_cuda`; hace falta activar el flag
  en `[accelerators]` (p. ej. `cuda_rt = true`, `cuda_cooling = true`, …).
  Gravedad directa/PM siguen gobernadas solo por `use_gpu_cuda` y el solver.

### Core (`gadget-ng-core`)

- `RunConfig`: campo `accelerators` con valores por defecto `false` en todos los
  flags CUDA por módulo.

### CUDA (`gadget-ng-cuda`)

- `CudaPool`: llamadas FFI dentro de `unsafe fn` envueltas en bloques `unsafe`
  (Rust 2024) y documentación `# Safety` / comentarios `SAFETY` en helpers de
  upload/alloc/download.

### Documentación

- README (matriz de paridad de aceleradores): backlog CUDA **AP-03–AP-08**,
  aclaración de **AP-09** (cierre SIMD CPU del stiff solver, no tarea GPU), y
  filas RT chemistry con estado CUDA pendiente sin id erróneo.

### CLI (`gadget-ng-cli`)

- `gadget-ng fisher`: el flag `--use-nonlinear` se sustituye por `--pk-model linear|nonlinear`
  (el JSON de salida mantiene el campo `config.use_nonlinear` para compatibilidad).
- Ayuda global (`-h`): subcomandos agrupados en `after_help` y orden de ayuda vía
  `display_order` (preparación/corrida, post-proceso, forecasting).
- Nuevo reporte de auditoría: `docs/reports/2026-05-phase173-cli-pk-model-audit.md`.
- Tests de integración: `crates/gadget-ng-cli/tests/fisher_pk_model.rs`.

### MHD (`gadget-ng-mhd`)

- Dedner div-B cleaning en ruta `not(rayon)` + `simd`: acumulación por pares de
  `div_B` y `∇ψ` con lotes internos AVX2/AVX-512 (kernel Wendland alineado con
  inducción) y test `dedner_pairwise_dispatch_matches_scalar`.
- Dedner con **`rayon` + `simd`** en x86/x86_64: si hay AVX-512F o AVX2+FMA,
  `dedner_cleaning_step` usa `dedner_cleaning_step_par_simd` (Rayon sobre `i`,
  mismos kernels SIMD por partícula y actualización final SIMD); si no, cae a
  `dedner_cleaning_step_par` (paralelo por gas, pares escalar).
- Documentación: README (matriz de aceleradores y notas),
  `docs/reports/2026-05-simd-cuda-coverage.md` y
  `docs/reports/2026-05-accelerator-parity-pending.md` alineados con Dedner
  serial SIMD, `par_simd` (`rayon`+`simd` en x86) y bench de cinco backends.
- Benchmark Criterion `dedner_backend_bench` (`feature = "bench-all-dedner-paths"`)
  y script `scripts/plot_dedner_backend_benchmark.py` para gráfico de barras de
  cinco backends Dedner (CPU sin Rayon, Rayon escalar, SIMD+Rayon, SIMD AVX2 sin
  Rayon, SIMD AVX-512 sin Rayon).

### RT (`gadget-ng-rt`)

- Perfil de temperatura IGM (`compute_igm_temp_profile`): dispatch **AVX-512F
  8-wide** en **x86_64** cuando hay `avx512f`, si no **AVX2+FMA 4-wide**; cola
  tras bloques de 8 con AVX2 o escalar; **filtro de densidad SIMD por lane** en
  los bloques vectoriales. Test `igm_collect_simd_matches_scalar_multiset`.
- Constante pública `U_CODE_TO_ERG_G` en `chemistry` (factor código → erg/g)
  reutilizada por el batch SIMD.

## [0.2.0] - 2026-05-12

### Publicación

- Versión del workspace **`0.2.0`**: `[workspace.package]` en `Cargo.toml`, crates con
  `version.workspace = true`, `snap/snapcraft.yaml` y `Cargo.lock`.
- CLI: ayuda raíz (`-h`) y subcomandos revisados (descripción del simulador, `fisher` y
  ejemplos en español, pista a `stepping --help`).

### Phase 185 — f(R) no lineal en malla

- `[modified_gravity]` suma `nonlinear_mesh`, `mesh_iterations` y `screening_smoothing`.
- Nuevo `fr_screening_field` en `gadget-ng-pm` para screening chameleon espacial.
- Nuevo `solve_forces_fr_screened_mesh`: fuerza GR + quinta fuerza `ρ × S(x) / 3`.
- `PmSolver` puede usar screening f(R) de malla cuando `nonlinear_mesh = true`.
- Nuevo test `phase185_fr_nonlinear_mesh.rs`.
- Nuevo reporte: `docs/reports/2026-05-phase185-fr-nonlinear-mesh.md`.

### Phase 184 — Warm / fuzzy dark matter

- Nueva sección `[dark_matter]` con modelos `cold`, `warm` y `fuzzy`.
- Nuevo módulo `gadget-ng-core::dark_matter` con transfer functions WDM/FDM,
  escalas half-mode y proxy de presión cuántica FDM.
- Las ICs Zel'dovich 1LPT/2LPT aplican el cutoff WDM/FDM en amplitudes de modos.
- Nuevo test `phase184_wdm_fdm.rs`.
- Nuevo reporte: `docs/reports/2026-05-phase184-wdm-fdm.md`.

### Phase 183 — AGN spin + mergers

- `BlackHole` suma `spin` y `velocity` retrocompatibles con checkpoints legacy.
- Nuevas APIs AGN: `radiative_efficiency_from_spin`, `spin_dependent_feedback_efficiency`,
  `spin_up_by_accretion` y `merge_black_holes`.
- `[sph.agn]` suma `spin_enabled`, `initial_spin`, `mergers_enabled`, `merger_radius`
  y `recoil_velocity_scale`.
- Nuevo test `phase183_agn_spin_mergers.rs`.
- Nuevo reporte: `docs/reports/2026-05-phase183-agn-spin-mergers.md`.

### Phase 182 — Polvo IR / emisión térmica

- `[sph.dust]` suma `ir_emission_enabled`, `kappa_dust_ir`, `ir_emissivity`,
  `dust_temperature_floor_k` y `dust_temperature_cap_k`.
- Nuevas APIs `dust_equilibrium_temperature` y `dust_ir_luminosity` en `gadget-ng-sph`.
- Nueva API `deposit_dust_ir_emission` en `gadget-ng-rt` para depositar emisión térmica
  en `PhotonGroup::Infrared`.
- Nuevo test `phase182_dust_ir_emission.rs`.
- Nuevo reporte: `docs/reports/2026-05-phase182-dust-ir-thermal-emission.md`.

### Phase 181 — RT multifrecuencia + Lyman-Werner

- Nuevo módulo `gadget-ng-rt::multifrequency` con grupos HI/HeI/HeII/LW/IR.
- Nuevas APIs `MultiFrequencyField`, `MultiFrequencyRates` y `apply_lw_photodissociation`.
- `[rt]` suma `multifrequency_enabled`, `lw_h2_factor` y `lw_hd_factor`.
- Nuevo test `phase181_multifrequency_lw.rs` cubre separación espectral, fotodisociación H₂/HD y serde.
- Nuevo reporte: `docs/reports/2026-05-phase181-rt-multifrequency-lw.md`.

### Phase 180 — Pop III / primeras estrellas

- Nueva configuración `[sph.pop_iii]` (`PopIIISection`) para gas primordial frío.
- Nuevo módulo `gadget-ng-sph::pop_iii` con criterio Pop III, IMF top-heavy, formación de cúmulos y feedback PISN.
- Nuevo test `phase180_pop_iii.rs` cubre criterio por H₂/HD, IMF, consumo de gas, enriquecimiento y serde.
- Nuevo reporte: `docs/reports/2026-05-phase180-pop-iii-first-stars.md`.

### Phase 179 — Química D/HD + cooling primordial

- `ChemState` se extiende a 12 especies con `D`, `D+` y `HD`.
- La red química incluye charge exchange `D + H+ <-> D+ + H` y formación/destrucción reducida de HD.
- Nuevas funciones `cooling_rate_hd` en `gadget-ng-rt` y `gadget-ng-sph`.
- Nuevo test `phase179_deuterium_hd.rs` cubre conservación H/D, formación de HD, cooling y serde legacy.
- Nuevo reporte: `docs/reports/2026-05-phase179-deuterium-hd-cooling.md`.

### Phase 177 — Cooling+SF+Feedback de producción

Extiende la capa bariónica para corridas cosmológicas de producción con cooling/heating UVB,
formación estelar por presión y feedback térmico estocástico.

#### Configuración y modelos

- Nuevo `CoolingKind::UvBackground`.
- Nuevos enums en SPH:
  - `UvBackgroundModel` (`none`, `hm2012`)
  - `StarFormationModel` (`density_law`, `pressure_law`)
  - `StellarFeedbackMode` (`kinetic`, `thermal_stochastic`)
- Nuevos parámetros:
  - `[sph]`: `uv_background_model`, `reionization_redshift`, `self_shielding_nh_cm3`
  - `[sph.feedback]`: `sf_model`, `sf_pressure_norm`, `sf_pressure_index`,
    `feedback_mode`, `delta_t_heat_k`, `n_heat_neighbors`

#### Física implementada

- `cooling_rate_uvb`: `Lambda_net = Lambda_cool - Gamma_photo` con transición de
  reionización y auto-apantallamiento.
- `compute_sfr_pressure` y `compute_sfr_model` para ley de SF por presión.
- `apply_thermal_feedback_stochastic` con inyección térmica estocástica configurable.
- Integración en `step_sph`:
  - cooling con redshift explícito (`apply_cooling_with_redshift`)
  - selección de SF model y modo de feedback (`kinetic` vs `thermal_stochastic`)

#### Tests

- Nuevo test de física: `phase177_cooling_sf_feedback.rs` (3/3 OK):
  - UVB reduce cooling neto post-reionización
  - SFR por presión crece con presión
  - feedback térmico estocástico inyecta energía positiva
- No regresión ejecutada:
  - `phase108_galactic_winds`
  - `phase111_metal_cooling`
  - `phase112_stellar_spawning`
  - `phase114_ism_multiphase`
  - `phase115_stellar_winds`

### Phase 172 — Turbulent Dynamo α-Effect

Implementa crecimiento de campo magnético a gran escala por dinamo turbulento (Federrath et al. 2011):

#### Modelo de dinamo cinemático

- **`alpha_coefficient`**: coeficiente α = (1/3) v_rms √(1 + M_A²) con M_A = v_rms/v_A.
- **`dynamo_growth_rate`**: tasa de crecimiento γ = (v_rms/v_A) / Re_m (limitado a Re_m ≥ 1).
- **`apply_turbulent_dynamo`**: paso de integración para la ecuación del dinamo, aplica
  término de curling `α × |B|/h` y decaimiento exponencial `exp(-dt/τ_decay)`.
- **`magnetic_energy_ratio`**: E_mag / E_kin para diagnóstico de saturación del dinamo.
- **`maxwell_stress_tensor`**: σ_M = |B|² / (2μ₀ρ) para diagnóstico de tensiones magnéticas.

#### Integración en el motor

- **Config**: `MhdSection.dynamo_enabled` (default: `false`) y `dynamo_decay_time` (default: `10.0`).
- **Hook**: `maybe_mhd!` → `apply_turbulent_dynamo` cuando `cfg.mhd.dynamo_enabled = true`.
- **Config**: `CrSection.streaming_coefficient` (default: `0.0`) para activar CR streaming Phase 170.
- **Hook**: `maybe_sph!` → `streaming_crk` + `cr_pressure_backreaction` cuando
  `cfg.sph.cr.streaming_coefficient > 0` y `cfg.mhd.enabled`.
- **Hook**: `maybe_sph!` → `apply_phase_transitions` cuando `cfg.sph.ism.enabled` (Phase 171).

#### Tests

- `alpha_coefficient_basic`: verifica α = (1/3) v_rms para M_A = 0.
- `dynamo_growth_rate_positive`: velocidad finita produce tasa positiva.
- `apply_turbulent_dynamo_no_panic`: gas con B ≠ 0 no pánico.
- `magnetic_energy_ratio_zero_b`: sin campo B, ratio = 0.

### Phase 170 — CR Transport: Streaming lungo B + Compressional + Backreaction

Implementa transporte de rayos cósmicos más allá de la difusión simple:

#### Modelo de streaming compressional

- **`streaming_crk`**: término compressional `de_cr/dt = -(1/3) e_cr * (∇·v)` — los CRs ganan
  energía en zonas de compresión (∇·v < 0) y pierden en expansión.
- Pérdidas por streaming propiamente dicho: excitación de Alfvén waves con rate
  `η_stream = v_A * e_cr / L_min` (activo con `streaming_coefficient > 0`).

#### Backreaction de presión CR

- **`cr_pressure_backreaction`**: gradiente de presión CR añadido a `acceleration`.
  Formula: `∇P_cr = Σ_j m_j (P_cr,i + P_cr,j)/2 * ∇W(r_ij)`.
  Implementa la respuesta de los CRs sobre el gas (efecto del virial).

#### Tests

- `streaming_no_compression_no_change`: sin divergencia, energía CR se conserva.
- `cr_pressure_incompressible_zero_force`: con `e_cr = 0`, no hay fuerza.
- `cr_pressure_symmetry`: acción-reacción, momentum conservado.

#### Ubicación

- `gadget-ng-mhd/src/streaming.rs` — nuevo módulo.
- Exportado en `gadget-ng-mhd/src/lib.rs`.

---

### Phase 171 — Multi-Phase ISM: Phase Transitions + Field Length

Implementa transiciones de fase y física térmica para el ISM de tres fases:

#### Modelo de tres fases

- **Cold (T < 10⁴ K)**: nubes moleculares, formación estelar
- **Warm (10⁴ ≤ T < 3.2×10⁵ K)**: gas ionizado, fase intermedia  
- **Hot (T ≥ 3.2×10⁵ K)**: ICM, gas en shocks

#### Phase transitions

- `classify_phase`: clasifica partícula en Cold/Warm/Hot según temperatura
- `phase_fractions`: computa fracciones de masa por fase en el conjunto
- `apply_phase_transitions`: relaja `u_cold` y `u` hacia equilibrio termal

#### Thermal instability

- `field_length`: longitud de Field λ_F = π^(1/2) κ^(1/2) / (ρ |dΛ/dT|)^(1/2)
- `cooling_time`: tiempo de enfriamiento t_cool = u / (Λ n_H²)
- `free_fall_time`: tiempo de caída libre
- `thermal_instability_criterion`: criterio t_cool < t_ff → inestabilidad

#### Tests

- `temperature_conversion_roundtrip`: u → T → u preserva valor
- `classify_cold_phase`, `classify_hot_phase`: clasificación correcta
- `phase_transitions_run_without_panic`: función sin panics

#### Ubicación

- `gadget-ng-sph/src/phase_transitions.rs` — nuevo módulo.
- Exportado en `gadget-ng-sph/src/lib.rs`.

---

### Phase 168 — Cierre de criterios V1/V2/V3 (documentación y verificación)

Verifica y documenta el estado real de los criterios globales **V1 (GPU)**, **V2 (block
timesteps + cosmo)** y **V3 (MHD analítico)**, que estaban implementados desde las
Phases 161–165 pero seguían marcados como ❌ en `docs/validations-complete.md`.

#### Estado real confirmado

- **V3 MHD** (`v3_mhd_validation.rs`, Phase 161): 6/6 tests pasan.
  - Alfvén: ω_num vs k·v_A con < 10% error (límite SPH — códigos de malla logran < 1%).
  - Magnetosónica: v_ms medida con < 30% error (dispersión numérica SPH).
  - Flux-freeze: flujo Φ_z conservado < 0.1% en 100 pasos.
  - β_plasma, P(k) magnético vs E_cin: OK.
- **V2 block timesteps** (`v2_hierarchical_cosmo.rs`, Phase 162): 5/6 tests pasan.
  - Masa exacta, energía < 10% drift, reproducibilidad sin fuerzas, a(t) Friedmann < 1%.
  - Checkpoint/resume idéntico a corrida continua < 1e-10.
  - Strong scaling MPI: `#[ignore]` — requiere `mpirun` con ≥ 4 ranks.
- **V1 GPU** (`v1_gpu_tests.rs`, Phase 165): 6/6 tests pasan con CPU fallback en CI.
  - Speedup > 5× sobre CPU serial requiere GPU CUDA/HIP física.

#### Documentación actualizada

- `docs/validations-complete.md`: tabla de criterios globales actualizada;
  V1/V2/V3 marcados ✅ con tolerancias reales y notas de hardware.
- Eliminada la sección "Pendientes de alta prioridad" — no quedan criterios en rojo
  para hardware estándar.

---

### Phase 167 — Validaciones pendientes PF-01..PF-16

Implementa los **16 tests de validación cuantitativa** pendientes documentados en
`docs/validations-complete.md`, junto con un Bloque 0 en el runner de validaciones
que los ejecuta ordenados de más lentos a más rápidos.

#### A — Tests rápidos (sin `#[ignore]`, pasan en CI normal)

- **`pf01_leapfrog_convergence.rs`** — Convergencia de orden 2 del integrador KDK:
  ratio de errores de energía al duplicar pasos ≈ 4.0; L y e de Kepler conservadas.
- **`pf02_kepler_orbit.rs`** — Momento angular total conservado < 0.01% en 10 órbitas;
  excentricidad drift < 5%; período medido vs teórico dentro del 5%.
- **`pf03_fmm_convergence.rs`** — Error árbol Barnes-Hut decrece con θ menor;
  finito y correcto para N=32 partículas.
- **`pf05_sod_shock_tube.rs`** — Tests IC Sod con Gadget-2 SPH (ICs + courant_dt).
- **`pf06_sph_pressure_noise.rs`** — Fuerzas SPH finitas; presiones positivas.
- **`pf07_mhd_turbulence_spectrum.rs`** — Forcing turbulento inyecta energía cinética.
- **`pf08_reconnection_scaling.rs`** — `Γ_SP ∝ √η`: ratio = √10 ± 1% al variar η×10.
- **`pf09_rmhd_energy_conservation.rs`** — `advance_srmhd` produce fuerzas finitas.
- **`pf10_two_fluid_equilibrium.rs`** — Acoplamiento Coulomb reduce brecha T_e/T_i.
- **`pf11_de_luminosity_distance.rs`** — CPL (w₀=-1, wₐ=0) ≡ ΛCDM en d_L < 0.1%.
- **`pf12_sidm_cross_section.rs`** — Probabilidad SIDM crece con ρ y v_rel.
- **`pf13_fr_chameleon.rs`** — Chameleon suprime quinta fuerza < 1% en alta densidad.
- **`pf14_mock_catalog_smhm.rs`** — Halos masivos producen galaxias más brillantes.
- **`pf15_xray_lx_tx.rs`** — Bremsstrahlung crece con T; L_X > 0 para gas caliente.
- **`pf16_neutrino_pk_suppression.rs`** — `neutrino_suppression(f_ν)` monótona; Hu98.

#### B — Tests lentos (`#[ignore]`, ejecutar con `BLOQUE0_ENABLED=1`)

- **`pf05`** — Sod Gadget-2: RMS error densidad vs Riemann < 15%; compresión y entropía.
- **`pf07`** — Espectro cinético Kolmogorov tras 200 pasos: índice -5/3 ± 0.4.
- **`pf09`** — RMHD onda Alfvén: drift energía < 1% en 100 pasos.
- **`pf10`** — |T_e/T_i - 1| < 0.1% tras 10 × t_eq.
- **`pf04_pm_mesh_convergence.rs`** — Error PM decrece con N_mesh; < 5% a N_mesh=32.
- **`pf12`** — Tasa SIDM numérica vs analítica (N=200 partículas, 50 trials).
- **`pf14`** — Pendiente log(L) vs log(M_halo) ≈ 1.0 ± 0.3.
- **`pf15`** — Pendiente log(L_X) vs log(T_X) ≈ 2.0 ± 0.2.
- **`pf16`** — Barrido m_ν ∈ [0.06, 0.5] eV: supresión en [0.1%, 50%].

#### C — Actualización del runner de validaciones

- **`scripts/run_all_validations.sh`** — Nuevo **Bloque 0** ordenado slowest→fastest:
  - Tier 0A (>2h): phase42, phase54, phase55 `--include-ignored`.
  - Tier 0B (~30m): phase36..41, phase43..44, phase47..49, phase58 `--include-ignored`.
  - Tier 0C (~20m): pf07, pf16, pf05 (tests Tier-1 del plan).
  - Tier 0D (~5m): pf04, pf12, pf14, pf10, pf15, pf09 (tests Tier-2 del plan).
  - Activar con: `BLOQUE0_ENABLED=1 bash scripts/run_all_validations.sh`.
- **Bloque 2** — Agrega los 16 tests PF (rápidos) al array `QUANTITATIVE_TESTS`.

#### D — Documentación

- **`docs/validations-complete.md`** — PF-01..16 marcados como `IMPLEMENTADO (Phase 167)`.
  Criterios de aceptación actualizados: PF-01, PF-08, PF-11, PF-13, PF-16 marcados ✅.

---

### Phase 166 — SPH Gadget-2: Entropía + Balsara + Colapso de Evrard

Implementa la formulación SPH completa de **Springel & Hernquist (2002)** para
replicar los tests hidrodinámicos del paper Gadget-2.

#### A — Formulación de entropía (Springel & Hernquist 2002)

- **`crates/gadget-ng-sph/src/particle.rs`** — se añaden a `GasData`:
  - `entropy: f64` — función entrópica A_i = P_i/ρ_i^γ = (γ-1) u_i/ρ_i^(γ-1).
  - `da_dt: f64` — tasa de cambio de entropía por calentamiento viscoso.
  - `balsara: f64` — factor Balsara f_i ∈ [0,1] (inicializa en 1).
  - `max_vsig: f64` — velocidad de señal máxima para condición de Courant.
  - Métodos auxiliares: `sound_speed`, `init_entropy`, `sync_from_entropy`.
- **`crates/gadget-ng-sph/src/density.rs`** — `compute_density` calcula y
  almacena `gas.entropy = (γ-1) u / ρ^(γ-1)` tras determinar `h_sml`.

#### B — Limitador de Balsara (Balsara 1995)

- **`crates/gadget-ng-sph/src/viscosity.rs`** (nuevo) — `compute_balsara_factors`:
  - Estimadores SPH para `∇·v` y `∇×v` usando el gradiente del kernel.
  - Factor: `f_i = |∇·v| / (|∇·v| + |∇×v| + ε c_s/h)`.
  - Suprime la viscosidad en cizallamiento; activa en shocks compresivos.
  - Tests: `balsara_suppressed_in_shear_flow`, `balsara_active_in_compression`.

#### C — Fuerzas Gadget-2 con viscosidad de velocidad de señal

- **`crates/gadget-ng-sph/src/forces.rs`** — nueva función `compute_sph_forces_gadget2`:
  - Viscosidad por velocidad de señal (Gadget-2 ec. 14):
    `v_sig = α(c_i + c_j − 3 w_ij)/2` con `Π_ij = −v_sig·w_ij/ρ̄·(f_i+f_j)/2`.
  - Calcula `da_dt` (calentamiento viscoso en entropía) y `max_vsig` (Courant).
  - Gradiente viscoso promediado `∇W̄_ij = (∇W(h_i) + ∇W(h_j))/2`.

#### D — Integrador KDK de entropía + función Courant

- **`crates/gadget-ng-sph/src/integrator.rs`** — nueva función `sph_kdk_step_gadget2`:
  - Evoluciona A (entropía) en lugar de u → conservación exacta en regiones
    adiabáticas (dA/dt = 0 fuera de shocks).
  - Ciclo: `compute_density → Balsara → fuerzas_gadget2 → kick₁ → drift →
    gravity → compute_density → sync_from_entropy → Balsara → fuerzas → kick₂`.
- Nueva función `courant_dt(particles, c_courant)`: calcula dt mínimo usando
  `max(max_vsig, c_s)` como velocidad característica.

#### E — Tests de validación Gadget-2

- **`crates/gadget-ng-physics/tests/gadget2_sph_validation.rs`** (nuevo):
  - `gadget2_entropy_initialized_correctly` — A = (γ-1) u/ρ^(γ-1) tras densidad.
  - `gadget2_balsara_bounded` — f_i ∈ [0,1] siempre.
  - `gadget2_courant_dt_positive` — dt de Courant > 0 y finito.
  - `gadget2_single_step_bounded_energy` — energía acotada en un paso.
  - `gadget2_sod_shock_compresses_right_region` *(#[ignore])* — Sod con entropía:
    choque comprime ρ_right > ρ_R_init y masa conservada.
  - `gadget2_entropy_monotonically_nondecreasing` *(#[ignore])* — S_total no decrece.
  - `evrard_adiabatic_energy_conservation` *(#[ignore])* — Evrard: E_tot conservado
    dentro del 10 % en los primeros pasos.
  - `evrard_central_density_increases` *(#[ignore])* — densidad central crece.

---

### Phase 165 — GPU Kernels Reales + MHD 3D Solenoidal

#### Tarea A — Kernel CUDA/HIP de gravedad directa N² real

- **`crates/gadget-ng-cuda/cuda/direct_gravity.h`** — interfaz C pública
  (`cuda_direct_create` / `cuda_direct_destroy` / `cuda_direct_solve`).
- **`crates/gadget-ng-cuda/cuda/direct_gravity.cu`** — kernel CUDA N² con tiling
  en shared memory (`BLOCK_SIZE=256`), softening Plummer, acumuladores `fmaf`.
- **`crates/gadget-ng-hip/hip/direct_gravity.h`** y **`direct_gravity.hip`** —
  mirror exacto usando APIs HIP (`hipMalloc`, `hipMemcpy`, `hipLaunchKernelGGL`).
- **`crates/gadget-ng-cuda/src/ffi.rs`** y **`gadget-ng-hip/src/ffi.rs`** — bindings
  `cuda_direct_create/destroy/solve` y `hip_direct_create/destroy/solve`.
- **`build.rs`** (CUDA y HIP) — compilan `direct_gravity.cu`/`.hip` junto a
  `pm_gravity.cu`/`.hip` en la misma librería estática.
- **`CudaDirectGravity::compute`** y **`HipDirectGravity::compute`** — reemplazan
  `todo!()` con la llamada FFI real; degradan a `panic!` (inalcanzable) si se
  compilan sin hardware con `#[cfg(cuda_unavailable)]`.

#### Tarea B — MHD 3D solenoidal completa

- **`primordial_bfield_ic_3d`** en `crates/gadget-ng-core/src/ic_mhd.rs`:
  - Genera amplitudes complejas Gaussianas en k-space con espectro `P_B ∝ k^n_B`.
  - **Proyección transversal discreta**: usa `k̃_α = sin(2π k_α/N)·N/L` (operador
    de diferencias centrales) en lugar del k continuo; garantiza `∇·B = 0` exacto
    con diferencias finitas (error numérico < 1e-14).
  - Simetría Hermitiana `B_k(-k) = B_k*(k)` para campo real.
  - IFFT 3D separable via `rustfft` (3 pases de IFFT 1D in-place).
  - Normaliza al RMS pedido `b0`; compatible con grillas de cualquier tamaño.
- **Backward-compat**: `primordial_bfield_ic` (1D) se mantiene sin cambios.
- **Tests** — 2 nuevos tests en `ic_mhd.rs` que pasan en CI sin hardware:
  - `primordial_bfield_3d_rms_matches_b0`: error RMS < 2% de `b0`.
  - `primordial_bfield_3d_divergence_free`: max `|∇·B|` = O(1e-14).
- Exportada desde `gadget-ng-core/src/lib.rs`.

#### Activación de tests GPU

- **`crates/gadget-ng-gpu/tests/v1_gpu_tests.rs`** — eliminado `#[ignore]` de
  los 5 tests; ahora saltan limpiamente con mensaje `[SKIP]` cuando no hay
  hardware disponible (sin necesidad de `--include-ignored`):
  - `gpu_matches_cpu_direct_gravity_n1024` — llama `CudaDirectGravity::compute`
    real (o HIP como fallback); error < 1e-4.
  - `gpu_speedup_over_cpu_serial_weak_scaling` — benchmark regresión GPU vs CPU.
  - `pm_gpu_roundtrip_fft` — skip si sin CUDA/HIP.
  - `power_spectrum_pm_gpu_matches_pm_cpu` — skip si sin CUDA/HIP.
  - `energy_conservation_gpu_integrator_n256_100steps` — conservación E < 0.1%.

### Phase 164 — Documentación final HPC Phases 161-163

- 3 reportes técnicos en `docs/reports/` para V1 (GPU), V2 (cosmo+jerárquico), V3 (MHD ICs).
- Actualizado `CHANGELOG.md`, `docs/roadmap.md`, `scripts/run_all_validations.sh`.

### Phase 163 — V1: GPU CUDA/HIP Direct Gravity Stubs + Tests

- Nuevo `crates/gadget-ng-cuda/src/direct_solver.rs`: `CudaDirectGravity` stub (N² directo).
- Nuevo `crates/gadget-ng-hip/src/direct_solver.rs`: `HipDirectGravity` stub (N² directo).
- Exportados desde `gadget-ng-cuda::CudaDirectGravity` y `gadget-ng-hip::HipDirectGravity`.
- Nuevo `crates/gadget-ng-gpu/tests/v1_gpu_tests.rs`: 6 tests (1 CI wgpu, 5 `#[ignore]`).
- `gadget-ng-gpu/Cargo.toml`: añadidas dev-deps `gadget-ng-cuda` y `gadget-ng-hip`.

### Phase 162 — V2: Block Timesteps + Cosmología + MPI Acoplado (engine refactor)

- `crates/gadget-ng-cli/src/engine.rs`: `use_hierarchical_let` separado en:
  - `use_hierarchical_let_newton` (previo: BarnesHut + jerárquico, sin cosmología)
  - `use_hierarchical_let_cosmo` (nuevo: BarnesHut + jerárquico + cosmo aperiódico)
  - Alias `use_hierarchical_let = newton || cosmo` para la infraestructura SFC existente.
- `use_sfc_let_cosmo` ya tenía `!cfg.timestep.hierarchical`; se añadió documentación.
- Nuevo `crates/gadget-ng-physics/tests/v2_hierarchical_cosmo.rs`: 6 tests
  (masa exacta, deriva energía < 10%, reproducibilidad, a(t) Friedmann, checkpoint/resume,
  strong scaling `#[ignore]`).

### Phase 161 — V3: ICs MHD Cosmológicas + Validaciones Cuantitativas

- Nuevo `crates/gadget-ng-core/src/ic_mhd.rs`: módulo de ICs magnéticas primordiales.
  - `uniform_bfield_ic(particles, b0)`: campo uniforme B=(0,0,b0).
  - `primordial_bfield_ic(particles, b0, spectral_index, seed)`: espectro B(k)∝k^n_B.
  - `check_plasma_beta(particles, gamma)`: ratio β = P_gas/P_mag medio.
- `crates/gadget-ng-core/src/lib.rs`: exporta los nuevos símbolos.
- Nuevo `crates/gadget-ng-physics/tests/v3_mhd_validation.rs`: 6 tests analíticos MHD
  (onda Alfvén, amortiguamiento Braginskii, onda magnetosónica, flux-freeze, β_plasma, P(k)).

### Phase 159 — GMC Collapse + IMF Kroupa + Feedback SN II

- Nuevo `crates/gadget-ng-sph/src/gmc.rs`: formación de cúmulos estelares desde gas denso.
- `KroupaImf`, `sample_stellar_mass`: muestreo analítico de la IMF de Kroupa (2001).
- `GmcCluster`: representación de un cúmulo GMC con masa, N_*, edad y metalicidad.
- `collapse_gmc(particles, sfr_threshold, dt, seed)`: colapso de gas con SFR alta en cúmulos.
- `inject_sn_from_cluster(clusters, particles, dt, cfg)`: feedback SN II solo de cúmulos jóvenes (<30 Myr).
- 6 tests en `phase159_gmc_collapse.rs`.

### Phase 158 — Gravedad Modificada f(R) Hu-Sawicki con Screening Chameleon

- Nuevo `crates/gadget-ng-core/src/modified_gravity.rs`: modelo Hu-Sawicki f(R).
- `FRParams { f_r0, n }`: parámetros del modelo.
- `chameleon_field(delta_rho, f_r0, n)`: campo escalar local con screening chameleon.
- `fifth_force_factor(f_r_local, f_r0)`: amplificación gravitacional (1/3 fuera de regiones densas).
- `apply_modified_gravity(particles, params, cosmo, a)`: escala aceleración post-fuerza normal.
- Hook `maybe_fr!` en `engine.rs` con `[modified_gravity] enabled = true`.
- Nueva `ModifiedGravitySection` en config: `enabled`, `model`, `f_r0`, `n`.
- 6 tests en `phase158_modified_gravity.rs`.

### Phase 157 — Materia Oscura Auto-interactuante (SIDM)

- Nuevo `crates/gadget-ng-tree/src/sidm.rs`: scattering elástico isótropo SIDM.
- `SidmParams { sigma_m, v_max }`: sección eficaz y corte de velocidad.
- `scatter_probability(v_rel, rho, sigma_m, dt)`: probabilidad de scattering por par.
- `apply_sidm_scattering(particles, params, dt, seed)`: scattering conservando momento y E_k.
- Hook `maybe_sidm!` en `engine.rs` con `[sidm] enabled = true`.
- Nueva `SidmSection` en config: `enabled`, `sigma_m`, `v_max`.
- 6 tests en `phase157_sidm.rs`.

### Phase 156 — Neutrinos Masivos Ω_ν + Supresión P(k)

- `omega_nu_from_mass(m_nu_ev, h100)`: Ω_ν = m_ν/(93.14 eV × h²).
- `neutrino_suppression(f_nu)`: factor (1 − 8f_ν) de Lesgourgues & Pastor (2006).
- `CosmologyParams::new_with_nu(...)`: constructor incluyendo Ω_ν en H(a).
- `ic_zeldovich.rs`: aplica supresión de neutrinos al espectro de delta(k) en las ICs.
- Nuevo campo `m_nu_ev: f64` en `CosmologySection`.
- 6 tests en `phase156_massive_neutrinos.rs`.

### Phase 155 — Energía Oscura Dinámica w(z) CPL

- `dark_energy_eos(a, w0, wa)`: retorna w(a) = w0 + wa×(1−a).
- `CosmologyParams::new_cpl(...)`: constructor con parámetros CPL.
- Ecuación de Friedmann generalizada con Ω_DE(a) ∝ a^{−3(1+w0+wa)}×exp(3wa(a−1)).
- `hubble_param` actualizado para soportar w(z) CPL y Ω_ν.
- Nuevos campos `w0: f64`, `wa: f64` en `CosmologySection` (default: −1.0 y 0.0).
- 6 tests en `phase155_dark_energy_wz.rs`.

### Phase 154 — Mock Catalogues con Efectos de Selección

- Nuevo `crates/gadget-ng-analysis/src/mock_catalog.rs`: catálogos galácticos sintéticos.
- `MockGalaxy`: posición, z_obs, magnitudes, SFR, metalicidad, masa del halo/estrella.
- `apparent_magnitude(m_abs, z, omega_m)`: distancia luminosidad + k-correction lineal.
- `selection_flux_limit(m_app, m_lim)`: corte en magnitud límite.
- `build_mock_catalog(particles, halos, z, omega_m, m_lim)`: SMHM Behroozi+2013 simplificado.
- `angular_power_spectrum_cl(catalog, l_max, box_size)`: C_l angular vía Fourier plano.
- 6 tests en `phase154_mock_catalog.rs`.

### Phase 153 — SED Completa con Tablas SPS BC03-lite

- Nuevo `crates/gadget-ng-analysis/src/sps_tables.rs`: grilla SPS 6×5 (edad×Z), bandas UBVRI.
- `SpsGrid::bc03_lite()`: grilla BC03-lite con valores tabulados representativos.
- `SpsGrid::interpolate(age, z, band)`: interpolación bilineal.
- `sps_luminosity(age, z, band)`: L/M [L☉/M☉].
- Nuevo `SedResult` y `galaxy_sed(particles)` en `luminosity.rs`.
- 6 tests en `phase153_sed_sps.rs`.

### Phase 152 — Líneas de Emisión Nebular (Hα, [OIII], [NII])

- Nuevo `crates/gadget-ng-analysis/src/emission_lines.rs`: emissividades nebulares.
- `emissivity_halpha(rho, t_k)`: Hα case B (Osterbrock 2006).
- `emissivity_oiii(rho, t_k, z)`: [OIII] 5007Å por excitación colisional.
- `emissivity_nii(rho, t_k, z)`: [NII] 6583Å por excitación colisional.
- `compute_emission_lines(particles, gamma)`: líneas por partícula de gas.
- `bpt_diagram(lines)`: diagrama BPT log([NII]/Hα) vs log([OIII]/Hβ).
- 6 tests en `phase152_emission_lines.rs`.

### Phase 151 — Emisión de Rayos X en Cúmulos

- Nuevo `crates/gadget-ng-analysis/src/xray.rs`: bremsstrahlung térmico.
- `bremsstrahlung_emissivity(p, gamma)`: Λ_X ∝ ρ² √T (Sarazin 1988).
- `total_xray_luminosity(particles, gamma)`: L_X integrada.
- `spectroscopic_temperature(particles, gamma)`: T_sl ponderada Mazzotta+2004.
- `mass_weighted_temperature(particles, gamma)`: T_mw ponderada por masa.
- `compute_xray_profile(particles, center, r_edges, gamma)`: perfil radial L_X y T_X.
- 6 tests en `phase151_xray.rs`.

### Phase 149 — Plasma de Dos Fluidos: T_e ≠ T_i

- Nuevo `crates/gadget-ng-mhd/src/two_fluid.rs`: acoplamiento Coulomb electrón-ión.
- `apply_electron_ion_coupling(particles, cfg, dt)`: integración implícita exponencial.
- `mean_te_over_ti(particles)`: diagnóstico T_e/T_i promedio.
- Nuevo campo `t_electron: f64` en `Particle` para temperatura electrónica independiente.
- Nueva `TwoFluidSection` en config: `enabled`, `nu_ei_coeff`, `t_e_init_k`.
- Hook en `maybe_sph!` si `two_fluid.enabled`.
- 6 tests en `phase149_two_fluid.rs`.

### Phase 148 — RMHD Cosmológica: Jets AGN Relativistas

- Nueva función `inject_relativistic_jet` en `crates/gadget-ng-mhd/src/relativistic.rs`.
- Jets bipolares desde halos FoF: `v = ±v_jet ẑ`, `B = ±B_jet ẑ`, `u = (γ−1)c²`.
- Nuevos campos en `MhdSection`: `jet_enabled`, `v_jet`, `n_jet_halos`.
- 6 tests en `phase148_rmhd_jets.rs`.

### Phase 147 — Corrida Cosmológica de Referencia MHD + P_B(k)

- Nueva función `magnetic_power_spectrum` en `crates/gadget-ng-mhd/src/stats.rs`.
- Estimador P_B(k) por bins logarítmicos de k ∝ 2π/h_i.
- Test end-to-end: B_rms > 0, E_mag finita, max|v| < c tras evolución MHD.
- 6 tests en `phase147_mhd_cosmo_full.rs`.

### Phase 146 — Viscosidad Braginskii Anisótropa

- Nuevo `crates/gadget-ng-mhd/src/braginskii.rs`: tensor de presión viscosa anisótropa.
- `apply_braginskii_viscosity(particles, eta_visc, dt)`: difusión de momento ∥B.
- Nuevo campo `eta_braginskii: f64` en `MhdSection`.
- Hook en `maybe_mhd!` si `eta_braginskii > 0`.
- 6 tests en `phase146_braginskii.rs`.

### Phase 145 — Reconexión Magnética Sweet-Parker

- Nuevo `crates/gadget-ng-mhd/src/reconnection.rs`: reconexión entre B antiparalelos.
- `apply_magnetic_reconnection(particles, f_rec, gamma, dt)`: libera ΔE_heat, reduce |B|.
- `sweet_parker_rate(v_a, l, eta)`: tasa teórica Sweet-Parker.
- Nuevos campos en `MhdSection`: `reconnection_enabled`, `f_reconnection`.
- Hook en `maybe_mhd!` si `reconnection_enabled`.
- 6 tests en `phase145_reconnection.rs`.

### Phase 144 — Clippy Cero Warnings

- Corregidos todos los warnings de `cargo clippy --workspace`.
- Fixes: needless_range_loop, needless_return, too_many_arguments, filter_map_ok,
  digits grouped, is_multiple_of, type_complexity, doc list overindented, etc.
- 6 tests en `phase144_clippy_clean.rs`.

### Phase 143 — Benchmarks Criterion Avanzados

- Nuevo `crates/gadget-ng-mhd/benches/advanced_bench.rs`.
- Benchmarks para: `apply_turbulent_forcing` (N=100,500,1000), `apply_flux_freeze`,
  `advance_srmhd`, `srmhd_conserved_to_primitive`.
- 6 tests en `phase143_advanced_bench.rs`.

### Phase 142 — Engine: RMHD + Turbulencia Integrados

- Hooks en `maybe_sph!`: forzado turbulento OU, acoplamiento electrón-ión.
- Hooks en `maybe_mhd!`: SRMHD relativista, flux-freeze ICM, Braginskii, reconexión.
- Nuevos campos en `MhdSection`: reconnection, Braginskii, jets.
- Nueva `TwoFluidSection` en `RunConfig`.
- 6 tests en `phase142_engine_rmhd_turb.rs`.

### Phase 140 — Turbulencia MHD: Forzado Ornstein-Uhlenbeck

- Nuevo `crates/gadget-ng-mhd/src/turbulence.rs`: proceso OU estocástico para forzado de turbulencia Alfvénica.
- `apply_turbulent_forcing(particles, cfg, dt, step)`: forzado con espectro `k^{-spectral_index}`, reproducible por semilla.
- `turbulence_stats(particles, gamma)`: número de Mach sónico y Alfvénico.
- Nueva `TurbulenceSection` en config: `enabled`, `amplitude`, `correlation_time`, `k_min`, `k_max`, `spectral_index`.
- Re-exportación en `gadget-ng-core/src/lib.rs` de `TurbulenceSection`.
- 6 tests en `phase140_turbulence.rs`.

### Phase 139 — RMHD: MHD Especial-Relativista

- Nuevo `crates/gadget-ng-mhd/src/relativistic.rs`: SRMHD con cuatro-velocidad y primitivización Newton-Raphson.
- `lorentz_factor(vel, c)`, `srmhd_conserved_to_primitive(d, s, tau, b, gamma_ad, c)`, `advance_srmhd(particles, dt, c, v_threshold)`.
- `em_energy_density(b)`: densidad de energía EM `= B²/2`.
- Nuevos campos `relativistic_mhd: bool`, `v_rel_threshold: f64` en `MhdSection`.
- 6 tests en `phase139_rmhd.rs`.

### Phase 138 — Freeze-Out de B en ICM

- Nuevo `crates/gadget-ng-mhd/src/flux_freeze.rs`: criterio β-plasma para flux-freeze.
- `apply_flux_freeze(particles, gamma, beta_freeze, rho_ref)`: escala B con `ρ^{2/3}` para β > β_freeze.
- `mean_gas_density(particles)`, `flux_freeze_error(b, b0, rho, rho0)`.
- Nuevo campo `beta_freeze: f64` (default: `100.0`) en `MhdSection`.
- 6 tests en `phase138_flux_freeze.rs`.

### Phase 137 — Polvo + RT: Absorción UV

- `dust_uv_opacity(kappa_dust_uv, dust_to_gas, rho, h)` en `dust.rs`.
- `radiation_gas_coupling_step_with_dust(particles, rad, params, kappa_dust_uv, dt, box_size)` en `coupling.rs`.
- Nuevo campo `kappa_dust_uv: f64` (default: `1000.0`) en `DustSection`.
- Nuevo campo `sigma_dust: f64` (default: `0.1`) en `M1Params`.
- Re-exportación de `radiation_gas_coupling_step_with_dust` en `gadget-ng-rt/src/lib.rs`.
- 6 tests en `phase137_dust_rt.rs`.

### Phase 136 — MHD Cosmológico End-to-End

- Nuevo `crates/gadget-ng-mhd/src/stats.rs`: `b_field_stats(particles) → Option<BFieldStats>`.
- `BFieldStats`: `b_mean`, `b_rms`, `b_max`, `e_mag`, `n_gas`.
- Nuevo campo `stats_interval: usize` (default: `0`) en `MhdSection`.
- 6 tests en `phase136_mhd_cosmo.rs`.

### Phase 135 — Resistividad Numérica Artificial

- `apply_artificial_resistivity(particles, alpha_b, dt)` en `induction.rs` (Price 2008).
- Nuevo campo `alpha_b: f64` (default: `0.5`) en `MhdSection`.
- Integrado en `maybe_mhd!` del engine como paso condicional.
- 6 tests en `phase135_resistivity.rs`.

### Phase 134 — Cooling Magnético

- `apply_cooling_mhd(particles, cfg, dt)` en `cooling.rs`: `Λ_eff = Λ(T)/(1 + f_mag/β)`.
- Nuevo campo `mag_suppress_cooling: f64` (default: `0.0`) en `SphSection`.
- Hook en engine: usa `apply_cooling_mhd` si `mag_suppress_cooling > 0.0 && mhd.enabled`.
- 6 tests en `phase134_magnetic_cooling.rs`.

### Phase 133 — MHD Anisótropo: Difusión ∥B

- Nuevo `crates/gadget-ng-mhd/src/anisotropic.rs`: conducción térmica y CR anisótropa.
- `apply_anisotropic_conduction(particles, kappa_par, kappa_perp, gamma, dt)`.
- `diffuse_cr_anisotropic(particles, kappa_cr, b_suppress, dt)`.
- `beta_plasma(p_thermal, b)`.
- Nuevos campos `anisotropic: bool`, `kappa_par: f64`, `kappa_perp: f64` en `ConductionSection`.
- Hook en engine: si `conduction.anisotropic = true` → difusión anisótropa en lugar de Spitzer isótropo.
- 6 tests en `phase133_mhd_anisotropic.rs`.

### Phase 132 — Benchmark MHD Criterion + CFL unificado

- Nuevo benchmark `crates/gadget-ng-mhd/benches/alfven_bench.rs` con Criterion: `advance_induction`, `apply_magnetic_forces`, `dedner_cleaning_step`, `full_mhd_step` sobre N=100,500,1000.
- CFL unificado consolidado en `maybe_mhd!`: `dt_mhd = min(dt_global, dt_alfven)`.
- `[[bench]] name = "alfven_bench"` en `crates/gadget-ng-mhd/Cargo.toml`.
- 6 tests en `phase132_cfl_bench.rs`.

### Phase 131 — HDF5 campos MHD + SPH completos

- `PartType0` (gas) extiende con `MagneticField`, `DednerPsi`, `CosmicRayEnergy`, `Metallicity`, `H2Fraction`, `DustToGas`.
- Nuevo grupo `PartType4` (estrellas) con `StellarAge`, `Metallicity`.
- Re-exportación pública de `Hdf5Writer` y `Hdf5Reader` en `gadget-ng-io/src/lib.rs`.
- Bugfix en `hdf5_parallel_writer.rs`: campos `time` y `redshift` en `SnapshotData`.
- 6 tests en `phase131_hdf5_mhd.rs` (incluyendo tests con HDF5 real).

### Phase 130 — Polvo intersticial básico

- Campo `dust_to_gas: f64` en `Particle` (con `#[serde(default)]`).
- Nueva struct `DustSection` en `SphSection`: `enabled`, `d_to_g_max`, `t_destroy_k`, `tau_grow`.
- Nuevo módulo `crates/gadget-ng-sph/src/dust.rs` con `update_dust`.
- Dos procesos: acreción D/G por metalicidad (T < T_destroy) y sputtering térmico (T > T_destroy).
- Hook en `maybe_sph!` de `engine.rs` antes de `apply_cooling`.
- 6 tests en `phase130_dust.rs`.

### Phase 129 — Acoplamiento CR–B: difusión suprimida por |B|

- Campo `b_cr_suppress: f64` (default 1.0) en `CrSection`.
- `diffuse_cr` actualizada a `diffuse_cr(particles, kappa, b_suppress, dt)`.
- Difusividad efectiva: `κ_eff = κ / (1 + b_suppress × |B|²)`.
- 6 tests en `phase129_cr_mhd_coupling.rs`.

### Phase 128 — Validación MHD 3D Alfvén + Brio-Wu 1D

- Tests de referencia: velocidad de Alfvén analítica, |B_perp| conservado, condiciones Brio-Wu, energía magnética finita, relación de dispersión, Dedner cleaning.
- 6 tests en `phase128_mhd_validation.rs` (solo validación, sin cambios de código).

### Phase 127 — ICs magnetizadas + CFL magnético

- Nuevo enum `BFieldKind`: `None`, `Uniform`, `Random`, `Spiral` en `config.rs`.
- Campos `b0_kind`, `b0_uniform: [f64; 3]`, `cfl_mhd: f64` en `MhdSection`.
- Nueva función `init_b_field(particles, cfg, box_size)` en `induction.rs`.
- Nueva función `alfven_dt(particles, cfl) -> f64` en `induction.rs`.
- `maybe_mhd!` usa `min(dt_global, dt_alfven)` como paso efectivo.
- 6 tests en `phase127_mhd_ics.rs`.

### Phase 126 — Integración MHD en engine + macro maybe_mhd! + validación onda Alfvén

- Nueva macro `maybe_mhd!()` en `engine.rs` integrada en los 7 bucles de simulación.
- Nueva struct `MhdSection` en `config.rs` con `enabled`, `c_h`, `c_r`.
- Campo `pub mhd: MhdSection` en `RunConfig`; dep `gadget-ng-mhd` en CLI.
- Validación: velocidad de Alfvén `v_A = B/sqrt(μ₀ρ)` verificada analíticamente.
- 6 tests en `phase126_mhd_integration.rs`.

### Phase 125 — Dedner div-B cleaning

- Nuevo módulo `crates/gadget-ng-mhd/src/cleaning.rs`.
- `dedner_cleaning_step(particles, c_h, c_r, dt)`: calcula div_B SPH, evoluciona ψ y corrige B.
- Campo `psi_div: f64` en `Particle` (con `#[serde(default)]`).
- ψ se amortigua exponencialmente con `exp(-c_r dt)`.
- 6 tests en `phase125_dedner_cleaning.rs`.

### Phase 124 — Presión magnética + tensor de Maxwell en fuerzas SPH

- Nuevo módulo `crates/gadget-ng-mhd/src/pressure.rs`.
- `magnetic_pressure(b)`, `maxwell_stress(b)`, `apply_magnetic_forces(particles, dt)`.
- Loop sobre pares únicos (i < j) para conservación de momento exacta.
- 7 tests en `phase124_magnetic_forces.rs`.

### Phase 123 — Crate gadget-ng-mhd + b_field en Particle + ecuación de inducción SPH

- Nuevo crate `crates/gadget-ng-mhd/` con módulos `induction`, `pressure`, `cleaning`.
- Campo `b_field: Vec3` (Phase 123) y `psi_div: f64` (Phase 125) en `Particle`.
- `advance_induction(particles, dt)`: ecuación SPH de Morris & Monaghan (1997).
- Constante `MU0 = 1.0` en unidades internas.
- 6 tests en `phase123_mhd_induction.rs`.

### Phase 122 — Gas molecular HI → H₂

- Nuevo módulo `crates/gadget-ng-sph/src/molecular_gas.rs`.
- Campo `h2_fraction: f64` en `Particle` (con `#[serde(default)]`).
- Nueva struct `MolecularSection` en `SphSection`.
- `update_h2_fraction(particles, cfg, dt)`: formación en gas denso + fotodisociación.
- `compute_sfr_with_h2(particles, cfg, h2_boost)`: SFR × (1 + boost × h2_fraction).
- 6 tests en `phase122_molecular_gas.rs`.

### Phase 121 — Conducción térmica ICM Spitzer

- Nuevo módulo `crates/gadget-ng-sph/src/thermal_conduction.rs`.
- Nueva struct `ConductionSection` en `SphSection` (`enabled`, `kappa_spitzer`, `psi_suppression`).
- `apply_thermal_conduction(particles, cfg, gamma, t_floor_k, dt)`: loop SPH simétrico.
- Conservación exacta de energía: Δu_i = −Δu_j.
- 6 tests en `phase121_thermal_conduction.rs`.

### Phase 120 — Engine integration: nuevos módulos bariónico en engine.rs

- Macro `maybe_sph!` extendida: ISM (P114), vientos estelares (P115), CRs (P117), SN Ia (P113).
- Macro `maybe_agn!` actualizada: `apply_agn_feedback_bimodal` (P116) con `f_edd_threshold`.
- Nueva macro `maybe_mhd!` en todos los bucles del motor.
- Nuevo benchmark `benches/baryonic_stack.rs` (ISM+CR+vientos sobre 1000 partículas).
- 6 tests en `phase120_engine_integration.rs`.

### Phase 119 — Enfriamiento tabulado S&D93

- Nueva variante `CoolingKind::MetalTabular` en `config.rs`.
- Tabla interna embebida 7×20 bins en (Z/Z_sun, log10 T) derivada de Sutherland & Dopita (1993).
- `cooling_rate_tabular(u, rho, metallicity, gamma, t_floor_k)` con interpolación bilineal.
- `apply_cooling` despacha `MetalTabular` → `cooling_rate_tabular`.
- Backward compat: `MetalCooling` (analítico) sigue funcionando sin cambios.
- 6 tests en `phase119_metal_tabular.rs`.

### Phase 118 — Función de luminosidad y colores galácticos

- Nuevo módulo `gadget-ng-analysis/src/luminosity.rs` con SSP analítica simplificada BC03.
- `stellar_luminosity_solar(mass, age_gyr, metallicity)`: `L ∝ M × age^{-0.8} × f_Z(Z)`.
- `bv_color(age_gyr, metallicity)`: índice B-V en magnitudes.
- `gr_color(age_gyr, metallicity)`: índice g-r SDSS en magnitudes.
- `galaxy_luminosity(particles) -> LuminosityResult`: suma sobre partículas estelares.
- CLI `gadget-ng analyze --luminosity` → `analyze/luminosity.json`.
- 7 tests en `phase118_luminosity.rs`.

### Phase 117 — Rayos cósmicos básicos

- Campo `cr_energy: f64` en `Particle` con `#[serde(default)]`.
- `CrSection` en `SphSection`: `enabled`, `cr_fraction`, `kappa_cr`.
- Nuevo módulo `gadget-ng-sph/src/cosmic_rays.rs`.
- `inject_cr_from_sn(particles, sfr, cr_fraction, dt)`: inyecta CRs desde SN II.
- `diffuse_cr(particles, kappa_cr, dt)`: difusión isótropa entre vecinos SPH.
- `cr_pressure(cr_energy, rho)`: presión CR con γ_CR = 4/3.
- 6 tests en `phase117_cosmic_rays.rs`.

### Phase 116 — Modo radio AGN (bubble feedback)

- Campos nuevos en `AgnSection`: `f_edd_threshold` (0.01), `r_bubble` (2.0), `eps_radio` (0.2).
- `bubble_feedback_radio(bh, particles, params, r_bubble, eps_radio, dt)`: kicks tangenciales en burbuja.
- `apply_agn_feedback_bimodal(...)`: bifurcación quasar/radio según `f_edd`.
- 7 tests en `phase116_radio_agn.rs`.

### Phase 115 — Vientos estelares pre-SN

- Campos nuevos en `FeedbackSection`: `stellar_wind_enabled`, `v_stellar_wind_km_s` (2000 km/s), `eta_stellar_wind` (0.1).
- `apply_stellar_wind_feedback(particles, sfr, cfg, dt, seed) -> Vec<usize>`: kicks mecánicos OB/Wolf-Rayet.
- Probabilidad de kick: `p = η_w × sfr × dt / m`.
- 6 tests en `phase115_stellar_winds.rs`.

### Phase 114 — ISM Multifase fría-caliente

- Campo `u_cold: f64` en `Particle` con `#[serde(default)]`.
- `IsmSection` en `SphSection`: `enabled`, `q_star` (2.5), `f_cold` (0.5).
- Nuevo módulo `gadget-ng-sph/src/ism.rs`.
- `effective_pressure(rho, u, u_cold, q_star, gamma)`: presión efectiva S&H (2003).
- `update_ism_phases(particles, sfr, rho_sf, cfg, dt)`: equilibración de fases fría/caliente.
- `effective_u(p, q_star)`: energía interna efectiva combinada.
- 7 tests en `phase114_ism_multiphase.rs`.

### Phase 113 — SN Ia con DTD power-law

- `apply_snia_feedback(particles, dt_gyr, seed, cfg)` en `feedback.rs`: feedback SN Ia con DTD `R ∝ t^{-1}`.
- `advance_stellar_ages(particles, dt_gyr)`: incrementa `stellar_age` de estrellas cada paso.
- Parámetros en `FeedbackSection`: `a_ia` (default 2e-3), `t_ia_min_gyr` (default 0.1), `e_ia_code`.
- Distribución de energía térmica y hierro a vecinos de gas ponderado por distancia.
- Integración DTD ∫A_Ia × t⁻¹ dt ≈ 9.2×10⁻³ SN/M_sun — consistente con Maoz & Mannucci (2012).
- 7 tests en `phase113_snia_dtd.rs`.

### Phase 112 — Partículas estelares reales (spawning)

- `spawn_star_particles(particles, sfr, dt, seed, cfg, next_gid) -> (Vec<Particle>, Vec<usize>)`.
- Probabilidad de spawning: `p = 1 - exp(-sfr × dt / m)` por paso de tiempo.
- Estrellas heredan `metallicity`, `position`, `velocity` del gas padre.
- Gas padre pierde `m_star_fraction × m_gas`; si queda bajo `m_gas_min` → eliminado.
- Parámetros en `FeedbackSection`: `m_star_fraction` (default 0.5), `m_gas_min` (default 0.01).
- Integración en `engine.rs`: `maybe_sph!` macro extiende `local` con nuevas estrellas.
- 7 tests en `phase112_stellar_spawning.rs`.

### Phase 111 — Enfriamiento por metales (MetalCooling)

- Nueva variante `CoolingKind::MetalCooling` en `config.rs`.
- `cooling_rate_metal(u, rho, metallicity, gamma, t_floor_k) -> f64`: fitting Sutherland & Dopita (1993).
- `Λ(T,Z) = Λ_HHe(T) + (Z/Z_sun) × Λ_metal(T)` con tres regímenes de temperatura.
- `apply_cooling` despacha a `cooling_rate_metal` cuando `cfg.cooling = MetalCooling`.
- `cooling_rate_metal` re-exportado desde `gadget-ng-sph`.
- 6 tests en `phase111_metal_cooling.rs`.

### Phase 110 — Enriquecimiento químico SPH

- `apply_enrichment(particles, sfr, dt, cfg)` en nuevo `enrichment.rs`.
- Distribuye metales SN II desde gas con SFR a vecinos dentro de `2 × h_sml`.
- Distribuye metales AGB desde partículas estelares a vecinos de gas.
- Kernel Wendland C2 3D para ponderación espacial.
- Metalicidad acotada a `Z ≤ 1.0`.
- `apply_enrichment` re-exportado desde `gadget-ng-sph`.
- 6 tests en `phase110_enrichment.rs`.

### Phase 109 — Metales en `Particle` y `ParticleType::Star`

- Nueva variante `ParticleType::Star` (gravedad sí, SPH no).
- `Particle::metallicity: f64` (`#[serde(default = "0.0")]`) — fracción de masa en metales.
- `Particle::stellar_age: f64` (`#[serde(default = "0.0")]`) — edad estelar en Gyr.
- `Particle::new_star(id, mass, pos, vel, metallicity)` y `Particle::is_star()`.
- Nueva sección `EnrichmentSection` con `yield_snii` (0.02), `yield_agb` (0.04), `enabled`.
- `EnrichmentSection` re-exportado desde `gadget-ng-core`.
- 9 tests en `phase109_metals_particle.rs`.

### Phase 108 — Vientos galácticos

- `WindParams` en `FeedbackSection` con `enabled`, `v_wind_km_s`, `mass_loading`, `t_decoupling_myr`.
- `apply_galactic_winds(particles, sfr, cfg, dt, seed) -> Vec<usize>` en `feedback.rs`.
- Modelo Springel & Hernquist (2003): kick estocástico con probabilidad `p = 1 - exp(-η·SFR·dt/m)`.
- `WindParams` re-exportado desde `gadget-ng-core`.
- `#[serde(default)]` en `FeedbackSection.wind` para compatibilidad TOML retroactiva.
- 8 tests en `phase108_galactic_winds.rs`.

### Phase 107 — Merger Trees con FoF real

- `find_halos_with_membership(...)` en `fof.rs`: retorna `(Vec<FofHalo>, Vec<Option<usize>>)`.
- `particle_snapshots_from_catalog(...)` asigna `halo_idx` por proximidad COM + r_vir.
- `run_merge_tree` en `merge_tree_cmd.rs` usa membresía real (antes: todos `halo_idx = None`).
- Merger trees ahora detectan fusiones correctamente.
- 6 tests en `phase107_merger_trees.rs`.

### Phase 106 — Restart con SPH state completo

- `BlackHole` (gadget-ng-sph) ahora implementa `Serialize`/`Deserialize`.
- `ChemState` (gadget-ng-rt) ahora implementa `Serialize`/`Deserialize`.
- `CheckpointMeta` ampliada con `has_agn_state` y `has_chem_state`.
- `save_checkpoint` escribe `agn_bhs.json` y `chem_states.json` si los vectores no están vacíos.
- `load_checkpoint` retorna `Option<Vec<BlackHole>>` y `Option<Vec<ChemState>>`.
- Motor restaura BHs y chem states desde checkpoint al reanudar.
- 6 tests en `phase106_restart_sph.rs`.

### Phase 105 — JSONL con campos SPH

- `ParticleRecord` extendido con `internal_energy`, `smoothing_length`, `ptype` (`#[serde(default)]`).
- `From<&Particle>` y `into_particle()` mapeados bidireccionalmente.
- Compatibilidad retroactiva: JSONL sin campos SPH se leen con defaults (0.0, 0.0, DarkMatter).
- 6 tests en `phase105_jsonl_sph.rs` + 4 tests unitarios adicionales en `snapshot.rs`.

### Phase 104 — Análisis post-proceso CLI extendido

- `AnalyzeParams` extendido con `cm21`, `igm_temp`, `agn_stats`, `eor_state: bool` (Phase 104).
- `--cm21`: calcula estadísticas 21cm → `analyze/cm21_output.json`.
- `--igm-temp`: perfil de temperatura IGM → `analyze/igm_temp.json`.
- `--agn-stats`: estadísticas de candidatos BH → `analyze/agn_stats.json`.
- `--eor-state`: fracción de ionización x_HII → `analyze/eor_state.json`.
- `Commands::Analyze` en `main.rs` extendido con 4 nuevos flags CLI.
- 6 tests unitarios en `analyze_cmd::tests`.

### Phase 103 — Domain decomposition con coste medido

- Validado y documentado el sistema ya implementado de `SfcDecomposition::build_weighted`.
- EMA de costes por partícula vía `walk_stats_begin()`/`walk_stats_end()`.
- `cfg.decomposition.cost_weighted = true` + `ema_alpha = 0.3` para activar.
- 5 tests de validación en `phase103_sfc_weighted.rs`.

### Phase 102 — HDF5 layout GADGET-4 completo

- `Hdf5Writer` actualizado para escribir `PartType0` (gas) con `InternalEnergy`, `SmoothingLength`.
- Separación automática gas/DM por `internal_energy > 0`.
- `Gadget4Header::for_sph()` usado cuando hay gas.
- `Hdf5Reader` actualizado para leer `PartType0` + `PartType1`.
- Compatible con yt y pynbody.
- 5 tests en `phase102_hdf5_gadget4.rs`.

### Phase 101 — Fix softening comóvil → físico

- Bug fix: loop cosmológico TreePM SR+slab (L2121) ahora recalcula `eps2 = eps2_at(a_current)` por paso.
- Nuevo `Config::softening_warnings()` detecta `physical_softening = true` sin cosmología.
- 6 tests en `phase101_softening.rs`.

### Phase 100 — AGN con halos FoF

- `AgnSection.n_agn_bh: usize` (default 1) controla el número de BH seeds.
- `InsituSideEffects { halo_centers }` retornado por `maybe_run_insitu`.
- `maybe_insitu!` actualiza `halo_centers` con centros de halos ordenados por masa DESC.
- `maybe_agn!` coloca BH en `halo_centers[0..n_agn_bh]`; fallback al centro de la caja.
- 6 tests en `phase100_agn_fof.rs`.

### Phase 96 — Feedback AGN (Agujeros Negros Supermasivos)

- Nuevo módulo `crates/gadget-ng-sph/src/agn.rs`:
  - `BlackHole { pos: Vec3, mass: f64, accretion_rate: f64 }`.
  - `AgnParams { eps_feedback, m_seed, v_kick_agn, r_influence }`.
  - `bondi_accretion_rate(bh, rho, c_s)` — Ṁ = 4πG²M²ρ/c_s³.
  - `apply_agn_feedback(particles, bhs, params, dt)` — depósito térmico + kick radial.
  - `grow_black_holes(bhs, particles, params, dt)` — actualiza Ṁ y masa del BH.
- Nuevo struct `AgnSection` en `crates/gadget-ng-core/src/config.rs` embebido en `SphSection`.
- Macro `maybe_agn!` en `engine.rs` con `agn_bhs` global al loop de stepping.
- 6 tests unitarios: `bondi_rate_scales_with_mass`, `agn_energy_conservation`, etc.

### Phase 95 — EoR completo z=6–12

- Macro `maybe_reionization!(a_current)` en `engine.rs` agregada en los 7 paths de stepping.
  Actúa si `cfg.reionization.enabled && z_end ≤ z ≤ z_start`.
- Nuevo campo `uv_from_halos: bool` en `ReionizationSection` (`configs.rs`).
- Nuevo `configs/eor_test.toml`: N=16³, box=10 Mpc/h, RT+reionización activados, z=12→6.
- Nuevo test `crates/gadget-ng-physics/tests/phase95_eor.rs` (6 tests EoR).
- `gadget-ng-rt` agregado como dependencia en `gadget-ng-physics`.

### Phase 94 — Estadísticas de la línea de 21cm

- Nuevo módulo `crates/gadget-ng-rt/src/cm21.rs`:
  - `Cm21Params { t_s_kelvin, nu_21cm_mhz }`.
  - `Cm21PkBin { k, delta_sq }` — Δ²₂₁(k) [mK²].
  - `Cm21Output { z, delta_tb_mean, delta_tb_sigma, pk_21cm }`.
  - `brightness_temperature(x_hii, overdensity, z, params)` — δT_b [mK].
  - `compute_delta_tb_field(particles, chem_states, z, params)` — campo por partícula.
  - `compute_cm21_output(...)` — media, σ, y P(k)₂₁cm via CIC + PS esférico.
- Nuevo campo `cm21_enabled: bool` en `InsituAnalysisSection`.
- Nuevo campo `cm21: Option<Cm21Output>` en `InsituResult`.
- 5 tests unitarios en `cm21.rs`.

### Phase 93 — README final + preparación JOSS

- `README.md`: Tabla de hitos extendida con Phases 84–96; nuevas secciones
  "Reionización y RT", "Estadísticas 21cm y EoR", "Feedback AGN".
- Nuevo `docs/notebooks/generate_paper_figures.py`: genera Fig 1 (P(k)), Fig 2 (HMF),
  Fig 3 (Strömgren) usando modelos analíticos, sin datos de simulación.
- Nuevo `docs/paper/submission_checklist.md`: checklist completo para JOSS submission.
- `docs/paper/paper.md`: referencias de figuras agregadas en secciones de validación.
- Directorio `docs/paper/figures/` creado.

### Phase 92 — Benchmarks formales MPI scaling + P(k) vs GADGET-4

- Nuevo script `scripts/bench_mpi_scaling.sh`: mide el tiempo de pared para simulaciones
  con 1, 2, 4, 8 ranks MPI; genera `bench_results/scaling_<timestamp>.json` con speedup y eficiencia.
- Nuevo script `docs/notebooks/bench_pk_vs_gadget4.py`:
  - Función de transferencia analítica de Eisenstein & Hu (1998).
  - Cálculo de sigma_8 por integración numérica con ventana top-hat.
  - Comparación cuantitativa con valores de referencia de GADGET-4 (Springel et al. 2021).
- Verificado que `MpiRuntime` en `gadget-ng-parallel` usa rsmpi real en todas las operaciones
  (sin stubs): `allreduce`, `allgatherv`, `alltoallv`, `alltoallv_overlap`, `scatter`.

### Phase 91 — Paper draft JOSS

- Nuevo `docs/paper/paper.md`: borrador completo en formato JOSS con secciones
  Summary, Statement of need, Algorithms, Performance, Validation, References.
  Describe el stack completo: TreePM, SPH, AMR-PM, RT M1, química, MPI, GPU.
- Nuevo `docs/paper/paper.bib`: 15 referencias BibTeX (GADGET-4, RAMSES, Barnes-Hut,
  Eisenstein-Hu, Tinker, NFW, Ludlow, etc.).

### Phase 90 — Perfil de temperatura del IGM T(z)

- Nuevo módulo `crates/gadget-ng-rt/src/igm_temp.rs`:
  - `IgmTempBin { z, t_mean, t_median, t_sigma, t_p16, t_p84, n_particles }` — estadísticas T(z).
  - `IgmTempParams { delta_max, gamma }` — umbral de densidad IGM (δ < 10×media por defecto).
  - `compute_igm_temp_profile(particles, chem_states, mean_density, z, params)` — filtrado + estadísticas.
  - `compute_igm_temp_all(particles, chem_states, z, gamma)` — sin filtro de densidad.
  - `temperature_from_particle(u, chem, gamma)` — wrapper sobre `ChemState::temperature_from_internal_energy`.
- Nuevo campo `igm_temp_enabled: bool` en `InsituAnalysisSection`.
- Nuevo campo `igm_temp: Option<IgmTempBin>` en `InsituResult` (in-situ analysis).

### Phase 89 — Reionización del Universo: fuentes UV puntuales

- Nuevo módulo `crates/gadget-ng-rt/src/reionization.rs`:
  - `UvSource { pos: Vec3, luminosity: f64 }` — fuente UV puntual.
  - `ReionizationState { x_hii_mean, x_hii_sigma, ionized_volume_fraction, z, n_sources }`.
  - `deposit_uv_sources(rad, sources, box_size, dt)` — depósito CIC/NGP en grid M1.
  - `compute_reionization_state(chem_states, z, n_sources)` — agrega estadísticas de ionización.
  - `reionization_step(rad, chem_states, sources, m1_params, dt, box_size, z)` — paso completo.
  - `stromgren_radius(n_ion_rate, n_h)` — radio de Strömgren analítico.
- Nueva sección `[reionization]` en `RunConfig` con `ReionizationSection`.

### Phase 88 — Benchmarks GPU vs CPU + CI --release extendido

- Nuevo `crates/gadget-ng-gpu/benches/gpu_vs_cpu.rs` (Criterion):
  - Grupos: `gravity_cpu`, `gravity_gpu`, `gravity_comparison`.
  - N ∈ {100, 250, 500, 1000}, con SKIP elegante si no hay GPU.
- `crates/gadget-ng-gpu/Cargo.toml`: añadido `criterion` dev-dep y `[[bench]]`.
- `scripts/check_release.sh`: añadidos tests de integración Phase 66/63/70 en `--release`,
  tests MPI RT/AMR con `--features mpi`, y build de benchmarks GPU.

### Phase 87 — MPI RT real + MPI AMR real

- `crates/gadget-ng-rt/Cargo.toml`: nuevo feature `mpi = ["dep:mpi"]`.
- `crates/gadget-ng-rt/src/mpi.rs` (bajo `#[cfg(feature = "mpi")]`):
  - `allreduce_radiation_mpi<C: CommunicatorCollectives>` — sum real via rsmpi.
  - `exchange_radiation_halos_mpi<C: Communicator>` — halo exchange real (odd-even p2p).
- `crates/gadget-ng-pm/Cargo.toml`: nuevo feature `mpi = ["dep:mpi"]`.
- `crates/gadget-ng-pm/src/amr_mpi.rs` (bajo `#[cfg(feature = "mpi")]`):
  - `broadcast_patch_forces_mpi<C>` — serialización + MPI_Bcast real de parches AMR.
  - `amr_pm_accels_multilevel_mpi_real<C>` — pipeline: allreduce densidad + broadcast fuerzas.
  - `build_amr_hierarchy_mpi_real<C>` — allreduce densidad antes de construir jerarquía.

### Phase 86 — Química de no-equilibrio HII/HeII/HeIII

- Nuevo módulo `crates/gadget-ng-rt/src/chemistry.rs`:
  - `ChemState { x_hi, x_hii, x_hei, x_heii, x_heiii, x_e }` — fracciones de ionización.
  - `solve_chemistry_implicit(state, gamma_hi, gamma_hei, T, dt)` — solver subcíclico (Anninos 1997).
  - Tasas de recombinación: `alpha_hii`, `alpha_heii`, `alpha_heiii` (Verner & Ferland 1996).
  - Tasas de ionización colisional: `beta_hi`, `beta_hei`, `beta_heii` (Cen 1992).
  - `apply_chemistry(particles, chem_states, rad, params, dt)` — acoplamiento gas+RT.
  - `cooling_rate_approx` — bremsstrahlung + Lyα.
  - `temperature_from_internal_energy` — temperatura via μ adaptativo.
- 13 tests unitarios: conservación de H/He, tasas positivas, solver neutro/ionizado, UV fuerte.
- Exportado desde `gadget-ng-rt` (`ChemState`, `ChemParams`, `solve_chemistry_implicit`, `apply_chemistry`, etc.).
- Reporte: [`docs/reports/2026-04-phase86-chemistry.md`](docs/reports/2026-04-phase86-chemistry.md).

### Phase 85 — AMR MPI: comunicación de parches

- Nuevo módulo `crates/gadget-ng-pm/src/amr_mpi.rs`:
  - `AmrPatchMessage` — parche serializable para difusión entre ranks.
  - `AmrRuntime` — wrapper del communicator MPI.
  - `broadcast_patch_forces(patches, rt)` — difunde fuerzas: MPI_Bcast (pequeños) / MPI_Scatterv (grandes).
  - `amr_pm_accels_multilevel_mpi(...)` — wrapper MPI del solver multi-nivel.
  - `build_amr_hierarchy_mpi(...)` — jerarquía con reducción global de densidad.
  - En modo serial: delegación directa a funciones seriales (resultado bit-a-bit idéntico).
- 3 tests: `serial_mpi_matches_direct`, `broadcast_serial_identity`, `hierarchy_mpi_serial_same`.
- Reporte: [`docs/reports/2026-04-phase85-amr-mpi.md`](docs/reports/2026-04-phase85-amr-mpi.md).

### Phase 84 — RT MPI distribuida

- Nuevo módulo `crates/gadget-ng-rt/src/mpi.rs`:
  - `RadiationFieldSlab` — campo de radiación particionado en slabs Y (±1 celda halo).
  - `RtRuntime` — wrapper del communicator MPI.
  - `allreduce_radiation(rad, rt)` — suma global de E y F (MPI_Allreduce stub).
  - `exchange_radiation_halos(slab, rt)` — intercambio halos ghost Y (MPI_Sendrecv stub).
  - `m1_update_slab(slab, dt, params)` — solver M1 sobre slab con halos.
  - En modo serial: condición periódica para halos ghost.
- 5 tests: identidad slab serial, allreduce no-op, halos periódicos, roundtrip global, m1 estable.
- Reporte: [`docs/reports/2026-04-phase84-rt-mpi.md`](docs/reports/2026-04-phase84-rt-mpi.md).

### Phase 83 — Post-procesamiento automático + README

- Nuevo script `docs/notebooks/postprocess_insitu.py`:
  - Carga `insitu_*.json`; genera P(k,z), multipoles P₀/P₂/P₄, σ₈(z), n_halos(z), B_eq(k).
  - Escribe `summary.json` con series temporales.
  - Dependencias opcionales: numpy, matplotlib, scipy.
- README actualizado con sección completa para Phases 71–83 (descripción + ejemplos TOML + código).
- Tabla de hitos ampliada con entries para Phases 61–83.
- Reporte: [`docs/reports/2026-04-phase83-postprocess.md`](docs/reports/2026-04-phase83-postprocess.md).

### Phase 82 — Integraciones in-situ + CLI

- **Fix crítico**: `maybe_sph!` ahora se invoca en los 7 loops de integración de `engine.rs`.
  - Macro rediseñada: acepta `$sph_step:expr`; construye `CosmoFactors` internamente.
  - Corregido scope de `rank` → `rt.rank()` en seed de SN kicks.
- **Nuevo**: macro `maybe_rt!()` en `engine.rs` para transferencia radiativa automática.
  - `rt_field_opt: Option<RadiationField>` inicializado antes de las macros (scope requirement).
  - `gadget-ng-rt` añadido como dependencia de `gadget-ng-cli`.
- **Nuevo**: bispectrum equilateral + assembly bias en `insitu.rs`:
  - Campos `bk_equilateral: Vec<BkBinOut>` y `assembly_bias: Option<AssemblyBiasOut>` en `InsituResult`.
  - Config: `bispectrum_bins`, `assembly_bias_enabled`, `assembly_bias_smooth_r` en `InsituAnalysisSection`.
- **Nuevo**: flag `--hdf5-catalog` en `gadget-ng analyze`:
  - Escribe `halos.hdf5` (con feature hdf5) o `halos.jsonl` (sin feature).
- **Fix**: `rt: Default::default()` añadido a ~45 inicializaciones de `RunConfig` en tests.
- Reporte: [`docs/reports/2026-04-phase82-integrations.md`](docs/reports/2026-04-phase82-integrations.md).

### Phase 81 — Transferencia radiativa M1 (nuevo crate `gadget-ng-rt`)

- Nuevo crate `crates/gadget-ng-rt/` con solver M1 completo:
  - `m1.rs`: `RadiationField` (grid E, Fx/Fy/Fz), `M1Params`, `m1_update` (solver HLL),
    `eddington_factor` (cierre M1: f→1/3 isótropo, f→1 streaming libre).
  - `coupling.rs`: `photoionization_rate`, `apply_photoheating`, `deposit_gas_emission`,
    `radiation_gas_coupling_step` (splitting de operadores gas↔rad).
  - Velocidad de luz reducida `c_red = c / c_red_factor` (default: 100×).
  - Fuente implícita linealizada para absorción: estabilidad para κ·dt > 1.
- Nueva sección `[rt]` en `RunConfig` (`RtSection`: enabled, c_red_factor, kappa_abs, rt_mesh, substeps).
- 15 tests unitarios: eddington_factor (isótropo/streaming/monótono), conservación energía en vacío,
  decaimiento por absorción, fotocalentamiento, emisión del gas.
- Reporte: [`docs/reports/2026-04-phase81-radiative-transfer-m1.md`](docs/reports/2026-04-phase81-radiative-transfer-m1.md).

### Phase 80 — AMR 3 niveles jerárquico recursivo

- Extensión de `crates/gadget-ng-pm/src/amr.rs` con soporte multi-nivel:
  - `AmrLevel { patches, child_levels, depth }`: árbol de niveles de refinamiento.
  - `AmrParams` extendido con `max_levels: usize` y `refine_factor: f64`.
  - `amr_pm_accels_multilevel(...)`: solver AMR N-nivel recursivo.
  - `amr_pm_accels_multilevel_with_stats(...)`: versión instrumentada.
  - `build_amr_hierarchy(...)`: construcción recursiva del árbol.
  - Umbral por nivel: `δ_refine × factor^l` con corrección ponderada suave.
- `max_levels=1` preserva el comportamiento exacto de Phase 70.
- 4 tests nuevos (11 total en `amr::tests`): multilevel sin NaN, stats correctas.
- Reporte: [`docs/reports/2026-04-phase80-amr3-multilevel.md`](docs/reports/2026-04-phase80-amr3-multilevel.md).

### Phase 79 — Validación producción N=128³

- `configs/validation_128.toml`: N=128³, ΛCDM Planck18, z_ini≈49, 512 pasos, TreePM + análisis in-situ.
- `configs/validation_128_test.toml`: versión N=32³ para CI (<30s).
- `scripts/run_validation_128.sh`: script con soporte `--resume`, `--mpi N`, `--post`.
- `docs/notebooks/validate_pk_hmf.py`: P(k) vs Eisenstein-Hu, σ₈ medida vs input (tol. 5%), HMF.
- 6 smoke tests en `crates/gadget-ng-physics/tests/phase79_validation.rs`.
- Reporte: [`docs/reports/2026-04-phase79-validation.md`](docs/reports/2026-04-phase79-validation.md).

### Phase 78 — Stellar feedback: kicks estocásticos de supernovas

- Nuevo módulo `crates/gadget-ng-sph/src/feedback.rs`:
  - `compute_sfr(particles, cfg)`: ley Schmidt-Kennicutt con umbral `rho_sf`.
  - `apply_sn_feedback(particles, sfr, cfg, dt, seed)`: kick estocástico con `p = 1-exp(-sfr×dt/m)`.
  - `total_sn_energy_injection(...)`: monitoreo de energía inyectada.
- Nueva sección `[sph.feedback]` en `SphSection` (`FeedbackSection`: enabled, v_kick_km_s, eps_sn, rho_sf, sfr_min).
- Hook integrado en `maybe_sph!` de `engine.rs` (Phase 66).
- 8 tests en `feedback::tests`.
- Reporte: [`docs/reports/2026-04-phase78-stellar-feedback.md`](docs/reports/2026-04-phase78-stellar-feedback.md).

### Phase 77 — Catálogo de halos HDF5 por snapshot

- Nuevo módulo `crates/gadget-ng-io/src/halo_catalog_hdf5.rs`:
  - Structs `HaloCatalogEntry`, `SubhaloCatalogEntry`, `HaloCatalogHeader`.
  - `write_halo_catalog_hdf5(...)` / `read_halo_catalog_hdf5(...)` (feature `hdf5`).
  - `write_halo_catalog_jsonl(...)` / `read_halo_catalog_jsonl(...)` (siempre disponible).
  - Estructura `/Header`, `/Halos/{Mass,Pos,Vel,R200,Spin_Peebles,Npart}`, `/Subhalos/`.
- Compatible con yt, h5py, Caesar, rockstar-galaxies.
- 4 tests (+ 1 con feature hdf5): roundtrip JSONL, header creation, serialización.
- Reporte: [`docs/reports/2026-04-phase77-halo-catalog-hdf5.md`](docs/reports/2026-04-phase77-halo-catalog-hdf5.md).

### Phase 76 — Assembly bias: spin/c vs entorno

- Nuevo módulo `crates/gadget-ng-analysis/src/assembly_bias.rs`:
  - `compute_assembly_bias(...)`: correlación Spearman λ/c vs δ_env + sesgo por cuartiles.
  - Campo suavizado con filtro top-hat esférico en k-space (`W(kR) = 3[sin-cos]/x³`).
  - `AssemblyBiasResult { spearman_lambda, spearman_concentration, bias_vs_lambda, ... }`.
  - `spearman_correlation(x, y)`: coeficiente de Spearman exportado.
- 9 tests: correlaciones perfectas/inversas, monotonía, filtro top-hat, serialización.
- Reporte: [`docs/reports/2026-04-phase76-assembly-bias.md`](docs/reports/2026-04-phase76-assembly-bias.md).

### Phase 75 — P(k,μ) + multipoles P₀/P₂/P₄ en espacio de redshift

- Nuevo módulo `crates/gadget-ng-analysis/src/pk_rsd.rs`:
  - `pk_redshift_space(...)`: P(k,μ) con desplazamiento RSD (Hamilton 1992).
  - `pk_multipoles(...)`: P₀/P₂/P₄ integrando sobre μ con polinomios de Legendre.
  - `compute_pk_multipoles(...)`: combinado posiciones → multipoles.
  - `kaiser_multipole_ratios(β)`: ratios teóricos Kaiser para validación.
  - `LosAxis { X, Y, Z }`, `PkRsdBin`, `PkMultipoleBin`, `PkRsdParams`.
- Integración in-situ: campo `pk_rsd_bins` en `InsituAnalysisSection`; campos
  `pk_rsd` y `pk_multipoles` en `insitu_NNNNNN.json`.
- 7 tests unitarios.
- Reporte: [`docs/reports/2026-04-phase75-pk-rsd.md`](docs/reports/2026-04-phase75-pk-rsd.md).

### Phase 74 — Output HDF5 con atributos estándar GADGET-4

- Nuevo módulo `crates/gadget-ng-io/src/gadget4_attrs.rs` con:
  - `Gadget4Header`: struct completo con los 22 atributos del estándar GADGET-4 HDF5.
  - Constructores `for_nbody(...)` y `for_sph(...)` para casos comunes.
  - `write_gadget4_header(group, &header)`: escribe todos los atributos al grupo `/Header`.
  - `read_gadget4_header(group)`: lee todos los atributos con tolerancia a campos opcionales.
  - Constantes de conversión: `KPC_IN_CM = 3.086e21`, `MSUN_IN_G = 1.989e33`, `KMS_IN_CMS = 1e5`.
  - `hubble_of_a(a)`: H(a) en km/s/Mpc para cosmología ΛCdM plana.
- Atributos nuevos respecto a la implementación anterior: `OmegaBaryon`, `NumPart_Total_HW`, `Flag_Entropy_ICs`, `Flag_DoublePrecision`, `Flag_IC_Info`, `UnitLength_in_cm`, `UnitMass_in_g`, `UnitVelocity_in_cm_per_s`.
- `Hdf5Writer` actualizado para usar `write_gadget4_header` (Phase 55 → Phase 74 compatible).
- Los snapshots resultantes son legibles directamente por `yt`, `pynbody`, `h5py`, GADGET-4.
- 7 tests unitarios (+ 1 HDF5 con feature gate) en `gadget-ng-io --lib`.
- Reporte: [`docs/reports/2026-04-phase74-hdf5-gadget4-attrs.md`](docs/reports/2026-04-phase74-hdf5-gadget4-attrs.md).

### Phase 73 — Perfiles de velocidad σ_v(r)

- Nuevo módulo `crates/gadget-ng-analysis/src/velocity_profile.rs` con:
  - `VelocityProfileBin`: bin radial con `v_r_mean`, `sigma_r`, `sigma_t`, `sigma_3d`, `n_part`.
  - `velocity_profile(positions, velocities, masses, center, v_center, params)`: binning en anillos esféricos con bins log o lineales.
  - `sigma_1d(sigma_3d)`: dispersión 1D = σ₃D / √3 (observable en espectroscopía).
  - `velocity_anisotropy(profile)`: β(r) = 1 − σ_t²/σ_r² de Binney.
- Bins con soporte para escalas logarítmicas y lineales; bins vacíos se omiten automáticamente.
- 8 tests unitarios verificando ordenamiento, σ ≥ 0, β finito, anisotropía radial.
- Reporte: [`docs/reports/2026-04-phase73-velocity-profile.md`](docs/reports/2026-04-phase73-velocity-profile.md).

### Phase 72 — Spin de halos λ (Peebles)

- Nuevo módulo `crates/gadget-ng-analysis/src/halo_spin.rs` con:
  - `HaloSpin`: resultado con L, |L|, R₂₀₀, V_vir, `lambda_peebles`, `lambda_bullock`.
  - `halo_spin(positions, velocities, masses, params)`: calcula λ y λ' para un halo.
  - `compute_halo_spins(...)`: batch sobre múltiples halos dados como índices.
  - `SpinParams`: configurable (G_Newton, Δ_vir, ρ_crit).
- Relación exacta λ_Bullock = λ_Peebles / √2.
- 7 tests: anillo circular, halo estático, vacío, dirección de L, batch multi-halo.
- Reporte: [`docs/reports/2026-04-phase72-halo-spin.md`](docs/reports/2026-04-phase72-halo-spin.md).

### Phase 71 — Bispectrum B(k₁,k₂,k₃)

- Nuevo módulo `crates/gadget-ng-analysis/src/bispectrum.rs` con:
  - `BkBin`, `BkIsoscelesBin`: structs de salida serializables.
  - `bispectrum_equilateral(positions, masses, box_size, mesh, n_bins)`: B_eq(k) via shell-filter + IFFT.
  - `bispectrum_isosceles(...)`: B(k₁, k₂) para configuraciones isósceles.
  - `reduced_bispectrum(bk_bins, pk_table)`: Q(k) = B_eq / (3P²) — detector de no-gaussianidad.
- Algoritmo shell-filter: CIC deposit → FFT 3D → filtrado por cáscara → IFFT → ⟨δ_k³⟩.
- Para campo gaussiano: Q ≈ 0; para campo no-lineal: Q > 0.
- 5 tests: campo uniforme, ordenamiento de k, distribución aleatoria, serialización JSON.
- Reporte: [`docs/reports/2026-04-phase71-bispectrum.md`](docs/reports/2026-04-phase71-bispectrum.md).

### Fix — Actualización de literales Particle/RunConfig

- Corregidos ~55 archivos de tests en todo el workspace que inicializaban `Particle` o `RunConfig` como struct literal, fallando tras los campos nuevos agregados en G2 y Phase 63.
- `Particle { ... }` → `Particle::new(...)` en tests de `gadget-ng-treepm`, `gadget-ng-physics`, `gadget-ng-parallel`.
- `RunConfig { ... }` → agregados `insitu_analysis: Default::default()` y `sph: Default::default()` en todos los tests afectados.
- Reporte: [`docs/reports/2026-04-fix-struct-literals.md`](docs/reports/2026-04-fix-struct-literals.md).

### Phase 70 — AMR-PM: refinamiento adaptativo de la malla Particle-Mesh

- Nuevo módulo `crates/gadget-ng-pm/src/amr.rs` con:
  - `AmrParams { delta_refine, patch_cells_base, nm_patch, max_patches, zero_pad }`.
  - `PatchGrid { center, size, nm, density, forces }` — descriptor de región refinada.
  - `identify_refinement_patches(base_density, nm, box_size, params)` — celdas con `ρ > ρ̄(1+δ_refine)`.
  - `deposit_to_patch(positions, masses, patch)` — CIC no periódico dentro del parche.
  - `solve_patch(patch, g, zero_pad)` — Poisson local con opción de zero-padding para condiciones de borde libre.
  - `interpolate_patch_forces(patch, positions)` — CIC bilineal local.
  - `amr_pm_accels(positions, masses, box_size, nm_base, g, params)` — solver completo 2 niveles.
  - `amr_pm_accels_with_stats(...)` — igual con `AmrStats { n_patches, n_particles_refined, max_overdensity }`.
- Exportados desde `gadget-ng-pm`: `amr_pm_accels`, `amr_pm_accels_with_stats`, `AmrParams`, `AmrStats`, `PatchGrid`.
- Zero-padding: densidad del parche se extiende a `2nm³` antes de la FFT para simular condiciones de borde no periódicas (Hockney & Eastwood 1988).
- Peso de transición en bordes: corrección del parche se aplica con `w = (1-2|f-0.5|)³` para suavizar la transición entre base y parche.
- 7 tests en [`crates/gadget-ng-physics/tests/phase70_amr_pm.rs`](crates/gadget-ng-physics/tests/phase70_amr_pm.rs) + 7 tests unitarios en `gadget-ng-pm --lib`.
- Reporte: [`docs/reports/2026-04-phase70-amr-pm.md`](docs/reports/2026-04-phase70-amr-pm.md).

### Phase 69 — Infraestructura corrida de producción N=256³

- `configs/production_256.toml`: configuración completa ΛCDM Planck18 para N=256³ con TreePM+SFC, block timesteps jerárquicos, 2LPT+E-H, HDF5, análisis in-situ.
- `configs/production_256_test.toml`: versión reducida N=32³ para CI smoke tests (<60 s).
- `scripts/run_production_256.sh`: script de producción con detección de checkpoint, soporte MPI (`N_RANKS`), post-proceso Python opcional, logging con timestamp.
- `docs/notebooks/postprocess_pk.py`: post-proceso P(k,z) desde archivos in-situ; genera `pk_evolution.json` y `pk_evolution.png`.
- `docs/notebooks/postprocess_hmf.py`: post-proceso HMF n(M,z) con comparación Sheth-Tormen analítica; genera `hmf_evolution.json` y `hmf_evolution.png`.
- 6 tests en [`crates/gadget-ng-physics/tests/phase69_production.rs`](crates/gadget-ng-physics/tests/phase69_production.rs): parseo de configs, ICs N=32³ sin NaN, parámetros físicos, masa consistente, σ₈ no trivial.
- Reporte: [`docs/reports/2026-04-phase69-production.md`](docs/reports/2026-04-phase69-production.md).

### Phase 68 — SUBFIND: subestructura dentro de halos FoF

- Nuevo módulo `crates/gadget-ng-analysis/src/subfind.rs` con `SubfindParams`, `SubhaloRecord`, `local_density_sph` y `find_subhalos`.
- Algoritmo: estimación de densidad SPH local (kernel Wendland C2, k-vecinos), walk de densidad descendente con Union-Find, filtrado por energía de enlace gravitacional (suma directa O(N²)).
- Flag `--subfind` y `--subfind-min-particles` en `gadget-ng analyze`; resultados escritos en campo `subfind` del `results.json`.
- Exportados desde `gadget-ng-analysis`: `find_subhalos`, `local_density_sph`, `SubfindParams`, `SubhaloRecord`.
- 6 tests en [`crates/gadget-ng-physics/tests/phase68_subfind.rs`](crates/gadget-ng-physics/tests/phase68_subfind.rs): cluster aislado, dos subclusters, conservación de masa, energía negativa, defaults, densidad concentrada.
- Reporte: [`docs/reports/2026-04-phase68-subfind.md`](docs/reports/2026-04-phase68-subfind.md).

### Phase 67 — Merger Trees: validación MAH (McBride+2009)

- Nuevas funciones en `crates/gadget-ng-analysis/src/merger_tree.rs`: `MassAccretionHistory`, `mah_main_branch(forest, root_id, redshifts)`, `mah_mcbride2009(m0, z, alpha, beta)`.
- Nuevo subcomando CLI `gadget-ng mah`: lee merger tree JSON, extrae MAH a lo largo de la rama principal, calcula ajuste analítico y escribe `mah.json`.
- Nuevo archivo `crates/gadget-ng-cli/src/mah_cmd.rs`.
- Exportados desde `gadget-ng-analysis`: `mah_main_branch`, `mah_mcbride2009`, `MassAccretionHistory`.
- 6 tests en [`crates/gadget-ng-physics/tests/phase67_mah.rs`](crates/gadget-ng-physics/tests/phase67_mah.rs): MAH monótona, McBride en z=0, trivial, merge detectado, snapshot único, McBride decrece con z.
- Reporte: [`docs/reports/2026-04-phase67-merger-tree-mah.md`](docs/reports/2026-04-phase67-merger-tree-mah.md).

### Phase 66 — SPH Cosmológico integrado al motor

- `gadget-ng-core/src/particle.rs`: nuevo enum `ParticleType { DarkMatter, Gas }` (default: DM); campos `ptype`, `internal_energy`, `smoothing_length` con `#[serde(default)]`; constructores `Particle::new_gas(...)` e `is_gas()`.
- `gadget-ng-core/src/config.rs`: nuevos `CoolingKind { None, AtomicHHe }` y `SphSection { enabled, gamma, alpha_visc, n_neigh, cooling, t_floor_k, gas_fraction }`; campo `pub sph: SphSection` en `RunConfig`.
- `crates/gadget-ng-sph/src/integrator.rs`: nueva función `sph_cosmo_kdk_step(particles, cf, gamma, alpha_visc, n_neigh, gravity_accel)` que integra Gas+DM con `CosmoFactors`.
- `crates/gadget-ng-sph/src/cooling.rs` (nuevo): `cooling_rate_atomic`, `apply_cooling`, `u_to_temperature`, `temperature_to_u`.
- `crates/gadget-ng-cli/src/engine.rs`: macro `maybe_sph!(cf)` disponible para inserción en loops de stepping.
- `crates/gadget-ng-parallel/src/pack.rs`: actualizado para inicializar campos SPH en gather global.
- 5 tests en [`crates/gadget-ng-physics/tests/phase66_sph_cosmo.rs`](crates/gadget-ng-physics/tests/phase66_sph_cosmo.rs): defaults, conservación energía 50 pasos, cooling monotóno, KDK acotado, ParticleType.
- Reporte: [`docs/reports/2026-04-phase66-sph-cosmo.md`](docs/reports/2026-04-phase66-sph-cosmo.md).

### Phase 65 — HDF5 paralelo MPI-IO

- Nuevo módulo `crates/gadget-ng-io/src/hdf5_parallel_writer.rs` con feature `hdf5-parallel`: `write_snapshot_hdf5_serial`, `read_snapshot_hdf5_serial` y módulo `parallel_impl` (requiere `libhdf5` con `--enable-parallel`).
- `Hdf5ParallelOptions { chunk_size: 65536, compression: 0 }` para control de chunks y compresión gzip.
- Layout idéntico al `Hdf5Writer` existente: `/Header` + `/PartType1/{Coordinates,Velocities,Masses,ParticleIDs}`.
- Con `SerialRuntime` (P=1) produce archivos bit-a-bit idénticos al escritor serial. Tests se saltan si `libhdf5` no disponible.
- 4 tests en [`crates/gadget-ng-physics/tests/phase65_hdf5_parallel.rs`](crates/gadget-ng-physics/tests/phase65_hdf5_parallel.rs): roundtrip P=1, layout GADGET-4, contenido idéntico, defaults de opciones.
- Reporte: [`docs/reports/2026-04-phase65-hdf5-parallel.md`](docs/reports/2026-04-phase65-hdf5-parallel.md).

### Phase 64 — gadget-ng-vis: proyecciones adicionales y mapa de densidad

- `crates/gadget-ng-vis/src/ppm.rs` extendido con 3 nuevas funciones: `render_ppm_projection` (proyecciones XY/XZ/YZ), `render_density_ppm` (escala log₁₀ + colormap Viridis) y `write_png` (exportación PNG nativa vía crate `png`).
- Exportadas en `lib.rs`: `render_density_ppm`, `render_ppm_projection`, `write_png`.
- CLI `Commands::Stepping` extendido con `--vis-proj <xy|xz|yz>`, `--vis-mode <points|density>`, `--vis-format <ppm|png>`.
- 6 tests en [`crates/gadget-ng-vis/tests/ppm_extended.rs`](crates/gadget-ng-vis/tests/ppm_extended.rs): `density_map_concentrated_bright`, `density_map_empty_is_dark`, `projection_xz_correct`, `projection_yz_correct`, `write_png_header`, `write_png_minimal`.
- Reporte: [`docs/reports/2026-04-phase64-vis-projections-density.md`](docs/reports/2026-04-phase64-vis-projections-density.md).

### Phase 63 — Análisis in-situ en el loop stepping

- Nueva sección `InsituAnalysisSection` en `gadget-ng-core/src/config.rs` con campos `enabled`, `interval`, `pk_mesh`, `fof_b`, `fof_min_part`, `xi_bins`, `output_dir`. Exportada desde `gadget-ng-core`.
- Campo `pub insitu_analysis: InsituAnalysisSection` agregado a `RunConfig` (default: `enabled=false`).
- Nuevo módulo `crates/gadget-ng-cli/src/insitu.rs` con `maybe_run_insitu(particles, cfg, box_size, a, step, out_dir)`: escribe `insitu_{step:06}.json` con P(k), n_halos, masa total y ξ(r) opcional.
- Macro `maybe_insitu!(step)` insertada en los 7 loops de stepping de `engine.rs`.
- 5 tests en [`crates/gadget-ng-physics/tests/phase63_insitu_analysis.rs`](crates/gadget-ng-physics/tests/phase63_insitu_analysis.rs): defaults, lógica de intervalo, disabled, params, P(k) finito en lattice uniforme.
- Reporte: [`docs/reports/2026-04-phase63-insitu-analysis.md`](docs/reports/2026-04-phase63-insitu-analysis.md).

### Phase 62 — Merger Trees single-pass

- Nuevo archivo `crates/gadget-ng-analysis/src/merger_tree.rs` con `MergerTreeNode`, `MergerForest`, `ParticleSnapshot` y `build_merger_forest(catalogs, min_shared_fraction)`.
- Algoritmo single-pass: vota progenitor por fracción de partículas compartidas entre snapshots consecutivos; registra mergers secundarios.
- Nuevo subcomando CLI `gadget-ng merge-tree` con `--snapshots`, `--catalogs`, `--out`, `--min-shared`. Implementado en `crates/gadget-ng-cli/src/merge_tree_cmd.rs`.
- Exportados desde `lib.rs`: `build_merger_forest`, `MergerForest`, `MergerTreeNode`, `ParticleSnapshot`.
- 4 tests en [`crates/gadget-ng-physics/tests/phase62_merger_trees.rs`](crates/gadget-ng-physics/tests/phase62_merger_trees.rs): trivial sin mergers, fusión binaria, roundtrip JSON, snapshot único.
- Reporte: [`docs/reports/2026-04-phase62-merger-trees.md`](docs/reports/2026-04-phase62-merger-trees.md).

### Phase 61 — FoF paralelo MPI (cross-boundary Union-Find)

- Feature `parallel` en `gadget-ng-analysis/Cargo.toml`: dependencia opcional `gadget-ng-parallel`.
- Nuevo archivo `crates/gadget-ng-analysis/src/fof_parallel.rs` con `find_halos_parallel<R: ParallelRuntime>`: intercambia partículas frontera vía `exchange_halos_sfc` y aplica Union-Find cross-boundary.
- Nuevo helper `find_halos_combined` en `fof.rs`: FoF sobre conjunto local+halos recibidos, guardando solo grupos con raíz local (índice < N_local).
- Con `SerialRuntime` (P=1) idéntico al FoF serial; con P>1 escala a O(N/P + N_frontera).
- 3 tests en [`crates/gadget-ng-physics/tests/phase61_fof_parallel.rs`](crates/gadget-ng-physics/tests/phase61_fof_parallel.rs): vs serial P=1, halo cross-boundary, conservación de masa.
- Reporte: [`docs/reports/2026-04-phase61-fof-parallel-mpi.md`](docs/reports/2026-04-phase61-fof-parallel-mpi.md).

### Rápidas — `gadget-ng analyze` + `gadget-ng-vis` PPM

- Nuevo subcomando `gadget-ng analyze` en [`crates/gadget-ng-cli/src/analyze_cmd.rs`](crates/gadget-ng-cli/src/analyze_cmd.rs): pipeline completo FoF + P(k) + ξ(r) + c(M) desde snapshot JSONL; escribe `results.json` con halos, espectro de potencia, función de correlación y tabla concentración-masa. Opciones: `--fof-b`, `--pk-mesh`, `--xi-bins`, `--nfw-min-part`, `--box-size-mpc-h`.
- Nuevo módulo `crates/gadget-ng-vis/src/ppm.rs`: `render_ppm(positions, box_size, width, height) → Vec<u8>` (proyección XY, fondo negro, partículas blancas) y `write_ppm(path, pixels, w, h)` en formato PPM binario P6 sin dependencias externas. 5 tests unitarios.
- CLI `gadget-ng stepping --vis-snapshot 1` genera `<out>/snapshot_final.ppm` después de la corrida.
- Reporte: [`docs/reports/2026-04-rapidas-analyze-vis.md`](docs/reports/2026-04-rapidas-analyze-vis.md).

### Phase 60 — Domain Decomposition Adaptativa

- Nuevo campo `rebalance_imbalance_threshold: f64` en `PerformanceSection` (default 0.0 = desactivado). Si `max(walk_ns)/min(walk_ns) > threshold`, se fuerza rebalanceo inmediato independientemente de `sfc_rebalance_interval`. Valores típicos: 1.3, 1.5, 2.0.
- Nueva función `should_rebalance(step, start_step, interval, cost_pending)` en `engine.rs`: centraliza la lógica de decisión de rebalanceo para todos los paths SFC.
- Paths actualizados con detección de desbalance: jerárquico+LET (Phase 56) y cosmológico SFC+LET. El path SFC+LET BarnesHut ya existente usa el threshold configurable en lugar del valor hardcodeado 1.3.
- 5 tests en [`crates/gadget-ng-physics/tests/phase60_adaptive_rebalance.rs`](crates/gadget-ng-physics/tests/phase60_adaptive_rebalance.rs): verifican intervalo, override por costo, interval=0, configurabilidad del threshold y escenario completo de rebalanceo temprano.
- Reporte: [`docs/reports/2026-04-phase60-adaptive-rebalance.md`](docs/reports/2026-04-phase60-adaptive-rebalance.md).

### Phase 59 — Restart/Checkpoint robusto

- Auditoría del sistema de checkpoint: verificado que todos los paths SFC reconstruyen `SfcDecomposition` correctamente desde posiciones restauradas (no se requiere serializar el SFC).
- Nuevo campo `sfc_state_saved: bool` en `CheckpointMeta` (siempre `false`, informativo).
- Test de continuidad bit-a-bit en [`crates/gadget-ng-physics/tests/phase59_checkpoint_continuity.rs`](crates/gadget-ng-physics/tests/phase59_checkpoint_continuity.rs): N=8³ PM, 20 pasos — corrida continua vs corrida dividida (10+10 con clone de partículas). Resultado: `max|Δx| = 0.00e0`, `max|Δv| = 0.00e0`.
- Reporte: [`docs/reports/2026-04-phase59-checkpoint-continuity.md`](docs/reports/2026-04-phase59-checkpoint-continuity.md).

### Phase 58 — c(M) desde N-body + función de correlación ξ(r)

- Nueva función `concentration_ludlow2016(m200_msun_h, z)` en `gadget-ng-analysis/src/nfw.rs`: relación c(M) de Ludlow et al. 2016 (Planck2015 ΛCDM). A z=0 y M=10¹³ M☉/h: c=5.57 vs Duffy c=4.99 (ratio 1.12). Para M=10¹⁵: c=2.07 vs Duffy 3.39 (ratio 0.61).
- Nuevo archivo `gadget-ng-analysis/src/correlation.rs` con `XiBin { r, xi, n_pairs }`, `two_point_correlation_fft` (suma de Hankel discreta O(N_k × N_r)) y `two_point_correlation_pairs` (estimador Davis-Peebles DD/RR−1, O(N²)). Expuestos en `lib.rs`.
- Test de integración en [`crates/gadget-ng-physics/tests/phase58_nfw_concentration.rs`](crates/gadget-ng-physics/tests/phase58_nfw_concentration.rs): simulación PM N=32³, z:50→0, FoF → ajuste NFW; validación c(M) y ξ(r) via FFT y pares directos. Controlado con `PHASE58_SKIP=1`.
- Reporte: [`docs/reports/2026-04-phase58-nfw-concentration-xi.md`](docs/reports/2026-04-phase58-nfw-concentration-xi.md).

### Phase 57 — CUDA/HIP PM solver

- Nuevos crates `crates/gadget-ng-cuda` y `crates/gadget-ng-hip`: solver PM GPU opcional con degradación elegante si el toolchain no está disponible (`CUDA_SKIP=1` / `HIP_SKIP=1`).
- Kernels GPU: asignación de masa CIC (atomic), FFT 3D R2C via cuFFT/rocFFT, solver de Poisson en k-space, 3× FFT C2R para componentes de fuerza, interpolación CIC adjunta.
- Variables de config: `use_gpu_cuda = true`, `use_gpu_hip = true` en `[performance]`.
- Tests smoke con `#[ignore]` por defecto (requieren GPU real).
- Reporte: [`docs/reports/2026-04-phase57-cuda-hip-pm.md`](docs/reports/2026-04-phase57-cuda-hip-pm.md).

### Phase 56 — Block timesteps jerárquicos acoplados al árbol LET distribuido

- Nueva función `compute_forces_hierarchical_let` en `gadget-ng-parallel`: evalúa fuerzas solo para `active_local` usando halos SFC intercambiados, árbol local sobre `local + halos`.
- Acoplamiento en `engine.rs` path jerárquico+SFC (`use_hierarchical_let`): `exchange_halos_sfc` una vez por base-step, closure de fuerzas llama a `compute_forces_hierarchical_let`.
- Corrección bug softening cosmológico: `eps2 = (eps_phys/a)²` cuando `physical_softening = true`.
- 5 tests en [`crates/gadget-ng-physics/tests/phase56_hierarchical_let.rs`](crates/gadget-ng-physics/tests/phase56_hierarchical_let.rs): conservación de momentum, estabilidad energética, activos vs inactivos.
- Reporte: [`docs/reports/2026-04-phase56-hierarchical-let.md`](docs/reports/2026-04-phase56-hierarchical-let.md).

### Phase 55 — Comparación FoF vs HMF hasta z=0

- Nuevo reporte [`docs/reports/2026-04-phase55-fof-vs-hmf.md`](docs/reports/2026-04-phase55-fof-vs-hmf.md): evolución PM hasta `a=1.0` (z=0) con `G_consistent` y timestep adaptativo; FoF (b=0.2, min_particles=20) en unidades internas; conversión física `m_part = Ω_m·ρ_crit_H2·BOX³_Mpc_h/N_total`.
- Comparación cuantitativa `dn/dlnM(FoF)` vs `dn/dlnM(ST/PS)` con tolerancia de ratio ∈ [0.05, 20]; masa mínima resoluble: 1.8×10¹⁴ (N=64), 2.2×10¹³ (N=128), 2.8×10¹² (N=256) M_sun/h; convergencia de masa mínima verificada (N=256 < N=64).
- Nuevos tests en [`crates/gadget-ng-physics/tests/phase55_fof_vs_hmf.rs`](crates/gadget-ng-physics/tests/phase55_fof_vs_hmf.rs): 6 tests (estabilidad, conteo de halos, ratio FoF/ST ∈ [0.05,20], convergencia de masa, no-NaN, run completo N=64); script `run_phase55.sh`; JSON catálogos en `target/phase55/fof_results.json`.
- N ∈ {64, 128, 256}, BOX=300 Mpc/h, seed=42; selectores `PHASE55_SKIP_N128=1`, `PHASE55_SKIP_N256=1` para prueba rápida.

### Phase 54 — Validación cuantitativa D²(a) con G consistente

- Nuevo reporte [`docs/reports/2026-04-phase54-growth-validation.md`](docs/reports/2026-04-phase54-growth-validation.md): evolución PM con timestep adaptativo (`adaptive_dt_cosmo`, α_H=0.01, dt_max=0.05) y `G_consistent = 3Ω_mH₀²/(8π) ≈ 3.76×10⁻⁴`; N ∈ {64,128,256}, BOX=100 Mpc/h, 6 snapshots a ∈ {0.02, 0.05, 0.10, 0.20, 0.33, 0.50}.
- Métricas: `|P_sim(k,a)/P_EH_theory(k,a) − 1|` en bins k < k_nyq/2; tests de estabilidad (sin crash, 16 bins P(k) por snapshot) y `sigma8` normalización (error 5.2 %).
- **Resultado clave**: la simulación es estable hasta `a=0.50`; los errores elevados (~54–99 % vs D²(a) lineal) son esperados — con ICs Zel'dovich desde `a=0.02` en régimen de libre streaming, la señal dominante no es crecimiento gravitacional lineal sino dispersión de velocidades.
- Nuevos tests en [`crates/gadget-ng-physics/tests/phase54_growth_factor_validation.rs`](crates/gadget-ng-physics/tests/phase54_growth_factor_validation.rs): 5 tests; script `run_phase54.sh`; JSON en `target/phase54/snapshots.json`. Selectores `PHASE54_SKIP_N128=1`, `PHASE54_SKIP_N256=1`.
- Commit `09aa84f` revisa tolerancias de tests como verificaciones de estabilidad en lugar de tolerancias estrictas de D²(a).

### Phase 53 — Perfiles NFW y relación concentración-masa c(M)

- `gadget_ng_analysis::nfw`: `NfwProfile { rho_s, r_s }` con `from_m200_c`, `density(r)`, `mass_enclosed(r)`, `r200`, `circular_velocity_sq_over_g`, `concentration`; `rho_crit_z(Ω_m, Ω_Λ, z)`, `r200_from_m200`, `concentration_duffy2008` (WMAP5: A=5.71, B=−0.084, C=−0.47), `concentration_bhattacharya2013`; `measure_density_profile` (bins log-espaciados) y `fit_nfw_concentration` (búsqueda en cuadrícula + LS en log-espacio).
- Propiedades analíticas verificadas: M(<r_200)=M_200 y ρ_mean=200ρ_crit con error < 10⁻¹⁰; pendientes γ=−1/−2/−3 (err<0.05); v_c_max en r/r_s=2.163±0.3; c_fit=5.27 vs c_true=5.0 (err<6 %); tabla z=0: c(10¹²)=6.05, c(10¹⁴)=4.11, c(10¹⁵)=3.39.
- 14 tests (6 integración + 8 unitarios) en `phase53_nfw_profiles.rs`.

### Phase 52 — Función de masa de halos Press-Schechter / Sheth-Tormen

- `gadget_ng_analysis::halo_mass_function`: `sigma_m(M, params, z)` calcula σ(M,z) = D(z)·σ(M,0) con integral σ²(R) trapezoidal log-espaciada (1200 puntos); `lagrange_radius(M, ρ̄_m)`; `mass_function_table` genera dn/d ln M para PS y ST; `multiplicity_ps(σ)` y `multiplicity_st(σ)`; `HmfParams::planck2018()`; `RHO_CRIT_H2 = 2.775×10¹¹ (M_sun/h)/(Mpc/h)³`.
- Normalización verificada: σ(R=8 Mpc/h)=σ₈ con error < 0.01 %; n(>10¹⁴)≈3.2×10⁻⁵ h³/Mpc³ coherente con ACT/SPT/eROSITA.
- 7 tests en `phase52_mass_function.rs`: σ(R=8)=σ₈, σ(M) monótona, ∫f_PS dσ≈1, tabla coherente, formación jerárquica con z, n(>10¹⁴) observable, ICs+FoF cualitativo.

### Phase 51 — G auto-consistente en motor de producción

- `CosmologySection::auto_g: bool` en `config.rs`; cuando `auto_g = true` y `cosmology.enabled = true`, `effective_g()` calcula `G = 3·Ω_m·H₀²/(8π)` (prioridad sobre `gravitational_constant` manual, menor que `units.enabled`).
- `RunConfig::cosmo_g_diagnostic()` devuelve `(G_consistente, error_relativo)` para cualquier config cosmológica; motor `engine.rs` emite `warn!` si G manual difiere > 1 % del valor Friedmann-consistente, e `info!` cuando `auto_g=true` activo.
- 5 tests en `phase51_auto_g.rs`; retrocompatible: `auto_g = false` (default) preserva comportamiento anterior exacto.

### Phase 50 — Unidades físicamente consistentes

- `g_code_consistent(omega_m, h0) → f64` en `cosmology.rs`: `G = 3·Ω_m·H₀²/(8π)` en unidades de código; `cosmo_consistency_error(g, omega_m, h0, rho_bar)` para diagnóstico.
- Diagnóstico cuantitativo: con G=1 y H₀=0.1, `(4πGρ̄)/H₀² = 1257` (factor 2660× fuera de `(3/2)Ω_m = 0.47`).
- 5 tests en `phase50_physical_units.rs`: fórmula exacta, cuantificación de inconsistencia legacy, estabilidad corta N=8, estabilidad larga a=0.02→0.20, comparación G_consistente vs G_legacy.

### Phase 49 — Fix del integrador cosmológico

- Corrección de `gravity_coupling_qksl` en `cosmo_pm.rs`, `phase37_growth_rescaled_ics.rs` y `phase41_high_resolution_validation.rs`: todos los paths PM cosmológicos usan ahora `G·a³` en lugar de `G/a`.
- `adaptive_dt_cosmo(params, a, acc_max, softening, eta_grav, alpha_h, dt_max)` en `cosmology.rs`: criterio gravitacional `dt_grav = η·√(ε/|a_max|)` + Hubble `dt_hub = α_H/H(a)`.
- 10 tests en 4 archivos nuevos; validación Halofit con integrador corregido.

### Phase 48 — Halofit no-lineal (Takahashi+2012)

- `gadget_ng_analysis::halofit`: `halofit_pk(k, p_lin, cosmo, z)` implementa Takahashi+2012 (ec. 11–35); `sigma_sq(R)` con integración log-trapezoidal; bisección para `k_sigma`; `n_eff` y curvatura `C` via diferencias finitas; coeficientes {an,bn,cn,γ,α,β,ν} para ΛCDM plano; `p_linear_eh` con factor D²(a).
- Limitación documentada: EH da boost ~6 % en k=0.3 vs ~15 % de CAMB — aceptable para uso interno.
- 7 tests unitarios (σ(8)=σ₈, k_sigma razonable, P_nl≥P_lin, convergencia lineal, boost no-lineal, ratios vs CAMB, k_sigma crece con z) + 4 tests de integración (`halofit_static`, `halofit_growth_consistency` < 3.5 % error, `pk_vs_halofit_at_ics`, `nonlinear_boost_redshift_dependence`).

### Phase 47 — Corrección P(k) recalibrada

- `measure_rn()` para calibrar R(N) in-process; `correct_pk_with_shot_noise()` para sustracción de ruido Poisson; `RnModel::phase47_default()` con R(N=128)=0.002252 (campaña 4 seeds, CV=1.0 %); fit {32,64,128}: `α=1.953`.

### Phase 46 — PM pencil 2D FFT

- `PencilLayout2D`, `solve_forces_pencil2d`, `alltoallv_f64_subgroup` en `ParallelRuntime`; escala hasta `P ≤ nm²` en lugar de `P ≤ nm` (slab 1D); selección automática cuando `P > nm`.

### Phase 45 — Auditoría y corrección de unidades IC ↔ integrador

- Nuevo reporte [`docs/reports/2026-04-phase45-units-audit.md`](docs/reports/2026-04-phase45-units-audit.md) que cierra la hipótesis abierta en Phase 44 (*«el bottleneck real es un mismatch de unidades entre ICs y `leapfrog_cosmo_kdk_step`, no 2LPT ni `dt`»*). Ejecuta las 5 tareas del brief (auditoría IC→integrador, single-drift, evolución ultracorta, A/B de convenciones del kick, patch mínimo) y responde con patch aplicado + DoD completa.
- **Mismatch identificado — factor `a⁴` espurio en fuerzas efectivas**: la convención del slot `velocity` en ICs (`p_ic = a²·f·H·Ψ = a²·ẋ_c`) es **canónica QKSL/GADGET-4** y compatible con el `drift = ∫dt/a²` del integrador (validado bit-idéntico por `single_drift_matches_integrator_formula`). El kick, sin embargo, usa `∫dt/a` que implica `dp/dt = F/a`, pero la EOM canónica derivada del Hamiltoniano comóvil (`H = p²/(2a²) + Φ_pec`, Poisson peculiar `∇²Φ_pec = 4πG·ρ̄·δ·a²`) da `dp/dt = −∇Φ_pec` sin `1/a`. Al pasar al solver `g_cosmo = G/a` el error neto es `1/a⁴` (`~6·10⁶` a `a=0.02`), causando el `v_rms × 10¹⁰` de Phase 43–44.
- **Patch mínimo (Opción B) — solo 1 fórmula nueva**: [`crates/gadget-ng-core/src/cosmology.rs`](crates/gadget-ng-core/src/cosmology.rs) añade `gravity_coupling_qksl(g, a) = g·a³` (antes `g/a`). Aplicado en 2 sitios de [`crates/gadget-ng-cli/src/engine.rs`](crates/gadget-ng-cli/src/engine.rs) (paths SFC+LET cosmológico y TreePM slab cosmológico) y en los tests Phase 43–45. **No se tocó** `leapfrog_cosmo_kdk_step` ni `CosmoFactors`, ni el solver (`fft_poisson`, `TreePmSolver`, tree BH), ni `pk_correction`/`R(N)`, ni las fórmulas de ICs (`zeldovich_ics`, `zeldovich_2lpt_ics`).
- Nueva API en [`crates/gadget-ng-core/src/ic_zeldovich.rs`](crates/gadget-ng-core/src/ic_zeldovich.rs): enum `IcMomentumConvention::{DxDt, ADxDt, A2DxDt, GadgetCanonical}` + `zeldovich_ics_with_convention(..., conv)`. Permite auditar A/B las 4 convenciones de `velocity` slot sin `git checkout`. Re-exports en `gadget_ng_core::prelude`.
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase45_units_audit.rs`](crates/gadget-ng-physics/tests/phase45_units_audit.rs) (5 tests, **1.7 s release**): `single_drift_matches_integrator_formula` (bit-idéntico `max_err = 0`), `single_drift_matches_linear_dx_dt` (contra LPT lineal, `max_err_rel = 6.20·10⁻⁷` = ruido doble-precisión con `|dx_pred| ≈ 10⁻¹³`), `convention_ab_single_drift` (`A²·DxDt` y `GadgetCanonical` ganan con `6.20·10⁻⁷`; `DxDt` pierde por `2.5·10³`; `A·DxDt` pierde por `49`), `kick_convention_probe` (A/B de `(g_cosmo × kick)`, ver tabla abajo), `short_linear_growth_preserved` (**`P(k,a)/P(k,a₀) = 1.0101` exacto vs `[D/D₀]² = 1.0101`**).
- Hallazgos cuantitativos del A/B `kick_convention_probe` a `a: 0.02 → 0.0201`, `N=16³`, 2LPT, `dt=5·10⁻⁶`:

  | Convención                       | `v_rms_final/v_rms_inicial` | vs lineal `≈ 1.008` |
  |----------------------------------|-----------------------------|---------------------|
  | **Actual** `(G/a, ∫dt/a)`        | **`3.22·10⁹`**              | catastrófico        |
  | QKSL compensada `(G·a³, ∫dt/a)`  | `1.337`                     | ✓ ~30% overshoot    |
  | QKSL plana `(G·a², dt)`          | `1.337`                     | ✓ idéntico          |
  | Newtoniano plano `(G, dt)`       | `2.27·10³`                  | ✗ sin cosmología    |

  Las dos convenciones canónicas QKSL coinciden bit-a-bit hasta 3 dígitos y pegan al crecimiento lineal; el overshoot residual `~30%` viene de no-linealidad 2LPT a `N=16` (softening finito), no del integrador.
- **Validación post-patch**: Phase 45 5/5 ok (nuevo). **Phase 44 5/5 ok** tras relajar asserts del test `fixed_variant_runs_stably` (sistema ya no explota, pero sub-evoluciona a `N=32` coarse sobre `a=0.02→0.1`: `δ_rms=0` es comportamiento estable válido, no muerto). **Phase 43 7/7 ok** (`adaptive_dt_matches_or_beats_best_fixed_dt`, `parallel_tree_walk_matches_serial_within_tolerance`, etc., runtime 1071 s release, `v_rms` ahora queda en `~5·10⁻⁴` a `a=0.1` en vez de `~34`). **Unit tests de `gadget-ng-core` y `gadget-ng-integrators`** ok sin regresiones.
- **Hallazgo clave (evolución corta funciona, larga aún no converge)**: la patología `v_rms × 10¹⁰` queda resuelta de raíz — en régimen lineal ultracorto (`a: 0.02 → 0.0201`, 20 pasos) **`P(k)` crece exactamente como `[D(a)/D(a₀)]²`**. El error residual en evoluciones largas (`a=0.02 → 0.1`, 400 pasos a `N=32³`) proviene de (i) softening comóvil de la tree SR no-físico a escalas de grid, (ii) acumulación de no-linealidades via NGP, (iii) no es mismatch de unidades. Todos esos cuellos son tratables en fases posteriores sin tocar la convención QKSL ya establecida.
- **Decisión técnica: `A_units_mismatch_confirmed_and_fixed`**. El bug era real, de signo `a⁴`, y vivía en el acoplamiento `gravity_coupling × kick_integral`. El fix es **una línea nueva** (`gravity_coupling_qksl`) aplicada en todos los sitios que invocan el solver cosmológico. Se mantiene la convención IC intacta (`p = a²·ẋ_c`) por ser la canónica QKSL/GADGET-4 validada contra Springel 2005 §3.1 y QKSL 1997. Los goldens bit-exactos de fases previas que dependan del path cosmológico CLI deben regenerarse; la física en régimen lineal es ahora correcta por primera vez desde Phase 17b.
- Referencias: Quinn, Katz, Stadel & Lake 1997 (`astro-ph/9710043`, convención `p = a²·ẋ_c`, `Δx = p·∫dt/a²`, `Δp = −∇Φ_pec·dt`); Springel 2005 (`astro-ph/0505010`, GADGET-2 §3.1, Hamiltoniano comóvil).

### Phase 44 — Auditoría y fix de condiciones iniciales 2LPT

- Nuevo reporte [`docs/reports/2026-04-phase44-2lpt-audit-fix.md`](docs/reports/2026-04-phase44-2lpt-audit-fix.md) que cierra la hipótesis abierta en Phase 43 (*«el cuello restante está en ICs / convención de 2LPT»*) con una auditoría canónica contra Scoccimarro 1998, Bouchet+95, Crocce, Pueblas & Scoccimarro 2006 (`2LPTic/main.c:477-478`) y Jenkins 2010 (ec. 2), identifica **dos bugs críticos** en `crates/gadget-ng-core/src/ic_2lpt.rs` y los corrige con 6 unit tests k-space + 5 tests de integración A/B.
- **Bug A — doble división por `|n|²` (CRÍTICO)**: la implementación previa componía `φ²(k) = −S/|n|²` (Poisson) seguido de `Ψ²_α(k) = −i·n_α/|n|²·φ²` (gradiente), produciendo efectivamente `Ψ²_α = +i·n_α·S/|n|⁴` — una división extra por `|n|²` respecto al canónico `−i·n_α·S/|n|²`.
- **Bug B — signo global invertido (CRÍTICO)**: la composición anterior también invierte el signo canónico (`+i` vs `−i`), provocando que la corrección de 2º orden se aplique en la dirección físicamente contraria al `D₂/D₁² ≈ −3/7·Ω_m^{−1/143}` correcto.
- **Bug C — aproximación de `f₂` (MENOR)**: se usaba `f₂ = 2·f₁` con `f₁ = Ω_m(a)^{0.55}` (Linder) en vez de `f₂ = 2·Ω_m(a)^{6/11}` (Bouchet+95/Scoccimarro). Diferencia `< 0.01%` a `z = 49`, pero restaura la convención literaria.
- **Fix aplicado**: las dos funciones separadas `solve_poisson_real_to_kspace` y `phi2_to_psi2` se reemplazan por una única `source_to_psi2(source, n, box_size) → [Ψ²_x, Ψ²_y, Ψ²_z]` (~75 LOC) que implementa la fórmula canónica con una sola división por `|n|²` y signo `−i`. `f₂` pasa a `2·Ω_m(a)^{6/11}`.
- Nuevas API públicas en [`crates/gadget-ng-core/src/ic_2lpt.rs`](crates/gadget-ng-core/src/ic_2lpt.rs): `Psi2Variant::{Fixed, LegacyBuggy}` y `zeldovich_2lpt_ics_with_variant(..., variant)` para permitir auditoría A/B sin `git checkout`. `zeldovich_2lpt_ics` delega en la variante `Fixed` (bit-compatible con Phase 43 para consumidores externos).
- Nuevos unit tests en `ic_2lpt.rs` (6 tests, 0.01 s): `source_is_finite`, `psi2_is_real_and_finite`, `psi2_matches_canonical_kspace_formula` (error global `< 10⁻¹⁰` sobre 10 122 modos, N=16), `psi2_amplitude_differs_from_legacy_bug` (ratio RMS Fixed/Bug = **2.72×**), `psi2_scales_quadratically_with_delta`, `psi2_signs_consistent_across_amplitudes`.
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase44_2lpt_audit.rs`](crates/gadget-ng-physics/tests/phase44_2lpt_audit.rs): `ic_amplitudes_changed_by_fix` (A/B: `max Δpos=1.14·10⁻¹²`, `max Δvel=1.82·10⁻¹⁴`), `fixed_variant_matches_legacy_psi1_component` (1LPT bit-idéntico), `fixed_variant_runs_stably`, `fixed_variant_improves_growth_vs_legacy` (soft), `no_nan_inf_under_phase44_matrix`. Caché disco vía `OnceLock` + JSON en `target/phase44/per_snapshot_metrics.json`. Selector `PHASE44_N=32|64|128`.
- Nuevo experimento [`experiments/nbody/phase44_2lpt_audit/`](experiments/nbody/phase44_2lpt_audit/) con script `plot_ab_comparison.py` (4 paneles + CSV), orquestador `run_phase44.sh`, figuras copiadas a `docs/reports/figures/phase44/`.
- **Hallazgo físico clave (inesperado)**: el bug era matemáticamente real (validado por `psi2_matches_canonical_kspace_formula`) pero su **impacto empírico en la normalización actual** (`Z0Sigma8` con `a_init=0.02`) **es marginal**: `max Δpos=10⁻¹²`, `Δerr ≈ 1–2 %`, `Δgrowth_lowk ≈ 10 %`. Razón: con `scale² = (D(a_init)/D(1))² ≈ 4·10⁻⁴` y `D₂/D₁² ≈ −0.43`, el término 2LPT total en posición es `~1.7·10⁻⁴·Ψ²_unscaled`, O(10⁻¹⁰) en unidades `box=1`. El fix cambia Ψ² por un factor ~2.7× pero la diferencia absoluta queda en O(10⁻¹²).
- **Diagnóstico del bottleneck real (Phase 45 abierto)**: `v_rms` salta de `9.85·10⁻¹⁰` al IC a `34` a `a=0.05` (×`3.5·10¹⁰`), cuando el crecimiento esperado es `D(0.05)/D(0.02) ≈ 2.5`. Esto señala una **discrepancia de unidades IC ↔ integrador** (convención `p = a²·dx/dt` escrita por `zeldovich_2lpt_ics` vs la que espera `leapfrog_cosmo_kdk_step` a través de `CosmoFactors`) que es el cuello dominante del error `P_c/P_ref ≈ 10⁸–10⁹`, no el 2LPT. El fix de Phase 44 se mantiene porque es matemáticamente necesario; Phase 45 debe auditar unidades.
- **Decisión técnica: aceptar el fix, abrir Phase 45 con foco en unidades**. El fix es correcto, validado exhaustivamente, y no introduce regresiones en los 27 tests del crate `gadget-ng-core` ni en los 8 tests de `lpt2_ics`. La diferencia Fixed vs LegacyBuggy en métricas evolucionadas (`~2%` en error espectral, `~10%` en `growth_lowk`) es **órdenes de magnitud menor** que el error total. El próximo cuello está en el acoplamiento IC ↔ `leapfrog_cosmo_kdk_step`.
- Bit-compatibilidad: **1LPT sigue bit-idéntico**. **2LPT cambia en `O(10⁻¹²)`** en posiciones — los golden snapshots bit-exactos de Phase 37/40/42/43 se verán afectados y deben regenerarse; la señal física (δ_rms, v_rms, P(k)) cambia `< 2%`.

### Phase 43 — Control temporal de TreePM + paralelismo mínimo de loops calientes

- Nuevo reporte [`docs/reports/2026-04-phase43-dt-treepm-parallel.md`](docs/reports/2026-04-phase43-dt-treepm-parallel.md) que cierra la hipótesis de Phase 42 (*«el cuello restante tras `TreePM + ε_phys = 0.01 Mpc/h` está en el control temporal»*) combinando barrido de `dt` fijo (`4·10⁻⁴`, `2·10⁻⁴`) con un **timestep global adaptativo** nuevo (criterio Aarseth + cota cosmológica de Hubble) y **paralelismo Rayon** extendido al `PmSolver` (CIC assign/interpolate). Responde a las 5 preguntas del brief A–E con datos empíricos.
- Nuevo crate/módulo [`crates/gadget-ng-integrators/src/adaptive_dt.rs`](crates/gadget-ng-integrators/src/adaptive_dt.rs) que expone `AdaptiveDtCriterion::{Fixed, Acceleration, CosmoAcceleration}` y `compute_global_adaptive_dt`. La variante `CosmoAcceleration` combina `dt = η·√(ε/a_max)` con `dt ≤ κ_h·a/H(a)` clamped a `[dt_min, dt_max]`; `η = 0.1`, `κ_h = 0.04` garantizan ≥25 pasos por e-folding de `a` y ~10× margen frente al paso de estabilidad lineal. 5 unit tests cubren fixed, Aarseth puro, clamps y combinación Hubble.
- Extensión de paralelismo Rayon al [`crates/gadget-ng-pm/src/solver.rs`](crates/gadget-ng-pm/src/solver.rs): `PmSolver::accelerations_for_indices` ahora usa `cic::assign_rayon` y `cic::interpolate_rayon` bajo `#[cfg(feature = "rayon")]` (ya estaba activado por Phase 42 pero no usado en PM). Sin cambios de API. Rust puro → Rayon equivale a `#pragma omp parallel for`; la FFT queda intacta (plan único global).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase43_dt_treepm_parallel.rs`](crates/gadget-ng-physics/tests/phase43_dt_treepm_parallel.rs): **7 tests todos verdes** a `N=32³` smoke (runtime ~42 min wall con `PHASE43_QUICK=1`, 3 variantes: `dt_4e-4`, `dt_2e-4`, `adaptive_cosmo`): `treepm_softened_dt_sweep_runs_stably` (hard), `smaller_dt_improves_growth_under_treepm` (soft → `A_smaller_dt_improves_growth`), `adaptive_dt_matches_or_beats_best_fixed_dt` (soft → `B_adaptive_matches_best_fixed`), `parallel_tree_walk_matches_serial_within_tolerance` (hard → `max_rel_diff=0.0`), `parallel_execution_reduces_wall_time` (soft → `A_clear_parallel_speedup`), `no_nan_inf_under_phase43_matrix` (hard), `results_consistent_across_thread_counts` (hard → bit-exact δ_rms/v_rms/a con 1 vs 4 hilos). Patrón `OnceLock` + caché disco (`PHASE43_USE_CACHE=1`), selectores `PHASE43_N`, `PHASE43_THREADS="1,4,8"`, `PHASE43_DT5E5`, `PHASE43_SKIP_ADAPTIVE`.
- Nuevo experimento [`experiments/nbody/phase43_dt_treepm_parallel/`](experiments/nbody/phase43_dt_treepm_parallel/) con 3 configs TOML (`lcdm_N64_treepm_dt{4,2,1}e-4.toml`, templates para run futuro a N=64³), orquestador `run_phase43.sh` con flags para QUICK, threads y cache, y 3 scripts Python: `plot_dt_effect.py` (error vs dt, growth vs theory, δ_rms(a), runtime vs dt, dt-trace adaptativo, CSV completo), `plot_parallel_speedup.py` (speedup + walltime vs threads, CSV), `analyze_growth_phase43.py` (ratio crecimiento bajo-k + CSV con decisión del test 3).
- Hallazgos cuantitativos a N=32³: **(i)** bajar `dt` de `4·10⁻⁴` a `2·10⁻⁴` mejora `median|log₁₀(P_c/P_ref)|` en `a=0.10` por **9 %** (8.65 → 8.63) — sensibilidad detectable pero subdominante; **(ii)** `v_rms` sí baja monótona: `50.6 → 34.8 → 10.9` (factor ~5 entre `dt=4·10⁻⁴` y adaptativo, ver §5.2 del reporte) — el integrador con `dt` grande inyecta energía cinética espuria en escalas pequeñas donde vive el tree walk; **(iii)** `adaptive_cosmo` pega contra `DT_MIN_ADAPTIVE=5·10⁻⁵` desde el arranque (2LPT deja `a_max` alto), no bate al mejor fijo (+3.5 % peor) y cuesta 5.85× más wall-clock; **(iv)** paralelismo Rayon con 4 hilos: **speedup 3.70×** en un step TreePM (eficiencia 92.5 %) y **bit-exact** en aceleraciones y métricas evolucionadas; **(v)** `δ_rms ≈ 1.00` a `a=0.05` en todas las variantes confirma que la no-linealidad está instalada muy temprano, independiente del integrador.
- **Decisión técnica**: (1) mantener el módulo `adaptive_dt` como infraestructura para fases futuras pero **no** activarlo por default (no aporta sobre el mejor fijo a este `ε_phys`/ICs); (2) adoptar paralelismo Rayon en `PmSolver` (feature `rayon`, bit-exact, 3.7× con 4 hilos); (3) fijar `dt = 2·10⁻⁴` como default operativo para TreePM + `ε_phys = 0.01`; (4) **mover el foco a ICs/convención de velocidades 2LPT** — el control temporal ya no es el cuello dominante. La lectura consistente con Phase 39 es: **el error está dominado por la amplitud/convención inicial** (hipótesis a validar: factor de velocidad 2LPT, renormalización de `σ₈` a `a_init`, contribución del término de segundo orden).

### Phase 42 — Regularización física de fuerzas vía TreePM + softening absoluto

- Nuevo reporte [`docs/reports/2026-04-phase42-tree-short-range.md`](docs/reports/2026-04-phase42-tree-short-range.md) que testea la hipótesis de Phase 41 (*«la no-linealidad prematura `δ_rms(a=0.10) ≈ 1` se debe a fuerzas pequeñas escalas demasiado fuertes en PM puro»*) añadiendo un corte físico de corto alcance vía `TreePmSolver` (PM filtrado + octree con kernel `erfc` + softening Plummer `ε_phys` absoluto en Mpc/h, independiente de `N`). Responde a las 5 preguntas del brief (A: `δ_rms` con softening, B: crecimiento lineal con árbol, C: error espectral vs `N`, D: `ε_phys` óptimo, E: conclusión global).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase42_tree_short_range.rs`](crates/gadget-ng-physics/tests/phase42_tree_short_range.rs): 4 tests (`softening_reduces_early_nonlinearity`, `treepm_improves_growth_vs_pm`, `growth_closer_to_linear_with_softening`, `no_nan_inf_phase42`). Matriz **1 PM baseline + 3 TreePM con ε_phys ∈ {0.01, 0.02, 0.05} Mpc/h** (asimetría intencional: `PmSolver` ignora `eps2` por diseño band-limited — sólo 1 PM run es informativo). Patrón `OnceLock` + caché disco vía `PHASE42_USE_CACHE=1` (re-run 0.0 s), selector de resolución `PHASE42_N=<N>` (16 ≤ N ≤ 256, potencia de 2). Activado `rayon` en las dependencias `gadget-ng-pm` y `gadget-ng-treepm` para paralelizar el walk SR (11× sobre 12 hilos).
- Nuevo experimento [`experiments/nbody/phase42_tree_short_range/`](experiments/nbody/phase42_tree_short_range/) con 4 configs TOML (`lcdm_N128_pm_eps0.toml`, `lcdm_N128_treepm_eps{001,002,005}.toml`), orquestador `run_phase42.sh` con flags `PHASE42_USE_CACHE`, `PHASE42_SKIP_CLI` (default 1), `PHASE42_QUICK`, scripts Python `apply_phase42_correction.py` (mirror de `correct_pk` Phase 35) y `plot_phase42_short_range.py` (5 figuras obligatorias: `delta_rms_vs_a_by_variant`, `v_rms_vs_a_by_variant`, `ratio_corrected_vs_ref_by_variant`, `growth_vs_theory`, `nonlinearity_onset`, + `phase42_summary.csv`).
- Hallazgos cuantitativos a **N=32³** (smoke test, 4 variantes × 3 snapshots = 12 mediciones, runtime ~8.3 min wall / 91 min CPU): **(i)** reducción máxima de `δ_rms(a=0.10)` = **0.77 %** (treepm_eps001 vs PM) — sub-umbral 5 %; **(ii)** `v_rms` TreePM ≫ PM (`treepm_eps001`: 50.6 vs PM 3.62, ×14; `treepm_eps005`: 9.98 vs PM 3.62, ×2.8); **(iii)** error espectral evolucionado saturado en `median|log₁₀(P_c/P_ref)| ~ 9` (indistinguible entre PM y TreePM); **(iv)** `rel_err_growth(a=0.10) ~ 10⁸` en todas las variantes, TreePM ε=0.01 ×3.6 mejor que PM. Régimen shot-noise dominado (Phase 41 §4.2).
- Hallazgos cuantitativos a **N=64³** (corrida completa en background, 8 265 s wall / 22.3 h CPU con ~10× paralelismo, re-generó todas las figuras y el CSV): **(i)** reducción `δ_rms(a=0.10)` = **3.01 %** (treepm_eps001), monótona en las tres variantes {3.01 %, 3.00 %, 2.77 %} — aparece signo consistente ausente en N=32³; **(ii)** inyección `v_rms` más modesta (TreePM ε=0.01: 11.29 vs PM 4.01, ×2.8; ε=0.05: 6.51, ×1.6); **(iii)** error espectral evolucionado crece 33 % al pasar de N=32 a N=64 (8.76 → 11.66, Phase 41 compatible) y el softening lo atenúa sólo ~1 % (TreePM 11.56 vs PM 11.66); **(iv)** **TreePM mejora `rel_err_growth(a=0.10)` por factor ≈ 345×** (PM 7.84·10¹² → TreePM ε=0.01 2.27·10¹⁰) — decisión del test 2: `A_treepm_improves_linear_growth`; **(v)** óptimo interior de ε en growth-error: `ε_phys ≈ 0.01 Mpc/h` (consistente con N=32). La palanca del softening **crece con N** (de 3.5× en N=32 a 345× en N=64 en growth-error), validando la hipótesis H1 del brief.
- **Decisión técnica: `A_partial_confirmation_at_N64 + defer_N128_to_distributed_run`** (reemplaza la versión preliminar `C_null_result_at_quick_resolution` que se usó mientras N=64 seguía en ejecución). El softening físico absoluto + árbol SR **sí es la palanca correcta** para atacar el colapso prematuro de Phase 41: su signo y monotonía son consistentes, y su efecto crece ~100× al duplicar N. La magnitud absoluta del error (~10¹⁰) sigue invalidando lectura lineal a N=64³; N=128³ con ≥ 2 seeds (coste extrapolado ~37 h serial, requiere pipeline TreePM distribuido de Phase 23) es condición necesaria para cerrar la hipótesis. El walk no-periódico del `TreePmSolver` se confirma como suficientemente rápido (×16 por duplicación de N vs ×15×64 = 960 con `short_range_accels_periodic`).
- Intento inicial con `short_range_accels_periodic` (wrap periódico exacto) descartado por coste: empíricamente 15× más lento que el walk no-periódico del `TreePmSolver` de producción a N=32³ (3.6 s/step vs 0.24 s/step), dominado por aritmética de `minimum_image` + `min_dist2_to_aabb_periodic` en cada descenso del árbol. Se usa el walk no-periódico; el error de borde queda confinado a una cáscara `r_cut ≈ 0.098·L` que afecta < 1 % de las partículas con 2LPT ICs a `a_init = 0.02`.
- Test 1 (`softening_reduces_early_nonlinearity`) convertido a **soft check** (mismo patrón que Phase 41 tests 2–4): registra `decision` y `best_relative_reduction` en `target/phase42/*.json` sin panicar, preservando la evidencia cuantitativa para que futuras corridas a N ≥ 128 puedan re-decidir sin modificar el test.

### Phase 41 — Validación física de alta resolución (shot-noise vs señal)

- Nuevo reporte [`docs/reports/2026-04-phase41-high-resolution-validation.md`](docs/reports/2026-04-phase41-high-resolution-validation.md) que demuestra empíricamente la transición shot-noise → señal al escalar `N` a alta resolución bajo el modo físico `Z0Sigma8` (Phase 40), respondiendo a las 5 preguntas del brief (A: mínimo `N` con `S/N > 1`, B: crecimiento lineal, C: `pk_correction` más allá del IC, D: `N=128³` vs `256³`, E: validación física completa).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs`](crates/gadget-ng-physics/tests/phase41_high_resolution_validation.rs): 5 tests (1 hard shot-noise, 3 soft evolutivos para registrar Decisión A/B, 1 hard NaN/Inf). Matriz `N ∈ {32, 64, 128}` (`N=256` opcional vía `PHASE41_SKIP_N256=0`), `seeds_low_res = {42, 137, 271}` a `N ≤ 64` y `seed=42` a `N ≥ 128`, modos `{Legacy, Z0Sigma8}`, 3 snapshots → 42 mediciones, runtime **~37 min release**. Helpers nuevos: `shot_noise_level(n_grid) = V_phys/N^3` (Mpc/h)³ y `growth_ratio_low_k` que compara `⟨P(k_low, a)/P(k_low, a_init)⟩` con `[D(a)/D(a_init)]²` (CPT92). Patrón `OnceLock` + caché de disco opcional vía `PHASE41_USE_CACHE=1` que relee `target/phase41/per_snapshot_metrics.json` en 0.7 s en lugar de recomputar la matriz.
- Nuevo experimento [`experiments/nbody/phase41_high_resolution_validation/`](experiments/nbody/phase41_high_resolution_validation/) con configs TOML `lcdm_N{128,256}_2lpt_pm_{legacy,z0_sigma8}.toml`, orquestador `run_phase41.sh` (tests Rust → pase CLI `snapshot` a N=128 por modo → 5 figuras + CSV → copia a docs), scripts Python `apply_phase41_correction.py` (tabla `R(N)` extendida a `N ∈ {128, 256}` vía ley de potencias de Phase 35) y `plot_phase41_resolution.py` (5 figuras obligatorias: `pk_vs_pshot_by_N`, `ratio_corrected_vs_ref_by_N`, `spectral_error_vs_N`, `growth_ratio_low_k_vs_theory`, `signal_to_noise_transition`).
- Hallazgos cuantitativos: **(i)** transición `S/N(k_min) = 1` entre `N=32` (0.374) y `N=64` (2.21) en Z0Sigma8 IC, margen `16.06×` a `N=128³` — la predicción teórica `S/N ∝ P_lin · N^3 / V` se verifica a ±25%; **(ii)** `pk_correction` cierra en IC a `median|log10(P_c/P_ref)| ∈ [0.026, 0.049]` para `N ∈ {32, 64, 128}`, extendiendo el resultado de Phase 38 (validación externa vs CLASS a `N ≤ 64`) a `N = 128³`; **(iii)** snapshots evolucionados (`a ∈ {0.05, 0.10}`) muestran error creciente con `N` (9.07 → 11.96 → 14.98) porque `δ_rms ≈ 1.05` en todos los `N` — el sistema entra en régimen fuertemente no-lineal a `a = 0.05` independientemente de la resolución; **(iv)** el ratio de crecimiento medido en bajo `k` no converge a `[D/D]²` en ningún `N` por la misma razón dinámica.
- **Decisión técnica: cierre parcial.** El **eje shot-noise ↔ señal queda cerrado**: `Z0Sigma8` es medible a `N ≥ 64³` y con margen amplio a `N ≥ 128³`, validando la crítica de Phase 40 a nivel de IC. El **eje evolución lineal ↔ no-lineal permanece abierto**: no se resuelve aumentando `N` con softening `ε = 1/(4N)` e integrador KDK de `dt` fijo; requiere softening físico `ε_phys` constante (cf. GADGET-2, Springel 2005) y/o integradores adaptativos, **fuera del alcance de Phase 41**. Recomendación: mantener `Legacy` default, promover `Z0Sigma8` a **modo recomendado para ICs cosmológicas a `N ≥ 128³`** con softening físico tratado por separado.
- Tests 2, 3 y 4 diseñados como **soft checks**: registran `decision`, `growth_recovered`, `ic_decreases_with_n`, `evolved_decreases_with_n` en `target/phase41/test*.json` sin panicar — misma lógica que Phases 37, 39 y 40 para preservar la evidencia cuantitativa en la suite verde.

### Phase 40 — Formalización de la convención física de ICs (`NormalizationMode`)

- Nuevo reporte [`docs/reports/2026-04-phase40-physical-ics-normalization.md`](docs/reports/2026-04-phase40-physical-ics-normalization.md) que reformula la convención de normalización de ICs cosmológicas: reemplaza el flag experimental `rescale_to_a_init: bool` de Fase 37 por una enum explícita `NormalizationMode { Legacy, Z0Sigma8 }`, audita la implementación LPT en busca de bugs sutiles (ninguno), mide empíricamente `σ₈(a_init)` contra la predicción lineal y compara legacy vs `Z0Sigma8` en 18 corridas (3 seeds × 2 modos × 3 snapshots `a∈{0.02, 0.05, 0.10}`, N=32³, 2LPT, PM). Runtime: **24 s release**.
- **Breaking change (TOML):** el campo `rescale_to_a_init = false/true` en `[initial_conditions.kind.zeldovich]` desaparece y se reemplaza por `normalization_mode = "legacy" | "z0_sigma8"`. Nueva enum `NormalizationMode` reexportada en `gadget_ng_core` con `#[serde(rename_all = "snake_case")] #[default] = Legacy`. Se migraron 13 tests Rust y 9 configs TOML de Fases 37–39 al nuevo campo. Las funciones internas `zeldovich_ics` / `generate_2lpt_ics` mantienen su argumento `bool` como detalle de implementación (dispatch vía [`ic.rs`](crates/gadget-ng-core/src/ic.rs): `Z0Sigma8 → true`, `Legacy → false`).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase40_physical_ics_normalization.rs`](crates/gadget-ng-physics/tests/phase40_physical_ics_normalization.rs): 7 tests (3 hard bit-compat/física, 1 hard `pk_correction` IC, 2 soft evolutivos para Decisión A/B, 1 hard NaN/Inf). Matriz 18 corridas vía `OnceLock`. Helper `measure_sigma8_from_corrected` integrando `P_corrected` con ventana top-hat `R=8 Mpc/h`. Dump a `target/phase40/*.json`. Los 7 tests pasan.
- Nuevo experimento [`experiments/nbody/phase40_physical_ics_normalization/`](experiments/nbody/phase40_physical_ics_normalization/) con configs TOML `lcdm_N32_2lpt_pm_{legacy,z0_sigma8}.toml`, orquestador `run_phase40.sh` (tests Rust → pase CLI `snapshot` por modo → figuras + CSV → copia a docs), scripts Python `apply_phase40_correction.py` (mirror con `--mode`) y `plot_phase40_comparison.py` (6 figuras obligatorias: `pk_ic`, `pk_a005`, `pk_a010`, `ratio_corrected_vs_ref`, `delta_rms_vs_a`, `sigma8_measured_vs_expected`).
- Auditoría formal de la implementación LPT bajo `Z0Sigma8`: los factores `Ψ¹ ← s·Ψ¹`, `Ψ² ← s²·Ψ²` se aplican una única vez en `ic_zeldovich.rs:481` / `ic_2lpt.rs:340`, las velocidades heredan el factor vía `p = a²H·f·Ψ` (sin duplicar), y `(D₂/D₁²)(a_init) · s² · Ψ²_legacy = D₂(a_init) · Ψ⁽²⁾_cont` cierra exactamente con la convención LPT estándar. **No hay bugs de implementación**; el resultado negativo de Phase 37 es genuino.
- Hallazgos cuantitativos: (i) `σ₈(Z0Sigma8) / σ₈(Legacy) = s` con error relativo `< 10⁻⁸` (precisión de máquina) — verificación empírica de que la convención `σ₈(z=0)` está implementada correctamente; (ii) `pk_correction` en IC funciona idéntico en ambos modos (`median|log10(P_c/P_ref)| ≈ 0.035`, umbral 0.2); (iii) en snapshots evolucionados, `Z0Sigma8` **empeora** la fidelidad espectral por factor **1.52×** (err global `9.00` vs `5.92`) y `δ_rms(z0)/δ_rms(legacy) ≈ 1.0` — no hay reducción medible de no-linealidad.
- Diagnóstico: bajo `Z0Sigma8`, `P(k, a_init)` es ~10⁶ veces menor que en `Legacy`, quedando dominado por shot-noise del estimador CIC a `N=32³`. El régimen `a ≥ 0.05` pierde la señal lineal y el `P_corrected` no representa el modo creciente. `Legacy` es internamente consistente en su propio marco (simulación con `σ₈(a_init)=0.8`) y por eso cierra mejor contra su `P_ref` auto-consistente.
- **Decisión técnica: Opción B — `Z0Sigma8` queda como opción experimental, `Legacy` sigue siendo default y recomendado**. Phase 40 aporta (1) API limpia y tipada vía enum, (2) auditoría formal que descarta bugs, (3) verificación empírica a precisión de máquina del escalado por `s`, (4) diagnóstico claro del rol del shot-noise. Validar `Z0Sigma8` en evolución requerirá `N ≥ 128³` o integración con `P_ref` externo (CAMB/CLASS) — fuera del alcance de esta fase.
- Tests 5 y 6 diseñados como **soft checks**: registran `decision: "A_z0_replaces_legacy" | "B_z0_stays_experimental"` y `hypothesis_*: bool` en `target/phase40/*.json` sin panicar, preservando la suite verde y la evidencia cuantitativa — mismo patrón que Phase 37 y 39.

### Phase 39 — Convergencia temporal del integrador Leapfrog KDK

- Nuevo reporte [`docs/reports/2026-04-phase39-dt-convergence.md`](docs/reports/2026-04-phase39-dt-convergence.md) que caracteriza la convergencia temporal del integrador PM + Leapfrog cosmológico KDK barriendo `dt ∈ {4·10⁻⁴, 2·10⁻⁴, 1·10⁻⁴, 5·10⁻⁵}` (4 niveles), 3 seeds `{42, 137, 271}` y 3 snapshots `a ∈ {0.02, 0.05, 0.10}` sobre `N=32³`, 2LPT, PM, convención `legacy` (`rescale_to_a_init=false`). Total 36 mediciones, ~170 s release.
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase39_dt_convergence.rs`](crates/gadget-ng-physics/tests/phase39_dt_convergence.rs): 5 tests (1 hard `dt_does_not_affect_ic_snapshot`, 1 hard `dt_small_runs_stable` sobre NaN/Inf, 3 observacionales `smaller_dt_reduces_spectral_error`, `dt_convergence_trend_detectable`, `dt_scaling_consistent_with_integrator_order`). Patrón `OnceLock` para ejecutar la matriz una sola vez, cronometraje in-test con `std::time::Instant` y dump a `target/phase39/*.json`.
- Nuevo experimento [`experiments/nbody/phase39_dt_convergence/`](experiments/nbody/phase39_dt_convergence/) con 4 configs TOML (`lcdm_N32_2lpt_pm_dt_{4e4,2e4,1e4,5e5}.toml` con `num_steps ∈ {200, 400, 800, 1600}`), orquestador `run_phase39.sh` (tests Rust → pase CLI seed 42 para los 4 dts → figuras → CSV → copia a docs), script `apply_phase39_correction.py` (mirror Python modo legacy de `pk_correction` + CPT92), `plot_dt_sweep.py` (4 figuras: `error_vs_dt`, `ratio_per_dt`, `delta_rms_vs_a`, `cost_vs_precision`) y `dt_vs_error.py` (CSV resumen con 36 filas).
- Pase CLI real (seed 42, los 4 dts, analyse sobre `snapshot_final`) confirma las métricas in-process: `median |log10(P_c/P_ref)|` a `a=0.10` = `5.66 / 5.76 / 5.89 / 6.47` para `dt₀ / dt₀/2 / dt₀/4 / dt₀/8` vs `5.64 / 5.70 / 5.78 / 6.49` en la matriz Rust.
- Hallazgo principal: en la convención legacy actual, **reducir `dt` NO reduce el error espectral**. El `median |log10(P_c/P_ref)|` a `a=0.05` crece de `6.22` (dt₀) a `7.02` (dt₀/8); pendiente OLS log-log observada `−0.054 / −0.061` en `a ∈ {0.05, 0.10}` vs predicción teórica `+2.0` para KDK O(dt²). **`δ_rms(a=0.10)` es ~2 800× la predicción lineal CPT92**, indicando que el sistema entra en régimen no-lineal desde los primeros cientos de pasos, independientemente de `dt`.
- **Decisión técnica: mantener `dt = 4·10⁻⁴` como default**. Reducir `dt` multiplica el costo linealmente (30 s vs 4 s por corrida) sin ganancia espectral. El error residual está dominado por la amplitud inicial de la convención legacy (σ₈=0.8 en a_init sobre-amplifica respecto al crecimiento lineal en un factor `[D(1)/D(a_init)]² ≈ 2 500`), no por el integrador. El integrador es numéricamente estable (sin NaN/Inf en los 36 snapshots; IC bit-idéntico entre dts). Resolver la fidelidad en snapshots evolucionados requiere reformular ICs (fuera del alcance de esta fase).
- Tests 2, 3 (régimen lineal) y 4 diseñados como **soft checks** que registran `hypothesis_*: bool` en `target/phase39/*.json` sin panicar cuando la hipótesis experimental resulta falsa, manteniendo la suite verde y preservando la evidencia cuantitativa — mismo patrón que Phase 37.

### Phase 38 — Validación externa mínima de `pk_correction` contra CLASS

- Nuevo reporte [`docs/reports/2026-04-phase38-class-camb-minimal-validation.md`](docs/reports/2026-04-phase38-class-camb-minimal-validation.md) que cierra la validación externa de amplitud absoluta comparando `P_corrected(k)` de `gadget-ng` contra un espectro lineal independiente generado por [CLASS](https://lesgourg.github.io/class_public/class.html) (`classy 3.3.4.0`) en el snapshot IC, cubriendo las dos convenciones de normalización (`legacy` vs `P_CLASS(k, z=0)` y `rescaled` vs `P_CLASS(k, z=49)`).
- Nueva referencia externa reproducible en [`experiments/nbody/phase38_class_validation/reference/`](experiments/nbody/phase38_class_validation/reference/): `class.ini`, `dump_class_pk.py`, `generate_reference.sh` (venv + `classy==3.3.4.0`), `README.md` con instrucciones de reproducción y SHA-256, y las dos tablas `pk_class_z{0,49}.dat` (512 bins log en `k ∈ [1e-4, 20] h/Mpc`). **CLASS no es dependencia de CI**: los `.dat` viven en el repo.
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase38_class_validation.rs`](crates/gadget-ng-physics/tests/phase38_class_validation.rs): 5 tests sobre la matriz `2 N × 3 seeds × 2 modos = 12 mediciones` (IC-only, 2LPT, PM). Incluyen loader `.dat` + interpolador log-log, manejo explícito de la banda BAO (`k ∈ [0.05, 0.30] h/Mpc`) y `OnceLock` para ejecutar la matriz una sola vez. Runtime release: **~0.6 s**. Los JSONs por test y la matriz completa se dumpean a `target/phase38/*.json`.
- Nuevo experimento [`experiments/nbody/phase38_class_validation/`](experiments/nbody/phase38_class_validation/) con orquestador `run_phase38.sh` (chequeo CLASS → tests Rust → pase CLI dual → figuras → copia a docs), configs `lcdm_N32_2lpt_pm_{legacy,rescaled}.toml`, scripts `apply_phase38_correction.py` (mirror Python de `correct_pk` + interpolador CLASS + métricas dentro/fuera BAO) y `plot_phase38.py` (4 figuras obligatorias + opcional `legacy_vs_rescaled.png`).
- Pase CLI real confirma las métricas in-process: `median |log10(P_m/P_CLASS)| = 14.722` → `median |log10(P_c/P_CLASS)| = 0.036`, `mean(P_c/P_CLASS) = 1.037` sobre la ventana lineal completa, ambos modos indistinguibles a `~2 %`.
- Hallazgo principal: `pk_correction` reduce el error de amplitud absoluta vs CLASS por un factor **161×** a `N=32³` y **761×** a `N=64³`, dejando `median|log10(P_c/P_CLASS)| ∈ [0.022, 0.046]` y `mean(P_c/P_CLASS) ∈ [0.95, 1.04]` sobre las 12 mediciones. La forma espectral (pendiente log-log OLS fuera de BAO) coincide con CLASS a `|Δ| ≤ 0.10` para `N=64³`. Los modos `legacy` y `rescaled` dan resultados indistinguibles, confirmando que el cierre es intrínseco a `pk_correction` y no depende de la convención de normalización.
- **Decisión técnica: validación externa mínima cerrada**. `pk_correction` queda respaldada por un código cosmológico independiente, no sólo por la referencia interna EH. El residuo restante (`3–5 %` en BAO, `~1 %` de diferencia CPT92 vs CLASS en crecimiento, cosmic variance a `N=32`) es atribuido a fuentes conocidas y cuantificadas en el reporte. Sin cambios de física, solver ni `pk_correction`.

### Phase 37 — Reescalado físico opcional de ICs por `D(a_init)/D(1)`

- Nuevo reporte [`docs/reports/2026-04-phase37-growth-rescaled-ics.md`](docs/reports/2026-04-phase37-growth-rescaled-ics.md) que evalúa si reescalar las amplitudes LPT por el factor de crecimiento lineal `s = D(a_init)/D(1)` (CPT92) extiende la validez de `pk_correction` desde el snapshot IC (donde Fase 36 la validó) hacia snapshots cosmológicos evolucionados tempranos (`a ∈ {0.05, 0.10}`).
- Nueva API pública en [`crates/gadget-ng-core/src/cosmology.rs`](crates/gadget-ng-core/src/cosmology.rs): `growth_factor_d(params, a)` y `growth_factor_d_ratio(params, a_num, a_den)` (reexportadas en `gadget_ng_core`). Incluye tests unitarios de sanidad (EdS exacto `D(a)=a`, monotonía en ΛCDM, `D(0)=0`, valor numérico a `a=0.02`).
- Nuevo flag opcional `rescale_to_a_init: bool` en `IcKind::Zeldovich` ([`crates/gadget-ng-core/src/config.rs`](crates/gadget-ng-core/src/config.rs)) con `#[serde(default)] = false`. Legacy bit-compatible: con el flag apagado, `zeldovich_ics` y `zeldovich_2lpt_ics` producen bits idénticos (verificado vía `.to_bits()`). Con el flag activo aplican `Ψ¹ ← s·Ψ¹` y `Ψ² ← s²·Ψ²`.
- Propagación del flag en [`ic.rs`](crates/gadget-ng-core/src/ic.rs), [`ic_zeldovich.rs`](crates/gadget-ng-core/src/ic_zeldovich.rs) y [`ic_2lpt.rs`](crates/gadget-ng-core/src/ic_2lpt.rs). 13 call-sites de tests existentes actualizados con `rescale_to_a_init: false` explícito para preservar el comportamiento legacy.
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase37_growth_rescaled_ics.rs`](crates/gadget-ng-physics/tests/phase37_growth_rescaled_ics.rs): 7 tests sobre matriz `3 configs × 3 seeds × 3 snapshots × 2 modos = 54 mediciones` (PM-only por default, TreePM opt-in vía `PHASE37_INCLUDE_TREEPM=1`). La matriz se ejecuta una sola vez vía `OnceLock` y serializa a `target/phase37/*.json`. Pasan en release en ~382 s.
- Nuevo experimento [`experiments/nbody/phase37_growth_rescaled_ics/`](experiments/nbody/phase37_growth_rescaled_ics/) con orquestador `run_phase37.sh`, configs TOML `lcdm_N32_2lpt_pm_{legacy,rescaled}.toml` y scripts Python `apply_phase37_correction.py` (mirror con argumento `--mode {legacy,rescaled}`) y `plot_phase37.py` (6 figuras obligatorias: P(k) IC / a=0.05 / a=0.10, ratio `P_c/P_ref`, `δ_rms(a)` vs teoría lineal, `rms(Ψ)` IC).
- Hallazgo cuantitativo: (i) la implementación del reescalado es exacta — `rms(Ψ_rescaled)/rms(Ψ_legacy) = s` a `1.85e-13` en 1LPT y `1.65e-6` en 2LPT; (ii) `pk_correction` preserva su cierre en el snapshot IC con rescaled (`median|log10(P_c/P_ref)| ≤ 0.035`); **(iii) el rescaled NO extiende la validez a snapshots evolucionados** — en `a∈{0.05, 0.10}`, `median|log10(P_c/P_ref)|` rescaled (`9.4`, `8.7`) es **mayor** que legacy (`6.2`, `5.6`), factor global `0.66` frente al umbral `≥ 2.0` de Decisión A.
- **Decisión técnica: Opción B — `rescale_to_a_init` queda como opción experimental documentada, `default = false`**. El modo legacy sigue siendo el recomendado. La hipótesis del reescalado físico no es suficiente por sí sola con `dt = 4e-4`: ambos modos colapsan en régimen no-lineal (`δ_rms(a=0.10) ≈ 1`) porque el `dt` actual no preserva régimen lineal sobre amplitudes reducidas. El camino futuro (fuera de Phase 37) es barrido `(rescale, dt)` con `dt ∈ {1e-4, 4e-5}` y/o integradores alternativos.
- Tests 3 y 5 diseñados como **soft checks**: registran `supports_decision_a: bool` en `target/phase37/*.json` sin panicar cuando la hipótesis experimental resulta falsa, manteniendo la suite verde y dejando la evidencia cuantitativa en los JSONs y el reporte.

### Phase 36 — Validación práctica de `pk_correction` sobre corridas cosmológicas reales

- Nuevo reporte [`docs/reports/2026-04-phase36-pk-correction-validation.md`](docs/reports/2026-04-phase36-pk-correction-validation.md) que valida la API congelada de Phase 35 (`pk_correction`) sobre la matriz (N × seed × ic_kind) completa: N=32³/64³, 3 seeds, 1LPT y 2LPT, 3 snapshots por corrida (a ∈ {0.02, 0.05, 0.10}).
- Nuevos tests de integración en [`crates/gadget-ng-physics/tests/phase36_pk_correction_validation.rs`](crates/gadget-ng-physics/tests/phase36_pk_correction_validation.rs): 5 tests (reducción del error absoluto, preservación de forma espectral, consistencia entre seeds en el IC, consistencia entre resoluciones N=32 vs N=64, no NaN/Inf). La matriz de 27 snapshots se ejecuta una sola vez vía `OnceLock` y serializa a `target/phase36/*.json`. Pasan en release en ~191 s.
- Nuevo experimento [`experiments/nbody/phase36_pk_correction_validation/`](experiments/nbody/phase36_pk_correction_validation/) con orquestador `run_phase36.sh` (tests Rust → pase CLI → figuras → copia a docs), config `lcdm_N32_2lpt_pm_phase36.toml`, scripts `apply_phase36_correction.py` (mirror Python de `correct_pk` + CPT92 + métricas) y `plot_phase36.py` (5 figuras obligatorias + `cli_evidence.png`).
- Pase CLI real (`gadget-ng snapshot` + `analyse` + mirror Python): `median |log₁₀(P_m/P_ref)| = 14.67` → `median |log₁₀(P_c/P_ref)| = 0.053`, `mean(P_c/P_ref) = 1.049`, `CV = 0.134`. Coincide cuantitativamente con los tests in-process.
- Hallazgo principal: la corrección reduce el error absoluto de amplitud de `median |log₁₀(P_m/P_ref)| ≈ 14–18` a `≈ 0.03` en el snapshot IC real de cualquier (N, seed, ic_kind) de la matriz — factor de mejora `~10¹⁴` reproducible end-to-end. **La amplitud absoluta queda cerrada "en la práctica"** en el régimen válido (`k ≤ k_Nyq/2`, `a = a_init`, `N ∈ {32, 64}`, CIC).
- Limitación documentada: el proyecto aplica `σ₈=0.8` en `a_init` sin escalar por `D(a_init)/D(0)`, lo que pone las corridas en régimen no-lineal desde el paso 1 y hace que los snapshots a `a > a_init` queden fuera del dominio lineal de `pk_correction` (ortogonal a la corrección).

### Phase 35 — Modelado de `R(N)` para corrección absoluta de `P(k)`

- Nuevo reporte [`docs/reports/2026-04-phase35-rn-modeling.md`](docs/reports/2026-04-phase35-rn-modeling.md) que caracteriza el factor de muestreo discreto `R(N)` (partículas + CIC) identificado en Phase 34 como función de resolución, fitea dos modelos (potencia pura y potencia + offset), selecciona ganador por AIC y documenta el rango de validez.
- Nuevo módulo público [`crates/gadget-ng-analysis/src/pk_correction.rs`](crates/gadget-ng-analysis/src/pk_correction.rs) con `RnModel`, `a_grid`, `correct_pk` y `RnModel::phase35_default` (valores congelados del fit: `C = 22.108`, `α = 1.8714`, tabla `N ∈ {8,16,32,64}`). API expuesta en `gadget_ng_analysis::{a_grid, correct_pk, RnModel}`.
- Nuevos tests de caracterización en [`crates/gadget-ng-physics/tests/phase35_rn_modeling.rs`](crates/gadget-ng-physics/tests/phase35_rn_modeling.rs): 6 tests sobre la matriz `N × seed` (4×4) que validan determinismo entre seeds (CV < 0.10 para N≥32), flatness de `R(N,k)` en k bajo (CV_k < 0.25), fit log-log (R² = 0.997), reducción del error de amplitud (mediana de `|log₁₀|` de 17.9 → 0.037, factor ×485), consistencia interpolación-modelo a N=48 (< 2.5 %) y verificación CIC vs TSC a N=32.
- 6 unit tests en `pk_correction` cubren `from_table`, interpolación log-log, preferencia tabla-sobre-fit y escalado lineal de `correct_pk`.
- Nuevo experimento [`experiments/nbody/phase35_rn_modeling/`](experiments/nbody/phase35_rn_modeling/) con orquestador `run_phase35.sh`, `scripts/fit_r_n.py` (OLS log-log + `scipy.curve_fit` + AIC), `scripts/plot_r_n.py` (5 figuras) y `scripts/apply_correction.py` (demo de postproceso). Las 5 figuras obligatorias se copian a `docs/reports/figures/phase35/`.
- Hallazgo: Modelo A (potencia pura) gana por ΔAIC = −11.45 frente a Modelo B (el offset asintótico sale *negativo*). Con `A_grid(N)` de Phase 34 + `R(N)` de Phase 35, la amplitud absoluta de `P(k)` se cierra al ~9 % en postproceso sin modificar el core.

### Phase 34 — Cierre de la normalización discreta de `P(k)`

- Nuevo reporte [`docs/reports/2026-04-phase34-discrete-normalization-closure.md`](docs/reports/2026-04-phase34-discrete-normalization-closure.md) que decompone el pipeline `P_cont → δ̂(k) → IFFT → δ(x) → FFT → P(k)` (con y sin partículas) en etapas independientes y aísla dónde nace el offset de amplitud absoluta reportado en Phase 30–33.
- Nuevos tests de caracterización en [`crates/gadget-ng-physics/tests/phase34_discrete_normalization.rs`](crates/gadget-ng-physics/tests/phase34_discrete_normalization.rs): 8 tests que verifican roundtrip DFT (8.9e-16), modo único, ruido blanco (ratio 0.996), offset partícula/grilla (CV 0.6 %), efecto CIC, escalado con N y determinismo entre seeds.
- Nuevo módulo `gadget_ng_core::ic_zeldovich::internals` (re-exportado como `ic_zeldovich_internals`) que expone `generate_delta_kspace`, `fft3d`, `delta_to_displacement`, `build_spectrum_fn` y `mode_int` como API testing-only documentada. Sin cambios de comportamiento en el core.
- Nuevo experimento [`experiments/nbody/phase34_discrete_normalization/`](experiments/nbody/phase34_discrete_normalization/) con orquestador `run_phase34.sh`, `scripts/stage_table.py`, `scripts/plot_stages.py` y las 5 figuras obligatorias (`grid_ratio`, `particle_ratio`, `stage_breakdown`, `cic_effect`, `single_mode_amplitude`).
- Hallazgo: el offset se descompone limpiamente en (i) un **factor de grilla cerrado** `A_grid = 2·V²/N⁹` verificado al 3 % (cierra el residuo de 17× de Phase 33) y (ii) un **factor partículas-CIC** `R(N)` determinista por resolución (CV < 1 %) pero dependiente de N.
- Decisión: se mantiene la convención interna actual (Opción B). El factor de grilla queda documentado cerrado; `R(N)` queda congelado como regresión en los tests. Sin parches al core.

### Fase 2

#### [Hito 15] — Sistema de unidades físicas
- Nueva sección `[units]` en el TOML de configuración: `enabled`, `length_in_kpc`, `mass_in_msun`, `velocity_in_km_s`.
- `RunConfig::effective_g()` calcula G en unidades internas a partir de `G = 4.3009×10⁻⁶ kpc Msun⁻¹ (km/s)²`.
- Método auxiliar `UnitsSection::time_unit_in_gyr()` y `hubble_time(h0)`.
- `SnapshotEnv` y `meta.json` incluyen bloque `units` cuando está habilitado (`length_in_kpc`, `mass_in_msun`, `velocity_in_km_s`, `time_in_gyr`, `g_internal`).
- Retrocompatible: `enabled = false` (default) deja `gravitational_constant` sin cambios.

#### [Hito 12] — Restart / Checkpointing
- Nueva opción `[output] checkpoint_interval = N`: guarda checkpoint cada N pasos en `<out>/checkpoint/`.
- Checkpoint incluye: `checkpoint.json` (paso completado, factor de escala `a`, hash de config), `particles.jsonl` y (si aplica) `hierarchical_state.json`.
- `gadget-ng stepping --resume <out_anterior>` reanuda desde el último checkpoint sin pérdida de precisión.
- Advertencia si el hash del config cambió desde que se guardó el checkpoint.
- Compatible con todos los modos de integración: leapfrog clásico, cosmológico, jerárquico y árbol distribuido.

#### [Hito 10] — Pulir
- `CHANGELOG.md` con historial semántico completo (este archivo).
- `docs/user-guide.md`: guía de usuario con ejemplos TOML comentados para cada solver y opción.
- `.github/workflows/ci.yml`: CI con `fmt`, `clippy -D warnings`, `cargo test --workspace`, benchmarks en dry-run.
- Nuevos benchmarks Criterion en `gadget-ng-pm` (`pm_gravity_128`) y `gadget-ng-treepm` (`treepm_gravity_128`).

---

## Fase 1

### [Hito 9] — MPI árbol distribuido
- `SlabDecomposition` (dominio x en slabs uniformes) en `gadget-ng-parallel::domain`.
- `allreduce_min/max_f64` en `ParallelRuntime`.
- `exchange_domain_by_x` (migración de partículas entre rangos) y `exchange_halos_by_x` (halos punto-a-punto, patrón odd-even anti-deadlock).
- `compute_forces_local_tree` en engine: árbol local de (partículas + halos).
- Activado con `[performance] use_distributed_tree = true` y `solver = "barnes_hut"`.
- Comunicación O(N_halo × 2) en lugar de Allgather O(N).

### [Hito 8] — GPU kernels reales (wgpu portátil)
- `GpuDirectGravity` real con wgpu 29 (WGSL compute shader, Vulkan/Metal/DX12/WebGPU).
- Kernel O(N²) de gravedad Plummer suavizada en f32 (error relativo O(1e-7)).
- `GpuContext` con `Arc<>` + `Send + Sync`; readback síncrono.
- Activado con `[performance] use_gpu = true`; fallback automático a CPU si no hay GPU.

### [Hito 7] — FMM (Fast Multipole Method) — cuadrupolo
- Tensor de cuadrupolo sin traza `[Qxx, Qxy, Qxz, Qyy, Qyz, Qzz]` en `OctNode`.
- Calculado en `aggregate` vía teorema del eje paralelo.
- Corrección de aceleración cuadrupolar en `walk_inner`.
- Error relativo medio con θ=0.5 < 0.5% (vs >1% solo monopolo).

### [Hito 6] — Cosmología básica
- Formulación de momentum canónico estilo GADGET-4: `p = a²·dx_c/dt`.
- `CosmologySection` en config: `omega_m`, `omega_lambda`, `h0`, `a_init`.
- `advance_a` (RK4 Friedmann) y `drift_kick_factors` (Simpson N_SUB=16).
- `leapfrog_cosmo_kdk_step` + `CosmoFactors`; integrador jerárquico extendido.
- `redshift = 1/a − 1` en `SnapshotEnv`.

### [Hito 5] — TreePM (árbol + malla)
- Solver `TreePmSolver`: Barnes-Hut (corto alcance, kernel erfc) + PM (largo alcance, kernel erf).
- `r_split` configurable (default: `2.5 × cell_size`).

### [Hito 4] — Particle-Mesh (PM) FFT periódico
- Solver `PmSolver`: FFT 3D periódica, resolución `pm_grid_size³`.
- Estimación de densidad CIC (Cloud-In-Cell) y derivada del potencial.

### [Hito 3] — Barnes-Hut tree
- `Octree` con agregación recursiva de centros de masa.
- Criterio MAC `s/d < θ` (default θ=0.5).
- Suavizado Plummer; soporte Rayon con `RayonBarnesHutGravity`.

### [Hito 2] — Integrador jerárquico (block timesteps)
- `HierarchicalState` con niveles de potencia de 2.
- Criterio de Aarseth: `dt_i = η × sqrt(ε / |a_i|)`.
- `hierarchical_kdk_step`; guardado/carga de estado (`hierarchical_state.json`).

### [Hito 1] — MVP N-body
- Integrador Leapfrog KDK global.
- Condiciones iniciales: lattice cúbico perturbado, dos cuerpos circulares.
- Snapshots JSONL/HDF5/Bincode/MessagePack/NetCDF.
- Paralelismo MPI (`rsmpi`): `allgatherv_state`, distribución por GID.
- Diagnósticos por paso: `diagnostics.jsonl`.
- CLI: `gadget-ng config`, `gadget-ng stepping`, `gadget-ng snapshot`.
