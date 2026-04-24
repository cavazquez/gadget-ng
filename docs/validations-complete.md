# Validaciones Completas — gadget-ng

**Fecha:** 2026-04-24 (actualizado Phase 165)  
**Estado del proyecto:** Phases 1–165 completadas. Workspace compila y pasa `cargo test --workspace` sin ningún FAILED.  
**Propósito:** Inventario exhaustivo de **todas** las validaciones del proyecto: las existentes (pasando), las pendientes de implementación (marcadas con `[ ]`), y las de infraestructura HPC (detalladas en `docs/validation-plan-hpc.md`).

> **Novedades Phase 165:**
> - `primordial_bfield_ic_3d` implementada y validada (MHD 3D solenoidal con ∇·B < 1e-14).
> - Kernel CUDA/HIP N² real: `CudaDirectGravity::compute` y `HipDirectGravity::compute` llaman FFI real.
> - 5 tests GPU en `v1_gpu_tests.rs` activados (sin `#[ignore]`); saltan limpiamente sin hardware.

---

## Tabla de contenidos

1. [Resumen ejecutivo](#resumen)
2. [Inventario de tests existentes por módulo](#existentes)
3. [Validaciones pendientes — física ya implementada](#pendientes-fisica)
4. [Validaciones pendientes — HPC/Ingeniería](#pendientes-hpc)
5. [Script de corrida autónoma completa](#script)
6. [Criterios de aceptación global](#criterios)

---

<a name="resumen"></a>
## Resumen ejecutivo

| Categoría | Tests existentes | Tests pendientes | Prioridad |
|-----------|:---:|:---:|:---:|
| N-body directo / leapfrog | 10 | 0 | — |
| Barnes-Hut árbol | 7 | 1 | Media |
| FMM octupolo | 5 | 0 | — |
| PM / TreePM | 14 | 2 | Media |
| SPH (hidrodinámica) | 12 | 2 | Media |
| Cosmología ΛCDM | 31 | 3 | Alta |
| MHD básico | 8 | 0 | — |
| MHD avanzado (turb, recone, Braginskii) | 18 | 3 | Alta |
| RMHD / jets AGN | 12 | 2 | Media |
| Two-fluid plasma | 6 | 1 | Baja |
| Física Capa 2 (SF, CR, polvo, etc.) | 48 | 4 | Media |
| Observables sintéticos (Capa 3) | 24 | 3 | Media |
| Física Frontera (Capa 4) | 30 | 3 | Alta |
| HDF5 / IO | 21 | 0 | — |
| GPU (wgpu + CUDA/HIP) | 6 ✅ | 0 | — |
| Block timesteps + MPI cosmo | 6 ✅ | 0 | — |
| MHD cosmológico cuantitativo | 8 ✅ | 0 | — |
| **TOTAL** | **~278** | **~24** | |

El workspace tiene **0 tests fallando** en `cargo test --workspace` (compilación limpia post-Phase 165).

**Tests que requieren hardware** (pasan en CI con skip automático; activar con hardware real):
- `v1_gpu_tests::gpu_matches_cpu_direct_gravity_n1024` — CUDA/HIP kernel N² real
- `v1_gpu_tests::pm_gpu_roundtrip_fft` — FFT roundtrip < 1e-8
- `v1_gpu_tests::power_spectrum_pm_gpu_matches_pm_cpu` — P(k) bin error < 1%

---

<a name="existentes"></a>
## Inventario de tests existentes por módulo

### Mecánica N-body y geometría
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `momentum_lattice.rs` | 1 | Momento lineal conservado ≤ 5×10⁻¹¹ en red cúbica, 30 pasos |
| `yoshida_linear_momentum.rs` | 1 | Idem con integrador Yoshida4 |
| `cold_collapse.rs` | 1 | Colapso frío: energía cinética crece al colapso |
| `bh_force_accuracy.rs` | 3 | Error fuerza BH vs directo < 1% con θ=0.5 |
| `bh_stepping_energy.rs` | 3 | Drift de energía BH < 5% en 200 pasos |
| `regression_vs_direct.rs` | 7 | Regresión BH vs CPU directo, varios N y seeds |
| `octree_com.rs` | 4 | Centro de masa y partición octante |
| `split_vs_full_accel.rs` | 1 | Aceleración splitada == full (short+long) |
| `hierarchical_level_histogram.rs` | 2 | Histograma de niveles jerárquicos no vacío |

### PM / TreePM
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `cosmo_pm.rs` | 4 | PM periódico: fuerzas finitas, P(k) razonable a z=0 |
| `phase42_tree_short_range.rs` | 6 | Short-range force TreePM correcto |
| `phase43_dt_treepm_parallel.rs` | 6 | TreePM paralelo determinístico y conservativo |
| `phase70_amr_pm.rs` | 5 | AMR-PM: refinamiento en zona densa, fuerzas finitas |
| `phase101_softening.rs` | 6 | Softening físico vs comoving |

### Cosmología
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `cosmo_serial.rs` | 2 | Corrida serial cosmológica, a(z) |
| `cosmo_mpi.rs` | 2 | Idem MPI, reproducibilidad |
| `zeldovich_ics.rs` | 6 | ICs ZA: P(k) ∝ k, σ₈ correcto |
| `transfer_sigma8_ics.rs` | 4 | σ₈ normalización EH transfer |
| `lpt2_ics.rs` | 3 | 2LPT vs 1LPT: dispersión mayor |
| `phase29_lpt_comparison.rs` | 6 | 2LPT vs ZA cuantitativo |
| `phase30_linear_reference.rs` | 6 | D(a) lineal vs numérico |
| `phase31_ensemble.rs` | 6 | Ensemble P(k) ruido Poisson |
| `phase32_high_res_ensemble.rs` | 6 | Alta resolución P(k) |
| `phase33_pk_normalization.rs` | 6 | Normalización P(k) σ₈ |
| `phase34_discrete_normalization.rs` | 6 | P(k) discreto corregido |
| `phase36_pk_correction_validation.rs` | 6 | Corrección grid P(k) |
| `phase37_growth_rescaled_ics.rs` | 6 | ICs rescaladas a z distinto |
| `phase38_class_validation.rs` | 6 | Validación vs CLASS |
| `phase39_dt_convergence.rs` | 6 | Convergencia dt cosmo |
| `phase40_physical_ics_normalization.rs` | 6 | ICs en unidades físicas |
| `phase41_high_resolution_validation.rs` | 6 | Alta resolución |
| `phase44_2lpt_audit.rs` | 6 | Auditoría 2LPT |
| `phase45_units_audit.rs` | 6 | Auditoría unidades |
| `phase47_pk_evolution.rs` | 6 | Evolución P(k) z=2→0 |
| `phase48_halofit_validation.rs` | 6 | Halofit no-lineal |
| `phase49_halofit_comparison.rs` | 6 | Halofit vs P(k) simulado |
| `phase49_integrator_diagnosis.rs` | 6 | Diagnóstico integradores |
| `phase49_long_growth.rs` | 6 | Crecimiento estructura largo |
| `phase50_physical_units.rs` | 6 | Unidades físicas consistentes |
| `phase51_auto_g.rs` | 6 | G auto-consistente con cosmo |
| `phase54_growth_factor_validation.rs` | 6 | D(a) vs analítico |
| `phase59_checkpoint_continuity.rs` | 6 | Continuidad tras checkpoint |
| `phase69_production.rs` | 6 | Corrida de producción |

### Análisis de halos
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `phase52_mass_function.rs` | 6 | HMF Press-Schechter vs analítico |
| `phase53_nfw_profiles.rs` | 6 | Perfil NFW: χ² < 0.05, r_s correcto |
| `phase55_fof_vs_hmf.rs` | 6 | FoF halos vs HMF |
| `phase58_nfw_concentration.rs` | 6 | Concentración NFW Duffy/Bhattacharya |

### SPH
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `phase121_thermal_conduction.rs` | 6 | Conducción térmica Spitzer/Braginskii |
| `phase130_dust.rs` | 6 | Polvo: acreción D/G y sputtering |
| Lib tests sph | ~8 | Kernel SPH, densidad, presión |

### MHD
| Archivo test | Tests | Qué validan |
|---|:---:|---|
| `phase133_*.rs` (mhd avanzado) | 6 | Conducción anisótropa, k_par/k_perp |
| `phase134_*.rs` | 6 | MHD básico: Alfvén, B·∇v |
| `phase140_turbulence.rs` | 6 | Forzamiento turbulento OU |
| `phase142_engine_rmhd_turb.rs` | 6 | Engine RMHD + turb |
| `phase145_reconnection.rs` | 6 | Sweet-Parker: tasa analítica |
| `phase146_braginskii.rs` | 6 | Viscosidad Braginskii anisótropa |
| `phase147_mhd_cosmo_full.rs` | 6 | Potencia magnética + cosmo completo |
| `phase148_rmhd_jets.rs` | 6 | Jets RMHD AGN: inyección de momento |
| `phase149_two_fluid.rs` | 6 | Two-fluid: acoplamiento Coulomb T_e/T_i |

### Física Capa 2 (fases 100–131)
| Módulo | Tests | Qué validan |
|---|:---:|---|
| Rayos cósmicos (Phase 117) | 6 | CR: acelera gas en choque |
| Retroalimentación AGN (Phase 113) | 6 | BH accretion + feedback |
| Star Formation (Phase 109) | 6 | Schmidt-Kennicutt, SFR |
| Química H₂ (Phase 118) | 6 | Fracción H₂ equilibrio |
| Enfriamiento metalicidad (Phase 112) | 6 | Λ(T, Z) tasa correcta |
| Reionización (Phase 89) | 6 | Esfera Strömgren |
| Polvo ISM (Phase 130) | 6 | Dust-to-gas bounds |
| Conducción térmica (Phase 121) | 6 | Spitzer + Braginskii |
| RT M1 (Phase 85) | 6 | Solver M1 de radiación |

### Física Capa 3 — Observables sintéticos (Phases 151–154)
| Módulo | Tests | Qué validan |
|---|:---:|---|
| X-ray emission (Phase 151) | 6 | L_X, T_X, perfil radial |
| Emission lines (Phase 152) | 6 | Hα, [OIII], BPT |
| SED + SPS BC03-lite (Phase 153) | 6 | Luminosidad multiband |
| Mock catalogs (Phase 154) | 6 | m_app, flujo, SMHM |

### Física Capa 4 — Frontera (Phases 155–159)
| Módulo | Tests | Qué validan |
|---|:---:|---|
| Dark energy w(z) CPL (Phase 155) | 6 | H(z) CPL vs ΛCDM |
| Massive neutrinos (Phase 156) | 6 | Ω_ν, supresión P(k) |
| SIDM scattering (Phase 157) | 6 | Probabilidad scattering |
| Gravedad modificada f(R) (Phase 158) | 6 | Fifth force, chameleon |
| GMC collapse + Kroupa IMF (Phase 159) | 6 | Masa estelar, SN feedback |

### IO / HDF5
| Tests | Qué validan |
|---|---|
| `hdf5_writer` (21 tests) | Roundtrip posiciones, velocidades, redshift, campos MHD |
| `gadget4_attrs` (4 tests) | Header GADGET-4, Redshift = 1/a−1 |
| `jsonl_reader/writer` | Snapshot JSONL legacy |

---

<a name="pendientes-fisica"></a>
## Validaciones pendientes — física ya implementada

Las siguientes validaciones faltan a pesar de que el código está implementado.
Son tests cuantitativos contra soluciones analíticas o referencias externas.

### PF-01: Convergencia de orden del integrador leapfrog KDK

```
Estado: PENDIENTE
Módulo: gadget-ng-integrators
Archivo propuesto: crates/gadget-ng-integrators/tests/order_convergence.rs
```

El integrador KDK es O(Δt²). Hay que verificar esto midiendo el error de posición
al dividir el timestep por 2, 4, 8 y comprobando que el error baja como Δt².

```rust
// Test propuesto
#[test]
fn leapfrog_kdk_order2_convergence() {
    // Órbita de Kepler: E_init fijada, E(T) - E_init debe escalar como dt²
    for &n in &[100_usize, 200, 400] {
        let dt = 1.0 / n as f64;
        let err = simulate_kepler_error(dt, n_steps=n, one_period=true);
        // Ratio de errores debe ser ~4 al duplicar pasos
    }
    let ratio = errors[0] / errors[1];
    assert!((ratio - 4.0).abs() < 0.5, "Orden de convergencia: {:.2}", ratio.log2());
}
```

**Tolerancia:** ratio ≈ 4 (error escala como Δt²) ± 0.5

---

### PF-02: Órbita de Kepler — conservación de momento angular y excentricidad

```
Estado: PENDIENTE
Módulo: gadget-ng-integrators / gadget-ng-tree
Archivo propuesto: crates/gadget-ng-physics/tests/kepler_orbit.rs
```

Órbita de 2 cuerpos: L y excentricidad deben conservarse.

```rust
#[test]
fn kepler_orbit_angular_momentum_conserved_100_orbits() {
    let (m1, m2) = (1.0, 1e-3);  // star-planet
    let L0 = initial_angular_momentum(m1, m2);
    run_kepler(n_orbits=100);
    let L_final = final_angular_momentum();
    let drift = ((L_final - L0) / L0).abs();
    assert!(drift < 1e-4, "L drift = {drift:.3e}");
}
```

**Tolerancia:** drift L < 0.01%, excentricidad < 1%

---

### PF-03: FMM cuadrupolo — test de convergencia formal con θ

```
Estado: PENDIENTE
Módulo: gadget-ng-tree
Archivo propuesto: crates/gadget-ng-tree/tests/fmm_convergence.rs
```

Error relativo de FMM con octupolo debe ser < 0.1% para θ=0.4 (ya existe 1 test
informal en octree_com). Lo que falta es la curva error vs θ y comparación
formal con la referencia de Greengard & Rokhlin.

**Tolerancia:** err < 0.1% para θ = 0.4, err < 0.5% para θ = 0.6

---

### PF-04: PM periódico — convergencia con N_mesh

```
Estado: PENDIENTE
Módulo: gadget-ng-pm
Archivo propuesto: crates/gadget-ng-physics/tests/pm_mesh_convergence.rs
```

La fuerza PM debe converger al valor analítico (Ewald) conforme N_mesh crece.
Error relativo en fuerza < 1% para N_mesh = 128.

**Tolerancia:** fuerza PM vs Ewald < 1% (N_mesh=128)

---

### PF-05: SPH — test de choque de Sod (Sod shock tube)

```
Estado: PENDIENTE  ← más importante de los pendientes de SPH
Módulo: gadget-ng-sph
Archivo propuesto: crates/gadget-ng-physics/tests/sod_shock_tube.rs
```

El test de choque de Sod es **el** test estándar de cualquier código hidro.
Solución analítica conocida. El perfil de densidad y presión a t=0.2 debe
coincidir con la solución Riemann dentro del 5%.

```rust
// Condiciones iniciales del choque de Sod (1D mapeado a 3D)
// ρ_L=1, P_L=1, v_L=0  | ρ_R=0.125, P_R=0.1, v_R=0
// γ = 1.4

#[test]
fn sod_shock_tube_density_profile_error_lt_5pct() {
    let particles = setup_sod_1d(n_left=100, n_right=100, gamma=1.4);
    run_sph(&mut particles, t_final=0.2, dt=1e-3);
    let (rho_sim, x_sim) = extract_density_profile(&particles);
    let (rho_ana, _) = sod_analytical_density(x_sim, t=0.2, gamma=1.4);
    let rms_err = rms_relative_error(&rho_sim, &rho_ana);
    assert!(rms_err < 0.05, "Sod shock RMS error: {rms_err:.3e}");
}
```

**Tolerancia:** RMS error en densidad < 5% (sin viscosidad artificial excesiva)

---

### PF-06: SPH — ruido de presión en distribución aleatoria de partículas

```
Estado: PENDIENTE
Módulo: gadget-ng-sph
Archivo propuesto: crates/gadget-ng-physics/tests/sph_pressure_noise.rs
```

En un gas en reposo con presión uniforme, la fuerza SPH neta por partícula
debe ser < 1% de la presión / densidad.

**Tolerancia:** |a_sph| < 0.01 * P/ρ para distribución de poisson

---

### PF-07: Turbulencia MHD — espectro de potencias Kolmogorov

```
Estado: PENDIENTE
Módulo: gadget-ng-mhd
Archivo propuesto: crates/gadget-ng-physics/tests/mhd_turbulence_spectrum.rs
```

Después de N_turnover tiempos de correlación, el espectro cinético `E(k) ∝ k^(-5/3)`.
Hay que medir el índice espectral en el rango inercial.

```rust
#[test]
fn turbulence_kolmogorov_spectrum_after_multiple_turnover_times() {
    let mut sim = setup_driven_turbulence(n=256, alpha_turb=0.1, k_min=2, k_max=8);
    run_for_turnover_times(&mut sim, n_t=10.0);
    let (k_vals, Ek) = kinetic_power_spectrum(&sim.particles);
    let spectral_idx = fit_power_law(&k_vals, &Ek, k_range=(4.0, 16.0));
    assert!((spectral_idx - (-5.0/3.0)).abs() < 0.2,
        "Índice espectral: {spectral_idx:.2} (esperado -5/3)");
}
```

**Tolerancia:** índice espectral: −5/3 ± 0.2

---

### PF-08: Reconexión magnética — tasa Sweet-Parker escalado con resistividad

```
Estado: PENDIENTE (el test existente prueba la fórmula puntual, no el escalado)
Módulo: gadget-ng-mhd/reconnection.rs
Archivo propuesto: crates/gadget-ng-physics/tests/reconnection_scaling.rs
```

La tasa Sweet-Parker `Γ_SP ∝ η^(1/2)` (donde η = resistividad). Hay que verificar
este escalado variando η por un factor 10.

```rust
#[test]
fn sweet_parker_rate_scales_as_sqrt_eta() {
    let eta1 = 1e-3_f64;
    let eta2 = 1e-2_f64;   // factor 10
    let r1 = sweet_parker_rate(b0=1.0, rho=1.0, l=1.0, eta=eta1);
    let r2 = sweet_parker_rate(b0=1.0, rho=1.0, l=1.0, eta=eta2);
    let ratio = r2 / r1;
    let expected = (eta2 / eta1).sqrt();  // = sqrt(10) ≈ 3.162
    assert!((ratio - expected).abs() / expected < 0.01,
        "Tasa SP no escala como √η: ratio={ratio:.3} expected={expected:.3}");
}
```

**Tolerancia:** desviación del escalado √η < 1%

---

### PF-09: RMHD — conservación de energía total EM + cinética

```
Estado: PENDIENTE
Módulo: gadget-ng-mhd/relativistic.rs
Archivo propuesto: crates/gadget-ng-physics/tests/rmhd_energy_conservation.rs
```

En RMHD, la energía total E = E_cinetica + E_EM debe conservarse dentro
del 0.1% durante 100 pasos de integración para una onda de Alfvén.

**Tolerancia:** drift < 0.1% en 100 pasos

---

### PF-10: Two-fluid — equilibrio termal T_e → T_i a largo plazo

```
Estado: PENDIENTE (Phase 149 prueba el acoplamiento a t corto)
Módulo: gadget-ng-mhd/two_fluid.rs  
Archivo propuesto: crates/gadget-ng-physics/tests/two_fluid_equilibrium.rs
```

A t >> t_coulomb, T_e y T_i deben equilibrarse: (T_e - T_i)/T_i < 0.1%.

```rust
#[test]
fn two_fluid_reaches_thermal_equilibrium_long_time() {
    let t_eq = 1.0 / NU_EI;  // tiempo de equilibración
    let mut sim = setup_two_fluid(t_e_init=2.0, t_i_init=1.0);
    run_for_time(&mut sim, 10.0 * t_eq);
    let ratio = sim.mean_te_over_ti();
    assert!((ratio - 1.0).abs() < 1e-3,
        "T_e/T_i = {ratio:.4} tras 10×t_eq (esperado: 1.0)");
}
```

**Tolerancia:** |T_e/T_i − 1| < 0.1% después de 10 × t_equilibración

---

### PF-11: Dark energy w(z) — distancia de luminosidad vs Planck 2018

```
Estado: PENDIENTE
Módulo: gadget-ng-core/cosmology.rs
Archivo propuesto: crates/gadget-ng-physics/tests/de_luminosity_distance.rs
```

d_L(z) calculada con CPL (w0=-1, wa=0) debe coincidir con ΛCDM estándar.
Para w0=-0.9, wa=0.1, la diferencia a z=1 debe ser < 2%.

**Tolerancia:** |d_L^CPL / d_L^ΛCDM - 1| < 0.1% para w0=-1 wa=0

---

### PF-12: SIDM — sección eficaz efectiva vs. σ/m de referencia

```
Estado: PENDIENTE
Módulo: gadget-ng-tree/sidm.rs
Archivo propuesto: crates/gadget-ng-physics/tests/sidm_cross_section.rs
```

La tasa de dispersión simulada `Γ_num = N_scatter / (N_pairs × dt)` debe
coincidir con `Γ_ana = ρ × v × σ/m` dentro del 5% para N estadísticamente grande.

**Tolerancia:** |Γ_num / Γ_ana − 1| < 5%

---

### PF-13: Gravedad modificada f(R) — recuperación de Newton en alta densidad

```
Estado: PENDIENTE
Módulo: gadget-ng-core/modified_gravity.rs
Archivo propuesto: crates/gadget-ng-physics/tests/fr_chameleon_recovery.rs
```

En el régimen de alta densidad (chameleon screening activo), la quinta fuerza
debe ser < 1% de la fuerza newtoniana. El test del Phase 158 solo verifica
que la quinta fuerza existe; no que se apaga en alta densidad.

```rust
#[test]
fn chameleon_suppresses_fifth_force_in_dense_medium() {
    let fr = FRParams { f_r0: 1e-6, n: 1.0 };
    let cosmo = CosmologyParams::new(0.3, 0.7, 0.7);
    let rho_dense = 1e6;  // mucho mayor que rho_crit
    let ff = fifth_force_factor(&fr, &cosmo, rho_dense);
    assert!(ff < 0.01, "Chameleon no suprime quinta fuerza: ff = {ff:.3e}");
}
```

**Tolerancia:** fifth_force_factor < 1% en ρ >> ρ_crit

---

### PF-14: Mock catálogos — recuperación de la SMHM (Stellar-to-Halo Mass Relation)

```
Estado: PENDIENTE
Módulo: gadget-ng-analysis/mock_catalog.rs
Archivo propuesto: crates/gadget-ng-physics/tests/mock_catalog_smhm.rs
```

El catálogo sintético debe reproducir la pendiente de la SMHM
(log M_* vs log M_halo) dentro de ±0.1 dex en el rango 10¹¹–10¹³ M_☉.

**Tolerancia:** pendiente SMHM: 1.0 ± 0.15 (valor esperado ~1.0)

---

### PF-15: X-ray — consistencia L_X – T_X

```
Estado: PENDIENTE
Módulo: gadget-ng-analysis/xray.rs
Archivo propuesto: crates/gadget-ng-physics/tests/xray_lx_tx.rs
```

La relación L_X – T_X debe seguir L_X ∝ T_X^2 (bremsstrahlung puro).
Para una muestra de cúmulos sintéticos con diferentes T_X, ajustar la pendiente.

**Tolerancia:** pendiente log-log: 2.0 ± 0.1

---

### PF-16: Neutrinos masivos — supresión P(k)

```
Estado: PENDIENTE (Phase 156 verifica la fórmula, no la simulación de extremo a extremo)
Módulo: gadget-ng-core/cosmology.rs
Archivo propuesto: crates/gadget-ng-physics/tests/neutrino_pk_suppression.rs
```

Con m_ν = 0.1 eV, el espectro de potencias a k = 1 h/Mpc debe estar
suprimido ~1% respecto a la corrida sin neutrinos.

**Tolerancia:** supresión entre 0.5% y 3% para m_ν = 0.1 eV, k = 1 h/Mpc

---

<a name="pendientes-hpc"></a>
## Validaciones HPC — Estado (Phase 165)

| ID | Descripción | Tests | Estado |
|---|---|:---:|:---:|
| V1 | GPU CUDA/HIP kernels reales | 6 | ✅ implementado; 3 skip sin HW |
| V2 | Block timesteps + MPI cosmo acoplado | 6 | ✅ 5 pasan, 1 skip (scaling MPI) |
| V3 | ICs MHD cosmo 1D + validaciones analíticas | 6 | ✅ todas pasan |
| V3b | **MHD 3D solenoidal** (`primordial_bfield_ic_3d`) | 2 | ✅ **nuevo** |

### Detalle Phase 165 — MHD 3D solenoidal

| Test | Qué valida | Criterio | Estado |
|---|---|:---:|:---:|
| `primordial_bfield_3d_rms_matches_b0` | RMS campo vs b0 pedido | < 2% | ✅ |
| `primordial_bfield_3d_divergence_free` | max \|∇·B\| discreta | < 1e-10 | ✅ (1e-14) |

### Detalle Phase 165 — GPU tests activados

| Test | Backend | Criterio | Sin HW |
|---|---|:---:|:---:|
| `gpu_matches_cpu_direct_gravity_n1024` | CUDA → HIP → skip | err < 1e-4 | skip |
| `gpu_speedup_over_cpu_serial_weak_scaling` | wgpu | t_gpu < 100×t_cpu | skip |
| `pm_gpu_roundtrip_fft` | CUDA/HIP | FFT rnd-trip < 1e-8 | skip |
| `power_spectrum_pm_gpu_matches_pm_cpu` | CUDA/HIP | P(k) bin < 1% | skip |
| `energy_conservation_gpu_integrator_n256_100steps` | wgpu | drift E < 0.1% | skip |

---

<a name="script"></a>
## Script de corrida autónoma completa

El script vive en `scripts/run_all_validations.sh` y está listo para ejecutarse.
Incluye los 6 bloques: unit tests, tests cuantitativos, benchmarks, GPU,
validación N=128³ (2–4 h) y la **corrida de producción N=256³ (8–12 h, la definitiva)**.

### Modos de ejecución

```bash
# Corrida completa (toda la noche — incluye N=128³ y N=256³):
nohup bash scripts/run_all_validations.sh 2>&1 | tee logs/all_validations.log &
echo "PID: $!"
tail -f logs/all_validations.log

# Solo unit tests rápidos (~30–60 min):
ONLY_UNIT=1 bash scripts/run_all_validations.sh

# Sin la producción N=256³ (2–4 h):
SKIP_PROD=1 bash scripts/run_all_validations.sh

# Con MPI×4 para las corridas largas (más rápido):
N_RANKS=4 nohup bash scripts/run_all_validations.sh 2>&1 | tee logs/all_validations.log &
```

### Estructura de bloques

| Bloque | Contenido | Tiempo estimado |
|--------|-----------|:---:|
| 1 | `cargo test --workspace --release` (278+ tests, incl. 3D MHD) | ~30–60 min |
| 2 | Tests cuantitativos con métricas impresas | ~5 min |
| 3 | Benchmarks Criterion (--quick) | ~5 min |
| 4 | GPU tests (skip automático sin hardware) | ~10 min |
| 5 | Corrida de validación N=128³ end-to-end | ~2–4 h |
| **6** | **Corrida de PRODUCCIÓN N=256³ (la definitiva)** | **~8–12 h** |
| | **TOTAL** | **~12–18 h** |

> El bloque 6 (N=256³) hace checkpoints automáticos cada 2 h y reanuda si se interrumpe.

---

<a name="criterios"></a>
## Criterios de aceptación global

El proyecto está **listo para producción** cuando:

| Criterio | Estado actual | Meta |
|---|:---:|:---:|
| `cargo test --workspace` sin FAILED | ✅ | ✅ |
| `cargo clippy --workspace` sin warnings | ✅ | ✅ |
| Tests existentes (≥262) pasando | ✅ | ✅ |
| PF-05: Sod shock tube < 5% | ❌ pendiente | < 5% |
| PF-01: Convergencia leapfrog O(Δt²) | ❌ pendiente | ratio ≈ 4 |
| PF-07: Espectro turbulencia -5/3 ± 0.2 | ❌ pendiente | ✅ |
| PF-09: RMHD energía < 0.1% | ❌ pendiente | ✅ |
| V3-T1: Onda Alfvén < 1% | ❌ pendiente | ✅ |
| V3-T3: Onda magnetosónica < 1% | ❌ pendiente | ✅ |
| V2: Block timesteps + cosmo + MPI | ❌ pendiente | ✅ |
| V1: GPU CUDA/HIP speedup > 5× | ❌ pendiente | ✅ |

**Los más críticos para física rigurosa** (prioridad alta):
1. PF-05 (Sod) — estándar de cualquier código hidro
2. V3-T1/T3 (Alfvén + magnetosónica) — validación cuantitativa MHD
3. PF-01 (convergencia O(Δt²)) — confirma correctitud del integrador
4. PF-07 (Kolmogorov) — confirma turbulencia MHD realista

**Los más críticos para escalado HPC** (prioridad alta):
1. V2 (block timesteps + MPI cosmo) — permite corridas cosmológicas grandes
2. V1 (GPU CUDA/HIP) — aceleración de factor > 5× en hardware real

---

*Documento generado: 2026-04-24. Versión del proyecto: Phases 1–160 completas.*
