# AP-16 — Cierre completo de CUDA (2026-05-16)

## Resumen ejecutivo

Este reporte documenta la implementación y validación en hardware de los 6
últimos módulos CUDA pendientes: 3 kernels nuevos y 3 wirings de kernels
existentes sin conexión al motor de stepping/análisis.

Hardware: NVIDIA GTX 1060, sm_61, CUDA 12.4.  
Todos los tests `cargo test -p gadget-ng-cuda -- --ignored` pasan.

---

## Parte 1 — Kernels nuevos

### 1a. RT IGM temperatura (`rt_igm_temp_kernel`)

**Archivo:** `crates/gadget-ng-cuda/cuda/rt_kernels.cu`  
**Wrapper Rust:** `CudaRtSolver::try_igm_temp_profile`  
**Engine hook:** `analyze_cmd.rs` bajo `params.cuda_analysis`

Reducción paralela en dos pasadas (atomicAdd sobre doubles):
- Filtro IGM: `rho_sph = mass/h³ < delta_max × mean_density`.
- μ calculada usando los mismos campos almacenados en `ChemState` que la CPU:
  `free_h = x_hi + x_hii + x_e`, `deut = x_d`, `he_particles = x_hei + x_heii + x_heiii`.
- T = u_cgs × (γ-1) × μ × m_p / k_B.
- Salida: `t_mean`, `t_sigma` (exactas), `t_median` = t_mean (aproximación; sort en GPU no implementado), `n_particles`.

**Resultado hardware:**
```
cuda_rt_igm_temp_match_cpu ... ok
  t_mean GPU vs CPU: rel < 2%
  t_sigma GPU vs CPU: rel < 10%
  n_particles: coincide (sin filtro densidad en el test de paridad)
```

### 1b. MHD difusión ambipolar (`mhd_ambipolar_kernel`)

**Archivo:** `crates/gadget-ng-cuda/cuda/mhd_kernels.cu`  
**Wrapper Rust:** `CudaMhdSolver::try_ambipolar_diffusion`  
**Engine hook:** `context.rs::step_mhd` bajo `[accelerators] cuda_mhd = true`

Replica `gadget_ng_mhd::apply_ambipolar_diffusion` por partícula:
- `x_i = collisional × dust_suppression` (proxy ionización)
- `rate = η_ad × (1/x_i − 1)`
- `B *= exp(-rate × dt)`, `u += heat_eff × ΔB²/2m`

**Resultado hardware:**
```
cuda_mhd_ambipolar_match_cpu ... ok
  b_field.x rel: < 1% para todos los 256 test particles
```

### 1c. MHD two-fluid acoplamiento e-i (`mhd_two_fluid_kernel`)

**Archivo:** `crates/gadget-ng-cuda/cuda/mhd_kernels.cu`  
**Wrapper Rust:** `CudaMhdSolver::try_electron_ion_coupling`  
**Engine hook:** `context.rs::step_sph` bajo `[accelerators] cuda_mhd = true`

Replica `gadget_ng_mhd::apply_electron_ion_coupling` por partícula:
- `ν_ei = ν_coeff × ρ / T_e^{3/2}`
- `T_e += (T_i − T_e) × (1 − exp(−ν_ei × dt))`

**Resultado hardware:**
```
cuda_mhd_two_fluid_match_cpu ... ok
  t_electron rel: < 2% para todos los 256 test particles
```

---

## Parte 2 — Wiring de kernels existentes

### 2a. RT 21cm en engine (`step_reionization`)

`CudaRtSolver::try_cm21_field` ya existía y tenía test. Se cableó en
`step_reionization` bajo `[accelerators] cuda_rt_chem = true`. El resultado
se descarta actualmente (log solamente); integración con `InsituResult` pendiente.

### 2b. Conducción anisótropa / difusión CR (`step_sph`)

`CudaMhdSolver::try_scalar_diffusion` (campo-medio) se cableó como fallback CUDA
para `apply_anisotropic_conduction` y `diffuse_cr_anisotropic` en `step_sph`
bajo `[accelerators] cuda_mhd = true`.

**Nota:** el kernel CUDA usa difusión de campo-medio (no pares SPH), por lo que
es una aproximación. La precisión es menor que el path CPU para configuraciones
de campo muy inhomogéneas. Documentado en comentarios del código.

### 2c. Análisis CUDA en CLI (`analyze_cmd.rs`)

- Nuevo campo `[accelerators] cuda_analysis = true` en `AcceleratorsSection`.
- `analyze_cmd::AnalyzeParams` tiene campo `cuda_analysis: bool` (default false).
- `--igm-temp`: intenta `CudaRtSolver::try_igm_temp_profile` antes del path CPU.
- `--luminosity`: intenta `CudaAnalysisSolver::try_galaxy_luminosity` antes del path CPU.

---

## Parte 3 — Tests hardware

Suite completa ejecutada en GTX 1060 sm_61:

```
cargo test -p gadget-ng-cuda -- --ignored
```

**Resultado total:** 36 tests, 0 failed.

### Tests nuevos añadidos en AP-16

| Test | Archivo | Resultado |
|------|---------|-----------|
| `cuda_rt_igm_temp_match_cpu` | `cuda_rt_smoke.rs` | ✅ ok |
| `cuda_mhd_ambipolar_match_cpu` | `cuda_mhd_smoke.rs` | ✅ ok |
| `cuda_mhd_two_fluid_match_cpu` | `cuda_mhd_smoke.rs` | ✅ ok |

---

## Parte 4 — Resumen de estado CUDA tras AP-16

| Módulo | Kernel | Estado |
|--------|--------|--------|
| Gravedad directa | `cuda_direct_*` | ✅ producción |
| PM solver | `cuda_pm_*` | ✅ producción |
| Tree monopolar | `cuda_tree_*` | ✅ producción |
| SPH densidad/fuerzas | `cuda_sph_*` | ✅ producción |
| SPH dust/H2/cooling | `cuda_dust/h2/cooling_*` | ✅ validado hw |
| MHD inducción/fuerzas | `mhd_induction_*` | ✅ validado hw |
| MHD flux-freeze/stats | `mhd_flux_freeze/density/b_stats` | ✅ validado hw |
| MHD Braginskii | `mhd_braginskii_kernel` | ⚠️ smoke/parity |
| MHD reconexión/dynamo | `mhd_reconnection_*` | ⚠️ smoke/parity |
| MHD ambipolar | `mhd_ambipolar_kernel` | ✅ validado hw (AP-16) |
| MHD two-fluid | `mhd_two_fluid_kernel` | ✅ validado hw (AP-16) |
| MHD anisotropic/CR diff | `mhd_scalar_diffusion` | ⚠️ aprox campo-medio (AP-16) |
| MHD CR streaming | `mhd_cr_streaming_o2_kernel` | ✅ validado hw |
| MHD CR backreaction | `mhd_cr_backreaction_kernel` | ✅ validado hw |
| RT M1 diagnostics/adv | `rt_*` | ✅ validado hw |
| RT chemistry rates | `rt_chemistry_rates_kernel` | ✅ validado hw |
| RT chemistry stiff | `rt_chemistry_stiff_kernel` | ✅ validado hw |
| RT reionization | `rt_reionization_stats_kernel` | ✅ validado hw |
| RT 21cm | `rt_cm21_field_kernel` | ✅ validado hw + wired (AP-16) |
| RT IGM temperatura | `rt_igm_temp_kernel` | ✅ validado hw (AP-16) |
| Analysis halo spin | `cuda_analysis_halo_spin` | ⚠️ smoke/parity |
| Analysis luminosity | `cuda_analysis_luminosity` | ⚠️ smoke/parity + wired (AP-16) |
| Analysis X-ray | `cuda_analysis_xray` | ⚠️ smoke/parity |

---

## Gaps restantes

1. **MHD anisotropic conduction/CR diffusion CUDA** — el kernel actual es campo-medio
   (sin pares SPH). Para precisión completa se requiere un kernel O(N²) similar
   a `mhd_cr_streaming_o2_kernel`. Pendiente en backlog.
2. **t_median/percentiles IGM en GPU** — sort GPU (thrust/cub) o histograma
   adaptativo. Actualmente t_median = t_mean.
3. **MHD Braginskii + reconexión SIMD-without-Rayon** — gaps pendientes en
   `docs/reports/2026-05-clippy-all-targets-backlog.md`.
4. **cuda_analysis halo spin + X-ray** — wired en `CudaAnalysisSolver` pero
   no conectados a `analyze_cmd.rs`; pendiente en próximo sprint.
